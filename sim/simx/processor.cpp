// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "processor.h"
#include "processor_impl.h"
#include "emulator.h"
#include "core.h"
#include <unordered_map>
#include <cstdio>

// diagnostic (2026-11-16, Fig. 2b apply lines): VORTEX_MEM_PAGEHIST=<file> counts DRAM reads/writes per 4 KB page
// over the same window as perf_mem_reads_/perf_mem_writes_ (cleared by reset) and writes "page reads writes" lines
// to <file> when the processor is destroyed. Unset = nothing is counted.
static const char* g_page_hist_file = getenv("VORTEX_MEM_PAGEHIST");
static std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> g_page_hist;

// diagnostic (2026-11-16, Fig. 2b floor): VORTEX_EPOCH_RANGES=lo-hi[,lo-hi...] (hex byte ranges) counts, between global
// barrier releases, the distinct 64 B lines of those ranges that cores read and those they write (an atomic does both),
// summed over epochs: the lines that cross DRAM if each line touched in a round crosses once per round. Printed at exit.
#include <unordered_set>
#include <vector>
static std::vector<std::pair<uint64_t, uint64_t>> epoch_ranges() {
  std::vector<std::pair<uint64_t, uint64_t>> v; const char* e = getenv("VORTEX_EPOCH_RANGES");
  while (e && *e) { char* x; uint64_t lo = strtoull(e, &x, 16); uint64_t hi = strtoull(x + 1, &x, 16); v.push_back({lo, hi}); e = *x ? x + 1 : x; }
  return v;
}
static const std::vector<std::pair<uint64_t, uint64_t>> g_ep_ranges = epoch_ranges();
static std::unordered_set<uint64_t> g_ep_rd, g_ep_wr;
static uint64_t g_ep_n = 0, g_ep_rsum = 0, g_ep_wsum = 0;
void epoch_touch(uint64_t addr, uint32_t bytes, bool rd, bool wr) {
  if (g_ep_ranges.empty()) return;
  for (auto& r : g_ep_ranges) if (addr >= r.first && addr < r.second) {
    for (uint64_t l = addr >> 6; l <= (addr + bytes - 1) >> 6; ++l) { if (rd) g_ep_rd.insert(l); if (wr) g_ep_wr.insert(l); }
    return;
  }
}
void epoch_barrier() {
  if (g_ep_ranges.empty()) return;
  g_ep_rsum += g_ep_rd.size(); g_ep_wsum += g_ep_wr.size(); ++g_ep_n; g_ep_rd.clear(); g_ep_wr.clear();
}

using namespace vortex;

ProcessorImpl::ProcessorImpl(const Arch& arch)
  : arch_(arch)
  , clusters_(arch.num_clusters())
{
  SimPlatform::instance().initialize();

	assert(PLATFORM_MEMORY_DATA_SIZE == MEM_BLOCK_SIZE);

  // create memory simulator
  memsim_ = MemSim::Create("dram", MemSim::Config{
    PLATFORM_MEMORY_NUM_BANKS,
    L3_MEM_PORTS,
    MEM_BLOCK_SIZE,
    MEM_CLOCK_RATIO
  });

  // create clusters
  for (uint32_t i = 0; i < arch.num_clusters(); ++i) {
    clusters_.at(i) = Cluster::Create(i, this, arch, dcrs_);
  }

  // create L3 cache
  l3cache_ = CacheSim::Create("l3cache", CacheSim::Config{
    !L3_ENABLED,
    log2ceil(L3_CACHE_SIZE),  // C
    log2ceil(MEM_BLOCK_SIZE), // L
    log2ceil(L2_LINE_SIZE),   // W
    log2ceil(L3_NUM_WAYS),    // A
    log2ceil(L3_NUM_BANKS),   // B
    XLEN,                     // address bits
    L3_NUM_REQS,              // request size
    L3_MEM_PORTS,             // memory ports
    L3_WRITEBACK,             // write-back
    false,                    // write response
    L3_MSHR_SIZE,             // mshr size
    2,                        // pipeline latency
    }
  );

  // connect L3 core interfaces
  for (uint32_t i = 0; i < arch.num_clusters(); ++i) {
    for (uint32_t j = 0; j < L2_MEM_PORTS; ++j) {
      clusters_.at(i)->mem_req_ports.at(j).bind(&l3cache_->CoreReqPorts.at(i * L2_MEM_PORTS + j));
      l3cache_->CoreRspPorts.at(i * L2_MEM_PORTS + j).bind(&clusters_.at(i)->mem_rsp_ports.at(j));
    }
  }

  // connect L3 memory interfaces
  for (uint32_t i = 0; i < L3_MEM_PORTS; ++i) {
    l3cache_->MemReqPorts.at(i).bind(&memsim_->MemReqPorts.at(i));
    memsim_->MemRspPorts.at(i).bind(&l3cache_->MemRspPorts.at(i));
  }

  // set up memory profiling
  for (uint32_t i = 0; i < L3_MEM_PORTS; ++i) {
    memsim_->MemReqPorts.at(i).tx_callback([&](const MemReq& req, uint64_t cycle){
      perf_mem_reads_  += !req.write;
      perf_mem_writes_ += req.write;
      if (g_page_hist_file) { auto& e = g_page_hist[req.addr >> 12]; (req.write ? e.second : e.first) += 1; }
      // diagnostic: VORTEX_MEM_SAMPLE=<cycles> prints cumulative DRAM reads/writes over time
      static const uint64_t mem_sample = getenv("VORTEX_MEM_SAMPLE") ? strtoull(getenv("VORTEX_MEM_SAMPLE"), nullptr, 0) : 0;
      static uint64_t mem_next = 0;
      if (mem_sample && cycle >= mem_next) {
        printf("MEMS %lu %lu %lu\n", (unsigned long)cycle, (unsigned long)perf_mem_reads_, (unsigned long)perf_mem_writes_);
        mem_next = (cycle / mem_sample + 1) * mem_sample;
      }
      perf_mem_pending_reads_ += !req.write;
    });
    memsim_->MemRspPorts.at(i).tx_callback([&](const MemRsp&, uint64_t cycle){
      __unused (cycle);
      --perf_mem_pending_reads_;
    });
  }

#ifndef NDEBUG
  // dump device configuration
  std::cout << "CONFIGS:"
            << " num_threads=" << arch.num_threads()
            << ", num_warps=" << arch.num_warps()
            << ", num_cores=" << arch.num_cores()
            << ", num_clusters=" << arch.num_clusters()
            << ", socket_size=" << arch.socket_size()
            << ", local_mem_base=0x" << std::hex << arch.local_mem_base() << std::dec
            << ", num_barriers=" << arch.num_barriers()
            << std::endl;
#endif
  // reset the device
  this->reset();
}

ProcessorImpl::~ProcessorImpl() {
  if (!g_ep_ranges.empty()) {
    epoch_barrier();   // the last epoch (after the final barrier)
    fprintf(stderr, "EPOCH epochs=%lu read_lines=%lu write_lines=%lu\n", (unsigned long)g_ep_n, (unsigned long)g_ep_rsum, (unsigned long)g_ep_wsum);
  }
  if (g_page_hist_file) {
    if (FILE* f = fopen(g_page_hist_file, "w")) {
      for (auto& kv : g_page_hist) fprintf(f, "%lx %lu %lu\n", (unsigned long)kv.first, (unsigned long)kv.second.first, (unsigned long)kv.second.second);
      fclose(f);
    }
  }
  SimPlatform::instance().finalize();
}

void ProcessorImpl::attach_ram(RAM* ram) {
  for (auto cluster : clusters_) {
    cluster->attach_ram(ram);
  }
}
#ifdef VM_ENABLE
void ProcessorImpl::set_satp(uint64_t satp) {
  for (auto cluster : clusters_) {
    cluster->set_satp(satp);
  }
}
#endif

int ProcessorImpl::run() {
  SimPlatform::instance().reset();
  this->reset();

  bool done;
  int exitcode = 0;
  do {
    SimPlatform::instance().tick();
    done = true;
    for (auto cluster : clusters_) {
      if (cluster->running()) {
        done = false;
        continue;
      }
      exitcode |= cluster->get_exitcode();
    }
    perf_mem_latency_ += perf_mem_pending_reads_;
  } while (!done);

  return exitcode;
}

void ProcessorImpl::reset() {
  g_page_hist.clear();
  g_ep_rd.clear(); g_ep_wr.clear(); g_ep_n = g_ep_rsum = g_ep_wsum = 0;
  perf_mem_reads_ = 0;
  perf_mem_writes_ = 0;
  perf_mem_latency_ = 0;
  perf_mem_pending_reads_ = 0;
}

void ProcessorImpl::dcr_write(uint32_t addr, uint32_t value) {
  dcrs_.write(addr, value);
}

ProcessorImpl::PerfStats ProcessorImpl::perf_stats() const {
  ProcessorImpl::PerfStats perf;
  perf.mem_reads   = perf_mem_reads_;
  perf.mem_writes  = perf_mem_writes_;
  perf.mem_latency = perf_mem_latency_;
  perf.l3cache     = l3cache_->perf_stats();
  perf.memsim      = memsim_->perf_stats();
  return perf;
}

// Advance the simulation by one cycle for SST - code adapted from run() method
bool ProcessorImpl::cycle() {
  if (!is_cycle_initialized_) {
    std::cout << "ProcessorImpl: Initializing cycle()\n";
    SimPlatform::instance().reset();
    this->reset();
    is_cycle_initialized_ = true;
  }

  SimPlatform::instance().tick();
  bool anyRunning = false;
  for (auto cluster : clusters_) {
    if (cluster->running()) {
      anyRunning = true;
      break;
    }
  }
  perf_mem_latency_ += perf_mem_pending_reads_;
  return anyRunning;
}

Emulator* ProcessorImpl::get_first_emulator() const {
  if (clusters_.empty()) {
    return nullptr;
  }
  auto& cluster = clusters_.at(0);
  if (!cluster || cluster->sockets().empty()) {
    return nullptr;
  }
  auto& socket = cluster->sockets().at(0);
  if (!socket || socket->cores().empty()) {
    return nullptr;
  }
  auto& core = socket->cores().at(0);
  if (!core) {
    return nullptr;
  }
  return &core->emulator();
}

///////////////////////////////////////////////////////////////////////////////

Processor::Processor(const Arch& arch)
  : impl_(new ProcessorImpl(arch))
{
#ifdef VM_ENABLE
  satp_ = NULL;
#endif
}

Processor::~Processor() {
  delete impl_;
#ifdef VM_ENABLE
  if (satp_ != NULL)
    delete satp_;
#endif
}

void Processor::attach_ram(RAM* mem) {
  impl_->attach_ram(mem);
}

int Processor::run() {
  try {
    return impl_->run();
  } catch (const std::exception& e) {
    std::cerr << "Error: exception: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Error: unknown exception." << std::endl;
  }
  return -1;
}

void Processor::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}

// advance the simulation by one cycle for SST
bool Processor::cycle() {
  try {
    return impl_->cycle();
  } catch (const std::exception& e) {
    std::cerr << "Error: exception: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Error: unknown exception." << std::endl;
  }
  return false;
}

Emulator* Processor::get_first_emulator() const {
  return impl_->get_first_emulator();
}

#ifdef VM_ENABLE
int16_t Processor::set_satp_by_addr(uint64_t base_addr) {
  uint16_t asid = 0;
  satp_ = new SATP_t (base_addr,asid);
  if (satp_ == NULL)
    return 1;
  uint64_t satp = satp_->get_satp();
  impl_->set_satp(satp);
  return 0;
}
bool Processor::is_satp_unset() {
  return (satp_== NULL);
}
uint8_t Processor::get_satp_mode() {
  assert (satp_!=NULL);
  return satp_->get_mode();
}
uint64_t Processor::get_base_ppn() {
  assert (satp_!=NULL);
  return satp_->get_base_ppn();
}
#endif
