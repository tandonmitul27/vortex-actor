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

#include <cstdlib>
#include "cluster.h"

using namespace vortex;

Cluster::Cluster(const SimContext& ctx,
                 uint32_t cluster_id,
                 ProcessorImpl* processor,
                 const Arch &arch,
                 const DCRS &dcrs)
  : SimObject(ctx, StrFormat("cluster%d", cluster_id))
  , mem_req_ports(L2_MEM_PORTS, this)
  , mem_rsp_ports(L2_MEM_PORTS, this)
  , cluster_id_(cluster_id)
  , processor_(processor)
  , sockets_(NUM_SOCKETS)
  , barriers_(arch.num_barriers(), 0)
  , cores_per_socket_(arch.socket_size())
{
  char sname[100];

  uint32_t sockets_per_cluster = sockets_.size();

  // create sockets

  for (uint32_t i = 0; i < sockets_per_cluster; ++i) {
    uint32_t socket_id = cluster_id * sockets_per_cluster + i;
    sockets_.at(i) = Socket::Create(socket_id, this, arch, dcrs);
  }

  // Create l2cache

  snprintf(sname, 100, "%s-l2cache", this->name().c_str());
  CacheSim::Config l2cfg{
    !L2_ENABLED,
    log2ceil(L2_CACHE_SIZE),// C
    log2ceil(MEM_BLOCK_SIZE),// L
    log2ceil(L1_LINE_SIZE), // W
    log2ceil(L2_NUM_WAYS),  // A
    log2ceil(L2_NUM_BANKS), // B
    XLEN,                   // address bits
    L2_NUM_REQS,            // request size
    L2_MEM_PORTS,           // memory ports
    L2_WRITEBACK,           // write-back
    false,                  // write response
    L2_MSHR_SIZE,           // mshr size
    2,                      // pipeline latency
    // P4 on chip transport: allocate a line on a write miss without fetching it.
    // Env knob so a policy sweep needs no rebuild. Meaningful only with
    // L2_WRITEBACK=1, which is a legal machine here because the L2 is the
    // coherence point (the per core L1s stay write through).
    (L2_WRITEBACK != 0) && (getenv("VORTEX_L2_NOFETCH") != nullptr)
                        && (getenv("VORTEX_L2_NOFETCH")[0] == '1'),
    // T38 ceiling probe. Queue storage never leaves the L2. Not a machine, and
    // it needs no write-back, since it removes the traffic rather than moving it.
    (getenv("VORTEX_L2_QORACLE") != nullptr) && (getenv("VORTEX_L2_QORACLE")[0] == '1'),
  };
  l2cfg.wack_sink = isa2_on() && L2_ENABLED;   // ISA2: the L2 acknowledges the writes it performs
  if (l2cfg.wack_sink) wack_enable();
  l2cache_ = CacheSim::Create(sname, l2cfg);
  if (isa2_on()) fprintf(stderr, "[ISA2] append+count instructions, L1-skipping atomics, acked fence, L1INV; queue layout by the program\n");
  // An ablation that changes nothing is void (see the ablation-validity note),
  // so both L2 policies announce themselves and the arm log carries the proof.
  if (getenv("VORTEX_L2_NOFETCH") && getenv("VORTEX_L2_NOFETCH")[0] == '1')
    fprintf(stderr, "[L2 POLICY] write_nofetch on queue storage ACTIVE (writeback=%d)\n", (int)L2_WRITEBACK);
  if (getenv("VORTEX_L2_QORACLE") && getenv("VORTEX_L2_QORACLE")[0] == '1')
    fprintf(stderr, "[L2 POLICY] queue-storage ORACLE ACTIVE, size=%u bytes ways=%u\n",
            (unsigned)L2_CACHE_SIZE, (unsigned)L2_NUM_WAYS);
  // The machine this libsimx was BUILT for. A kernel built for one shape and run
  // on a simulator built for another does not merely mis time, it can drop every
  // append and still print a plausible cycle count. Every arm log now carries the
  // simulator's own answer to "which machine am I".
  if (cluster_id == 0)
    fprintf(stderr, "[MACHINE] cores=%u warps=%u threads=%u L2=%s size=%u writeback=%d\n",
            (unsigned)NUM_CORES, (unsigned)NUM_WARPS, (unsigned)NUM_THREADS,
            L2_ENABLED ? "on" : "off", (unsigned)L2_CACHE_SIZE, (int)L2_WRITEBACK);

  // connect l2cache core interfaces
  for (uint32_t i = 0; i < sockets_per_cluster; ++i) {
    for (uint32_t j = 0; j < L1_MEM_PORTS; ++j) {
      sockets_.at(i)->mem_req_ports.at(j).bind(&l2cache_->CoreReqPorts.at(i * L1_MEM_PORTS + j));
      l2cache_->CoreRspPorts.at(i * L1_MEM_PORTS + j).bind(&sockets_.at(i)->mem_rsp_ports.at(j));
    }
  }

  // connect l2cache memory interfaces
  for (uint32_t i = 0; i < L2_MEM_PORTS; ++i) {
    l2cache_->MemReqPorts.at(i).bind(&this->mem_req_ports.at(i));
    this->mem_rsp_ports.at(i).bind(&l2cache_->MemRspPorts.at(i));
  }
}

Cluster::~Cluster() {
  //--
}

void Cluster::reset() {
  for (auto& barrier : barriers_) {
    barrier.reset();
  }
}

void Cluster::tick() {
  //--
}

void Cluster::attach_ram(RAM* ram) {
  for (auto& socket : sockets_) {
    socket->attach_ram(ram);
  }
}

#ifdef VM_ENABLE
void Cluster::set_satp(uint64_t satp) {
  for (auto& socket : sockets_) {
    socket->set_satp(satp);
  }
}
#endif

bool Cluster::running() const {
  for (auto& socket : sockets_) {
    if (socket->running())
      return true;
  }
  return false;
}

int Cluster::get_exitcode() const {
  int exitcode = 0;
  for (auto& socket : sockets_) {
    exitcode |= socket->get_exitcode();
  }
  return exitcode;
}

void Cluster::barrier(uint32_t bar_id, uint32_t count, uint32_t core_id) {
  auto& barrier = barriers_.at(bar_id);

  auto sockets_per_cluster = sockets_.size();
  auto cores_per_socket = cores_per_socket_;

  uint32_t cores_per_cluster = sockets_per_cluster * cores_per_socket;
  uint32_t local_core_id = core_id % cores_per_cluster;
  barrier.set(local_core_id);

  DP(3, "*** Suspend core #" << core_id << " at barrier #" << bar_id);

  if (barrier.count() == (size_t)count) {
      // resume all suspended cores
      for (uint32_t s = 0; s < sockets_per_cluster; ++s) {
        for (uint32_t c = 0; c < cores_per_socket; ++c) {
          uint32_t i = s * cores_per_socket + c;
          if (barrier.test(i)) {
            DP(3, "*** Resume core #" << i << " at barrier #" << bar_id);
            sockets_.at(s)->resume(c);
          }
        }
      }
      barrier.reset();
    }
}

Cluster::PerfStats Cluster::perf_stats() const {
  PerfStats perf_stats;
  perf_stats.l2cache = l2cache_->perf_stats();
  return perf_stats;
}