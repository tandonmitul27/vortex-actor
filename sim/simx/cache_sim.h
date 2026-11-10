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

#pragma once

#include <simobject.h>
#include "mem_sim.h"

namespace vortex {

class CacheSim : public SimObject<CacheSim> {
public:
	struct Config {
		bool    bypass;         // cache bypass
		uint8_t C;              // log2 cache size
		uint8_t L;              // log2 line size
		uint8_t W;              // log2 word size
		uint8_t A;              // log2 associativity
		uint8_t B;              // log2 number of banks
		uint8_t addr_width;     // word address bits
		uint8_t num_inputs;     // number of inputs
		uint8_t mem_ports;      // memory ports
		bool    write_back;     // is write-back
		bool    write_reponse;  // enable write response
		uint16_t mshr_size;     // MSHR buffer size
		uint8_t latency;        // pipeline latency
		bool    write_nofetch;  // P4: on a write miss, ALLOCATE the line without
		                        // fetching it from memory. Legal because this
		                        // cache tracks dirty bytes when it is write-back
		                        // (L2_DIRTYBYTES in VX_config.vh), which is what
		                        // lets a line be valid in only the bytes written.
		                        // Write-back only; ignored otherwise.
		bool    q_oracle;       // ORACLE PROBE, NOT A MACHINE. Queue storage is
		                        // answered by this cache unconditionally: never a
		                        // miss, never allocated, never evicted, never sent
		                        // to memory. It prices the CEILING of P4, what a
		                        // transport that stays on chip is worth, with no
		                        // capacity assumption at all. Everything else in
		                        // the kernel keeps the real cache it had.
		bool    l1skip_ops = false;   // ISA2, the L1: requests marked l1skip take the uncached path
		bool    wack_sink = false;    // ISA2, the L2: each write it performs acknowledges its core (wack_done)
	};

	struct PerfStats {
		uint64_t reads;
		uint64_t writes;
		uint64_t read_misses;
		uint64_t write_misses;
		uint64_t evictions;
		uint64_t write_nofetch_allocs;
		uint64_t q_oracle_hits;
		uint64_t bank_stalls;
		uint64_t mshr_stalls;
		uint64_t mem_latency;

		PerfStats()
			: reads(0)
			, writes(0)
			, read_misses(0)
			, write_misses(0)
			, evictions(0)
			, write_nofetch_allocs(0)
			, q_oracle_hits(0)
			, bank_stalls(0)
			, mshr_stalls(0)
			, mem_latency(0)
		{}

		PerfStats& operator+=(const PerfStats& rhs) {
			this->reads += rhs.reads;
			this->writes += rhs.writes;
			this->read_misses += rhs.read_misses;
			this->write_misses += rhs.write_misses;
			this->evictions += rhs.evictions;
			this->write_nofetch_allocs += rhs.write_nofetch_allocs;
			this->q_oracle_hits += rhs.q_oracle_hits;
			this->bank_stalls += rhs.bank_stalls;
			this->mshr_stalls += rhs.mshr_stalls;
			this->mem_latency += rhs.mem_latency;
			return *this;
		}
	};

	std::vector<SimPort<MemReq>> CoreReqPorts;
	std::vector<SimPort<MemRsp>> CoreRspPorts;
	std::vector<SimPort<MemReq>> MemReqPorts;
	std::vector<SimPort<MemRsp>> MemRspPorts;

	CacheSim(const SimContext& ctx, const char* name, const Config& config);
	~CacheSim();

	void reset();

	void tick();

	PerfStats perf_stats() const;

	void invalidate_all();   // ISA2 L1INV: every valid line not waiting on a fill becomes invalid

private:
	class Impl;
	Impl* impl_;
};

}