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

// Per-PC counters for fine-grained communication studies.
//
// Nothing is added to the kernel: each core counts per instruction address,
// and the analysis maps addresses to source lines through the kernel's debug
// info. Counting never feeds back into timing. Every counter lives in the core
// that updates it, so nothing is shared between simulation domains.
// VORTEX_FGC_STATS=<file> turns the counters on and writes them as JSON at the
// end of each run.
//
// Every cycle, each active warp is charged to exactly one PC and state:
//   issue  the warp issued a uop of this PC
//   scrb   the PC heads the warp's buffer and waits on the scoreboard
//   head   the PC heads the buffer and waits for anything else
//          (arbitration, a full unit queue, staging)
//   fetch  the buffer is empty and this PC is being fetched
//   wait   the buffer is empty and fetch is held by this PC, in flight
//          (a branch, a barrier or another warp-control op)
//   idle   the buffer is empty and the warp could fetch this PC but was not
//          picked
// so the states sum to the core's active warp-cycles.

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace vortex {
namespace fgc {

// Output file, or nullptr when the counters are off.
inline const char* out_path() {
  static const char* path = std::getenv("VORTEX_FGC_STATS");
  return path;
}

enum State : uint8_t { ST_NONE, ST_ISSUE, ST_SCRB };

struct PcStats {
  uint64_t issue = 0;       // warp-cycles issuing (= uops issued)
  uint64_t lanes = 0;       // active lanes over the issued uops
  uint64_t scrb = 0;        // warp-cycles at the head, blocked on the scoreboard
  uint64_t head = 0;        // warp-cycles at the head, blocked on anything else
  uint64_t fetch = 0;       // warp-cycles fetching this PC
  uint64_t wait = 0;        // warp-cycles with fetch held by this PC
  uint64_t idle = 0;        // warp-cycles ready to fetch this PC, not picked
  uint64_t commits = 0;     // uops committed
  uint64_t latency = 0;     // issue-to-commit cycles over the committed uops
  uint64_t disp = 0;        // cycles a collected uop waits for its unit's queue
};

using PcMap = std::unordered_map<uint64_t, PcStats>;

// Address tags name ranges of device memory. Tag 0 is everything untagged,
// tag 1 the thread stacks; the rest come from the file VORTEX_FGC_TAGS names,
// one "name base size" line per range, written by the host before launch.
// The table changes only between launches.
struct Tag {
  std::string name;
  uint64_t base;
  uint64_t size;
};

inline std::vector<Tag>& tags() {
  static std::vector<Tag> table;
  return table;
}

inline uint32_t tag_of(uint64_t addr) {
  auto& t = tags();
  for (uint32_t i = 1; i < t.size(); ++i) {
    if (addr - t[i].base < t[i].size) {
      return i;
    }
  }
  return 0;
}

// Requests through one side of a memory level: what arrives from above (in)
// or what leaves for the level below (out).
struct Flow {
  uint64_t rd = 0;
  uint64_t wr = 0;
  uint64_t wr_bytes = 0;  // enabled bytes over the writes
  uint64_t amo = 0;

  void count(bool is_amo, bool is_write, uint64_t byteen) {
    if (is_amo) {
      ++amo;
    } else if (is_write) {
      ++wr;
      wr_bytes += __builtin_popcountll(byteen);
    } else {
      ++rd;
    }
  }
  Flow& operator+=(const Flow& o) {
    rd += o.rd; wr += o.wr; wr_bytes += o.wr_bytes; amo += o.amo;
    return *this;
  }
};

enum Side { IN = 0, OUT = 1 };

// One memory level's requests by address tag and by the PC that issued them.
// A request carries the PC of its load or store from the LSU down to the
// fills it causes; write-backs carry none (PC 0).
struct MemStats {
  std::vector<std::array<Flow, 2>> by_tag;
  std::unordered_map<uint64_t, std::array<Flow, 2>> by_pc;

  void count(Side side, uint64_t addr, uint64_t pc, bool is_amo, bool is_write, uint64_t byteen) {
    uint32_t t = tag_of(addr);
    if (by_tag.size() <= t) {
      by_tag.resize(t + 1);
    }
    by_tag[t][side].count(is_amo, is_write, byteen);
    by_pc[pc][side].count(is_amo, is_write, byteen);
  }
  MemStats& operator+=(const MemStats& o) {
    if (by_tag.size() < o.by_tag.size()) {
      by_tag.resize(o.by_tag.size());
    }
    for (size_t t = 0; t < o.by_tag.size(); ++t) {
      by_tag[t][IN] += o.by_tag[t][IN];
      by_tag[t][OUT] += o.by_tag[t][OUT];
    }
    for (auto& [pc, f] : o.by_pc) {
      auto& d = by_pc[pc];
      d[IN] += f[IN];
      d[OUT] += f[OUT];
    }
    return *this;
  }
};

} // namespace fgc
} // namespace vortex
