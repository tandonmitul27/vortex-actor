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

#include "types.h"
#include <list>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <cstdlib>
#include <cstdio>

using namespace vortex;

namespace vortex {
// Modeled on-chip message memory window (see types.h). Read once from the
// environment; size 0 leaves the model disabled and behaviour unchanged.
static uint64_t env_u64(const char* name) {
  const char* s = getenv(name);
  return s ? strtoull(s, nullptr, 0) : 0;
}
uint64_t g_msgmem_base = env_u64("VORTEX_MSGMEM_BASE");
uint64_t g_msgmem_size = env_u64("VORTEX_MSGMEM_SIZE");
// Remote scratchpad push window (see types.h). Size 0 disables.
uint64_t g_rpush_base = env_u64("VORTEX_RPUSH_BASE");
uint64_t g_rpush_size = env_u64("VORTEX_RPUSH_SIZE");
// Combining append queue window (see types.h). Size 0 disables.
uint64_t g_aq_base = env_u64("VORTEX_AQ_BASE");
uint64_t g_aq_size = env_u64("VORTEX_AQ_SIZE");
AqCtrl g_aq_ctrl = {0, 0, 0, 0, 0};
AqPool g_aq_pool = {0, 0, 0, 0, 0, 0};
std::unordered_map<uint32_t, std::vector<uint64_t>> g_aq_ovf;
uint64_t g_aq_ovf_chunks = 0;
std::vector<uint32_t> g_aq_tail;
std::vector<uint32_t> g_aq_fill;
uint64_t g_aq_appends = 0;
uint64_t g_aq_dropped = 0;
uint64_t g_aq_flushes = 0;
uint64_t g_aq_tail_flushes = 0;
std::vector<uint32_t> g_aq_flushed;
// High water marks. A chunked round (P3) empties every queue between chunks, so
// the live fill array is zero at exit and the report would say no queue was ever
// used. These survive the reset and are what the report prints.
uint64_t g_aq_used_hw = 0;
uint64_t g_aq_maxfill_hw = 0;
// DIAGNOSTIC knob, default 3: bit 0 charges the append as a scratchpad access,
// bit 1 issues the combined line writes. 0 makes appends free (ceiling).
static uint64_t g_aq_charge = getenv("VORTEX_AQ_CHARGE") ? env_u64("VORTEX_AQ_CHARGE") : 3;
// VORTEX_AQ_FINAL (2026-11-03, L2_APPEND.md "Final mechanism"): the decided L2 append is the DEFAULT
// whenever the L2-append model runs (AQ_CHARGE bit 2) without software queues (VORTEX_SWQ=1 = no unit):
// AQ_SET + AQ_HASH + AQ_SPILL=l2 + a pinned counter area of 2 B per queue used (sized as queues appear).
// VORTEX_AQ_FINAL=0 turns it off; an explicit VORTEX_AQ_SET / _HASH / _SPILL / _PIN_BYTES always wins.
// Reads only the environment (a function-local static), so other files' static initialisers may call it.
bool aq_final_on() {
  static const bool on = [] {
    const char* c = getenv("VORTEX_AQ_CHARGE"); const char* s = getenv("VORTEX_SWQ"); const char* f = getenv("VORTEX_AQ_FINAL");
    return c && (strtoull(c, nullptr, 0) & 4) && !(s && s[0] == '1') && !(f && f[0] == '0');
  }();
  return on;
}
uint64_t g_aq_qhw = 0;   // highest queue number appended to, + 1: sizes the auto pinned counter area
// ISA2 (see types.h). Function-local static: other files' static initialisers may call it.
bool isa2_on() {
  static const bool on = !(getenv("VORTEX_LEGACY") && getenv("VORTEX_LEGACY")[0] == '1');
  return on;
}
uint64_t g_isa2_appends = 0, g_isa2_counts = 0, g_isa2_l1skips = 0, g_isa2_l1invs = 0, g_isa2_fence_wait = 0;
// Write acknowledgements (ISA2). A core's writes are counted from the LSU (lanes) through the switch and the
// coalescer (requests) to the L2, which performs each one and counts it off; the ack then needs WACK_LAT cycles
// to come back. A fence waits until its core has nothing outstanding and the last ack has arrived.
static constexpr uint64_t WACK_LAT = 4;
static std::vector<int64_t>  g_wack_pend;
static std::vector<uint64_t> g_wack_at;
static uint64_t g_wack_neg = 0;
static bool g_wack_on = false;   // set when an L2 that acknowledges writes exists (no L2: a fence waits for loads only)
void wack_enable() { g_wack_on = isa2_on(); }
static inline void wack_size(uint32_t cid) {
  if (cid >= g_wack_pend.size()) { g_wack_pend.resize(cid + 1, 0); g_wack_at.resize(cid + 1, 0); }
}
void wack_add(uint32_t cid, int64_t n) { if (!g_wack_on || n == 0) return; wack_size(cid); g_wack_pend[cid] += n; }
void wack_done(uint32_t cid) {
  if (!g_wack_on) return;
  wack_size(cid);
  if (--g_wack_pend[cid] < 0) { ++g_wack_neg; g_wack_pend[cid] = 0; }
  uint64_t t = SimPlatform::instance().cycles() + WACK_LAT;
  if (t > g_wack_at[cid]) g_wack_at[cid] = t;
}
bool wack_idle(uint32_t cid) {
  if (!g_wack_on) return true;
  wack_size(cid);
  return g_wack_pend[cid] == 0 && SimPlatform::instance().cycles() >= g_wack_at[cid];
}
// L2-append realism (2026-11-02, L2_APPEND_NOVELTY.md). Both default off.
// VORTEX_AQ_HOME=1: queue q's lines all live in the tail's L2 bank (q % L2_NUM_BANKS); the
//   timing address of every queue-storage access (appends and receiver reads) is remapped,
//   bijectively within the region. Overflow-pool chunks are not remapped.
// VORTEX_AQ_TAILREQ=1: each append also sends a write to its tail word (tailbase + 4q), so an
//   append costs two L2 requests (tail in an ordinary L2 line, or a slice-to-slice forward).
// VORTEX_AQ_SET=1 (design A, "the tail is the open line"): queue q's lines all live in ONE L2 set
//   of its home bank (bank = q % NB, set = (q / NB) % NS; the line number goes to the tag bits), so
//   the slice finds q's open line by a tag search. Implies AQ_HOME. Also: a line made full by an
//   append is marked evict-first in the L2, and an open (part-filled) line evicted from the L2
//   charges a tail spill + refill (two DRAM writes to the tail word, a bandwidth proxy).
static bool g_aq_set = getenv("VORTEX_AQ_SET") ? getenv("VORTEX_AQ_SET")[0] == '1' : aq_final_on();
static bool g_aq_home = g_aq_set || (getenv("VORTEX_AQ_HOME") && getenv("VORTEX_AQ_HOME")[0] == '1');
static const uint64_t AQ_NS = (uint64_t)L2_CACHE_SIZE / ((uint64_t)MEM_BLOCK_SIZE * L2_NUM_WAYS * L2_NUM_BANKS);
uint64_t g_aq_open_evicts = 0, g_aq_evict_first = 0, g_aq_protect_saves = 0;
static std::unordered_set<uint64_t> g_aq_done_lines;   // timing line addresses made full, not yet seen by the L2
// VORTEX_AQ_CNTMASK=1 (2026-11-06, experiment E2a in L2_APPEND.md): no 4-bit count per line, the slot is the popcount
// of the line's byte write mask. An open line evicted early is re-read from DRAM by the next append to it (its first
// slots then count as written) instead of being allocated without a fetch. The saved counter still keeps the tail.
// VORTEX_AQ_LASTBYTE=1 (2026-11-10): no count per line either, but the slot is the position after the LAST written byte of
// the line, not the number of written bytes. The L2 fills a queue's slots in order, so an open line's written bytes are
// one run that ends at the next free slot. An open line evicted early is reopened WITHOUT a fetch: its first append goes
// to the saved slot (the 2 B saved counter, charged at the eviction as spill + refill) and the run restarts there. Such a
// line holds only the messages appended since; a read that finds it still in the L2 fetches the line first and merges
// (one DRAM read). Exclusive with CNTMASK and REOPENBIT. ISA2 default (2026-11-10, with open lines in every way below);
// VORTEX_AQ_LASTBYTE=0 VORTEX_AQ_OPENWAYS=4 gives back the e1w4a count (popcount, refetch on reopen, ways 0-3).
static const bool g_aq_lastbyte = getenv("VORTEX_AQ_LASTBYTE") ? getenv("VORTEX_AQ_LASTBYTE")[0] == '1'
    : (isa2_on() && !(getenv("VORTEX_AQ_CNTMASK") && getenv("VORTEX_AQ_CNTMASK")[0] == '1')
                 && !(getenv("VORTEX_AQ_REOPENBIT") && getenv("VORTEX_AQ_REOPENBIT")[0] == '1'));
static std::unordered_set<uint64_t> g_aq_partial_lines;   // LASTBYTE: reopened lines in the L2 without their first messages
static std::unordered_set<uint64_t> g_aq_merging_lines;   // LASTBYTE: partial lines whose merge read is in flight
static uint64_t g_aq_lb_reopens = 0, g_aq_lb_merges = 0;
static const bool g_aq_cntmask = !g_aq_lastbyte && (getenv("VORTEX_AQ_CNTMASK") ? getenv("VORTEX_AQ_CNTMASK")[0] == '1' : isa2_on());   // ISA2: e1w4a default
static std::unordered_set<uint64_t> g_aq_reopen_lines;   // open lines evicted early, not yet appended to again
static uint64_t g_aq_reopen_fetches = 0;
// VORTEX_AQ_REOPENBIT=1 (experiment E2b): popcount of the byte mask as above, plus one "reopened" bit per line. A
// reopened line is allocated without a fetch, and every append to it while it stays open reads the saved counter
// (slot = saved base + popcount), one more L2 access each. Exclusive with CNTMASK.
static const bool g_aq_reopenbit = getenv("VORTEX_AQ_REOPENBIT") && getenv("VORTEX_AQ_REOPENBIT")[0] == '1';
static std::unordered_set<uint64_t> g_aq_reopened_open;   // E2b: reopened lines still open
static uint64_t g_aq_reopenbit_allocs = 0, g_aq_reopenbit_reads = 0;
// VORTEX_AQ_OPENWAYS=W (experiment E1): a queue line is allocated only in ways 0..W-1 of its set (other data still
// uses those ways), so the open bit and count are needed in W of the ways only. 0 = off.
static const uint32_t g_aq_openways = getenv("VORTEX_AQ_OPENWAYS") ? (uint32_t)atoi(getenv("VORTEX_AQ_OPENWAYS")) : 0;   // ISA2: every way (2026-11-10)
uint64_t g_aq_openways_allocs = 0;
uint32_t aq_openways() { return g_aq_openways; }
void aq_reopen_note(uint64_t line_addr) {
  uint64_t a = line_addr & ~(uint64_t)(AQ_LINE - 1);
  if (g_aq_cntmask || g_aq_reopenbit || g_aq_lastbyte) g_aq_reopen_lines.insert(a);
  if (g_aq_reopenbit) g_aq_reopened_open.erase(a);
}
bool aq_lastbyte_alloc(uint64_t line_addr) {   // LASTBYTE: a no-fetch allocation of a reopened line leaves it partial
  uint64_t a = line_addr & ~(uint64_t)(AQ_LINE - 1);
  if (!g_aq_lastbyte || !g_aq_reopen_lines.erase(a)) return false;
  g_aq_partial_lines.insert(a); ++g_aq_lb_reopens; return true;
}
bool aq_lastbyte_read(uint64_t line_addr) {    // LASTBYTE: a read hits a partial line -> it must fetch and merge first
  uint64_t a = line_addr & ~(uint64_t)(AQ_LINE - 1);
  if (!g_aq_lastbyte || !g_aq_partial_lines.erase(a)) return false;
  g_aq_merging_lines.insert(a); ++g_aq_lb_merges; return true;
}
bool aq_lastbyte_filled(uint64_t line_addr) {  // LASTBYTE: true = this fill was a merge, the line keeps its dirty bytes
  return g_aq_lastbyte && g_aq_merging_lines.erase(line_addr & ~(uint64_t)(AQ_LINE - 1));
}
bool aq_lastbyte_on() { return g_aq_lastbyte; }
void aq_lastbyte_evicted(uint64_t line_addr) { // LASTBYTE: a partial line written back leaves its masked bytes in DRAM
  if (g_aq_lastbyte) g_aq_partial_lines.erase(line_addr & ~(uint64_t)(AQ_LINE - 1));
}
bool aq_reopenbit_alloc(uint64_t line_addr) {   // E2b: a no-fetch allocation; true = this append reads the saved base
  uint64_t a = line_addr & ~(uint64_t)(AQ_LINE - 1);
  if (!g_aq_reopenbit || !g_aq_reopen_lines.erase(a)) return false;
  g_aq_reopened_open.insert(a); ++g_aq_reopenbit_allocs; ++g_aq_reopenbit_reads; return true;
}
bool aq_reopenbit_hit(uint64_t line_addr) {     // E2b: an append hitting a reopened open line reads the saved base
  if (!g_aq_reopenbit || !g_aq_reopened_open.count(line_addr & ~(uint64_t)(AQ_LINE - 1))) return false;
  ++g_aq_reopenbit_reads; return true;
}
void aq_reopenbit_full(uint64_t line_addr) { if (g_aq_reopenbit) g_aq_reopened_open.erase(line_addr & ~(uint64_t)(AQ_LINE - 1)); }
bool aq_reopen_take(uint64_t line_addr) {
  if (!g_aq_cntmask || !g_aq_reopen_lines.erase(line_addr & ~(uint64_t)(AQ_LINE - 1))) return false;
  ++g_aq_reopen_fetches; return true;
}
void aq_reopen_clear() { g_aq_reopen_lines.clear(); g_aq_reopened_open.clear(); }
bool aq_set_on() { return g_aq_set; }
static bool g_aq_tailreq = getenv("VORTEX_AQ_TAILREQ") && getenv("VORTEX_AQ_TAILREQ")[0] == '1';
// VORTEX_AQ_OVFCHG=1: price the overflow path as the hardware would run it. An append past its queue's
// region reaches the queue's home slice, which reads and bumps the queue's entry in the pinned area
// (count + current chunk, 8 B instead of 2 B) and forwards the word to the chunk line: one extra L2 access,
// charged as a write to the queue's tail word. A new chunk also costs the pool atomic and the chunk-table
// write: two L2 writes to the table entry. Off by default.
// VORTEX_AQ_OVFCHG=2: the same, but the pinned entry stays 2 B (count only), so every overflow append also
// reads its current chunk from the chunk table: one more L2 access, charged at the table entry.
static int g_aq_ovfchg = getenv("VORTEX_AQ_OVFCHG") ? atoi(getenv("VORTEX_AQ_OVFCHG")) : 0;
static uint64_t g_aq_ovf_tabreads = 0;
static uint64_t g_aq_ovf_appends = 0, g_aq_ovf_newchunks = 0;
uint32_t aq_pin_entry_bytes() { return (g_aq_ovfchg == 1 && vortex::g_aq_pool.ctab != 0) ? 8 : 2; }
// VORTEX_AQ_HASH=1: queue -> (bank, set) through a bijective hash of the low bits (like an L2 slice
// hash), so strided queue numbers do not pile into a few sets. Off: plain low bits.
static bool g_aq_hash = getenv("VORTEX_AQ_HASH") ? getenv("VORTEX_AQ_HASH")[0] == '1' : aq_final_on();
static uint64_t aq_qhash(uint64_t q, uint64_t bs) {   // bs = banks * sets, a power of two
  uint64_t lo = q % bs;
  if (!g_aq_hash) return lo;
  uint64_t hi = q / bs;
  lo = (lo ^ (hi * 0x9E5u)) & (bs - 1);
  lo = (lo * 0x9E3779B1u) & (bs - 1);            // odd multiplier: a bijection mod 2^k
  return lo ^ (lo >> 7);                           // xorshift: also a bijection
}
static uint64_t aq_qunhash(uint64_t h, uint64_t hi, uint64_t bs) {
  if (!g_aq_hash) return h;
  static std::unordered_map<uint64_t, std::vector<uint32_t>> inv;   // per high part: hash -> low bits
  auto& v = inv[hi];
  if (v.size() != bs) { v.assign(bs, 0); for (uint64_t lo = 0; lo < bs; ++lo) v[aq_qhash(hi * bs + lo, bs)] = (uint32_t)lo; }
  return v[h % bs];
}
static uint64_t aq_home_addr(uint64_t addr) {
  if (!g_aq_home || isa2_on()) return addr;   // ISA2: ordinary accesses are never remapped
  for (auto it = vortex::g_aq_regions.rbegin(); it != vortex::g_aq_regions.rend(); ++it) {
    const auto& r = *it;
    if (r.stride == 0 || addr < r.lo || (r.stride % AQ_LINE) != 0) continue;
    uint64_t off = addr - r.lo, q = off / r.stride;
    if (q > r.maxq) continue;
    uint64_t within = off % r.stride, lq = r.stride / AQ_LINE, nb = L2_NUM_BANKS;
    uint64_t g = ((q / nb) * lq + within / AQ_LINE) * nb + (q % nb);
    if (g_aq_set) {
      uint64_t bs = nb * AQ_NS, h = aq_qhash(q, bs);
      g = (((q / bs) * lq + within / AQ_LINE) * AQ_NS + (h / nb) % AQ_NS) * nb + (h % nb);
    }
    return r.lo + g * AQ_LINE + (within % AQ_LINE);
  }
  return addr;
}
// ISA2: the program's own layout of queue storage (the paper's "the program lays out each queue's storage so that
// the L2's ordinary address mapping also places its lines in that slice and set"). Line k of queue q sits at
// qbase + 64 * (((q / BS) * LQ + k) * BS + hash(q)), BS = banks x sets, LQ = qstride / 64: the same address the
// legacy model reached by remapping, now the real address. Overflow chunks are ordinary addresses.
uint64_t aq_home_slot(uint32_t q, uint64_t slot) {
  uint64_t a = aq_slot_addr(q, slot);
  if (!(g_aq_ctrl.cap == 0 || slot < g_aq_ctrl.cap || g_aq_pool.ctab == 0)) return a;   // overflow chunk
  if (!g_aq_set || g_aq_ctrl.qstride == 0 || (g_aq_ctrl.qstride % AQ_LINE) != 0) return a;
  uint64_t off = slot * g_aq_ctrl.wsize, lq = g_aq_ctrl.qstride / AQ_LINE, bs = (uint64_t)L2_NUM_BANKS * AQ_NS;
  uint64_t g = (((uint64_t)q / bs) * lq + off / AQ_LINE) * bs + aq_qhash(q, bs);
  return g_aq_ctrl.qbase + g * AQ_LINE + (off % AQ_LINE);
}
// design A: timing address of a queue line -> (q, line k); only for the latest region
static bool aq_set_inverse(uint64_t addr, uint64_t* q, uint64_t* k) {
  if (vortex::g_aq_regions.empty()) return false;
  const auto& r = vortex::g_aq_regions.back();
  if (r.stride == 0 || addr < r.lo || (r.stride % AQ_LINE) != 0) return false;
  uint64_t g = (addr - r.lo) / AQ_LINE, nb = L2_NUM_BANKS, lq = r.stride / AQ_LINE;
  uint64_t bank = g % nb, set = (g / nb) % AQ_NS, tp = g / (nb * AQ_NS), bs = nb * AQ_NS;
  *q = (tp / lq) * bs + aq_qunhash(set * nb + bank, tp / lq, bs);
  *k = tp % lq;
  return *q <= r.maxq;
}
bool aq_take_done(uint64_t line_addr) { return g_aq_done_lines.erase(line_addr) != 0; }
uint64_t aq_open_tail_addr(uint64_t line_addr) {   // 0 = not an open line
  uint64_t q, k;
  if (!aq_set_inverse(line_addr, &q, &k) || q >= g_aq_fill.size() || g_aq_ctrl.wsize == 0) return 0;
  uint32_t per_line = AQ_LINE / (uint32_t)g_aq_ctrl.wsize, f = g_aq_fill[q];
  if (f % per_line == 0 || k != f / per_line) return 0;
  return g_aq_ctrl.tailbase + 4 * q;
}
// one line of statistics at exit, only when the model was used
struct AqReport {
  ~AqReport() {
    if (g_aq_appends == 0) return;
    uint64_t used = 0, maxfill = 0;
    for (auto f : g_aq_fill) { used += (f != 0); if (f > maxfill) maxfill = f; }
    if (g_aq_used_hw > used) used = g_aq_used_hw;
    if (g_aq_maxfill_hw > maxfill) maxfill = g_aq_maxfill_hw;
    printf("PERF: aq charge=%llu, appends=%llu, line writes=%llu, tail flushes=%llu, queues used=%llu of %llu, max queue=%llu, word=%llu, DROPPED=%llu\n",
           (unsigned long long)g_aq_charge,
           (unsigned long long)g_aq_appends, (unsigned long long)g_aq_flushes, (unsigned long long)g_aq_tail_flushes,
           (unsigned long long)used, (unsigned long long)g_aq_fill.size(),
           (unsigned long long)maxfill, (unsigned long long)g_aq_ctrl.wsize,
           (unsigned long long)g_aq_dropped);
    if (g_aq_set) printf("PERF: aq design A (set layout): evict-first marks=%llu, open-line evictions=%llu (each = 2 DRAM tail writes), protect saves=%llu, L2 sets/bank=%llu\n",
                         (unsigned long long)g_aq_evict_first, (unsigned long long)g_aq_open_evicts, (unsigned long long)g_aq_protect_saves, (unsigned long long)AQ_NS);
    if (g_aq_set) printf("PERF: aq final=%d, queues used (max q + 1)=%llu, auto pinned counter area=%llu bytes\n",
                         (int)aq_final_on(), (unsigned long long)g_aq_qhw,
                         (unsigned long long)((aq_final_on() && !getenv("VORTEX_AQ_PIN_BYTES")) ? aq_pin_entry_bytes() * g_aq_qhw : 0));
    if (g_aq_cntmask) printf("PERF: aq cntmask=1, reopen fetches=%llu\n", (unsigned long long)g_aq_reopen_fetches);
    if (g_aq_lastbyte) printf("PERF: aq lastbyte=1, reopens without fetch=%llu, merge reads=%llu\n",
                              (unsigned long long)g_aq_lb_reopens, (unsigned long long)g_aq_lb_merges);
    if (isa2_on()) printf("PERF: isa2 appends=%llu, counts=%llu\n", (unsigned long long)g_isa2_appends, (unsigned long long)g_isa2_counts);
    if (g_aq_reopenbit) printf("PERF: aq reopenbit=1, reopened allocs=%llu, saved-base reads=%llu\n",
                               (unsigned long long)g_aq_reopenbit_allocs, (unsigned long long)g_aq_reopenbit_reads);
    if (g_aq_openways) printf("PERF: aq openways=%u, restricted allocs=%llu\n", g_aq_openways, (unsigned long long)g_aq_openways_allocs);
    if (g_aq_ovf_chunks) printf("PERF: aq overflow chunks=%llu of %llu\n", (unsigned long long)g_aq_ovf_chunks, (unsigned long long)g_aq_pool.nch);
    if (g_aq_ovfchg) printf("PERF: aq ovfchg=%d, overflow appends charged=%llu, new chunks charged=%llu, table reads charged=%llu, pin entry=%u B\n",
                            g_aq_ovfchg, (unsigned long long)g_aq_ovf_appends, (unsigned long long)g_aq_ovf_newchunks,
                            (unsigned long long)g_aq_ovf_tabreads, aq_pin_entry_bytes());
    fflush(stdout);
  }
};
static AqReport g_aq_report;
// ISA2: one line at exit, so every arm log proves which model ran (and that acks balanced: wack_neg must be 0)
struct Isa2Report {
  ~Isa2Report() {
    if (!isa2_on()) return;
    int64_t pend = 0; for (auto p : g_wack_pend) pend += p;
    printf("PERF: isa2=1 appends=%llu counts=%llu l1skip_lanes=%llu l1invs=%llu fence_wait_ticks=%llu wack_pending=%lld wack_neg=%llu\n",
           (unsigned long long)g_isa2_appends, (unsigned long long)g_isa2_counts, (unsigned long long)g_isa2_l1skips,
           (unsigned long long)g_isa2_l1invs, (unsigned long long)g_isa2_fence_wait, (long long)pend, (unsigned long long)g_wack_neg);
    fflush(stdout);
  }
};
static Isa2Report g_isa2_report;

// ============================================================================
// Apply engine (typed drains). docs/apply_engine_plan.md.
// ============================================================================
static uint64_t g_ap_lines = getenv("VORTEX_APPLY_LINES") ? env_u64("VORTEX_APPLY_LINES") : 256;
static uint64_t g_ap_rate  = getenv("VORTEX_APPLY_RATE")  ? env_u64("VORTEX_APPLY_RATE")  : 16;
static uint64_t g_ap_cmds = 0, g_ap_words = 0, g_ap_qreads = 0, g_ap_fills = 0,
                g_ap_writes = 0, g_ap_evicts = 0, g_ap_busy_ticks = 0;
static std::deque<ApTrace>                 g_ap_traces;    // core*16+lane, stable refs
static std::deque<std::array<uint64_t,8>>  g_ap_cmdw;
static std::deque<uint32_t>                g_ap_dones;
static std::deque<uint32_t>                g_ap_pend;
}  // namespace vortex

namespace vortex {
ApTrace& ap_trace(uint32_t core, uint32_t lane) {
  size_t i = (size_t)core * AP_NLANES + lane;
  while (i >= g_ap_traces.size()) g_ap_traces.emplace_back();
  return g_ap_traces[i];
}
uint64_t* ap_cmdwords(uint32_t core, uint32_t lane) {
  size_t i = (size_t)core * AP_NLANES + lane;
  while (i >= g_ap_cmdw.size()) g_ap_cmdw.push_back({0,0,0,0,0,0,0,0});
  return g_ap_cmdw[i].data();
}
uint32_t& ap_done(uint32_t core) {
  while (core >= g_ap_dones.size()) g_ap_dones.push_back(1);   // idle reads as done
  return g_ap_dones[core];
}
uint32_t& ap_pending(uint32_t core) {
  while (core >= g_ap_pend.size()) g_ap_pend.push_back(0);
  return g_ap_pend[core];
}
}  // namespace vortex

namespace {

constexpr uint32_t AP_TAG_BIT = 0x40000000u;

struct ApLane {
  bool     active = false;
  ApTrace* tr = nullptr;
  uint32_t wi = 0;               // next word to process
  uint32_t qreq = 0;             // queue lines requested so far (prefix of tr->qlines)
  std::vector<uint8_t> qarr;     // per queue line: arrived
  bool     blocked = false;      // waiting on an ADD fill
};
struct ApRead { uint64_t addr; uint32_t lane; uint32_t kind; uint32_t idx; };  // kind 0 qline, 1 fill

struct ApEngine {
  std::vector<ApLane> lanes;
  std::list<uint64_t> lru;                                   // MRU at front
  std::unordered_map<uint64_t, std::pair<std::list<uint64_t>::iterator, bool>> cache;
  std::deque<uint64_t> wq;                                   // line writes to issue
  std::deque<uint64_t> tq;                                   // partial line writes from aq_flush (tail charge)
  std::deque<ApRead>   rq;                                   // reads to issue
  std::unordered_map<uint64_t, ApRead> outstanding;          // (tag<<4|slot) -> read
  uint32_t seq = 0;
  uint32_t rr = 0;
  bool     flushing = false;
  bool     dirty_work = false;                               // something ran since the last flush
  ApEngine() : lanes(AP_NLANES) {}

  bool any_active() const { for (auto& l : lanes) if (l.active) return true; return false; }

  void touch(uint64_t line, uint32_t op, uint32_t lane) {
    auto it = cache.find(line);
    if (it != cache.end()) {
      lru.erase(it->second.first);
      lru.push_front(line);
      it->second.first = lru.begin();
      it->second.second = true;
      return;
    }
    if (cache.size() >= g_ap_lines && !lru.empty()) {
      uint64_t victim = lru.back(); lru.pop_back();
      auto v = cache.find(victim);
      if (v != cache.end()) { if (v->second.second) { wq.push_back(victim); ++g_ap_evicts; } cache.erase(v); }
    }
    lru.push_front(line);
    cache[line] = {lru.begin(), true};
    if (op == 1) {                                           // ADD needs the line's contents
      rq.push_back({line, lane, 1, 0});
      lanes[lane].blocked = true;
    }
  }

  void prefetch(uint32_t li) {
    ApLane& l = lanes[li];
    if (!l.active) return;
    uint32_t need = (l.wi < l.tr->qidx.size()) ? l.tr->qidx[l.wi] : (uint32_t)l.tr->qlines.size();
    while (l.qreq < l.tr->qlines.size() && l.qreq <= need + 1) {
      rq.push_back({l.tr->qlines[l.qreq], li, 0, l.qreq});
      l.qreq++;
    }
  }

  void activate(uint32_t core, uint32_t lane) {
    ApTrace& tr = ap_trace(core, lane);
    if (!tr.pending) return;
    tr.pending = false;
    if (ap_pending(core) > 0) ap_pending(core)--;
    ApLane& l = lanes[lane];
    l.active = true; l.tr = &tr; l.wi = 0; l.qreq = 0; l.blocked = false;
    l.qarr.assign(tr.qlines.size(), 0);
    ++g_ap_cmds;
    dirty_work = true;
    flushing = false;
    if (tr.tlines.empty()) l.active = false;                 // empty command
    prefetch(lane);
  }

  bool consume_rsp(const LsuRsp& rsp) {
    if (!(rsp.tag & AP_TAG_BIT)) return false;
    for (uint32_t i = 0; i < rsp.mask.size(); ++i) {
      if (!rsp.mask.test(i)) continue;
      // 8 bits for the lane index, not 4: at NUM_THREADS=32 a request carries 32
      // lanes, so a 4 bit field overflowed into the tag and two different reads
      // collided. The engine then never saw one of them complete, its lane stayed
      // blocked, the done flag never rose and the kernel spun forever. Found by
      // the exploration sweep, 2026-09-11. Harmless at 16 lanes or fewer.
      uint64_t key = ((uint64_t)rsp.tag << 8) | (i & 0xffu);
      auto it = outstanding.find(key);
      if (it == outstanding.end()) continue;
      ApRead r = it->second; outstanding.erase(it);
      ApLane& l = lanes[r.lane];
      if (r.kind == 0) { if (r.idx < l.qarr.size()) l.qarr[r.idx] = 1; }
      else             { l.blocked = false; }
    }
    return true;
  }

  // one tick: process words, then (if the port is free) issue one request
  void tick(uint32_t core, bool port_free, SimPort<LsuReq>& ReqDC, uint32_t delay, uint32_t nlanes) {
    // ---- words ----
    uint32_t budget = (uint32_t)g_ap_rate;
    uint32_t tries = 0;
    bool ran = false;
    while (budget > 0 && tries < AP_NLANES) {
      uint32_t li = rr;
      rr = (rr + 1) % AP_NLANES;
      ApLane& l = lanes[li];
      if (!l.active || l.blocked) { ++tries; continue; }
      uint32_t qi = l.tr->qidx[l.wi];
      if (!l.qarr[qi]) { ++tries; continue; }
      touch(l.tr->tlines[l.wi], l.tr->op, li);
      l.wi++; ++g_ap_words; --budget; ran = true; tries = 0;
      if (l.wi >= l.tr->tlines.size()) { l.active = false; }
      else prefetch(li);
    }
    if (ran) ++g_ap_busy_ticks;
    // ---- completion: everything applied, nothing in flight -> write back, then done ----
    if (dirty_work && !any_active() && outstanding.empty() && ap_pending(core) == 0 && !flushing) {
      for (auto& kv : cache) if (kv.second.second) wq.push_back(kv.first);
      cache.clear(); lru.clear();
      flushing = true;
    }
    // ---- issue ----
    if (port_free) {
      if (!wq.empty()) {
        LsuReq req(nlanes);
        req.write = true; req.tag = AP_TAG_BIT | (seq++ & 0xffffu); req.cid = core; req.uuid = 0;
        uint32_t n = 0;
        while (!wq.empty() && n < nlanes) { req.mask.set(n); req.addrs.at(n) = wq.front(); wq.pop_front(); ++n; ++g_ap_writes; }
        wack_add(core, n);
        ReqDC.push(req, delay);
      } else if (!tq.empty()) {                                // tail charge: after the engine's own writes
        LsuReq req(nlanes);
        req.write = true; req.tag = AP_TAG_BIT | (seq++ & 0xffffu); req.cid = core; req.uuid = 0;
        uint32_t n = 0;
        while (!tq.empty() && n < nlanes) { req.mask.set(n); req.addrs.at(n) = tq.front(); tq.pop_front(); ++n; }
        wack_add(core, n);
        ReqDC.push(req, delay);
      } else if (!rq.empty()) {
        LsuReq req(nlanes);
        req.write = false; req.tag = AP_TAG_BIT | (seq++ & 0xffffu); req.cid = core; req.uuid = 0;
        uint32_t n = 0;
        while (!rq.empty() && n < nlanes) {
          ApRead r = rq.front(); rq.pop_front();
          req.mask.set(n); req.addrs.at(n) = r.addr;
          outstanding[((uint64_t)req.tag << 8) | (n & 0xffu)] = r;
          if (r.kind == 0) ++g_ap_qreads; else ++g_ap_fills;
          ++n;
        }
        ReqDC.push(req, delay);
      }
    }
    // Raise done only if no command was traced while the flush drained. Several
    // warps share the flag: another warp's trigger can execute (done = 0,
    // pending++) after the flush began and before its store reaches the engine,
    // and raising done then released every waiting warp early. The flag stays
    // low and the pending command, once activated, raises it at its own end.
    if (flushing && wq.empty()) {
      flushing = false; dirty_work = false;
      if (ap_pending(core) == 0 && !any_active()) ap_done(core) = 1;
    }
  }
};
static std::deque<ApEngine> g_ap_engines;
static ApEngine& ap_engine(uint32_t core) {
  while (core >= g_ap_engines.size()) g_ap_engines.emplace_back();
  return g_ap_engines[core];
}
struct ApReport {
  ~ApReport() {
    if (g_ap_cmds == 0) return;
    printf("PERF: apply cmds=%llu, words=%llu, qline reads=%llu, fill reads=%llu, writes=%llu, evictions=%llu, busy ticks=%llu, lines=%llu, rate=%llu\n",
           (unsigned long long)g_ap_cmds, (unsigned long long)g_ap_words, (unsigned long long)g_ap_qreads,
           (unsigned long long)g_ap_fills, (unsigned long long)g_ap_writes, (unsigned long long)g_ap_evicts,
           (unsigned long long)g_ap_busy_ticks, (unsigned long long)g_ap_lines, (unsigned long long)g_ap_rate);
    fflush(stdout);
  }
};
static ApReport g_ap_report;
}

LocalMemSwitch::LocalMemSwitch(
  const SimContext& ctx,
  const char* name,
  uint32_t delay,
  uint32_t core_id
) : SimObject<LocalMemSwitch>(ctx, name)
  , ReqIn(this)
  , RspIn(this)
  , ReqLmem(this)
  , RspLmem(this)
  , ReqDC(this)
  , RspDC(this)
  , delay_(delay)
  , core_id_(core_id)
{}

void LocalMemSwitch::reset() {}

// Tail charge (log T36a). An append is charged a line write on every
// (AQ_LINE/wsize)'th append to its queue, so a queue whose tail sits mid line
// at the end of a round has a partial line still to write. aq_flush (a store to
// control word 5) and a geometry change charge one line write per such queue,
// issued through the cores' engines round-robin by queue index, and remember
// the fill so a queue that has not moved since is not charged twice. v1: the
// lanes do not wait for these writes; the traffic is charged, the latency
// overlaps with whatever follows the barrier.
std::vector<AqRegion> vortex::g_aq_regions;

// Called on every append. The region a queue belongs to is the one installed
// when it was written, which is always the last one, so the search is a walk
// backwards that stops at the first match.
void vortex::aq_note_queue(int64_t q) {
  if (q < 0 || g_aq_regions.empty()) return;
  auto& r = g_aq_regions.back();
  if ((uint64_t)q > r.maxq) r.maxq = (uint64_t)q;
}

bool vortex::aq_is_qstorage(uint64_t addr) {
  if (g_aq_pool.nch != 0 && addr >= g_aq_pool.base
   && addr < g_aq_pool.base + g_aq_pool.nch * g_aq_pool.ch * (g_aq_ctrl.wsize ? g_aq_ctrl.wsize : 4))
    return true;                                           // overflow chunks are queue storage too
  for (const auto& r : g_aq_regions) {
    if (r.stride == 0 || addr < r.lo) continue;
    uint64_t off = addr - r.lo;
    uint64_t q   = off / r.stride;
    if (q > r.maxq) continue;
    // The gap between one queue's capacity and the next queue's start is
    // padding the transport never writes, so it is not queue storage.
    if (r.cap_bytes != 0 && (off % r.stride) >= r.cap_bytes) continue;
    return true;
  }
  return false;
}

bool vortex::aq_in_qregion(uint64_t addr) {
  if (g_aq_pool.nch != 0 && addr >= g_aq_pool.base
   && addr < g_aq_pool.base + g_aq_pool.nch * g_aq_pool.ch * (g_aq_ctrl.wsize ? g_aq_ctrl.wsize : 4))
    return true;
  for (const auto& r : g_aq_regions)
    if (r.stride != 0 && addr >= r.lo && (addr - r.lo) / r.stride < (r.maxq / L2_NUM_BANKS + 1) * L2_NUM_BANKS) return true;   // rounded up: AQ_HOME lines
  return false;
}

void vortex::aq_tail_flush() {
  size_t n = g_aq_fill.size();
  if (g_aq_flushed.size() != n) g_aq_flushed.assign(n, 0);
  if (g_aq_ctrl.wsize == 0) return;
  uint32_t per_line = AQ_LINE / (uint32_t)g_aq_ctrl.wsize;
  if (per_line == 0) return;
  for (size_t q = 0; q < n; ++q) {
    uint32_t f = g_aq_fill[q];
    if (f == g_aq_flushed[q]) continue;
    g_aq_flushed[q] = f;
    if (f % per_line == 0) continue;
    if (!(g_aq_charge & 2)) continue;
    uint64_t addr = g_aq_ctrl.qbase + (uint64_t)q * g_aq_ctrl.qstride
                  + (uint64_t)(f - f % per_line) * g_aq_ctrl.wsize;
    ap_engine((uint32_t)(q % (NUM_CORES))).tq.push_back(addr);
    ++g_aq_tail_flushes;
  }
}

void LocalMemSwitch::tick() {
  // process outgoing responses
  if (!RspLmem.empty()) {
    auto& out_rsp = RspLmem.front();
    DT(4, this->name() << "-lmem-rsp: " << out_rsp);
    RspIn.push(out_rsp, 1);
    RspLmem.pop();
  }
  if (!RspDC.empty()) {
    auto& out_rsp = RspDC.front();
    DT(4, this->name() << "-dc-rsp: " << out_rsp);
    if (!ap_engine(core_id_).consume_rsp(out_rsp)) {   // engine reads never reach the LSU
      RspIn.push(out_rsp, 1);
    }
    RspDC.pop();
  }

  bool dc_pushed = false;
  uint32_t nlanes = NUM_LSU_LANES;

  // process incoming requests
  if (!ReqIn.empty()) {
    auto& in_req = ReqIn.front();
    nlanes = in_req.mask.size();

    const bool isa2 = isa2_on();
    LsuReq out_dc_req(in_req.mask.size());
    out_dc_req.write = in_req.write;
    out_dc_req.amo   = in_req.amo;
    out_dc_req.tag   = in_req.tag;
    out_dc_req.cid   = in_req.cid;
    out_dc_req.uuid  = in_req.uuid;
    out_dc_req.aqop  = in_req.aqop;
    // VORTEX_PROBE_AMO_L1=1: TIMING PROBE ONLY, not a machine. Ordinary atomics go through the L1 again (an L1 hit is
    // answered there, as the pre-ISA2 model did, which is incoherent across cores); appends and counts still skip it.
    static const bool probe_amo_l1 = getenv("VORTEX_PROBE_AMO_L1") && getenv("VORTEX_PROBE_AMO_L1")[0] == '1';
    out_dc_req.l1skip = isa2 && ((in_req.amo && !probe_amo_l1) || in_req.aqop != 0);   // ISA2: atomics, appends, counts skip the L1

    LsuReq out_lmem_req(out_dc_req);
    LsuReq out_tail_req(out_dc_req);
    out_tail_req.write = true; out_tail_req.amo = false; out_tail_req.aqop = 0;
    out_tail_req.l1skip = isa2;   // the slice's own accesses (pinned entry, chunk table) never touch an L1
    LsuReq out_ovf_req(out_tail_req), out_ovf2_req(out_tail_req), out_tab_req(out_tail_req);

    for (uint32_t i = 0; i < in_req.mask.size(); ++i) {
      if (in_req.mask.test(i) && in_req.aqop == 1) {
        // ISA2 APPEND: the lane carries the queue number. The queue's slice places the message at its next slot
        // (timing: the slot's address in the program's layout), exactly as the legacy CHARGE=4 path did.
        uint32_t q = (uint32_t)in_req.addrs.at(i);
        if ((size_t)q < g_aq_fill.size() && g_aq_ctrl.wsize != 0) {
          uint32_t f = ++g_aq_fill[q];
          ++g_aq_appends; ++g_isa2_appends;
          if ((uint64_t)q + 1 > g_aq_qhw) g_aq_qhw = (uint64_t)q + 1;
          uint32_t per_line = AQ_LINE / (uint32_t)g_aq_ctrl.wsize;
          out_dc_req.mask.set(i);
          out_dc_req.addrs.at(i) = aq_home_slot(q, (uint64_t)(f - 1));
          if (g_aq_set && f % per_line == 0)
            g_aq_done_lines.insert(out_dc_req.addrs.at(i) & ~(uint64_t)(AQ_LINE - 1));
          if (g_aq_ovfchg && g_aq_pool.ctab != 0 && g_aq_ctrl.cap != 0 && g_aq_ctrl.tailbase != 0
           && (uint64_t)(f - 1) >= g_aq_ctrl.cap) {
            out_tail_req.mask.set(i);
            out_tail_req.addrs.at(i) = g_aq_ctrl.tailbase + 4 * (uint64_t)q;
            ++g_aq_ovf_appends;
            uint64_t e = (uint64_t)(f - 1) - g_aq_ctrl.cap;
            if (g_aq_pool.ch != 0 && e % g_aq_pool.ch == 0) {
              uint64_t ca = g_aq_pool.ctab + 8 * ((uint64_t)q * g_aq_pool.maxc + e / g_aq_pool.ch);
              out_ovf_req.mask.set(i);  out_ovf_req.addrs.at(i) = ca;
              out_ovf2_req.mask.set(i); out_ovf2_req.addrs.at(i) = ca;
              ++g_aq_ovf_newchunks;
            } else if (g_aq_ovfchg == 2 && g_aq_pool.ch != 0) {
              out_tab_req.mask.set(i);
              out_tab_req.addrs.at(i) = g_aq_pool.ctab + 8 * ((uint64_t)q * g_aq_pool.maxc + e / g_aq_pool.ch);
              ++g_aq_ovf_tabreads;
            }
          }
          if (f % per_line == 0) ++g_aq_flushes;
        }
        continue;
      }
      if (in_req.mask.test(i) && in_req.aqop == 2) {
        // ISA2 COUNT: a request to the queue's slice (any line of the queue maps there); answered without DRAM
        uint32_t q = (uint32_t)in_req.addrs.at(i);
        out_dc_req.mask.set(i);
        out_dc_req.addrs.at(i) = aq_home_slot(q, 0) & ~(uint64_t)(AQ_LINE - 1);
        ++g_isa2_counts;
        continue;
      }
      if (in_req.mask.test(i)) {
        auto type = get_addr_type(in_req.addrs.at(i));
        // MsgMem is timed like the on-chip scratchpad (banked SRAM, no cache or
        // memory-controller traffic). Its DATA still lives in the global image,
        // so it stays coherent across cores; only the timing path differs.
        // RemoteLmem (v0 ceiling) is charged the same way: the push is timed as
        // a local scratchpad access, standing in for a NoC that accepts one
        // flit per cycle. Destination bank contention is NOT charged.
        if (type == AddrType::Shared || type == AddrType::MsgMem ||
            type == AddrType::RemoteLmem) {
          out_lmem_req.mask.set(i);
          out_lmem_req.addrs.at(i) = in_req.addrs.at(i);
        } else if (type == AddrType::AppendQ) {
          // an append is charged as a scratchpad access on the queue's line;
          // every (AQ_LINE/wsize)'th append to a queue also sends one full
          // line write down the dcache path. Writes need no response tag
          // anywhere in the LSU, the coalescer or the switch, so the extra
          // lane bit is all it takes.
          int64_t q = aq_queue(in_req.addrs.at(i));
          // apply engine: the trigger word's store activates the lane's command
          if (in_req.write && ap_is_window(in_req.addrs.at(i))
           && ap_line(in_req.addrs.at(i)) < AP_NLANES
           && (in_req.addrs.at(i) & (AQ_LINE - 1)) == AP_TRIG_OFF) {
            ap_engine(core_id_).activate(core_id_, ap_line(in_req.addrs.at(i)));
          }
          // The charge knob prices the APPEND, which is a write. A READ of this
          // window still has to be serviced, above all the apply engine's done
          // flag, which shares the window: dropping it leaves the lane waiting
          // for a response that never comes, and the arm hangs rather than
          // reporting a free append. Cost the diagnostic knob two lost days
          // once; it is a knob, so it must be inert on everything it does not
          // claim to price.
          if ((g_aq_charge & 1) || !in_req.write) {
            // Charge the append on the queue's COMBINING LINE, one line per
            // queue, interleaved across the banks so that appends to different
            // queues proceed in parallel and appends to one queue serialise.
            // The window itself is laid out 64 bytes a queue for addressing,
            // but LocalMem banks by (addr >> 3) & (banks-1), so handing it the
            // raw window address would put every queue in one of two banks and
            // serialise every warp's appends 8 to 16 ways. That is an artifact
            // of the window stride, not a property of the design.
            out_lmem_req.mask.set(i);
            out_lmem_req.addrs.at(i) = (uint64_t)LMEM_BASE_ADDR
                                     + ((q >= 0 ? (uint64_t)q : 0ull) << 3);
          }
          if (in_req.write && q >= 0 && (size_t)q < g_aq_fill.size() && g_aq_ctrl.wsize != 0) {
            uint32_t f = ++g_aq_fill[q];
            ++g_aq_appends;
            if ((uint64_t)q + 1 > g_aq_qhw) g_aq_qhw = (uint64_t)q + 1;
            uint32_t per_line = AQ_LINE / (uint32_t)g_aq_ctrl.wsize;
            // VORTEX_AQ_CHARGE bit 2 (4): the L2-APPEND design. No combining in the core: every
            // append is one write request to the L2 at its slot (the L2 slice holds the tail and
            // places the word); lines merge in the L2 (no-fetch allocation) and leave when evicted.
            if (g_aq_charge & 4) {
              out_dc_req.mask.set(i);
              out_dc_req.addrs.at(i) = aq_home_addr(aq_slot_addr((uint32_t)q, (uint64_t)(f - 1)));
              if (g_aq_set && f % per_line == 0)
                g_aq_done_lines.insert(out_dc_req.addrs.at(i) & ~(uint64_t)(AQ_LINE - 1));
              if (g_aq_tailreq) {
                out_tail_req.mask.set(i);
                out_tail_req.addrs.at(i) = g_aq_ctrl.tailbase + 4 * (uint64_t)q;
              }
              if (g_aq_ovfchg && g_aq_pool.ctab != 0 && g_aq_ctrl.cap != 0 && g_aq_ctrl.tailbase != 0
               && (uint64_t)(f - 1) >= g_aq_ctrl.cap) {
                out_tail_req.mask.set(i);
                out_tail_req.addrs.at(i) = g_aq_ctrl.tailbase + 4 * (uint64_t)q;
                ++g_aq_ovf_appends;
                uint64_t e = (uint64_t)(f - 1) - g_aq_ctrl.cap;
                if (g_aq_pool.ch != 0 && e % g_aq_pool.ch == 0) {
                  uint64_t ca = g_aq_pool.ctab + 8 * ((uint64_t)q * g_aq_pool.maxc + e / g_aq_pool.ch);
                  out_ovf_req.mask.set(i);  out_ovf_req.addrs.at(i) = ca;
                  out_ovf2_req.mask.set(i); out_ovf2_req.addrs.at(i) = ca;
                  ++g_aq_ovf_newchunks;
                } else if (g_aq_ovfchg == 2 && g_aq_pool.ch != 0) {   // 2 B entry: find the current chunk in the table
                  out_tab_req.mask.set(i);
                  out_tab_req.addrs.at(i) = g_aq_pool.ctab + 8 * ((uint64_t)q * g_aq_pool.maxc + e / g_aq_pool.ch);
                  ++g_aq_ovf_tabreads;
                }
              }
            }
            if (f % per_line == 0) {
              ++g_aq_flushes;
              if (g_aq_charge & 2) {
                out_dc_req.mask.set(i);
                out_dc_req.addrs.at(i) = g_aq_ctrl.qbase + (uint64_t)q * g_aq_ctrl.qstride
                                       + (uint64_t)(f - per_line) * g_aq_ctrl.wsize;
              }
            }
          }
        } else {
          out_dc_req.mask.set(i);
          out_dc_req.addrs.at(i) = aq_home_addr(in_req.addrs.at(i));
        }
      }
    }

    if (in_req.write) {   // ISA2 write acks: the lanes counted at the LSU become the requests sent on (IO excepted)
      int64_t n = 0;
      for (const LsuReq* r : {&out_dc_req, &out_tail_req, &out_ovf_req, &out_ovf2_req, &out_tab_req})
        for (uint32_t i = 0; i < r->mask.size(); ++i)
          if (r->mask.test(i) && get_addr_type(r->addrs.at(i)) != AddrType::IO) ++n;
      wack_add(in_req.cid, n - (int64_t)in_req.mask.count());
    }
    if (out_dc_req.l1skip && !out_dc_req.mask.none()) g_isa2_l1skips += out_dc_req.mask.count();

    if (!out_dc_req.mask.none()) {
      ReqDC.push(out_dc_req, delay_);
      dc_pushed = true;
      DT(4, this->name() << "-dc-req: " << out_dc_req);
    }

    if (!out_tail_req.mask.none()) {
      ReqDC.push(out_tail_req, delay_);
      dc_pushed = true;
    }
    if (!out_ovf_req.mask.none())  { ReqDC.push(out_ovf_req, delay_);  dc_pushed = true; }
    if (!out_ovf2_req.mask.none()) { ReqDC.push(out_ovf2_req, delay_); dc_pushed = true; }
    if (!out_tab_req.mask.none())  { ReqDC.push(out_tab_req, delay_);  dc_pushed = true; }

    if (!out_lmem_req.mask.none()) {
      ReqLmem.push(out_lmem_req, delay_);
      DT(4, this->name() << "-lmem-req: " << out_lmem_req);
    }
    ReqIn.pop();
  }

  // the apply engine shares this core's dcache port with the lanes
  if (g_aq_size != 0) {
    ap_engine(core_id_).tick(core_id_, !dc_pushed, ReqDC, delay_, nlanes);
  }
}

///////////////////////////////////////////////////////////////////////////////

LsuMemAdapter::LsuMemAdapter(
  const SimContext& ctx,
  const char* name,
  uint32_t num_inputs,
  uint32_t delay
) : SimObject<LsuMemAdapter>(ctx, name)
  , ReqIn(this)
  , RspIn(this)
  , ReqOut(num_inputs, this)
  , RspOut(num_inputs, this)
  , delay_(delay)
{}

void LsuMemAdapter::reset() {}

void LsuMemAdapter::tick() {
  uint32_t input_size = ReqOut.size();

  // process outgoing responses
  for (uint32_t i = 0; i < input_size; ++i) {
    if (RspOut.at(i).empty())
      continue;
    auto& out_rsp = RspOut.at(i).front();
    DT(4, this->name() << "-rsp" << i << ": " << out_rsp);

    // build memory response
    LsuRsp in_rsp(input_size);
    in_rsp.mask.set(i);
    in_rsp.tag = out_rsp.tag;
    in_rsp.cid = out_rsp.cid;
    in_rsp.uuid = out_rsp.uuid;

    // include other responses with the same tag
    for (uint32_t j = i + 1; j < input_size; ++j) {
      if (RspOut.at(j).empty())
        continue;
      auto& other_rsp = RspOut.at(j).front();
      if (out_rsp.tag == other_rsp.tag) {
        in_rsp.mask.set(j);
        RspOut.at(j).pop();
      }
    }

    // send memory response
    RspIn.push(in_rsp, 1);

    // remove input
    RspOut.at(i).pop();
    break;
  }

  // process incoming requests
  if (!ReqIn.empty()) {
    auto& in_req = ReqIn.front();
    assert(in_req.mask.size() == input_size);
    for (uint32_t i = 0; i < input_size; ++i) {
      if (in_req.mask.test(i)) {
        // build memory request
        MemReq out_req;
        out_req.write = in_req.write;
        out_req.amo   = in_req.amo;
        out_req.aqop  = in_req.aqop;
        out_req.l1skip = in_req.l1skip;
        out_req.addr  = in_req.addrs.at(i);
        out_req.type  = get_addr_type(in_req.addrs.at(i));
        out_req.tag   = in_req.tag;
        out_req.cid   = in_req.cid;
        out_req.uuid  = in_req.uuid;
        // send memory request
        ReqOut.at(i).push(out_req, delay_);
        DT(4, this->name() << "-req" << i << ": " << out_req);
      }
    }
    ReqIn.pop();
  }
}