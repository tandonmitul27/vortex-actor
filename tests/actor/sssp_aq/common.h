#ifndef LMEM_BYTES
#define LMEM_BYTES 16384   // scratchpad bytes per core (sensitivity runs pass -DLMEM_BYTES; 2026-11-15)
#endif
// SSSP, Bellman-Ford, after bale (jdevinney/bale, src/bale_classic/apps/sssp_src): sssp_bellman_conveyor.upc
// (the message-passing version) and its README (the AGP version needs an atomic min of a double). Input: bale's sssp
// default, a directed weighted Erdos-Renyi graph without loops: n = SS_LROWS * N vertices, every off-diagonal (v, w)
// an edge with probability nz / n (SS_NZ, default 10), weights uniform in [0, 1); source 0. Vertex v belongs to PE
// v % P at local index v / P; each PE holds the out-edges of its own vertices (oned CSR) and its tentative weights.
// Bellman-Ford as bale's: each round, only the vertices whose weight changed in the previous round relax their
// out-edges (v, w, c): {w, tent(v) + c} to w's owner, whose handler does tent(w) = min(tent(w), tw); stop when a round
// changes nothing.
// ACTOR=1 (P = N actors): the conveyor version on the append queue, the recipe: one 64-bit word {tw rounded to its top
//          64 - TB bits | w's in-tile index} (tw >= 0, so its bits order as an unsigned integer) to the queue of (w's
//          owner, tile) at parity (r & 1) ^ 1, queue ((parity * NT) + tile) * N + owner, monotone (tails only grow,
//          each consumer keeps its heads): round r drains parity r & 1 and appends parity (r & 1) ^ 1, one barrier per
//          round. The handler runs on the owner's warp, 16 messages at a time, into a TILE_W scratchpad tile loaded
//          from tent (an atomic min: amominu.d); the tile is compared and written back as whole lines, and the
//          vertices that changed (a warp ballot) are the warp's next senders, in the same round. One leaf per round.
// ACTOR=0 (P = 1): Bellman-Ford on one GPU, same rounds: a warp per changed vertex (grid stride over a global
//          frontier queue), lanes over its out-edges; tw < tent(w) is tested first, then an atomic min (amominu.d) on
//          the global tent; a vertex whose weight dropped is queued once per round (a round stamp, then a fetch-and-
//          add on the next queue's end). One barrier per round.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>
#include "../probe_apply.h"   // PROBE_APPLYT, ROUND1 (measurement only)
#ifndef NASSUME
#define NASSUME    256      // actors
#endif
#ifndef SPLIT
#define SPLIT      4        // warps per block (16 actors per block)
#endif
#ifndef ACTOR
#define ACTOR      1
#endif
#define LANES      16
#define GLANES     16
#define OW         ((GLANES) / (SPLIT))         // actors per warp
#ifndef AQ_BASE
#define AQ_BASE    0xD0000000u
#endif
#ifndef AQ_SIZE
#define AQ_SIZE    0x100000u
#endif
#ifndef TILE_W
#define TILE_W     512      // weights per scratchpad tile: SPLIT tiles fill the core's 16 KB local memory
#endif
// Ablation switches (results/mapping/ABLATION.md, ../abl.h). Both default off (= the full design). Either one
// drains all the warp's queues first, then sends; the dropped vertices are counted into a per-actor counter.
#ifndef NOTILE
#define NOTILE     0        // 1: piece B off: one queue per (parity, owner); the handler is an L2 atomic min on tent in
#endif                      //    memory, and a round stamp queues a dropped vertex once. TILE_W >= SS_L is the field width.
#ifndef LANE_RECV
#define LANE_RECV  0        // 1: piece C off: a lane drains a whole queue by itself; with tiles each lane has its own
#endif                      //    TILE_W / 16-weight tile
static_assert((TILE_W & (TILE_W - 1)) == 0 && (NOTILE || (TILE_W) * (SPLIT) * 8 <= (LMEM_BYTES)), "TILE_W: a power of 2, SPLIT tiles in 16 KB");
#define KW         ((NOTILE || !LANE_RECV) ? (TILE_W) : (TILE_W) / (GLANES))   // vertices per queue key (tile)
constexpr uint32_t ss_log2(uint32_t x) { return x <= 1 ? 0 : 1 + ss_log2(x / 2); }
#define TB         (ss_log2(KW))       // bits of the in-tile index
static_assert(((NASSUME) & ((NASSUME) - 1)) == 0, "NASSUME: a power of 2");
#ifndef SS_L
#define SS_L       1000     // ACTOR=1: vertices per actor, compile time (the host checks it is SS_LROWS)
#endif
#define VL         (SS_L)
#define NT         (((SS_L) + (KW) - 1) / (KW))   // tiles per actor
static_assert(!(NOTILE) || NT == 1, "NOTILE needs TILE_W >= SS_L");
#ifndef Q_CAP
#define Q_CAP      4096     // per-queue region (64-bit messages); the rest overflows into pool chunks
#endif
#define QS64       (((((Q_CAP) * 8 + 63) / 64) | 1) * 8)   // whole odd lines, in 64-bit words
#ifndef SWQ
#define SWQ        0        // 1: software append queue, no L2 append unit (../swq.h)
#endif
#ifndef OVF_CH
#define OVF_CH     512
#endif
#ifndef WC_U
#define WC_U       4
#endif
#ifndef HWBAR
#define HWBAR      0        // 1: the machine's global barrier (vx_barrier, global bit); 0: the software barrier
#endif
#define NBLK       ((NASSUME) / (GLANES))      // blocks = cores (one block of SPLIT warps per core)
#ifndef MAXR
#define MAXR       4096     // rounds, safety cap
#endif
#define SS_INF     0x7FF0000000000000ull       // +inf
#ifndef SEND_LANE
#define SEND_LANE  0        // senders: 1 a lane per vertex (its out-edges in a loop), 0 a warp per vertex (lanes over them)
#endif

typedef struct { uint64_t drain, send, total, msgs, rounds, t0, t1, pad;
#if PROBE_APPLYT
                 uint64_t pa_slot[2], pa_h, pa_s;   // drain = the receive calls, send = the sends (no handler sends)
#endif
} phase_cycles_t;

typedef struct {
    uint64_t off_addr;        // ACTOR=1: uint32_t[P][VL + 1] PE p's out-edge offsets (absolute); ACTOR=0: uint32_t[n + 1]
    uint64_t nbr_addr;        // uint32_t[m] heads (global ids), PE by PE, vertex by vertex
    uint64_t wt_addr;         // double[m] weights
    uint64_t tent_addr;       // uint64_t[P * lrows] tentative weights (double bits), PE p's at p * lrows
    uint64_t fr_addr;         // ACTOR=1: uint32_t[P][VL] each PE's senders (local ids); ACTOR=0: uint32_t[2][n] queues
    uint64_t stamp_addr;      // ACTOR=0: uint32_t[n] round + 1 a vertex was last queued; ACTOR=1: [P * lrows] (NOTILE: round)
    uint64_t cnt_addr;        // ACTOR=0: uint32_t[3] queue ends (round r appends to [(r + 1) % 3]); ACTOR=1: [P] senders per actor
    uint64_t flag_addr;       // ACTOR=1: uint32_t[MAXR] round r appended something
    uint64_t qs_addr;         // ACTOR=1: queue regions, 2 * NT * N x QS64 x 8 B
    uint64_t tails_addr;      // ACTOR=1: uint32_t[2 * NT * N] (model written)
    uint64_t heads_addr;      // ACTOR=1: uint32_t[2 * NT * N] (each consumer's own)
    uint64_t pool_addr;       // ACTOR=1: overflow chunks
    uint64_t ctab_addr;       // ACTOR=1: uint64_t[2 * NT * N * ovf_maxc]
#if SWQ
    uint64_t swq_addr;        // SWQ: the pool counter (zeroed)
#endif
    uint64_t phase_addr;      // phase_cycles_t[warps]
    uint64_t bar_addr;        // volatile uint32_t[8]
    uint32_t N;               // actors (the kernel's P when ACTOR=1)
    uint32_t n;               // vertices
    uint32_t lrows;           // vertices per PE
    uint32_t src;             // source vertex
    uint32_t ovf_maxc;        // ACTOR=1: chunks per queue (host: exact need + 1)
    uint32_t ovf_nch;         // ACTOR=1: pool size in chunks (host: exact need + slack)
} args_t;
#endif
