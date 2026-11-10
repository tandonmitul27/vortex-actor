// PageRank (push), after HClib-Actor's bale_actor tests (srirajpaul/hclib, branch bale3_actor):
// test/pagerank_push_LU_selector.cpp (the actor version) and test/pagerank_push_LU_agi.cpp (the non-actor baseline).
// Input: the tests' graph, bale's erdos_renyi_random_graph(n, p, UNDIRECTED, NOLOOPS) (strictly lower L) made
// symmetric, A2 = L + U, with n = PR_LROWS * N rows (PR_LROWS is the tests' -n, rows per PE) and
// p = 2 (PR_NZ - 1) / n (the tests fix nz_per_row = 35). Rows are distributed cyclically: row r belongs to PE r % P
// at local row r / P. Ranks start at 1/n; per iteration: curr = 0; every nonzero (y, col) adds pr[y] / deg(y) to
// curr[col]; curr = (1 - 0.85) / n + 0.85 curr; error = sum |curr - pr| (all PEs); stop if error < 1e-5, else
// pr = curr. The answer is pr (the iterate before the one that converged).
// ACTOR=1 (P = N actors): pagerank_selector. Every local nonzero (y, col) sends {col / P, pr[y] / deg(y)} to owner
//          col % P; the handler adds it, curr[col / P] += prf. Handlers run on the owner's warp, 16 messages at a
//          time, so the += is an atomic on the owner's own array (a double: an amocas loop).
//          On the append queue as transpose's (WC_TILE + DIRECT_BT): the sender appends to the queue of (owner,
//          row tile), tile = (col / P) / TILE_W, queue t * N + owner, one 64-bit word: the double prf rounded to
//          its top 55 bits (TB low mantissa bits dropped, ~13 significant digits) | the in-tile row. The owner's
//          warp applies one tile queue at a time into a TILE_W-double scratchpad tile, writes it out to curr as
//          whole lines and clears it. Each round is one leaf function over the warp's actors.
// ACTOR=0 (P = 1): pagerank_agi on one GPU: a thread per row (grid stride), each nonzero an atomic double add
//          (amocas loop: the lock / get / put of the agi test) to curr[col] in global memory.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>
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
#ifndef Q_CAP
#define Q_CAP      4096     // per-queue region (64-bit messages); the rest overflows into pool chunks
#endif
#define QS64       (((((Q_CAP) * 8 + 63) / 64) | 1) * 8)   // whole odd lines, in 64-bit words
#ifndef SWQ
#define SWQ        0        // 1: software append queue, no L2 append unit (../swq.h)
#endif
#ifndef OVF_CH
#define OVF_CH     256
#endif
#ifndef WC_U
#define WC_U       4
#endif
#ifndef TILE_W
#define TILE_W     512      // doubles per scratchpad tile: SPLIT tiles fill the core's 16 KB local memory
#endif
// Ablation switches (results/mapping/ABLATION.md, ../abl.h). Both default off (= the full design).
#ifndef NOTILE
#define NOTILE     0        // 1: piece B off: one queue per owner; the handler adds into curr in memory (an L2
#endif                      //    amocas loop per message). TILE_W is then only the row field width and must be >= lrows.
#ifndef LANE_RECV
#define LANE_RECV  0        // 1: piece C off: a lane drains a whole queue by itself; with tiles each lane has its
#endif                      //    own TILE_W / 16-double tile
static_assert((TILE_W & (TILE_W - 1)) == 0 && (NOTILE || (TILE_W) * (SPLIT) * 8 <= 16384), "TILE_W: a power of 2, SPLIT tiles in 16 KB");
#define KW         ((NOTILE || !LANE_RECV) ? (TILE_W) : (TILE_W) / (GLANES))   // rows per queue key (tile)
constexpr uint32_t pr_log2(uint32_t x) { return x <= 1 ? 0 : 1 + pr_log2(x / 2); }
#define TB         (pr_log2(KW))       // bits of the in-tile row
#ifndef HWBAR
#define HWBAR      0        // 1: the machine's global barrier (vx_barrier, global bit); 0: the software barrier
#endif
#define NBLK       ((NASSUME) / (GLANES))      // blocks = cores (one block of SPLIT warps per core)
#ifndef MAXIT
#define MAXIT      200      // safety cap (the tests have none)
#endif
#define DAMPING    0.85
#define ERR_DELTA  0.00001

typedef struct { uint64_t send, recv, upd, total, msgs, iters, t0, t1; } phase_cycles_t;

typedef struct {
    uint64_t aoff_addr;       // int[P][lrows + 1]: PE p's row offsets of A2 (absolute, into anz)
    uint64_t anz_addr;        // int[nnz]: A2's column ids (global), PE by PE, row by row
    uint64_t pr_addr;         // double[P * lrows]: pr, PE p's rows at p * lrows (in: 1/n, out: the answer)
    uint64_t cur_addr;        // double[P * lrows]: curr
    uint64_t err_addr;        // double[warps * GLANES] lane partial errors, then double[warps] warp errors
    uint64_t qs_addr;         // ACTOR=1: queue regions, N * ntile x QS64 x 8 B (queue t * N + owner)
    uint64_t tails_addr;      // ACTOR=1: uint32_t[N * ntile] (model written)
    uint64_t pool_addr;       // ACTOR=1: overflow chunks
    uint64_t ctab_addr;       // ACTOR=1: uint64_t[N * ntile * ovf_maxc]
#if SWQ
    uint64_t swq_addr;        // SWQ: the pool counter (zeroed)
#endif
    uint64_t phase_addr;      // phase_cycles_t[warps]
    uint64_t bar_addr;        // volatile uint32_t[8]
    uint32_t N;               // actors (the kernel's P when ACTOR=1)
    uint32_t numrows;         // rows (= columns) of A2
    uint32_t lrows;           // rows per PE, ceil(numrows / P)
    uint32_t ovf_maxc;        // ACTOR=1: chunks per queue (host: exact need + 1)
    uint32_t ovf_nch;         // ACTOR=1: pool size in chunks (host: exact need + slack)
    uint32_t ntile;           // ACTOR=1: row tiles per actor, ceil(lrows / TILE_W)
} args_t;
#endif
