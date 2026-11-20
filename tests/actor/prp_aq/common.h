#ifndef LMEM_BYTES
#define LMEM_BYTES 16384   // scratchpad bytes per core (sensitivity runs pass -DLMEM_BYTES; 2026-11-15)
#endif
// PageRank (pull), after HClib-Actor's bale_actor tests (srirajpaul/hclib, branch bale3_actor):
// test/pagerank_pull_LU_selector.cpp (the actor version) and test/pagerank_pull_LU_agi.cpp (the non-actor baseline).
// Input and iteration as pr_aq (the push tests'): A2 = L + U of erdos_renyi_random_graph(n, p, UNDIRECTED, NOLOOPS),
// n = PR_LROWS * N, rows cyclic (row r on PE r % P at local row r / P). Per iteration every row y pulls
// curr[y] = sum over its nonzeros x of pr[x] / deg(x); curr = (1 - 0.85) / n + 0.85 curr; error = sum |curr - pr|;
// stop if error < 1e-5, else pr = curr. A2 is symmetric, so the answer equals push's.
// ACTOR=1 (P = N actors): pagerank_selector (pull), two message types per iteration:
//          REQUEST  row owner (asker) -> owner of x: {asker | y | x}. The handler reads the owner's pr[x] and deg(x)
//                   and replies (the division the asker's handler does is done here, so the reply is one value).
//          RESPONSE owner of x -> asker: {pr[x] / deg(x) | y}; the handler adds it, curr[y] += prf.
//          On the append queue as the index gather's best path: the asker appends its REQUEST straight to the queue
//          of (owner, source tile), tile = (x / P) / RQ_W, as one 32-bit word (REQ32: the tile is in the queue key);
//          the owner's warp serves one tile queue at a time (warp cooperative, reads confined to RQ_W rows) and
//          appends each RESPONSE to the queue of (asker, row tile), tile = y / TILE_W, as one 64-bit word (prf
//          rounded to its top 64 - TB bits | the in-tile row, as pr_aq). The asker's warp applies one row tile at a
//          time in a TILE_W-double scratchpad tile and writes it out as whole lines, the update and the error fused.
//          The two word sizes are two queue geometries (and two overflow pools), switched between rounds; each
//          consumer zeroes the tails of the queues it drained. Each round is one leaf function over the warp's actors.
// ACTOR=0 (P = 1): pagerank_agi (pull) on one GPU: a thread per row (grid stride) sums pr[x] / deg(x) over its
//          nonzeros (plain loads), then the update and error.
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
#ifndef RQ_W
#define RQ_W       256      // source rows per REQUEST queue (the serve's read window)
#endif
#ifndef TILE_W
#define TILE_W     512      // doubles per scratchpad tile: SPLIT tiles fill the core's 16 KB local memory
#endif
// Ablation switches (results/mapping/ABLATION.md, ../abl.h). Both default off (= the full design).
#ifndef NOTILE
#define NOTILE     0        // 1: piece B off: one REQUEST queue per owner (RQ_W >= lrows) and one RESPONSE queue per
#endif                      //    asker; the RESPONSE handler adds into curr in memory (an L2 amocas loop per message),
                            //    then the update runs over the rows. TILE_W is then only the row field width (>= lrows).
#ifndef LANE_RECV
#define LANE_RECV  0        // 1: piece C off: a lane serves / drains a whole queue by itself; with tiles each lane
#endif                      //    has its own TILE_W / 16-double RESPONSE tile
static_assert((TILE_W & (TILE_W - 1)) == 0 && (NOTILE || (TILE_W) * (SPLIT) * 8 <= (LMEM_BYTES)), "TILE_W: a power of 2, SPLIT tiles in 16 KB");
#define KW         ((NOTILE || !LANE_RECV) ? (TILE_W) : (TILE_W) / (GLANES))   // rows per RESPONSE queue key (tile)
static_assert(((RQ_W) & ((RQ_W) - 1)) == 0, "RQ_W: a power of 2");
static_assert(((NASSUME) & ((NASSUME) - 1)) == 0, "NASSUME: a power of 2");
constexpr uint32_t pr_log2(uint32_t x) { return x <= 1 ? 0 : 1 + pr_log2(x / 2); }
#define TB         (pr_log2(KW))       // bits of the in-tile row (RESPONSE)
#define RB         (pr_log2(RQ_W))     // bits of the in-tile source row (REQUEST)
#define LNB        (pr_log2(NASSUME))  // bits of the asker
#define YB         (32 - (LNB) - (RB)) // bits of the asker's row y (REQUEST)
#ifndef RQ_CAP
#define RQ_CAP     ((RQ_W) * 64)       // per REQUEST queue region (32-bit words; A2 has ~68 nonzeros a row): the rest overflows into pool chunks
#endif
#ifndef RP_CAP
#define RP_CAP     ((KW) * 64)         // per RESPONSE queue region (64-bit words)
#endif
#define RQS32      (((((RQ_CAP) * 4 + 63) / 64) | 1) * 16)   // whole odd lines, in 32-bit words
#define RPS64      (((((RP_CAP) * 8 + 63) / 64) | 1) * 8)    // whole odd lines, in 64-bit words
#ifndef SWQ
#define SWQ        0        // 1: software append queue, no L2 append unit (../swq.h)
#endif
#ifndef OVF_CH
#define OVF_CH     256
#endif
#ifndef WC_U
#define WC_U       4
#endif
#ifndef HWBAR
#define HWBAR      0        // 1: the machine's global barrier (vx_barrier, global bit); 0: the software barrier
#endif
#define NBLK       ((NASSUME) / (GLANES))      // blocks = cores (one block of SPLIT warps per core)
#ifndef MAXIT
#define MAXIT      200      // safety cap (the tests have none)
#endif
#define DAMPING    0.85
#define ERR_DELTA  0.00001

typedef struct { uint64_t req, srv, rep, total, msgs, iters, t0, t1; } phase_cycles_t;

typedef struct {
    uint64_t aoff_addr;       // int[P][lrows + 1]: PE p's row offsets of A2 (absolute, into anz)
    uint64_t anz_addr;        // int[nnz]: A2's column ids (global), PE by PE, row by row
    uint64_t pr_addr;         // double[P * lrows]: pr, PE p's rows at p * lrows (in: 1/n, out: the answer)
    uint64_t cur_addr;        // double[P * lrows]: curr
    uint64_t err_addr;        // double[warps * GLANES] lane partial errors, then double[warps] warp errors
    uint64_t rqs_addr;        // ACTOR=1: REQUEST queues, N * nrq x RQS32 x 4 B (queue t * N + owner)
    uint64_t rqtails_addr;    // ACTOR=1: uint32_t[N * nrq]
    uint64_t rqpool_addr;     // ACTOR=1: REQUEST overflow chunks
    uint64_t rqctab_addr;     // ACTOR=1: uint64_t[N * nrq * rq_maxc]
    uint64_t rps_addr;        // ACTOR=1: RESPONSE queues, N * nrp x RPS64 x 8 B (queue t * N + asker)
    uint64_t rptails_addr;    // ACTOR=1: uint32_t[N * nrp]
    uint64_t rppool_addr;     // ACTOR=1: RESPONSE overflow chunks
    uint64_t rpctab_addr;     // ACTOR=1: uint64_t[N * nrp * rp_maxc]
#if SWQ
    uint64_t swq_addr;        // SWQ: the two pool counters (REQUEST word 0, RESPONSE word 16), zeroed
#endif
    uint64_t phase_addr;      // phase_cycles_t[warps]
    uint64_t bar_addr;        // volatile uint32_t[8]
    uint32_t N;               // actors (the kernel's P when ACTOR=1)
    uint32_t numrows;         // rows (= columns) of A2
    uint32_t lrows;           // rows per PE, ceil(numrows / P)
    uint32_t nrq;             // ACTOR=1: source tiles per actor, ceil(lrows / RQ_W)
    uint32_t nrp;             // ACTOR=1: row tiles per actor, ceil(lrows / TILE_W)
    uint32_t rq_maxc, rq_nch; // ACTOR=1: REQUEST chunks per queue / pool chunks (host: exact need + 1 / + slack)
    uint32_t rp_maxc, rp_nch; // ACTOR=1: RESPONSE
} args_t;
#endif
