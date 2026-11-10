// Toposort, after HClib-Actor's bale_actor tests (srirajpaul/hclib, branch bale3_actor):
// test/toposort_selector.cpp (the actor version) and test/toposort_agi.cpp (the non-actor baseline).
// Input: the tests' matrix, generate_toposort_input: T = transpose(erdos_renyi_random_graph(n, p, UNDIRECTED, LOOPS,
// seed 5)) (upper triangular, full diagonal), M = T with random row and column permutations, n = TP_LROWS * N
// rows (TP_LROWS is the tests' -n) and p = 2 nz_per_row / (n - 1) (-Z, default 10). Rows of M belong to PE r % P
// (local row r / P), columns of M (rows of tmat = M^T) to PE c % P (local column c / P).
// Goal: rperm, cperm with M[rperm^-1][cperm^-1] upper triangular. rowcnt[r] = nonzeros left in row r, rowsum[r] =
// the sum of their column ids, so a row with rowcnt 1 names its last column (the tests' "cool trick").
// ACTOR=1 (P = N actors): toposort_matrix_selector, run level by level (bale's "Do it in levels"; the selector's
//          level field is then the level, so it is not carried). Two message types, alternating rounds:
//          MATCH  row owner -> owner of column c (the row's last column): {c / P}. The handler expands the local
//                 column: one DEC per nonzero row r of column c.
//          DEC    column owner -> owner of row r: {in-tile row | c}, to the queue of (owner, row tile). The
//                 handler: rowsum -= c, rowcnt--, and a row that reaches rowcnt 1 gets this level and sends its MATCH.
//          Handlers run on the owner's warp, 16 messages at a time, so the -= / -- are atomics on the owner's own
//          arrays. Round r drains one type and appends the other, so no queue is drained and appended in the same
//          round: the queues are monotone (tails only grow, each consumer keeps a head) and a round ends with one
//          tail flush. Then (the selector's second half): level histogram, prefix over PEs and levels, rperm =
//          (n - 1) - level_start[level]++, and CPERM messages {in-tile column | rperm} to the column owners,
//          applied in a scratchpad tile written out as whole lines.
// ACTOR=0 (P = 1): toposort_matrix_agi on one GPU, level by level: a thread per queued row (grid stride) claims its
//          position (fetch_and_inc), writes rperm / cperm, reads its column of tmat and does the two fetch_and_adds
//          per nonzero; a row reaching rowcnt 1 is queued (fetch_and_inc on the queue end).
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
#define Q_CAP      4096     // per-queue region (32-bit messages); the rest overflows into pool chunks
#endif
#ifndef MSG64
#define MSG64      0        // 1: 64-bit messages (B off at a size where {row, column} needs more than 32 bits)
#endif
#if MSG64
typedef uint64_t msg_t;
#else
typedef uint32_t msg_t;
#endif
#define QS32       (((((Q_CAP) * ((MSG64) ? 8 : 4) + 63) / 64) | 1) * 16)   // whole odd lines, in 32-bit words
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
#define TILE_W     1024     // rows per DEC queue tile; columns per CPERM scratchpad tile (SPLIT tiles in 16 KB)
#endif
// Ablation switches (results/mapping/ABLATION.md, ../abl.h). Both default off (= the full design).
#ifndef NOTILE
#define NOTILE     0        // 1: piece B off: one DEC and one CPERM queue per owner; CPERM is applied in memory.
#endif                      //    TILE_W >= TP_L is then only the index field width.
#ifndef LANE_RECV
#define LANE_RECV  0        // 1: piece C off: a lane drains a whole queue by itself; with tiles each lane has its own
#endif                      //    TILE_W / 16-word CPERM tile (and DEC queues are keyed by the same smaller tile)
static_assert((TILE_W & (TILE_W - 1)) == 0 && (NOTILE || (TILE_W) * (SPLIT) * 4 <= 16384), "TILE_W: a power of 2, SPLIT tiles in 16 KB");
#define KW         ((NOTILE || !LANE_RECV) ? (TILE_W) : (TILE_W) / (GLANES))   // rows per queue key (tile)
#ifndef NUM_CORES
#define NUM_CORES  16
#endif
#define TP_GPC     ((((NASSUME) / (GLANES)) + (NUM_CORES) - 1) / (NUM_CORES))   // groups per core (D off: 4); each has its own slice
#define SPW        ((NOTILE) ? 16384 / ((SPLIT) * 4 * (TP_GPC)) : (TILE_W))     // scratchpad words per warp
constexpr uint32_t tp_log2(uint32_t x) { return x <= 1 ? 0 : 1 + tp_log2(x / 2); }
#define TB         (tp_log2(KW))       // bits of the in-tile index
#define LNB        (tp_log2(NASSUME))
static_assert(((NASSUME) & ((NASSUME) - 1)) == 0, "NASSUME: a power of 2");
#ifndef HWBAR
#define HWBAR      0        // 1: the machine's global barrier (vx_barrier, global bit); 0: the software barrier
#endif
#define NBLK       ((NASSUME) / (GLANES))      // blocks = cores (one block of SPLIT warps per core)
#ifndef MAXR
#define MAXR       1024     // rounds (ACTOR=1) / levels (ACTOR=0), safety cap
#endif
#ifndef MAXL
#define MAXL       256      // levels (ACTOR=1 histogram)
#endif
static_assert(2 * (MAXL) <= (SPW), "the level counters live in the scratchpad tile");
#ifndef TP_L
#define TP_L       1000     // ACTOR=1: rows (= columns) per actor, compile time (the host checks it is TP_LROWS)
#endif
#define LR         (TP_L)
#define NT         (((TP_L) + (KW) - 1) / (KW))   // row tiles per actor
static_assert(!(NOTILE) || NT == 1, "NOTILE needs TILE_W >= TP_L");

#ifndef PROF
#define PROF       0        // 1: phase 1 cycles split into MATCH rounds, DEC rounds, barriers (diagnostic)
#endif
typedef struct { uint64_t p1, p2, total, msgs, rounds, levels, t0, t1, prof[8]; } phase_cycles_t;

typedef struct {
    uint64_t toff_addr;       // uint32_t[P][lrows + 1]: PE p's column offsets of tmat (absolute, into tnz)
    uint64_t tnz_addr;        // uint32_t[nnz]: tmat's row ids (global), PE by PE, column by column
    uint64_t cnt_addr;        // uint32_t[P * lrows]: rowcnt, PE p's rows at p * lrows
    uint64_t sum_addr;        // uint32_t[P * lrows]: rowsum
    uint64_t lev_addr;        // ACTOR=1: uint32_t[P * lrows]: level
    uint64_t mat_addr;        // ACTOR=1: uint32_t[P * lrows]: matched column
    uint64_t rperm_addr;      // uint32_t[P * lrows]
    uint64_t cperm_addr;      // uint32_t[P * lrows]
    uint64_t iq_addr;         // ACTOR=1: uint32_t[]: PE p's initial rowcnt-1 rows at iqoff[p]; ACTOR=0: the queue (n)
    uint64_t iqoff_addr;      // ACTOR=1: uint32_t[P + 1]
    uint64_t ctr_addr;        // ACTOR=0: uint32_t[2]: pivots, queue end
    uint64_t hist_addr;       // ACTOR=1: uint32_t[MAXL][N] level counts, then their exclusive prefix over PEs
    uint64_t ltot_addr;       // ACTOR=1: uint32_t[MAXL] level totals, then their exclusive prefix
    uint64_t flag_addr;       // ACTOR=1: uint32_t[MAXR]: round r appended something
    uint64_t sums_addr;       // uint32_t[warps * GLANES] warp scratch
    uint64_t qs_addr;         // ACTOR=1: queue regions, N * (1 + ntile) x QS32 x 4 B
    uint64_t tails_addr;      // ACTOR=1: uint32_t[N * (1 + ntile)] (model written)
    uint64_t heads_addr;      // ACTOR=1: uint32_t[N * (1 + ntile)] (each consumer's own)
    uint64_t pool_addr;       // ACTOR=1: overflow chunks
    uint64_t ctab_addr;       // ACTOR=1: uint64_t[N * (1 + ntile) * ovf_maxc]
#if SWQ
    uint64_t swq_addr;        // SWQ: the pool counter (zeroed)
#endif
    uint64_t phase_addr;      // phase_cycles_t[warps]
    uint64_t bar_addr;        // volatile uint32_t[8]
    uint32_t N;               // actors (the kernel's P when ACTOR=1)
    uint32_t numrows;         // n
    uint32_t lrows;           // rows (= columns) per PE, ceil(n / P)
    uint32_t ovf_maxc;        // ACTOR=1: chunks per queue (host: exact need + 1)
    uint32_t ovf_nch;         // ACTOR=1: pool size in chunks (host: exact need + slack)
    uint32_t ntile;           // ACTOR=1: row tiles per actor, ceil(lrows / TILE_W)
    uint32_t iq0;             // ACTOR=0: initial queue length
} args_t;
// queues (one geometry): MATCH to owner o = q o; DEC (phase 1) and CPERM (phase 2, after the queues are emptied)
// to (owner o, tile t) = q N + t * N + o
#endif
