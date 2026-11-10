// Sparse matrix transpose, after HClib-Actor's bale_actor tests (srirajpaul/hclib, branch bale3_actor):
// test/transpose_selector.cpp (the actor version) and test/transpose_agi.cpp (the non-actor baseline).
// Input: the tests' matrix, bale's erdos_renyi_random_graph(n, p, DIRECTED, NOLOOPS) with n = TR_LROWS * N rows
// (TR_LROWS is the tests' -n, rows per PE) and p = 2 (TR_NZ - 1) / n (TR_NZ is -Z). Rows of A, and rows of At
// (columns of A), are distributed cyclically: row r belongs to PE r % P at local row r / P.
// ACTOR=1 (P = N actors): transpose_matrix_selector. Phase 1: every local nonzero (row, col) sends col / P to
//          owner col % P; the handler counts it, lcounts[col]++. Each actor prefix-sums its lcounts into its At
//          offsets. Phase 2: every local nonzero sends {row, col / P} to owner col % P; the handler places it,
//          At.lnonzero[loffset[col] + wrkoff[col]++] = row. Handlers run on the owner's warp, 16 messages at a
//          time, so their ++ are atomics on the owner's own arrays. The selector's self-checks are the host's.
//          On the append queue as histo's best (WC_TILE + DIRECT_BT): the sender appends to the queue of (owner,
//          column tile), tile = (col / P) / TILE_W, queue t * N + owner, in 32-bit words (phase 1: the in-tile
//          column; phase 2: row << TB | in-tile column); the owner's warp applies one tile queue at a time with
//          scratchpad atomics on a TILE_W-word scratchpad tile: phase 1 counts into it and writes it out as whole
//          lines, phase 2 loads it from the offsets and takes each position from it (loffset[col] + wrkoff[col]).
//          Each round is one leaf function over the warp's actors (one prologue per round).
// ACTOR=0 (P = 1): transpose_matrix_agi on one GPU: fetch_and_inc on the column counts (a thread per nonzero,
//          grid stride), a prefix sum (a parallel scan: one PE holds every column), then a thread per row (grid
//          stride): pos = fetch_and_add(start[col], 1), At.nonzero[pos] = row.
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
#define MSG64      0        // 1: phase-2 messages are 64-bit (B off at a size where {row, column} needs more than 32 bits)
#endif
#define QS32       (((((Q_CAP) * ((MSG64) ? 8 : 4) + 63) / 64) | 1) * 16)  // whole odd lines, in 32-bit words
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
#define TILE_W     1024     // words per scratchpad tile: SPLIT tiles fill the core's 16 KB local memory
#endif
// Ablation switches (results/mapping/ABLATION.md). Both default off (= the full design).
#ifndef NOTILE
#define NOTILE     0        // 1: piece B off: one queue per owner; the handlers update the owner's arrays in memory
#endif                      //    (L2 atomics). TILE_W is then only the column field width and must be >= lnc.
#ifndef LANE_RECV
#define LANE_RECV  0        // 1: piece C off: a lane drains a whole queue by itself (lane ln takes the warp's queues
#endif                      //    ln, ln + 16, ...); with tiles each lane has its own TILE_W / 16-word tile
static_assert((TILE_W & (TILE_W - 1)) == 0 && (NOTILE || (TILE_W) * (SPLIT) * 4 <= 16384), "TILE_W: a power of 2, SPLIT tiles in 16 KB");
#define KW         ((NOTILE || !LANE_RECV) ? (TILE_W) : (TILE_W) / (GLANES))   // columns per queue key (tile)
constexpr uint32_t tr_log2(uint32_t x) { return x <= 1 ? 0 : 1 + tr_log2(x / 2); }
#define TB         (tr_log2(KW))       // bits of the in-tile column

typedef struct { uint64_t p1, scan, p2, total, msgs, t0, t1, pad; } phase_cycles_t;

typedef struct {
    uint64_t aoff_addr;       // int[P][lrows + 1]: PE p's row offsets of A (absolute, into anz)
    uint64_t anz_addr;        // int[nnz]: A's column ids (global), PE by PE, row by row
    uint64_t cnt_addr;        // ACTOR=1: uint32_t[N][lnc] lcounts; ACTOR=0: uint32_t[numcols] column counts
    uint64_t wrk_addr;        // ACTOR=1: uint32_t[N][lnc] wrkoff; ACTOR=0: uint32_t[numcols] next position per column
    uint64_t toff_addr;       // ACTOR=1: uint32_t[N][lnc + 1] At offsets (into the PE's segment); ACTOR=0: [numcols + 1]
    uint64_t tnz_addr;        // ACTOR=1: int[N][tcap] At row ids, PE p's segment at p * tcap; ACTOR=0: int[nnz]
    uint64_t sums_addr;       // scan scratch: uint32_t[threads] (+ uint32_t[warps] after it, ACTOR=0)
    uint64_t qs_addr;         // ACTOR=1: queue regions, N * ntile x QS32 x 4 B (queue t * N + owner)
    uint64_t tails_addr;      // ACTOR=1: uint32_t[N * ntile] (model written)
    uint64_t pool_addr;       // ACTOR=1: overflow chunks
    uint64_t ctab_addr;       // ACTOR=1: uint64_t[N * ntile * ovf_maxc]
#if SWQ
    uint64_t swq_addr;        // SWQ: the pool counter (zeroed)
#endif
    uint64_t phase_addr;      // phase_cycles_t[warps]
    uint64_t bar_addr;        // volatile uint32_t[8]
    uint32_t N;               // actors (the kernel's P when ACTOR=1)
    uint32_t numrows;         // rows (= columns) of A
    uint32_t lrows;           // rows of A per PE, ceil(numrows / P)
    uint32_t lnc;             // ACTOR=1: rows of At per PE, ceil(numrows / N)
    uint32_t tcap;            // ACTOR=1: At segment per PE (the largest PE's nonzeros)
    uint32_t ovf_maxc;        // ACTOR=1: chunks per queue (host: exact need + 1)
    uint32_t ovf_nch;         // ACTOR=1: pool size in chunks (host: exact need + slack)
    uint32_t ntile;           // ACTOR=1: column tiles per actor, ceil(lnc / KW)
} args_t;
#endif
