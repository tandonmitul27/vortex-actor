#ifndef LMEM_BYTES
#define LMEM_BYTES 16384   // scratchpad bytes per core (sensitivity runs pass -DLMEM_BYTES; 2026-11-15)
#endif
// Sparse matrix permutation (bale's permute_matrix, apps/permute_matrix_src): B = A with row i moved to row
// rperminv[i] and every column j renamed cperminv[j].
// Input: the bale_actor permute tests' inputs: rperminv = rand_permp(n, seed), cperminv = rand_permp(n, seed + 1) and
// bale's erdos_renyi_random_graph(n, p, DIRECTED, NOLOOPS, seed + 2), n = PM_LROWS * N (-n), p = 2 (PM_NZ - 1) / n (-Z).
// Rows of A and Ap, and the permutation arrays, are distributed cyclically: index r belongs to PE r % P at local
// index r / P.
// ACTOR=1 (P = N actors). Three rounds; every message is 32 bits and every queue is keyed (owner, tile).
//          1. Row lengths: every local row r sends {rperminv[r] / P << 16 | length} to owner rperminv[r] % P, which
//             stores it; each actor prefix-sums its counts into its Ap offsets.
//          2. Request: every local nonzero (r, c) sends {new row rperminv[r] | c's index in its table tile} to the
//             column's owner c % P, queue (owner, table tile (c / P) / TILE_W).
//          3. Forward: the column owner's warp serves one table tile at a time with that tile of its cperminv in the
//             scratchpad, and sends {new row's index in its row tile | cperminv[c]} to the new row's owner, queue
//             (owner, row tile of RTILE rows); the owner's warp places each nonzero with a scratchpad atomic on that row tile's
//             offsets (Ap.lnonzero[loffset[row] + wrkoff[row]++] = col). Two messages per nonzero and no reply.
//          Request and forward queues are one queue set (keys t * N + owner: request tiles first, then row tiles).
// ACTOR=0 (P = 1): bale's permute_matrix_agi on one GPU, grid stride: rperm[rperminv[i]] = i; the offsets of Ap (a
//          parallel scan of A's row lengths in rperm order); a thread per row of Ap copies row rperm[i] of A; a
//          thread per row of Ap replaces each column j by cperminv[j].
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
#define QS32       (((((Q_CAP) * 4 + 63) / 64) | 1) * 16)  // whole odd lines, in 32-bit words
#ifndef SWQ
#define SWQ        0        // 1: software append queue, no L2 append unit (../swq.h)
#endif
#ifndef OVF_CH
#define OVF_CH     256
#endif
#ifndef WC_U
#define WC_U       4
#endif
#define NBAR       12
#ifndef TILE_W
#define TILE_W     1024     // words per scratchpad tile (table tile and row tile): SPLIT tiles fill 16 KB
#endif
#ifndef PROBE_NOAPPLY
#define PROBE_NOAPPLY 0     // 1: measurement only, the receive reads every message and skips the apply (B off only;
#endif                      //    results are not checked, message counts are)
// Ablation switches (results/mapping/ABLATION.md). Both default off (= the full design).
#ifndef NOTILE
#define NOTILE     0        // 1: piece B off: one request and one forward queue per owner; the column owner reads its
#endif                      //    cperminv and the row owner takes positions (L2 atomics) in memory. TILE_W and RTILE are
                            //    then only the field widths and must be >= lrows.
#ifndef LANE_RECV
#define LANE_RECV  0        // 1: piece C off: a lane drains a whole queue by itself (lane ln takes the warp's queues
#endif                      //    ln, ln + 16, ...); with tiles each lane has its own TILE_W / 16-word tile
static_assert((TILE_W & (TILE_W - 1)) == 0 && (NOTILE || (TILE_W) * (SPLIT) * 4 <= (LMEM_BYTES)), "TILE_W: a power of 2, SPLIT tiles in 16 KB");
constexpr uint32_t pm_log2(uint32_t x) { return x <= 1 ? 0 : 1 + pm_log2(x / 2); }
#ifndef RTILE
#define RTILE      128       // rows per row tile (forward queues, placement): the Ap lines being filled at once are
#endif                       // (warps) x RTILE x (nonzeros per row) x 4 B = 64 x 128 x 18 x 4 B = 590 KB, inside the 1 MB L2
static_assert(((RTILE) & ((RTILE) - 1)) == 0 && (RTILE) <= (TILE_W), "RTILE: a power of 2, at most TILE_W");
#define KW         ((NOTILE || !LANE_RECV) ? (TILE_W) : (TILE_W) / (GLANES))   // columns per table tile (request key)
#define RK         ((RTILE) < (KW) ? (RTILE) : (KW))                            // rows per row tile (forward key)
#define TB         (pm_log2(KW))       // bits of an in-tile column (requests)
#define RB         (pm_log2(RK))       // bits of an in-tile row (forwards)
#define CB         (32 - (RB))         // bits of a global column in a forward (n <= 2^CB)

#ifndef PROBE_TIMERS
#define PROBE_TIMERS 0      // 1: measurement only, time the send calls and the receive calls (serve_fwd's forwards count as receive)
#endif
typedef struct { uint64_t p1, scan, p2, p3, total, msgs, t0, t1;
#if PROBE_TIMERS
                 uint64_t snd, rcv;
#endif
} phase_cycles_t;

typedef struct {
    uint64_t aoff_addr;       // int[P][lrows + 1]: PE p's row offsets of A (absolute, into anz)
    uint64_t anz_addr;        // int[nnz]: A's column ids (global), PE by PE, row by row
    uint64_t rpi_addr;        // int[P][lrows]: rperminv, PE p's part
    uint64_t cpi_addr;        // int[P][lrows]: cperminv, PE p's part
    uint64_t cnt_addr;        // ACTOR=1: uint32_t[N][lrows] row lengths; ACTOR=0: int[n] rperm
    uint64_t poff_addr;       // ACTOR=1: uint32_t[N][lrows + 1] Ap offsets (into the PE's segment); ACTOR=0: [n + 1]
    uint64_t pnz_addr;        // ACTOR=1: int[N][tcap] Ap column ids, PE p's segment at p * tcap; ACTOR=0: int[nnz]
    uint64_t sums_addr;       // scan scratch: uint32_t[threads] (+ uint32_t[warps] after it, ACTOR=0)
    uint64_t qs_addr;         // ACTOR=1: queue regions, N * (ntile3 + ntile) queues of QS32 x 4 B (the program's layout)
    uint64_t tails_addr;      // ACTOR=1: uint32_t[N * (ntile3 + ntile)] (counts without the count instruction)
    uint64_t pool_addr;       // ACTOR=1: overflow chunks
    uint64_t ctab_addr;       // ACTOR=1: uint64_t[N * (ntile3 + ntile) * ovf_maxc]
#if SWQ
    uint64_t swq_addr;        // SWQ: the pool counter (zeroed)
#endif
#if NOTILE
    uint64_t wrk_addr;        // NOTILE: uint32_t[N][lrows] wrkoff, the row owner's next position per row
#endif
    uint64_t phase_addr;      // phase_cycles_t[warps]
    uint64_t bar_addr;        // volatile uint32_t[NBAR]
    uint32_t N;               // actors (the kernel's P when ACTOR=1)
    uint32_t numrows;         // rows (= columns) of A
    uint32_t lrows;           // rows per PE, n / P
    uint32_t tcap;            // ACTOR=1: Ap segment per PE (the largest PE's nonzeros)
    uint32_t ovf_maxc;        // ACTOR=1: chunks per queue (host: exact need + 1)
    uint32_t ovf_nch;         // ACTOR=1: pool size in chunks (host: exact need + slack)
    uint32_t ntile;           // ACTOR=1: row tiles per actor, ceil(lrows / RK) (forward queues)
    uint32_t ntile3;          // ACTOR=1: table tiles per actor, ceil(lrows / KW) (request queues)
} args_t;
#endif
