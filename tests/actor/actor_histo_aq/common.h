#ifndef LMEM_BYTES
#define LMEM_BYTES 16384   // scratchpad bytes per core (sensitivity runs pass -DLMEM_BYTES; 2026-11-15)
#endif
// actor_histo_aq: Bale histogram with message transport on the combining
// append queue. This is mechanism 1 of docs/aq_mechanisms.md carried across
// from index gather to a second kernel, to test whether it is a general actor
// communication layer or an index gather trick.
//
// The histogram is one round. Actor s sends M updates; update i targets global
// bucket g = mix(s,i) % (L*N), which belongs to owner g % N at its local bucket
// g / N. The owner accumulates in scratchpad and writes its L counts once.
//
// WHAT THE QUEUE REPLACES. The dense kernel sorts the send pool by destination
// with an on chip counting sort, publishes it densely and records run starts.
// The SORT_OFF probe priced those sorts at 4.56M of the kernel's 11.15M cycles,
// which is 41%. Appending removes them: one instruction per message and no
// destination address computed anywhere.
//
// WHAT DOES NOT CARRY OVER. There is no reply round, so keying a reply mailbox
// by question tile has no counterpart here. Keying the owner's own inbox by
// table tile has no effect either while the owner's L buckets fit in
// scratchpad, which they do for any L up to about 250. Its value appears only
// at larger L, and larger L is exactly what freeing the scratchpad allows.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

#ifndef M
#define M           10000   // updates sent by each actor
#endif
#ifndef L
#define L           10      // buckets owned by each actor
#endif
#ifndef LANES
#define LANES       16
#endif
#define PW          1
#ifndef LMEMW
#define LMEMW       4096
#endif
#ifndef NASSUME
#define NASSUME     256
#endif
#ifndef AQ_BASE
#define AQ_BASE     0xD0000000u
#endif
#ifndef AQ_SIZE
#define AQ_SIZE     0x100000u
#endif
#ifndef PROFILE_SUB
#define PROFILE_SUB 0
#endif

#define AP_BASE    ((uint64_t)(AQ_BASE) + (uint64_t)(AQ_SIZE))   // apply engine window
#define AP_SPEC(keyshift, keybits, valshift, valbits, op, wsize, valconst) \
    ((uint64_t)(keyshift) | ((uint64_t)(keybits) << 8) | ((uint64_t)(valshift) << 16) \
     | ((uint64_t)(valbits) << 24) | ((uint64_t)(op) << 32) | ((uint64_t)(wsize) << 40) \
     | ((uint64_t)(valconst) << 48))
#define AP_STORE   0
#define AP_ADD     1
#define GLANES     ((LANES) * (PW))
#define NBA        ((NASSUME) / (GLANES) / (VACT) / (QSG))   /* sender keys */
// queue capacity: the mean is M * GLANES / N, allow 1.4x
#ifndef AQ_QCAP
#ifndef SWQ
#define SWQ 0             // 1: SOFTWARE queues: no append unit; slot = atomicAdd(&tail[q], 1), then a plain store
#endif                    //    into the queue storage (lines merge in a write-back L2 with no-fetch allocation)
#ifndef VPATH
#define VPATH 0           // 1: the actors-per-lane kernel (plain path, SPLIT=1): VACT actors per lane, QSG blocks per sender key
#endif
#ifndef VACT
#define VACT 1
#endif
#ifndef QSG
#define QSG 1
#endif
#ifndef AQ_QCAP
#define AQ_QCAP    ((((M) * (GLANES) * 14 * (VACT) * (QSG)) / ((NASSUME) * 10) + 63) / 64 * 64)
#endif
#endif

#ifndef CHUNKS
#define CHUNKS      1       // P3: send and receive in CHUNKS pieces of the actor's
                            //     own queries instead of one whole round, so the
                            //     mail in flight is M/CHUNKS per actor and the
                            //     queue storage is refilled each chunk rather than
                            //     sized for the whole run. The destination keys are
                            //     unchanged, so each queue still receives its whole
                            //     share of a chunk and combining is untouched; only
                            //     the distance between filling a line and reading it
                            //     shrinks. Pays only under a cache that ALLOCATES on
                            //     a write, which on this machine means
                            //     -DL2_WRITEBACK=1 with VORTEX_L2_NOFETCH=1: under
                            //     write through no write allocate the line never
                            //     enters the cache at all and residency is moot.
                            //     Costs two extra barriers and one partial line per
                            //     queue per chunk. 1 is the unchunked kernel.
#endif
#ifndef WC_RECV
#define WC_RECV 0         // 1: warp-cooperative receive (plain path): a warp reads ONE actor's queue at a time
#endif
#ifndef WC_U
#define WC_U 4            // warp-coop: queue lines loaded before any is used
#endif
#ifndef WC_TILE
#define WC_TILE 0         // 1: warp-cooperative tile apply (DIRECT_BT): a warp applies ONE actor's tile queue at a time
#endif                    //    into a per-warp scratchpad tile with scratchpad atomics; counts in natural layout
#ifndef HYB_OUT
#define HYB_OUT 0         // 1: per-lane tile receive (each lane its actor's tile) + warp-cooperative output: after
#endif                    //    each tile the warp writes its 16 actors' tiles as whole sequential lines (natural layout)
#ifndef RECV_U
#define RECV_U 1          // per-lane receive loop unroll (loads first)
#endif
#ifndef ACC_ODD
#define ACC_ODD 0         // 1: plain-path scratchpad rows at an ODD stride (L|1) so 16 lanes spread over the banks
#endif
#ifndef RECV_IL
#define RECV_IL 1         // 2: a lane drains two of its queues interleaved into two rows (two independent chains)
#endif
#define ACC_LS ((ACC_ODD) ? ((L) | 1) : (L))
static_assert((NASSUME / GLANES / SPLIT) % (RECV_IL) == 0, "RECV_IL must divide the queues per lane");
#ifndef LANE_TS
#define LANE_TS 0         // 1: every lane records 4 timestamps (start, send end, barrier exit, end) after phase[N]
#endif
#ifndef SPLIT
#define SPLIT       1       // LANES PER ACTOR. The actor count has always been welded
                            // to the machine shape, N = cores x warps x threads, so
                            // asking for warps to fill the pipeline also multiplied
                            // the actors and the working set, which is why four warps
                            // always measured as a loss. SPLIT breaks that: S lanes
                            // share one actor and each emits a DISJOINT slice of that
                            // actor's outgoing mail, so N = lanes / SPLIT and the warp
                            // count is free again. The model is untouched: sending
                            // reads only the actor's own input and writes only to
                            // other actors' mailboxes, and the receive stays bound to
                            // one owner per counter (the whole inbox for sub-lane 0,
                            // or disjoint bucket tiles when tiled). 1 is one actor per
                            // lane, exactly as before.
#endif

#ifndef UNROLL
#define UNROLL      1       // messages formed at once in the send loop. Forming one
                            // is an eighteen deep dependent chain and the machine is
                            // in order with one warp per core, so the chain runs at
                            // about eleven cycles an instruction with nothing to fill
                            // the gaps. Messages are independent, so UNROLL chains
                            // interleave. 1 is the original loop.
#endif

#ifndef TAILFLUSH
#define TAILFLUSH   1       // 0: skip the end of round aq_flush, which is the v0
                            //    queue model that never charged a queue's last
                            //    partial line. Ablation only, see log T36b.
#endif
#ifndef QMAJOR
#define QMAJOR      0       // 1: queue index = sender block * N + owner (SENDER major).
                            //    Owner major puts a lane's NB queues contiguous, so the
                            //    lane stride is NB x the queue stride, a multiple of 64
                            //    lines, and every lane's queue sb lands in the SAME L1
                            //    set: 16 streams into 4 ways at one warp (1.65 DRAM
                            //    reads per line measured), 64 at four warps (4% hit
                            //    ratio measured). Sender major makes the lane stride
                            //    ONE queue, and with QPAD an odd number of lines, so
                            //    consecutive lanes land in consecutive sets.
#endif
#ifndef QPAD
#define QPAD        0       // extra words of storage per queue (16 = one line) so the
                            //    stride is an odd number of lines. Capacity unchanged.
#endif
#ifndef QODD
#define QODD        0       // 1: tile queue stride rounded UP to an ODD number of lines
#endif
#ifndef QROWPAD
#define QROWPAD     0       // spare queues per tile row (QMAJOR). With QODD, 16 puts
                            // lane k of sub-lane j on set s*(k + 16 j) mod 64: the 64
                            // queue streams of a 4-warp core land in 64 distinct L1 sets
#endif
#define BQ_ROW(n)   ((n) + (QROWPAD))             // queues per tile row
#if QODD
#define QSTRIDE_W   (((((AQ_QCAP) + 15) / 16) | 1) * 16)   // odd number of lines (QODD)
#else
#define QSTRIDE_W   ((AQ_QCAP) + (QPAD))
#endif
static_assert((QSTRIDE_W) % 16 == 0 && (QPAD) % 16 == 0, "queue stride must be a whole number of lines");
#ifndef MEMACC
#define MEMACC      0       // 1: accumulate straight into counts[] in memory
                            //    instead of a scratchpad row. Required once
                            //    L * GLANES exceeds the PHYSICAL scratchpad,
                            //    which is 16 KB and is NOT raised by raising
                            //    LMEMW: that only silences the assert, and the
                            //    scratchpad address then wraps and corrupts the
                            //    counts (measured at L=2000, 446,464 wrong
                            //    buckets). This is the regime where the owner's
                            //    table stops fitting on chip and where keying
                            //    the inbox by bucket tile has something to do.
#endif
#ifndef BUCKET_TILE
#define BUCKET_TILE 0       // >0 with MEMACC: the owner re-appends its own inbox
                            //    to its own per bucket-tile queues before
                            //    applying, so each apply pass touches only
                            //    L / BUCKET_TILE of its counts. Mechanism 2b.
#endif
static_assert(!(BUCKET_TILE) || (MEMACC), "BUCKET_TILE needs MEMACC");
static_assert(!(BUCKET_TILE) || (L) % (BUCKET_TILE) == 0, "BUCKET_TILE must divide L");
#ifndef DIRECT_BT
#define DIRECT_BT   0       // 1 with BUCKET_TILE: the SENDER appends straight to the
                            //    owner's (owner, bucket tile) queue. No inbox queues and
                            //    no re-append pass: the sender holds the bucket, so it
                            //    holds the tile. Mechanism 2 keyed at the producer, the
                            //    histogram's P2.
#endif
static_assert(!(DIRECT_BT) || (BUCKET_TILE), "DIRECT_BT needs BUCKET_TILE");
#if BUCKET_TILE
#define BTILE_SZ   ((L) / (BUCKET_TILE))
#ifndef FINE_T
#define FINE_T     1       // WC_TILE: BUCKET_TILE keys the SEND (coarse tiles, few live queues);
                            //    the warp applies each coarse queue in FINE_T scratchpad passes
#endif
#define FTILE_SZ   ((BTILE_SZ) / (FINE_T))
#ifndef HYB_APPLY
#define HYB_APPLY  0       // WC_TILE: the OWNER picks how it applies each coarse queue to its own counts:
                            //    sparse (tail * HYB_K < BTILE_SZ) -> one atomic add per message, no tile
#endif
#ifndef HYB_K
#define HYB_K      32      // 32 = below 0.5 messages per owned line (tile cost ~1 line write per line)
#endif
static_assert(!(HYB_APPLY) || (WC_TILE), "HYB_APPLY needs WC_TILE");
#ifndef OVF
#define OVF        0       // skewed traffic: a full tile queue continues in chunks the L2 append unit takes
#endif                      //    from a shared pool (chunk addresses in ctab[q * OVF_MAXC + c]); 0 = drop
#ifndef OVF_CH
#define OVF_CH     256     // words per overflow chunk (whole lines, a multiple of GLANES)
#endif
#ifndef OVF_MAXC
#define OVF_MAXC   128     // chunks per queue: one chunk-table row
#endif
#ifndef OVF_NCH
#define OVF_NCH    ((NASSUME) * (M) / (OVF_CH) + 1024)   // pool chunks: every message could overflow
#endif
static_assert(!(OVF) || ((WC_TILE) && (DIRECT_BT)), "OVF needs WC_TILE + DIRECT_BT");
static_assert((OVF_CH) % 16 == 0, "OVF_CH: whole lines of the warp's 16 lanes");
static_assert((FINE_T) == 1 || (WC_TILE), "FINE_T needs WC_TILE");
static_assert((BTILE_SZ) % (FINE_T) == 0, "FINE_T must divide the coarse tile");
// An owner receives about M messages, so a tile queue holds about M/BUCKET_TILE.
// Sizing this at AQ_QCAP silently DROPS at small tile counts: at BUCKET_TILE=8
// the mean is 1250 against AQ_QCAP 896. Same bug class as the IG TQ_CAP fix.
// rounded up to 16 words so a queue's storage is LINE ALIGNED: the model's
// combining line write is one 64 B request at the line's address, which only
// stands for one DRAM write if the queue starts on a line boundary.
#define BQ_MEAN    (((((M) * 14) / ((BUCKET_TILE) * 10)) + 15) / 16 * 16)
#ifndef BQ_CAP              // overridable, like the index gather's TQ_CAP: the
                            // AQ_QCAP floor is sized for the INBOX and is wildly
                            // oversized once BUCKET_TILE is large. At L = 400000
                            // it would ask for 1.2 GB of queue storage. Pass
                            // mu + 8*sqrt(mu) rounded up, mu = M / BUCKET_TILE.
#define BQ_CAP     ((BQ_MEAN) > (AQ_QCAP) ? (BQ_MEAN) : (AQ_QCAP))
#endif
#if QODD
#define BQ_STRIDE_W (((((BQ_CAP) + 15) / 16) | 1) * 16)
#else
#define BQ_STRIDE_W ((BQ_CAP) + (QPAD))          // padded like the inbox queues
#endif
static_assert((BQ_STRIDE_W) % 16 == 0, "tile queue stride must be a whole number of lines");
static_assert(!(QROWPAD) || (QMAJOR), "QROWPAD pads the tile-major (QMAJOR) rows");
static_assert(!(DIRECT_BT) || (unsigned long long)(BUCKET_TILE) * ((NASSUME) + (QROWPAD)) + 1 <= (AQ_SIZE) / 64,
              "AQ window too small for the tile queue ids: raise AQ_SIZE");
#endif

#ifndef APPLY
#define APPLY       0       // 1: the receive round is ONE COMMAND per lane to the
                            //    apply engine (docs/apply_engine_plan.md): ADD every
                            //    word of my inbox queues into my own counts, which
                            //    live in memory bucket major. With BUCKET_TILE the
                            //    command is per tile so a tile's count lines fit the
                            //    engine's combining cache. The lanes issue and wait.
#endif
#if APPLY && !defined(TILE_ACC)
#define TILE_ACC    0       // the engine accumulates; no scratchpad tile
#endif
#ifndef TILE_ACC
#if BUCKET_TILE
#define TILE_ACC    1       // accumulate each bucket tile in SCRATCHPAD and write
#else                       //    its BTILE_SZ counts back once, instead of a
#define TILE_ACC    0       //    read-modify-write per update into memory. This
#endif                      //    is what makes the tile a unit of RESIDENCY
#endif                      //    rather than only a unit of read order: it
                            //    turns O(M) scattered DRAM writes into O(L).
                            //    -DTILE_ACC=0 reproduces the read-only 2b arm.
static_assert(!(TILE_ACC) || (BUCKET_TILE), "TILE_ACC needs BUCKET_TILE");

#ifndef NCONST
#define NCONST      0       // 1: do the routing arithmetic against the CONSTANT
                            //    NASSUME instead of the runtime N, so % N is an
                            //    AND, / N a shift and % (L*N) a mulhu. simx
                            //    charges div/rem XLEN+2 = 34 cycles and the send
                            //    loop has three of them per message. Host must
                            //    check N == NASSUME.
#endif

#ifndef TILE_NOWB
#define TILE_NOWB   0       // PROBE: 1 drops the tile write-back (result check fails)
#endif
#ifndef TILE_LEAF
#define TILE_LEAF   0       // 1: leaf tile apply (no per-call register spills); needs TILE_ACC + CNT_T
#endif
#ifndef CNT_T
#define CNT_T       0       // 1: store counts bucket-major, counts[b*N + owner],
                            //    so the 16 lanes of a core write one 64 B line per
                            //    bucket instead of 16 lines 4 * L bytes apart. The
                            //    writeback is the ONLY remaining O(L) write class
                            //    once TILE_ACC is on, so this is what makes it
                            //    cheap. Same output-layout argument as the index
                            //    gather tgt transpose. Layout of an output array
                            //    is ours to choose and no actor touches another's.
#endif
static_assert(!(CNT_T) || (TILE_ACC) || (APPLY), "CNT_T is only wired through the TILE_ACC and APPLY paths");
#if CNT_T || APPLY
#define CNT_IDX(owner, b)  ((size_t)(b) * (size_t)(N) + (size_t)(owner))   /* N is the host's runtime actor count */
#else
#define CNT_IDX(owner, b)  ((size_t)(owner) * (size_t)(L) + (size_t)(b))
#endif

static_assert(!(APPLY) || !(TILE_ACC), "APPLY and TILE_ACC both accumulate: pick one");

// The accumulator that must be resident. Without tiling that is all L buckets
// per lane, which is the L <= 256 wall. With tiling it is one tile, so the wall
// moves from L to L / BUCKET_TILE and BUCKET_TILE is free to grow.
#if TILE_ACC
// +1 word of padding so the 16 lanes' tile rows do not land on a 4-bank cycle
// during the zero and writeback sweeps (stride 40 puts all 16 lanes on 4 banks).
#define BACC_STRIDE ((BTILE_SZ) + 1)
#define BUF_WORDS  ((WC_TILE) ? (FTILE_SZ) * (SPLIT) : (BACC_STRIDE) * (GLANES) * (SPLIT))   /* one row per lane, sub-lanes included */
#elif APPLY || MEMACC
#define BUF_WORDS  (GLANES)
#else
#define BUF_WORDS  ((ACC_LS) * (GLANES) * (SPLIT) * ((WC_RECV) ? 2 : 1) * (RECV_IL))   /* one row per lane (+ a private row each under WC_RECV) */
#endif
static_assert(BUF_WORDS * 4 <= (LMEM_BYTES),
              "accumulator exceeds the 16 KB PHYSICAL scratchpad: raise BUCKET_TILE "
              "(LMEMW is only the assert's bound, the hardware masks and wraps)");
static_assert((CHUNKS) >= 1 && (M) % (CHUNKS) == 0, "CHUNKS must divide M");
static_assert((UNROLL) >= 1 && ((M) / (CHUNKS)) % (UNROLL) == 0,
              "UNROLL must divide the per chunk message count M/CHUNKS");
static_assert((SPLIT) >= 1 && ((M) / (CHUNKS)) % (SPLIT) == 0,
              "SPLIT must divide the per chunk message count M/CHUNKS");
static_assert((SPLIT) == 1 || (APPLY) || (TILE_ACC) || (!(MEMACC) && !(BUCKET_TILE)) || ((DIRECT_BT) && (BUCKET_TILE) == 1),
              "SPLIT is wired through the APPLY, TILE_ACC and plain scratchpad receives");
static_assert((SPLIT) == 1 || !(BUCKET_TILE) || (WC_TILE) || (BUCKET_TILE) % (SPLIT) == 0 || (BUCKET_TILE) == 1,
              "with tiles, SPLIT must divide BUCKET_TILE so every lane takes the same trip count");
// The non direct BUCKET_TILE path re-appends its own inbox between the rounds,
// with its own barriers and a geometry change, which chunking would interleave
// wrongly. The producer keyed paths have no such pass.
static_assert((CHUNKS) == 1 || (DIRECT_BT) || !(BUCKET_TILE),
              "chunked rounds need the producer keyed path (-DDIRECT_BT=1)");

static_assert((unsigned long long)NBA * NASSUME + 1 <= (AQ_SIZE) / 64,
              "AQ_SIZE too small for N*NB queues plus the control line");
static_assert(!(BUCKET_TILE) ||
              (unsigned long long)(BUCKET_TILE) * NASSUME + 1 <= (AQ_SIZE) / 64,
              "AQ_SIZE too small for the N*BUCKET_TILE tile queues");

typedef struct {
    uint64_t send;      // append every update
    uint64_t barrier;
    uint64_t recv;      // read my own queues, accumulate, write my counts
    uint64_t total;
    uint64_t t_start;   // ABSOLUTE core cycle at entry to the body and at its end,
    uint64_t t_end;     // so the host can see time spent outside the timed body (T42)
} phase_cycles_t;

typedef struct {
    uint64_t qs_addr;       // int[N*NB*AQ_QCAP]   queue storage, model written
    uint64_t tails_addr;    // uint32_t[N*NB]      queue lengths, model written
    uint64_t counts_addr;   // int[N*L]            counts[owner*L + bucket]
    uint64_t bs_addr;       // BUCKET_TILE: int[N*BUCKET_TILE*BQ_CAP]  tile queues
    uint64_t btails_addr;   // BUCKET_TILE: uint32_t[N*BUCKET_TILE]    tile lengths
    uint64_t phase_addr;    // phase_cycles_t[N]
    uint64_t bar_addr;      // volatile uint32_t[8]
    uint32_t N;
    uint64_t pool_addr;     // OVF: int[OVF_NCH * OVF_CH]   overflow chunks
    uint64_t ctab_addr;     // OVF: uint64_t[queues * OVF_MAXC]   chunk addresses, model written
} args_t;

#endif
