// actor_ig_aq: Bale index gather on a combining append queue (HW-SW co-design
// ladder step 2). See docs/append_queue_plan.md.
//
// The request round is ONE instruction per request: append(q, word) into the
// owner's queue, where q = (owner, asker block). No count, no scan, no place,
// no publish, no run table. The owner reads its (owner, asker block) queue as a
// contiguous per lane stream in pieces of PIECE words, and from pass A on the
// serve round, the reply round and the drain are exactly actor_ig_dense's.
// Every queue belongs to one owner and only that owner reads it; the asker
// block in the key is a mailbox slot the sender chooses, as it chooses the
// window today. The table slice and tgt are touched only by their owner.
//
// The queue is a window the simulator models (VORTEX_AQ_BASE / VORTEX_AQ_SIZE,
// see sim/simx/types.h): line 0 is a control block the kernel writes once,
// line q+1 is queue q. The storage the model appends into (qs) and the tails
// are ordinary global buffers the host allocates and audits.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

#ifndef M
#define M           10000   // queries per actor
#endif
#ifndef TABLE_SIZE
#define TABLE_SIZE  16000   // table entries owned by each actor
#endif
#ifndef LANES
#define LANES       16      // threads per warp; must equal NUM_THREADS
#endif
#define PW          1       // one warp per pool (SHFL and the timing model assume it)
#ifndef CAP2
#define CAP2        1200    // asker-sorted reply pool for one block pair and piece
#endif
#ifndef LMEMW
#define LMEMW       4096    // scratchpad words for one BLOCK (16 KB per core)
#endif
#ifndef APPLY
#define APPLY       0       // 1 with REPLY_TILE: the drain round is one apply engine
                            //    command per reply tile (docs/apply_engine_plan.md):
                            //    STORE each {question, value} word at
                            //    tgt[question*N + me]. The lanes issue and wait; the
                            //    engine's combining cache assembles each tgt line
                            //    from the 16 lanes' replies and writes it once.
#endif
#define AP_BASE    ((uint64_t)(AQ_BASE) + (uint64_t)(AQ_SIZE))
#define AP_SPEC(keyshift, keybits, valshift, valbits, op, wsize, valconst) \
    ((uint64_t)(keyshift) | ((uint64_t)(keybits) << 8) | ((uint64_t)(valshift) << 16) \
     | ((uint64_t)(valbits) << 24) | ((uint64_t)(op) << 32) | ((uint64_t)(wsize) << 40) \
     | ((uint64_t)(valconst) << 48))
#define AP_STORE   0
#define AP_ADD     1
#ifndef DIRECT_TILE
#define DIRECT_TILE 0       // 1 with TILE_AQ: the ASKER appends its request straight
                            //    to the owner's (owner, table tile) queue, so the
                            //    owner's self reorder pass and its two barriers
                            //    disappear. The asker already holds the table
                            //    index, so it already holds the locality key.
                            //    Mechanism 2 keyed at the producer (P2 in
                            //    docs/next_mechanisms.md). Self reorder remains
                            //    the fallback for a key the producer cannot see.
#endif
#ifndef NCONST
#define NCONST      0       // 1: route against the CONSTANT NASSUME so % N is an
                            //    AND instead of a 34 cycle remuw. One per query
                            //    in the request loop. Host checks N == NASSUME.
#endif
#ifndef NASSUME
#define NASSUME     256     // N used for the compile-time sizing
#endif
#ifndef PACK64
#define PACK64 1            // reply publish as 64 bit stores (see store_path_width)
#endif
#ifndef SHFL
#define SHFL 1              // warp shuffle scan for the 16 asker counters
#endif
#ifndef PROFILE_SUB
#define PROFILE_SUB 0
#endif
#ifndef TGT_T
#define TGT_T 1             // transposed tgt, as in the dense SOTA
#endif
#ifndef ORDER_CHECK
#define ORDER_CHECK 0       // host verifies FIFO per (owner, asker) pair
#endif
#ifndef AQ_BASE
#define AQ_BASE     0xD0000000u   // window; the run line must set VORTEX_AQ_BASE to it
#endif
#ifndef AQ_SIZE
#define AQ_SIZE     0x100000u     // 1 MB window = 16383 queues; VORTEX_AQ_SIZE
#endif
#if WIDEREQ
typedef uint64_t req_t;
#define AQ_WORD     8
#else
typedef uint32_t req_t;
#define AQ_WORD     4
#endif
#ifndef WIDEREQ
#define WIDEREQ     0       // 1: the request word is 64 bits instead of 32, so the
                            //    three packed fields are not confined to 32 bits and
                            //    M or TABLE_SIZE can scale. Costs half the combining
                            //    efficiency on the request round (8 words a line
                            //    instead of 16), which is 160K extra line writes at
                            //    M=10000. Required whenever QBITS+LBITS+ABITS > 32.
#endif
#ifndef PACKBITS
#define PACKBITS    1       // 1: the request word packs {asker lane, question,
                            //    table index} on BIT boundaries instead of by
                            //    multiplication, so unpacking is shifts and masks
                            //    rather than four constant divisions. A constant
                            //    divide is a multiply-high plus shifts, about six
                            //    instructions; there are four per request on the
                            //    serve path. See the divide-latency note.
#endif
#ifndef QBITS
#define QBITS       14      // bits for the question index; must hold M
#endif
#ifndef LBITS
#define LBITS       14      // bits for the table index; must hold TABLE_SIZE
#endif
#ifndef ABITS
#define ABITS       4       // bits for the asker's lane; must hold GLANES
#endif
// TILE QUEUE WORD, 64 bits: the asker's GLOBAL id in the high half, and only
// {question, table index} in the low half. The asker's lane used to sit in the
// low half next to them, which made the packed word carry QBITS + LBITS + ABITS
// and forced TABLE_SIZE down to 8192 at 32 threads and to 2048 at M = 100000.
// The high half held just the asker's block and was nearly empty, so the whole
// asker id moves up there and the low half needs only QBITS + LBITS.
#if PACKBITS
#define TK_MAKE(que, lix) ((uint32_t)(((uint32_t)(que) << (LBITS)) | (uint32_t)(lix)))
#define TK_QUE(w)         ((uint32_t)((uint32_t)(w) >> (LBITS)))
#define TK_LIX(w)         ((uint32_t)((uint32_t)(w) & (((uint32_t)1 << (LBITS)) - 1)))
#else
#define TK_MAKE(que, lix) ((uint32_t)((uint32_t)(que) * (uint32_t)(TABLE_SIZE) + (uint32_t)(lix)))
#define TK_QUE(w)         ((uint32_t)((uint32_t)(w) / (uint32_t)(TABLE_SIZE)))
#define TK_LIX(w)         ((uint32_t)((uint32_t)(w) % (uint32_t)(TABLE_SIZE)))
#endif

#if PACKBITS
#define PK_MAKE(aib, que, lix) ((((req_t)(aib)) << ((QBITS) + (LBITS))) \
                              | (((req_t)(que)) << (LBITS)) | ((req_t)(lix)))
#define PK_AIB(w)  ((uint32_t)((req_t)(w) >> ((QBITS) + (LBITS))))
#define PK_QUE(w)  ((uint32_t)(((req_t)(w) >> (LBITS)) & (((req_t)1 << (QBITS)) - 1)))
#define PK_LIX(w)  ((uint32_t)((req_t)(w) & (((req_t)1 << (LBITS)) - 1)))
#else
#define PK_MAKE(aib, que, lix) (((((req_t)(aib)) * (req_t)(M) + (req_t)(que)) \
                                 * (req_t)(TABLE_SIZE)) + (req_t)(lix))
#define PK_AIB(w)  ((uint32_t)(((req_t)(w) / (req_t)(TABLE_SIZE)) / (req_t)(M)))
#define PK_QUE(w)  ((uint32_t)(((req_t)(w) / (req_t)(TABLE_SIZE)) % (req_t)(M)))
#define PK_LIX(w)  ((uint32_t)((req_t)(w) % (req_t)(TABLE_SIZE)))
#endif
#ifndef TILE_NOGEO
#define TILE_NOGEO 0      // BISECTION ONLY. 1: keep the tile round's barriers but
                          // skip both geometry rewrites.
#endif
#ifndef TILE_NOAPPEND
#define TILE_NOAPPEND 0   // BISECTION ONLY. 1: run the tile round's geometry
                          // switches, barriers and inbox reads but skip the
                          // appends. The answer is wrong; the question is only
                          // whether it TERMINATES.
#endif
#ifndef REPLY_TILE
#define REPLY_TILE  0       // >0: the reply mailbox is keyed (asker, QUESTION tile)
                            //     instead of (asker, owner block). Costs nothing
                            //     extra to fill, and lets the drain sort a tile in
                            //     scratchpad and write tgt with all sixteen lanes
                            //     on the SAME question index, which is one cache
                            //     line per step (this is TGT_IDEAL, built).
                            //     Each question is asked exactly once, so a tile
                            //     queue holds exactly M/REPLY_TILE replies and the
                            //     capacity is exact. Needs REPLY_AQ. The scratchpad
                            //     is free now (64 B used), which is what makes the
                            //     tile sort possible at all.
#endif
#ifndef TILE_AQ
#define TILE_AQ     0       // >0: the owner re-appends its OWN inbox to its OWN
                            //     per table tile queues before serving (ladder
                            //     step 4), then serves tile by tile so the random
                            //     gather is confined to a TABLE_SIZE/TILE_AQ slice.
                            //     A pure self reorder: an actor appending its own
                            //     mail to its own queues. Needs REPLY_AQ.
                            //     The table is N*TABLE_SIZE*4 = 16 MB against a
                            //     1 MB L2, so the gather misses everything today;
                            //     tiling aims the per core working set at the L2.
#endif
#ifndef REPLY_AQ
#define REPLY_AQ    0       // 1: the REPLY round is on the queue too (ladder step 3).
                            //    The owner appends {question, value} as one 64 bit
                            //    word to queue (asker, my block); there is no reply
                            //    pool, no counting pass, no scan, no run table and no
                            //    publish, and the drain reads its own queues as
                            //    contiguous streams with ONE load per reply.
                            //    The two rounds have different word sizes, so the
                            //    kernel rewrites the control block between them: the
                            //    model resets its internal tails on a geometry change
                            //    and the request tails already in memory are untouched.
#endif

#define GLANES     ((LANES) * (PW))
#define NBA        ((NASSUME) / (GLANES))       // blocks, for static sizing only
#define PIECE      ((CAP2) / (GLANES))          // words of its own queue a lane
                                                // serves per piece; 16 lanes
                                                // then fill at most CAP2
#ifndef NCHUNK
// pieces per (owner, asker block) queue: the mean queue holds M*GLANES/N words
// (625 at M=10000, N=256); allow 1.4x the mean, rounded up to whole pieces
#define NCHUNK     ((((M) * (GLANES) * 14) / ((NASSUME) * 10) + (PIECE) - 1) / (PIECE))
#endif
#define AQ_QCAP    ((NCHUNK) * (PIECE))         // words of storage per queue
#define RQ_CAP     (AQ_QCAP)                    // replies per (asker, owner block) queue
#if REPLY_TILE
#define RTILE_SZ   ((M) / (REPLY_TILE))      // replies in one question tile
#define RT_CAP     (RTILE_SZ)                // exact: every question is asked once
#ifndef RQ_ALIGN
#define RQ_ALIGN   0       // 1: reply tile queues start on a line and sit an ODD number
                           //    of lines apart, so a warp's lanes cross line boundaries
                           //    together (one stall per line, not one per lane per line)
#endif
#if RQ_ALIGN
#define RT_SSTRIDE ((((((RT_CAP) * 8) + 63) / 64) | 1) * 8)   // uint64 words
#else
#define RT_SSTRIDE ((((RT_CAP) * 8 + 63) / 64) * 8)   // whole lines (the ISA2 queue layout needs them); = RT_CAP when RT_CAP % 8 == 0
#endif
#define NRQ        ((NASSUME) * (REPLY_TILE))
#else
#define NRQ        ((NASSUME) * (NBA))
#endif
#ifndef DRAIN_DIRECT
#define DRAIN_DIRECT 0    // 1 (REPLY_TILE, natural tgt): the warp writes each reply straight into tgt, one actor's
#endif                    //    reply tile at a time; no scratchpad reorder, so reply tiles can be large (L2 merges the lines)
#ifndef SWQ
#define SWQ 0             // 1: SOFTWARE queues: slot = atomicAdd(&tail[q], 1), then a plain store into the queue storage
#endif
#ifndef WC_DRAIN
#define WC_DRAIN 0        // 1: warp-cooperative drain_in: a warp reads ONE actor's reply tile at a time (a line per load)
#endif
#ifndef LANE_RECV
#define LANE_RECV 0       // 1: ablation piece C off (results/mapping/ABLATION.md): DRAIN_DIRECT by one lane per reply
#endif                    //    queue (with WC_SERVE=0: the serve is one lane per tile queue too)
#ifndef WC_SERVE
#define WC_SERVE 0        // 1: warp-cooperative serve: a warp serves ONE owner's tile queue at a time
#endif
#ifndef WC_U
#define WC_U 4            // warp-coop: queue lines loaded before any is used
#endif
#ifndef LANE_TS
#define LANE_TS 0         // 1: every lane records 6 timestamps (phase starts/ends, barrier exits) after phase[N]
#endif
#ifndef SPLIT
#define SPLIT       1       // LANES PER ACTOR (log T41). N used to be welded to the
                            // machine, N = cores x warps x threads, so warps could not
                            // be added without multiplying the actors and shrinking
                            // every tile. S lanes now share one actor. All three rounds
                            // split legally here because the owner's table is READ
                            // ONLY: the request round gives each sub-lane a disjoint
                            // slice of the actor's questions, the serve round a
                            // disjoint set of its table tiles, the drain a disjoint set
                            // of its reply tiles (or sub-lane 0 alone, DRAIN_SPLIT=0).
                            // No lane ever writes state another lane also writes.
#endif
#ifndef DRAIN_SPLIT
#define DRAIN_SPLIT 0       // 1: sub-lanes drain different reply tiles at once. That
                            // puts SPLIT tiles in flight per core, so the apply engine
                            // must hold SPLIT x RTILE_SZ tgt lines: choose REPLY_TILE
                            // so that stays under its 256 lines (the knee at cache =
                            // tile, now per core rather than per lane). 0: sub-lane 0
                            // drains alone and the others send empty commands, so the
                            // engine sees exactly what it saw before.
#endif

#ifndef TAILFLUSH
#define TAILFLUSH   1       // 0: skip the end of round aq_flush, which is the v0
                            //    queue model that never charged a queue's last
                            //    partial line. Ablation only, see log T36b.
#endif
#ifndef QMAJOR
#define QMAJOR      0       // 1: tile queue index = tile * N + owner (TILE major), so
                            //    the sixteen owners a core serves at one time sit one
                            //    queue apart and, with an odd line stride, in sixteen
                            //    different L1 sets. Owner major puts them TILE_AQ
                            //    queues apart, a multiple of 64 lines, in ONE set.
#endif
// STORAGE STRIDES are separate from capacities and are a whole, ODD number of
// lines: whole so the model's one combining line write stands for one DRAM
// write (a queue starting mid line would straddle two), odd so consecutive
// queues land in consecutive L1 sets.
#define AQ_QSTRIDE_W  (((((AQ_QCAP) * (AQ_WORD) + 63) / 64) | 1) * (64 / (AQ_WORD)))
#if TILE_AQ
#define TILE_SZ    ((TABLE_SIZE) / (TILE_AQ))   // table words in one tile
#ifndef FINE_T
#define FINE_T     1       // WC_SERVE without REQ32: serve each (coarse) tile queue in FINE_T passes
#endif
#define FTILE_SZ   ((TILE_SZ) / (FINE_T))
static_assert((TILE_SZ) % (FINE_T) == 0, "FINE_T must divide the tile");
// requests per (owner, tile) queue: the mean is M/TILE_AQ, allow 1.4x. Keyed
// at the producer (DIRECT_TILE) the floor is 256; re-appended by the owner the
// floor is the request queue capacity, since sizing it lower overflowed
// silently at TILE_AQ=8 (mean 1250 against 900 slots).
#define TQ_MEAN    ((((M) * 14) / ((TILE_AQ) * 10) + 7) / 8 * 8)
#ifndef TQ_CAP              // overridable: the 256 floor was sized for TILE_AQ=80
#if DIRECT_TILE             // (mean 125, max seen 175); at TILE_AQ=500 (mean 20)
#define TQ_CAP     ((TQ_MEAN) > 256 ? (TQ_MEAN) : 256)   // pass -DTQ_CAP=64
#else
#define TQ_CAP     ((TQ_MEAN) > (AQ_QCAP) ? (TQ_MEAN) : (AQ_QCAP))
#endif
#endif
#define TQ_STRIDE_W   (((((TQ_CAP) * 8 + 63) / 64) | 1) * 8)
#endif

// REQ32 (log T43c). The request word in 32 bits. Under DIRECT_TILE the queue is
// keyed by (owner, table tile), so the key already carries the high bits of the
// table index and the word need only say WHERE IN THE TILE: asker, question and
// in-tile index. R12 counts the request as 2 x w / 64 DRAM transactions per
// query, so halving w is the one lever left on the gather at its ceiling. The
// reply stays 64 bit: it carries a whole 32 bit value. Refused at compile time
// when the three fields do not fit, never silently truncated.
#ifndef REQ32
#define REQ32 0
#endif
#if REQ32
constexpr uint32_t ig_bits(uint32_t x) { return x <= 1 ? 0 : 1 + ig_bits((x + 1) / 2); }
#define R32_TB   (ig_bits((uint32_t)(TILE_SZ)))       // in-tile index
#define R32_QB   (ig_bits((uint32_t)(M)))             // question
#define R32_AB   (ig_bits((uint32_t)(NASSUME)))       // asker
static_assert((DIRECT_TILE) != 0 && (TILE_AQ) != 0, "REQ32 needs DIRECT_TILE: the queue key carries the tile");
static_assert((R32_AB) + (R32_QB) + (R32_TB) <= 32, "REQ32: asker, question and in-tile index exceed 32 bits");
#define R32_MAKE(asker, que, lix) ((uint32_t)(((uint32_t)(asker) << ((R32_QB) + (R32_TB))) \
                                  | ((uint32_t)(que) << (R32_TB)) | ((uint32_t)(lix) % (uint32_t)(TILE_SZ))))
#define R32_ASKER(w) ((uint32_t)(w) >> ((R32_QB) + (R32_TB)))
#define R32_QUE(w)   (((uint32_t)(w) >> (R32_TB)) & ((1u << (R32_QB)) - 1u))
#define R32_OFF(w)   ((uint32_t)(w) & ((1u << (R32_TB)) - 1u))
#define TQ_STRIDE32  (((((TQ_CAP) * 4 + 63) / 64) | 1) * 16)   // whole odd lines of 32 bit words
#endif
#define TSTRIDE    ((GLANES) + 1)               // run starts per region, plus the end

#ifndef REQ_PROBE
#define REQ_PROBE 0   // DIAGNOSTIC, answer WRONG (host FAILED): the append becomes a
                      // plain scratchpad store, so the request phase prices the
                      // hashing alone and v0 minus this is the append's cost.
#endif
#ifndef STAGE_PIECE
#define STAGE_PIECE 0 // 1: pass A copies the lane's piece into scratchpad and pass B
                      // reads it there, so the queue is read from memory once.
#endif
#if REPLY_AQ
// the serve round needs no scratchpad at all
#define SERVE_WORDS (GLANES)
#else
#define SERVE_WORDS (2 * (GLANES) + 2 * (CAP2) + ((STAGE_PIECE) ? (CAP2) : 0))
#endif
#if APPLY || DRAIN_DIRECT
// the engine drains / the direct drain writes tgt itself; no scratchpad rows
#define DRAIN_WORDS (GLANES)
#elif REPLY_TILE
// one question tile per lane, sorted in place, then written in lockstep
#define DRAIN_WORDS ((RTILE_SZ) * (GLANES) * (SPLIT))   /* one row per lane, sub-lanes included */
static_assert((REPLY_TILE) % (SPLIT) == 0, "software drain: SPLIT must divide REPLY_TILE");
#elif REPLY_AQ
#define DRAIN_WORDS (GLANES)
#else
#define DRAIN_WORDS ((NCHUNK) * (TSTRIDE))
#endif
#define BUF_WORDS   ((SERVE_WORDS) > (DRAIN_WORDS) ? (SERVE_WORDS) : (DRAIN_WORDS))

static_assert(!(REQ_PROBE) || (LMEMW) >= 2048, "REQ_PROBE scratch");
static_assert(BUF_WORDS <= LMEMW, "block scratchpad exceeded: lower CAP2");
static_assert(!(PACK64) || ((CAP2) % 2 == 0), "PACK64 pairs adjacent words: CAP2 must be even");
#if PACKBITS
static_assert(((unsigned long long)1 << QBITS) >= (unsigned long long)(M), "QBITS too small for M");
static_assert(((unsigned long long)1 << LBITS) >= (unsigned long long)(TABLE_SIZE), "LBITS too small for TABLE_SIZE");
static_assert((DIRECT_TILE) || ((unsigned long long)1 << ABITS) >= (unsigned long long)(GLANES),
              "ABITS too small for GLANES");
// The request queue word carries the asker's LANE and is only used when the
// asker does not key by tile itself. The tile word (TK_) never carries it.
static_assert((DIRECT_TILE) || (QBITS) + (LBITS) + (ABITS) <= (WIDEREQ ? 64 : 32),
              "packed request does not fit: set -DWIDEREQ=1");
static_assert(!(TILE_AQ) || (QBITS) + (LBITS) <= 32,
              "tile queue payload does not fit 32 bits: lower M or TABLE_SIZE");
#else
static_assert((DIRECT_TILE) || (WIDEREQ) || (unsigned long long)GLANES * M * TABLE_SIZE < 4294967296ULL,
              "packed request overflows 32 bits: set -DWIDEREQ=1 or lower M/TABLE_SIZE");
static_assert(!(TILE_AQ) || (unsigned long long)M * TABLE_SIZE < 4294967296ULL,
              "tile queue payload overflows 32 bits: lower M or TABLE_SIZE");
#endif
static_assert((unsigned long long)NBA * NASSUME + 1 <= (AQ_SIZE) / 64,
              "AQ_SIZE too small for N*NB queues plus the control line");
static_assert(!(TILE_AQ) || (unsigned long long)(NASSUME) * (TILE_AQ) + 1 <= (AQ_SIZE) / 64,
              "AQ_SIZE too small for N*TILE_AQ tile queues");
static_assert(!(TILE_AQ) || (TABLE_SIZE) % (TILE_AQ) == 0, "TILE_AQ must divide TABLE_SIZE");
static_assert((SPLIT) >= 1 && (M) % (SPLIT) == 0, "SPLIT must divide M");
static_assert((SPLIT) == 1 || ((DIRECT_TILE) && (REPLY_TILE)),
              "SPLIT is wired only through the producer keyed, reply tiled path (engine or software drain)");
static_assert((SPLIT) == 1 || (TILE_AQ) % (SPLIT) == 0 || (TILE_AQ) < (SPLIT),
              "SPLIT must divide TILE_AQ so every sub-lane serves the same number of tiles");
static_assert(!(DRAIN_SPLIT) || (REPLY_TILE) % (SPLIT) == 0,
              "DRAIN_SPLIT needs SPLIT to divide REPLY_TILE");
static_assert(!(TILE_AQ) || (REPLY_AQ), "TILE_AQ needs REPLY_AQ");
static_assert(!(DIRECT_TILE) || (TILE_AQ), "DIRECT_TILE needs TILE_AQ");
static_assert(!(APPLY) || (REPLY_TILE), "APPLY drains reply TILE queues");
static_assert(!(APPLY) || (TGT_T), "APPLY assembles transposed tgt lines");
static_assert(!(DIRECT_TILE) || (TILE_AQ), "DIRECT_TILE keys by table tile, so it needs TILE_AQ");
static_assert(!(REPLY_TILE) || (REPLY_AQ), "REPLY_TILE needs REPLY_AQ");
static_assert(!(REPLY_TILE) || (M) % (REPLY_TILE) == 0, "REPLY_TILE must divide M");
static_assert(!(REPLY_TILE) || (TGT_T) || (DRAIN_DIRECT), "REPLY_TILE only pays with the transposed tgt");
static_assert((unsigned long long)(NRQ) + 1 <= (AQ_SIZE) / 64, "AQ_SIZE too small for the reply queues");

#define TGT_IDX(actor, q, n) (TGT_T ? ((size_t)(q) * (size_t)(n) + (size_t)(actor)) \
                                    : ((size_t)(actor) * (size_t)(M) + (size_t)(q)))

typedef struct {
    uint64_t request;   // phase 1: append every request
    uint64_t barrier;
    uint64_t serve;     // phase 2: read my queues, look up my table, sort replies by asker
    uint64_t drain;     // phase 3: read my own replies, write tgt
    uint64_t total;
    uint64_t v_count, v_scan, v_place, v_pub;   // serve sub-stages
    uint64_t t_start, t_end;   // ABSOLUTE core cycles at body entry and end: the
                               // launch-free body window (T42)
} phase_cycles_t;

typedef struct {
    uint64_t table_addr;    // int[N*TABLE_SIZE]
    uint64_t qs_addr;       // req_t[N*NB*AQ_QCAP]         queue storage (model writes)
    uint64_t tails_addr;    // uint32_t[N*NB]              queue lengths (model writes)
    uint64_t rp_tag_addr;   // int[NB*NB*NCHUNK*CAP2]      reply question index
                            // REPLY_AQ: uint64_t[N*NB*RQ_CAP]   reply queue storage
    uint64_t rp_val_addr;   // int[NB*NB*NCHUNK*CAP2]      reply value
                            // REPLY_AQ: uint32_t[N*NB]          reply queue lengths
    uint64_t ptbl_addr;     // int[NB*NB*NCHUNK*TSTRIDE]   reply run starts
    uint64_t tgt_addr;      // int[N*M]
    uint64_t phase_addr;    // phase_cycles_t[N]
    uint64_t ts_addr;       // TILE_AQ: uint64_t[N*TILE_AQ*TQ_CAP]  tile queue storage
    uint64_t ttails_addr;   // TILE_AQ: uint32_t[N*TILE_AQ]         tile queue lengths
    uint64_t bar_addr;      // volatile uint32_t[8]; [2] = dropped replies, [3] = over-long queues
    uint32_t N;
    uint64_t pool_addr;     // OVF: overflow chunks (OVF_NCH x OVF_CH words of the request word size)
    uint64_t ctab_addr;     // OVF: uint64_t[TILE_AQ*NASSUME*OVF_MAXC] chunk addresses, model written
} args_t;
#ifndef OVF
#define OVF        0       // skewed traffic: a full request tile queue continues in pool chunks (see actor_histo_aq)
#endif
#ifndef OVF_CH
#define OVF_CH     256
#endif
#ifndef OVF_MAXC
#define OVF_MAXC   128
#endif
#ifndef OVF_NCH
#define OVF_NCH    ((NASSUME) * (M) / (OVF_CH) + 1024)
#endif

#endif
