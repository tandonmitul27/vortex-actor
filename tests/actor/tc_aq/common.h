// Triangle counting, after bale-gpu (src/trianglecounting-*.cu) and the HClib-Actor selector version.
// Input: a Matrix Market file holding L, the lower triangular adjacency matrix (TC_MTX env), read as
// bale-gpu reads it. Rows are distributed cyclically: row r belongs to PE r % P at local row r / P, and
// every PE holds its rows as its own CSR (loffset/lnonzero), sorted.
// ACTOR=1 (P = N actors): bulk-push's generate_pairs is the send, each pair (v, w) of a local row u
//          is the message {v_local, w} to owner(v), appended to owner(v)'s queue; tc_process is the
//          handler, binary_search(w) in the owner's row v_local, counted in the owner's own counter.
// ACTOR=0 (P = 1): bale-gpu's agp kernel on one GPU, where every row is local: for every row u (a
//          thread per row, grid stride) and every v in L[u], intersect(L[u], L[v]).
// ACTOR=2 (P = N actors, one thread each): the plain actor (HClib-Actor style, as actor_histo /
//          actor_ig): actor_ig's mailbox, one SPSC ring of CAP packets per sender->receiver pair, one
//          message per pair; a send that finds the ring full drains the actor's own inbox instead
//          (the runtime's progress), and every actor broadcasts done and polls until all are done.
// ACTOR=3 (P = 1): Polak's CalculateTriangles (github.com/adampolak/triangles, gpu.cu) on one GPU: edges
//          oriented from lower to higher (degree, id) as RemoveBackwardEdges, sorted by (tail, head) as
//          SortEdges, unzipped (edges[0..m) heads, edges[m..2m) tails) with CSR nodes over the tails; a
//          thread per edge (grid stride) merges the two endpoints' lists.
// ACTOR=4 (P = N actors): bale-gpu's bulk-pull. For each edge (u, v) of a local row u whose v another actor
//          owns, the request {me, u_local, v_local} is appended to owner(v)'s queue; owner(v) answers with L[v],
//          a variable-length message: it reserves len slots in the requester's data region and one in its header
//          region (an atomic add on each region's tail, the k-slot append; a tail starts at its region's start),
//          copies L[v] there (the warp's lanes together, one request at a time) and writes the header {u_local, base, len}. Then every actor
//          intersects L[u] with each received L[v], and with the L[v] it owns itself (tc_process3).
// ACTOR=5 (P = 1): wedge lookup with no actors, the control for ACTOR=1: a thread per edge (u, v) of L (grid
//          stride, as Polak), every w of L[u] before v looked up in L[v] by binary_search, as tc_process, read
//          directly in global memory. erow (host, untimed) gives each edge's row, as Polak's tails array.
// ACTOR=6 (P = 1): GroupTC (IPDPS'24 comparative study, github.com/good-ncu/TC-Compare approach/GroupTC/tc.cu)
//          verbatim: a block of B = blockDim threads takes B consecutive edges; warp-wide strided walk of the
//          2-hop list, binary search in the 1-hop table, table/key flip, offset reuse. Mirrored onto L (their
//          table = N(u) after v; ours = L[u] before v). src_list = erow. Shared arrays in the scratchpad.
// ACTOR=7 (P = 1): TRUST (Pandey et al. TPDS'21, the TC-Compare port approach/TRUST/tc.cu) verbatim: its preprocessing
//          on the host (CSR2TrustCSR: degree orientation, out-degree-descending ids, reassignID groups), vertices of
//          out-degree > 100 hashed + intersected by the block, the rest by a warp; hash in the scratchpad (TR_BB bins
//          x 6), overflow in global memory; dynamic vertex chunks by global atomics. Per-thread counters as the others.
// BAL=1 (ACTOR 0/1): the per-row pass skips rows heavier than W/(N*16); those are cut into chunks of
//          ~W/(4N) placed by the host, largest first, on the least loaded sender (an actor; a warp of
//          the agp kernel), whose lanes share each chunk. ACTOR=1 copies a chunk's row prefix into its
//          sender's own region. The receive side is unchanged.
// RTILE=1 (ACTOR 1): the recipe's receive tile. The host cuts each owner's rows into tiles that fit a warp's
//          TS_W-word scratchpad region (the tile's row offsets + its rows' columns); a row too long for it is a
//          tile of its own, searched in global memory. A pair (v, w) is the 32-bit message {v's in-tile row, w} to
//          the queue of (owner(v), v's tile), from the host's map rmap[v] = queue << 10 | in-tile row. The owner's
//          warp loads one tile into its scratchpad, drains the tile's queue (binary_search(w) in the tile) and
//          moves to the next. The sender reads v's queue and in-tile row from lmap[j] (rmap of the column, laid
//          out as the CSR, read in order).
// RTILE=2: the tile is a hash set of its edges {in-tile row, w} (the message word itself), HS slots, at most
//          HMAX edges (HS / 2 default), linear probing; the host lays out every tile's image and the warp copies it into its
//          scratchpad. The handler is a probe: hit = a triangle.
// RTILE=3: as RTILE=2, but the owner's warp builds each tile's hash set in its scratchpad from its own CSR rows
//          (in the timed run; no host images): lanes over the tile's rows insert each edge's key by linear probing,
//          a write then a read back deciding a slot two lanes raced for.
// HSIZE=1 (RTILE 3): a tile's hash set is sized to its edges, the smallest power of 2 >= 2 x edges (>= 16 slots), so
//          its clear and probes span only that.
// HADAPT=a (RTILE 3): the owner builds a tile's hash set only when the tile's queue holds >= a/4 messages per tile
//          edge (its tail and its rows' offsets, read in the run); a tile with fewer is searched in its rows in place
//          (binary_search in global memory, as a long row).
// SHEAVY=d (RTILE): the sender's lanes take its rows as before, but a row of more than d entries is left to the whole
//          warp, which walks its j and splits each j's pairs k < j over its lanes (k = first + lane, step GLANES); a
//          ballot over the warp's 16 rows names them.
// SEDGE=1 (RTILE): edge-centric generate_pairs: the sender's lanes take its edges j in CSR order (no per-row loop);
//          lmap[j] also carries j's position in its row (bits 24..31, 255 = longer: the row is found by binary search
//          in the offsets), so j's pairs are k = j - pos .. j - 1.
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
#ifndef TR_BB
#define TR_BB      512      // ACTOR=7: TRUST's block_bucketnum (theirs 1024 = 28 KB of shared memory; 512 = 14 KB fits the 16 KB scratchpad)
#endif
#ifndef BAL
#define BAL        0
#endif
#if MSG32 && ACTOR != 1
#error "MSG32 needs ACTOR 1"
#endif
#if BAL && ACTOR >= 2
#error "BAL needs ACTOR 0 or 1"
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
#ifndef RTILE
#define RTILE      0
#endif
#if RTILE && (ACTOR != 1 || MSG32 || BAL)
#error "RTILE needs ACTOR 1, without MSG32 and BAL"
#endif
#define TS_W       1024     // RTILE: scratchpad words per warp (one tile)
// Ablation (results/mapping/ABLATION.md, ../abl.h): piece B off = RTILE=0 (one queue per owner, binary search in
// memory); piece C off = LANE_RECV=1: a lane takes a whole tile queue, with its own TS_W / 16-slot hash set (the
// host cuts HMAX-edge tiles for it: the generator passes a 16x smaller HMAX), so the lmap row field is 4 bits.
#ifndef LANE_RECV
#define LANE_RECV  0
#endif
#define HS_B       ((LANE_RECV) ? 6 : 10)   // log2(HS)
#define HS         (1u << (HS_B))           // RTILE 2/3: hash slots per tile (per lane under LANE_RECV)
#define LMB        ((LANE_RECV) ? 4u : 10u) // lmap: bits of the in-tile row (the queue above it)
#define HS_EMPTY   0xFFFFFFFFu
#ifndef HMAX
#define HMAX       ((HS) / 2)   // RTILE=2: edges per tile at most (load factor HMAX / HS)
#endif
static_assert((HMAX) < (HS), "HMAX: a tile's hash set needs an empty slot");
#ifndef HSIZE
#define HSIZE      0
#endif
#ifndef HADAPT
#define HADAPT     0
#endif
#ifndef SHEAVY
#define SHEAVY     0
#endif
#if SHEAVY && !(RTILE)
#error "SHEAVY needs RTILE"
#endif
#ifndef SEDGE
#define SEDGE      0
#endif
#ifndef SEDGE_PMAX
#define SEDGE_PMAX 255      // SEDGE: positions >= this are found by binary search (lower it only to test that path)
#endif
#if SEDGE && (SHEAVY || MSG32 || BAL)
#error "SEDGE: without SHEAVY, MSG32, BAL"
#endif
#if LANE_RECV && (RTILE == 1 || RTILE == 2 || HSIZE || SHEAVY || MSG32 || BAL)
#error "LANE_RECV: RTILE 0 or 3, without HSIZE, SHEAVY, MSG32, BAL"
#endif
#if (HSIZE || HADAPT) && RTILE != 3
#error "HSIZE and HADAPT need RTILE 3"
#endif
#define HS_HASH(k) (((uint32_t)(k) * 0x9E3779B1u) >> (32 - (HS_B)))
static_assert(!(RTILE) || (SPLIT) * (TS_W) * 4 <= 16384, "the warps' tiles exceed the scratchpad");
#ifndef Q_CAP
#define Q_CAP      ((RTILE) ? 1024 : 4096)     // per-queue region (messages); the rest overflows into pool chunks
#endif
#ifndef MSG32
#define MSG32      0        // ACTOR=1: 32-bit messages {v_local low bits, w}; queue (owner, tile) carries v_local's high bits
#endif
#if MSG32 || RTILE
typedef uint32_t msg_t;
#else
typedef uint64_t msg_t;
#endif
#define QSTRIDE_E  ((((((Q_CAP) * (int)sizeof(msg_t) + 63) / 64) | 1) * 64) / (int)sizeof(msg_t))   // whole odd lines, in messages
#ifndef SWQ
#define SWQ        0        // 1: software append queue, no L2 append unit (../swq.h)
#endif
#ifndef OVF_CH
#define OVF_CH     256
#endif
#ifndef WC_U
#define WC_U       4
#endif

#ifndef CAP
#define CAP        16       // ACTOR=2 ring capacity (power of 2), as actor_histo / actor_ig
#endif
typedef struct {
    uint32_t sender;
    uint32_t done_flag;       // 0 = data, 1 = done
    int      idx;             // v_local
    int      value;           // w (global)
} packet_t;                   // actor_ig's packet
typedef struct {
    volatile uint32_t head;
    volatile uint32_t tail;
    packet_t slots[CAP];
} ring_t;

typedef struct { uint64_t send, barrier, recv, total, msgs, t0, t1, pad; } phase_cycles_t;

typedef struct {
    uint64_t loffset_addr;    // int[P][lrows + 1]: PE p's row offsets (absolute, into lnonzero)
    uint64_t lnonzero_addr;   // int[nnz]: column ids (global), PE by PE, row by row, sorted
    uint64_t cnt_addr;        // uint32_t[warps * 16] per-lane triangle counts (the host sums them)
    uint64_t qs_addr;         // ACTOR=1: queue regions, NQ x QSTRIDE_E messages
    uint64_t tails_addr;      // ACTOR=1: uint32_t[NQ] (model written)
    uint64_t pool_addr;       // ACTOR=1: overflow chunks
    uint64_t ctab_addr;       // ACTOR=1: uint64_t[NQ * ovf_maxc]
#if SWQ
    uint64_t swq_addr;        // SWQ: the pool counter (zeroed)
#endif
    uint64_t phase_addr;      // phase_cycles_t[warps]
    uint64_t bar_addr;        // volatile uint32_t[8]
    uint64_t grid_addr;       // ACTOR=2: ring_t[N*N], grid[sender*N + receiver]
    uint32_t N;               // actors (the kernel's P when ACTOR=1)
    uint32_t numrows;         // rows of L
    uint32_t lrows;           // rows per PE, ceil(numrows / P)
    uint32_t ovf_maxc;        // ACTOR=1: chunks per queue (host: exact need + 1)
    uint32_t ovf_nch;         // ACTOR=1: pool size in chunks (host: exact need + slack)
    uint32_t m;               // ACTOR=3: oriented edges (lnonzero_addr holds edges[2m], loffset_addr nodes[numrows+1])
    uint32_t wbits;           // MSG32: bits of w
    uint32_t vbits;           // MSG32: bits of v_local in the message (the rest pick one of the owner's 1 << tbits queues)
    uint32_t tbits;           // MSG32: queues per owner = 1 << tbits, queue o * (1 << tbits) + (v_local >> vbits)
    uint32_t pad_;
#if RTILE || SEDGE
    uint64_t lmap_addr;       // uint32_t[nnz]: rmap of each column (row v's queue << 10 | in-tile row), as lnonzero
#endif
#if RTILE
    uint64_t timg_addr;       // RTILE=2: uint32_t[NQ][HS] the tiles' hash images
    uint64_t tdesc_addr;      // uint32_t[NQ][2]: tile {first local row, rows} (rows bit 31: a long row, in global memory)
    uint64_t qofs_addr;       // uint32_t[N + 1]: each owner's first queue
#endif
#if ACTOR == 7
    uint64_t part_addr;       // int[NB * TR_BB * 100]: TRUST's partition (hash overflow in global memory), BIN_MEM
    uint64_t gidx_addr;       // int[3]: TRUST's G_INDEX (dynamic vertex chunks)
    uint32_t wfv;             // TRUST's warpfirstvertex (first vertex of out-degree <= USE_CTA)
#endif
#if ACTOR == 5 || ACTOR == 6
    uint64_t erow_addr;       // int[nnz]: the row u of each edge (u, v) of L
#endif
#if ACTOR == 4
    uint64_t data_addr;       // int[]: received rows, requester by requester (host: exact region sizes)
    uint64_t hdr_addr;        // uint32_t[][3]: headers {u_local, base (absolute in data), len}, requester by requester
    uint64_t hofs_addr;       // uint32_t[N + 1]: each requester's header region
    uint64_t rtail_addr;      // uint32_t[N][2]: each requester's {data, header} tails (atomic), absolute
#endif
#if BAL
    uint64_t rend_addr;       // int[P][lrows]: end of the per-row pass over a row (its start if the row is split)
    uint64_t ch_addr;         // uint32_t[nch][3]: chunks {row base, j begin, j end}, sender by sender
    uint64_t chofs_addr;      // uint32_t[senders + 1]: each sender's chunk range
    uint64_t rep_addr;        // ACTOR=1: int[]: the chunks' row prefixes, sender by sender (ACTOR=0: unused)
#endif
} args_t;
#endif
