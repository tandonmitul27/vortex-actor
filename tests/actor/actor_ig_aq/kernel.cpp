// See common.h. Request round on the append queue; serve, reply and drain as
// in actor_ig_dense, reading the queue in pieces instead of window runs.

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"
#ifndef ACTOR
#define ACTOR 1           // this kernel is always the actor version (the non-actor gather is baseline_ig)
#endif
#include "../isa2.h"
#if AQ_ON
// ISA2 converts the paths the paper's index gather runs use (requests queued by table tile, exact reply tiles)
static_assert((TILE_AQ) && (DIRECT_TILE) && (REPLY_AQ) && (REPLY_TILE) && (DRAIN_DIRECT) && (QMAJOR) && !(OVF) && !(APPLY),
              "ISA2: this index gather path still reads the old queue layout");
#endif
#ifndef BASELINE
#define BASELINE 0
#endif
#if BASELINE
// BASELINE=1: the baseline actor version, plain GPU code. Software queues only (SWQ), and no
// L2-append window: any reachable use of AQ_BASE fails to link.
static_assert(SWQ, "BASELINE: software queues");
extern "C" uint64_t baseline_has_no_aq_window(void);
#undef AQ_BASE
#define AQ_BASE (baseline_has_no_aq_window())
#endif
#include "../lmem_check.h"
#if !defined(ACTOR) || ACTOR == 1
LMEM_CHECK((BUF_WORDS) * 4, (NASSUME) / (GLANES));
#endif
#include "../skew.h"

static inline uint32_t mix(uint32_t a, uint32_t b) {
    uint32_t x = a * 0x9E3779B9u + b * 0x85EBCA77u;
    x ^= (x >> 13);
    x *= 0xC2B2AE3Du;
    x ^= (x >> 16);
    return x;
}

#define POOL_SYNC() vx_fence()
#ifndef DRAIN_PROBE
#define DRAIN_PROBE 0
#endif

// Sub-stage timers accumulate straight into the actor's phase record in
// memory: keeping four 64 bit accumulators live across the serve loops made
// the compiler spill inside pass A (see T11 in the log).
#if PROFILE_SUB
#define SUB_T(v)      v = csr_read(VX_CSR_MCYCLE)
#define SUB_ACC(m, v) ph->m += csr_read(VX_CSR_MCYCLE) - (v)
#else
#define SUB_T(v)      ((void)0)
#define SUB_ACC(m, v) ((void)0)
#endif

#ifndef BAR_TIMEOUT
#define BAR_TIMEOUT 0   // DIAGNOSTIC: give up after this many spins and record
                        // the slot in bar[7], so a stuck barrier reports itself
                        // instead of hanging the simulation.
#endif
// One arrival per WARP, so NB blocks of SPLIT warps each arrive NB x SPLIT times.
// Counting only lane 0 of the block (warp 0) let sibling warps through before
// the other blocks' sibling warps had finished the round: at 4 warps a drain
// was triggered before every reply was appended, and the functional apply
// missed them (T41c "got 0"). Identical to the old barrier at SPLIT=1.
static inline void gbarrier(volatile uint32_t* bar, int slot, int tid, int NB) {
#if ISA2
    vx_fence();                                           // ISA2: this warp's writes are acknowledged before it arrives
#endif
    if (tid % GLANES == 0) __atomic_fetch_add((uint32_t*)&bar[slot], 1, __ATOMIC_RELAXED);
    vx_fence();
#if BAR_TIMEOUT
    uint32_t spins = 0;
    while (!vx_vote_all(bar[slot] >= (uint32_t)NB * (uint32_t)(SPLIT))) {
        if (++spins > (uint32_t)BAR_TIMEOUT) {
            if (tid == 0) __atomic_fetch_add((uint32_t*)&bar[7], 1u << (slot * 4), __ATOMIC_RELAXED);
            break;
        }
    }
#elif ISA2
    // the arrival count is kept by L2 atomics: poll it with an atomic read, then drop the lines cached before
    while (!vx_vote_all(__atomic_fetch_add((uint32_t*)&bar[slot], 0u, __ATOMIC_RELAXED) >= (uint32_t)NB * (uint32_t)(SPLIT))) { }
    ISA2_L1INV();
#else
    while (!vx_vote_all(bar[slot] >= (uint32_t)NB * (uint32_t)(SPLIT))) { }
#endif
}

// append(q, word): one store into line q+1 of the window
#if SWQ
// software queue: storage, stride (bytes), tails, word size and capacity of the current
// geometry; every lane sets them (swq_phase) before it appends in a round
static char* g_sq_st; static uint32_t* g_sq_tl; static uint32_t g_sq_stride, g_sq_ws, g_sq_cap;
static inline void swq_phase(uint64_t st, uint32_t stride_b, uint64_t tl, uint32_t ws, uint32_t cap) {
    g_sq_st = (char*)st; g_sq_stride = stride_b; g_sq_tl = (uint32_t*)tl; g_sq_ws = ws; g_sq_cap = cap;
}
static inline void swq_put(uint32_t q, uint64_t w) {
    const uint32_t s = __atomic_fetch_add(&g_sq_tl[q], 1u, __ATOMIC_RELAXED);
    if (s < g_sq_cap) {
        char* p = g_sq_st + (size_t)q * g_sq_stride + (size_t)s * g_sq_ws;
        if (g_sq_ws == 8) *(uint64_t*)p = w; else *(uint32_t*)p = (uint32_t)w;
    }
}
static inline void aq_append(uint32_t q, req_t word) { swq_put(q, (uint64_t)word); }
static inline void aq_append64(uint32_t q, uint64_t word) { swq_put(q, word); }
#elif ISA2
static inline void aq_append(uint32_t q, req_t word) { aq_isa2_put(q, word); }
static inline void aq_append64(uint32_t q, uint64_t word) { vx_aq_append64(q, word); }
#else
static inline void aq_append(uint32_t q, req_t word) {
    *(volatile req_t*)(uintptr_t)((uint64_t)AQ_BASE + 64ull * ((uint64_t)q + 1ull)) = word;
}
static inline void aq_append64(uint32_t q, uint64_t word) {
    *(volatile uint64_t*)(uintptr_t)((uint64_t)AQ_BASE + 64ull * ((uint64_t)q + 1ull)) = word;
}
#endif
// write the queue geometry: which storage array, how far apart the queues sit,
// where the lengths go, and how wide an appended word is
#if REPLY_TILE
#define RQ_STRIDE (RT_SSTRIDE)
#else
#define RQ_STRIDE (RQ_CAP)
#endif
// Every lane of the block writes the SAME five words. Writing them from one
// lane (`if (tid == 0)`) is a divergent region, and with two such regions in
// one kernel the reconvergence broke: the trace showed a warp spinning in the
// barrier with lane 0 still inactive, so the one lane that increments the
// barrier never arrived and every block spun forever. Warp-uniform stores
// remove the divergence. The model ignores a rewrite that changes nothing, so
// the sixteen identical stores cost one state change.
static inline void aq_geometry(uint64_t qbase, uint64_t qstride, uint64_t tailbase,
                               uint64_t wsize, uint64_t cap) {
#if !BASELINE && !(ISA2 && SWQ)   // ISA2: the append off writes no L2 control register
    volatile uint64_t* c = (volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE;
    c[0] = qbase; c[1] = qstride; c[2] = tailbase; c[3] = wsize; c[4] = cap;
#endif
}

// End of an append round: the model charges the partial line of every queue
// (the tail charge, log T36a). Warp uniform, from block 0, after the barrier
// that ends the round.
static inline void aq_flush(void) {
#if !BASELINE && !ISA2   // ISA2: no flush (the barrier waits for every append's acknowledgement)
#if TAILFLUSH
    volatile uint64_t* c = (volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE;
    c[5] = 1;
#endif
#endif
}
// END OF ROUND (P3). Flushes the partial lines as aq_flush does, and then
// EMPTIES every queue, so the next chunk refills the same storage from the
// start. One store, warp uniform, issued by one block after the barrier that
// closed the chunk's receive. This is what makes the queue a small buffer
// reused CHUNKS times instead of one sized for the whole run, and so the only
// thing that can bound the live footprint (docs/next_mechanisms.md, P3).
static inline void aq_round(void) {
#if !BASELINE && !(ISA2 && SWQ)
    volatile uint64_t* c = (volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE;
    c[6] = 1;
#endif
}

// Which reply queue a reply belongs in. Keyed by the asker always; the second
// key is the question tile when REPLY_TILE is on and the owner's block
// otherwise. Both are partitions of ONE asker's mailbox, so the asker still
// reads only its own mail.
#if REPLY_TILE
#define RQ_ID(asker, que, nb, blk)  ((uint32_t)(asker) * (uint32_t)(REPLY_TILE) \
                                     + (uint32_t)(que) / (uint32_t)(RTILE_SZ))
#else
#define RQ_ID(asker, que, nb, blk)  ((uint32_t)(asker) * (uint32_t)(nb) + (uint32_t)(blk))
#endif

#if APPLY
// Apply engine (docs/apply_engine_plan.md). Each lane writes its own command
// line, a 16-lane store to 16 lines with no divergence; word 7 triggers. The
// done word is one address for the core, so the poll is one coalesced load.
static inline void ap_issue(uint32_t lane, uint64_t qbase, uint64_t qstride, uint64_t tailbase,
                            uint32_t q0, uint32_t nq, uint64_t base, uint64_t stride, uint64_t spec) {
    volatile uint64_t* c = (volatile uint64_t*)(uintptr_t)((uint64_t)AP_BASE + 64ull * (uint64_t)lane);
    c[0] = qbase; c[1] = qstride; c[2] = tailbase;
    c[3] = (uint64_t)q0 | ((uint64_t)nq << 32);
    c[4] = base;  c[5] = stride;  c[6] = spec;
    c[7] = 1;
}
// Command slot of this lane on its core: warp id x threads + lane. Several
// warps on a core share one engine and one done flag (line AP_NSLOTS).
#ifndef NUM_WARPS
#define NUM_WARPS 1
#endif
#ifndef NUM_THREADS
#define NUM_THREADS 16
#endif
#define AP_NSLOTS ((NUM_WARPS) * (NUM_THREADS))
static inline uint32_t ap_slot(uint32_t tid) {
    return (uint32_t)vx_warp_id() * (uint32_t)(NUM_THREADS) + tid;
}
static inline void ap_wait(void) {
    volatile uint32_t* d = (volatile uint32_t*)(uintptr_t)((uint64_t)AP_BASE + 64ull * (uint64_t)AP_NSLOTS);
    while (!vx_vote_all(*d != 0)) { }
}
#endif

#if TILE_AQ
// tile queue index of (owner, tile); see QMAJOR in common.h
static inline uint32_t tqid_of(uint32_t owner, uint32_t tile, uint32_t N) {
#if QMAJOR
    return tile * N + owner;
#else
    (void)N; return owner * (uint32_t)TILE_AQ + tile;
#endif
}
#endif

#if TILE_AQ
// REORDER, ladder step 4. My own inbox into my own per table tile queues. The
// asker block is not recoverable from the packed request once it leaves the
// queue it arrived in, so it rides along in the high half of the word.
static __attribute__((noinline)) void tile_reorder(const req_t* q, uint32_t tail, uint32_t ab,
                                                   uint32_t me, uint32_t N) {
    for (uint32_t i = 0; i < tail; i++) {
        req_t pay = q[i];
        uint32_t til = PK_LIX(pay) / (uint32_t)TILE_SZ;
#if !TILE_NOAPPEND
        // rebuild into the tile word: whole asker id up, {question, index} down
        aq_append64(tqid_of(me, til, N),
                    ((uint64_t)(ab * (uint32_t)GLANES + PK_AIB(pay)) << 32)
                  | (uint64_t)TK_MAKE(PK_QUE(pay), PK_LIX(pay)));
#else
        if (til == 0xFFFFFFFFu) aq_append64(tqid_of(me, 0, N), 0);   // keep the loop, drop the append
#endif
    }
}
#if REQ32
// SERVE from a tile queue of 32 bit words: the tile comes from the queue key,
// the word supplies only the offset inside it (common.h, REQ32).
static __attribute__((noinline)) void serve_tile32(const uint32_t* q, uint32_t tail, const int* tslice,
                                                   uint32_t tile, uint32_t NB, uint32_t blk) {
    const int* tp = tslice + (size_t)tile * (uint32_t)TILE_SZ;
    for (uint32_t i = 0; i < tail; i++) {
        uint32_t w   = q[QIX_(q, i)];
        uint32_t que = R32_QUE(w);
        uint32_t val = (uint32_t)tp[R32_OFF(w)];
        aq_append64(RQ_ID(R32_ASKER(w), que, NB, blk), ((uint64_t)que << 32) | (uint64_t)val);
    }
}
#endif
// SERVE from a tile queue: same work as serve_aq, but every table address in
// this call lies inside one TILE_SZ slice.
static __attribute__((noinline)) void serve_tile(const uint64_t* q, uint32_t tail, const int* tslice,
                                                 uint32_t NB, uint32_t blk) {
    for (uint32_t i = 0; i < tail; i++) {
        uint64_t w     = q[QIX_(q, i)];
        uint32_t asker = (uint32_t)(w >> 32);
        uint32_t pay   = (uint32_t)w;
        uint32_t que   = TK_QUE(pay);
        uint32_t val   = (uint32_t)tslice[TK_LIX(pay)];
        aq_append64(RQ_ID(asker, que, NB, blk), ((uint64_t)que << 32) | (uint64_t)val);
    }
}
#endif

#if REPLY_AQ
// SERVE, ladder step 3. My whole inbox, one asker block at a time: read the
// request, look up MY OWN table, append {question, value} to the ASKER's queue.
// No pool, no counting pass, no scan, no atomics, no publish, no run table.
static __attribute__((noinline)) void serve_aq(const req_t* q, uint32_t tail, const int* tslice,
                                               uint32_t rq0, uint32_t NB, uint32_t blk) {
    for (uint32_t i = 0; i < tail; i++) {
        req_t pay = q[i];
        uint32_t aib = PK_AIB(pay);
        uint32_t que = PK_QUE(pay);
        uint32_t val = (uint32_t)tslice[PK_LIX(pay)];
        aq_append64(RQ_ID(rq0 + aib, que, NB, blk), ((uint64_t)que << 32) | (uint64_t)val);
        (void)NB; (void)blk;
    }
}
#if REPLY_TILE
// DRAIN, ladder step 5. My tile queue holds exactly the replies whose question
// lies in this tile, so a lane drops each one at its question's position in a
// private scratchpad row. Then the sixteen lanes walk the tile together: at
// step j every lane writes tgt[(base+j)*N + me], and the block's sixteen
// actors are sixteen consecutive ints, which is exactly one 64 byte line.
static __attribute__((noinline)) void drain_tile_in(const uint64_t* r, int* row, uint32_t base) {
    for (uint32_t i = 0; i < (uint32_t)RTILE_SZ; i++) {
        uint64_t w = r[i];
        row[(uint32_t)(w >> 32) - base] = (int)(uint32_t)w;
    }
}
static __attribute__((noinline)) void drain_tile_out(const int* row, int* tgt, uint32_t base,
                                                     uint32_t me, uint32_t N) {
    for (uint32_t j = 0; j < (uint32_t)RTILE_SZ; j++)
        tgt[(size_t)(base + j) * N + me] = row[j];
}
#endif
// DRAIN, ladder step 3. One 8 byte load per reply instead of two 4 byte loads
// from two arrays, and no run table.
static __attribute__((noinline)) void drain_aq(const uint64_t* r, uint32_t tail, int* tgt,
                                               uint32_t me, uint32_t N) {
    for (uint32_t i = 0; i < tail; i++) {
        uint64_t w = r[i];
        tgt[TGT_IDX(me, (uint32_t)(w >> 32), N)] = (int)(uint32_t)w;
    }
}
#endif

// Each round is its own function so no live value spans rounds. With one
// warp per core the stack is global memory, and STACK_LOG2_SIZE=13 puts the
// sixteen lanes' stacks 8 KB apart, all in ONE set of the 16 KB 4-way L1:
// a spill reload inside a hot loop misses twelve of sixteen ways every
// iteration (measured: ld t3,0x148(sp) in the request loop cost 900 cycles
// per request). Check kernel.dump for (sp) inside every loop after a change.
static __attribute__((noinline)) void ask_round(args_t* __UNIFORM__ a, int me, int tid, int blk,
                                                int N, int NB, int* buf, int sub) {
    (void)buf;
    // Geometry first. Every block writes the same four words; the model resets
    // its tails only when the geometry changes, so this is idempotent.
#if DIRECT_TILE && REQ32
    // REQ32: the same tile queues, 32 bit words (common.h)
    aq_geometry(a->ts_addr, (uint64_t)TQ_STRIDE32 * 4ull, a->ttails_addr, 4, TQ_CAP);
#if SWQ
    swq_phase(a->ts_addr, TQ_STRIDE32 * 4u, a->ttails_addr, 4, TQ_CAP);
#endif
#elif DIRECT_TILE
    // The request round runs on the TILE geometry from the start: the asker
    // keys by (owner, table tile) itself, and the owner never reorders.
    aq_geometry(a->ts_addr, (uint64_t)TQ_STRIDE_W * 8ull, a->ttails_addr, 8, TQ_CAP);
#if SWQ
    swq_phase(a->ts_addr, TQ_STRIDE_W * 8u, a->ttails_addr, 8, TQ_CAP);   // was missing: SWQ never ran with 64-bit requests
#endif
#else
    aq_geometry(a->qs_addr, (uint64_t)AQ_QSTRIDE_W * sizeof(req_t), a->tails_addr, AQ_WORD, AQ_QCAP);
#endif
#if OVF
    { volatile uint64_t* c = (volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE;   // overflow pool (control word 7)
      c[7] = (1ull << 60) | a->pool_addr;
      c[7] = (3ull << 60) | ((uint64_t)(OVF_CH) << 32) | (uint64_t)(OVF_MAXC);
      c[7] = (4ull << 60) | (uint64_t)(OVF_NCH);
      c[7] = (2ull << 60) | a->ctab_addr; }
#endif
    (void)tid;
    vx_fence();
    // SPLIT: this sub-lane asks only its slice of the actor's questions. The
    // question index stays global to the actor, so the reply still lands on
    // the right tgt entry and the set of (actor, question) pairs is unchanged.
    // CONSTANT trip count, sub only offsets the question: bounds derived from
    // the thread id make the loop divergent to the compiler, which then
    // predicates every iteration with warp control instructions (T42).
    const uint32_t qlo = (uint32_t)sub * (uint32_t)((M) / (SPLIT));
    for (uint32_t k = 0; k < (uint32_t)((M) / (SPLIT)); k++) {
        const uint32_t q2 = qlo + k;
#if NCONST
        uint32_t dst  = mix(me, 2u*q2 + 1u) % (uint32_t)(NASSUME);
#else
        uint32_t dst  = mix(me, 2u*q2 + 1u) % (uint32_t)N;
#endif
        uint32_t lidx = mix(me, 2u*q2 + 2u) % (uint32_t)TABLE_SIZE;
#if SKEW > 1
        { const uint32_t k_ = skew_key(mix(me, 2u*q2 + 1u), (uint32_t)(NASSUME) * (uint32_t)TABLE_SIZE); dst = k_ % (uint32_t)(NASSUME); lidx = (k_ / (uint32_t)(NASSUME)) % (uint32_t)TABLE_SIZE; }
#endif
#if REQ_PROBE
        buf[(dst * (uint32_t)NB + (uint32_t)blk) & 2047u] =
                  (int)(((uint32_t)tid * M + q2) * (uint32_t)TABLE_SIZE + lidx);
#elif DIRECT_TILE && REQ32
#if SWQ
        swq_put(tqid_of(dst, lidx / (uint32_t)TILE_SZ, (uint32_t)N), (uint64_t)R32_MAKE(me, q2, lidx));
#elif ISA2
        vx_aq_append32(tqid_of(dst, lidx / (uint32_t)TILE_SZ, (uint32_t)N), R32_MAKE(me, q2, lidx));
#else
        *(volatile uint32_t*)(uintptr_t)((uint64_t)AQ_BASE + 64ull * ((uint64_t)tqid_of(dst, lidx / (uint32_t)TILE_SZ, (uint32_t)N) + 1ull))
            = R32_MAKE(me, q2, lidx);
#endif
#elif DIRECT_TILE
        // same word the reorder pass would have built: whole asker id up top
        aq_append64(tqid_of(dst, lidx / (uint32_t)TILE_SZ, (uint32_t)N),
                    ((uint64_t)(uint32_t)me << 32) | (uint64_t)TK_MAKE(q2, lidx));
#else
        aq_append(dst * (uint32_t)NB + (uint32_t)blk, PK_MAKE(tid, q2, lidx));
#endif
    }
    vx_fence();
}

// Hot loops as noinline leaves with at most eight arguments: inside them
// nothing else is live, so the compiler has no reason to spill (T11).
static __attribute__((noinline)) void pass_a(const int* q, uint32_t lo, uint32_t hi, int* cnt) {
    for (uint32_t i = lo; i < hi; i++) {
        uint32_t pay = (uint32_t)q[i];
        __atomic_fetch_add(&cnt[PK_AIB(pay)], 1, __ATOMIC_RELAXED);
    }
}
#if STAGE_PIECE == 1
static __attribute__((noinline)) void pass_a_stage(const int* q, uint32_t lo, uint32_t hi, int* cnt, int* st) {
    for (uint32_t i = lo; i < hi; i++) {
        uint32_t pay = (uint32_t)q[i];
        st[i - lo] = (int)pay;
        __atomic_fetch_add(&cnt[PK_AIB(pay)], 1, __ATOMIC_RELAXED);
    }
}
#endif
static __attribute__((noinline)) uint32_t pass_b(const int* q, uint32_t lo, uint32_t hi, const int* tslice,
                                                 int* cur, int* srtq, int* srtv) {
    uint32_t over = 0;
    for (uint32_t i = lo; i < hi; i++) {
        uint32_t pay = (uint32_t)q[i];
        uint32_t aib = PK_AIB(pay);
        uint32_t loc = (uint32_t)__atomic_fetch_add(&cur[aib], 1, __ATOMIC_RELAXED);
        if (loc < (uint32_t)CAP2) {
            srtq[loc] = (int)PK_QUE(pay);
            srtv[loc] = tslice[PK_LIX(pay)];
        } else over++;
    }
    return over;
}
static __attribute__((noinline)) void publish(int* rt, int* rv, const int* srtq, const int* srtv, int total, int tid) {
#if PACK64
    for (int i = 2 * tid; i < total; i += 2 * GLANES) {
        if (i + 1 < total) {
            *(uint64_t*)&rt[i] = *(const uint64_t*)&srtq[i];
            *(uint64_t*)&rv[i] = *(const uint64_t*)&srtv[i];
        } else {
            rt[i] = srtq[i];
            rv[i] = srtv[i];
        }
    }
#else
    for (int i = tid; i < total; i += GLANES) { rt[i] = srtq[i]; rv[i] = srtv[i]; }
#endif
}

#if STAGE_PIECE == 2
// The block loads lane k's piece with lockstep line loads. Its own function so
// the copy loops hold their few live values in registers.
static __attribute__((noinline)) void stage_pieces(const int* qs, const volatile uint32_t* tails, int* st,
                                                   uint32_t qid0, uint32_t NB, uint32_t ch, int tid) {
    for (int k = 0; k < GLANES; k++) {
        uint32_t qk = qid0 + (uint32_t)k * NB;
        uint32_t kt = tails[qk];
        if (kt > (uint32_t)AQ_QCAP) kt = AQ_QCAP;
        uint32_t klo = ch * PIECE, khi = klo + PIECE; if (khi > kt) khi = kt;
        const int* kq = qs + (size_t)qk * AQ_QSTRIDE_W;
        int* sk = st + k * PIECE;
        for (uint32_t i = klo + (uint32_t)tid; i < khi; i += GLANES) sk[i - klo] = kq[i];
    }
}
#endif

#ifndef SERVE_MLP
#define SERVE_MLP 0   // WC_SERVE: issue a step's WC_U table loads before its replies (memory-level parallelism)
#endif
#if WC_SERVE && REQ32
// warp-cooperative serve, the whole round in one leaf (one prologue per round: every
// callee save is 16 scattered DRAM writes on this machine). The warp serves the block's
// 16 owners one at a time; lane ln takes entries ln, ln+16, ... of the owner's tile queue.
// Loop bounds are warp uniform (tails read from uniform addresses), full 16-entry steps
// run unguarded, 4 queue lines are loaded before any is used (the memory-level
// parallelism the per-lane mapping got from 16 streams), one guarded step takes the rest.
#define WCS_DO(w) { const uint32_t que_ = R32_QUE(w); \
    aq_append64(RQ_ID(R32_ASKER(w), que_, NB, blk), ((uint64_t)que_ << 32) | (uint64_t)(uint32_t)tp[R32_OFF(w)]); }
// one owner's tile queue: a small leaf that fits the caller-saved registers, so a call
// costs no stack traffic
static inline __attribute__((always_inline)) void serve_q_wc(const uint32_t* q, const int* tp, uint32_t tl,
                                                 uint32_t ln, uint32_t NB, uint32_t blk, size_t S) {   // S: ISA2 step
    const uint32_t nf = tl / GLANES;
    uint32_t k = 0;
    for (; k + (WC_U) <= nf; k += (WC_U)) {
        uint32_t w[WC_U];
#pragma GCC unroll 8
        for (uint32_t u = 0; u < (uint32_t)(WC_U); u++) w[u] = q[(k + u) * S];
#if SERVE_MLP
        // every table load of the step issued before any reply: WC_U misses in flight per lane
        uint32_t v[WC_U];
#pragma GCC unroll 8
        for (uint32_t u = 0; u < (uint32_t)(WC_U); u++) v[u] = (uint32_t)tp[R32_OFF(w[u])];
#pragma GCC unroll 8
        for (uint32_t u = 0; u < (uint32_t)(WC_U); u++) { const uint32_t que_ = R32_QUE(w[u]);
            aq_append64(RQ_ID(R32_ASKER(w[u]), que_, NB, blk), ((uint64_t)que_ << 32) | (uint64_t)v[u]); }
#else
#pragma GCC unroll 8
        for (uint32_t u = 0; u < (uint32_t)(WC_U); u++) WCS_DO(w[u])
#endif
    }
    for (; k < nf; k++) { const uint32_t w = q[k * S]; WCS_DO(w) }
    if (nf * GLANES + ln < tl) { const uint32_t w = q[nf * S]; WCS_DO(w) }
}
static_assert(!(WC_SERVE) || (QMAJOR), "WC_SERVE assumes the tile-major queue index");
#endif
#if WC_SERVE && !REQ32
// warp-cooperative serve of 64 bit tile requests {asker | question, in-slice index}. The
// 64 bit word carries the whole in-slice index, so the queue tile can be COARSE (few live
// queues at any table size). FINE_T > 1 serves each coarse queue in FINE_T passes, one
// table slice per pass: the slice stays cached and the small queue is re-read from cache.
static_assert((DIRECT_TILE) && (TILE_AQ), "WC_SERVE without REQ32 needs DIRECT_TILE");
#endif
static_assert((FINE_T) == 1 || ((WC_SERVE) && !(REQ32)), "FINE_T needs WC_SERVE with 64 bit requests");
#if WC_SERVE && !REQ32
#define WCS64_DO(w) { const uint32_t p_ = (uint32_t)(w); const uint32_t x_ = TK_LIX(p_); \
    if ((FINE_T) == 1 || x_ - base < (uint32_t)(FTILE_SZ)) { const uint32_t que_ = TK_QUE(p_); \
    aq_append64(RQ_ID((uint32_t)((w) >> 32), que_, NB, blk), ((uint64_t)que_ << 32) | (uint64_t)(uint32_t)tb[x_]); } }
#if OVF
// a full coarse queue's overflow chunks for one pass (warp-uniform, rare: kept out of line)
static __attribute__((noinline, cold)) void serve_ovf64(const uint64_t* cr, uint32_t r, const int* tb,
                                                        uint32_t base, uint32_t ln, uint32_t blk) {
    constexpr uint32_t NB = (uint32_t)(NASSUME / GLANES);
    (void)base;
#pragma GCC unroll 1
    for (uint32_t cc = 0; r != 0; cc++) {
        const uint32_t n = r < (uint32_t)(OVF_CH) ? r : (uint32_t)(OVF_CH);
        const uint64_t* q = (const uint64_t*)(uintptr_t)cr[cc] + ln;
        const uint32_t nf = n / GLANES;
#pragma GCC unroll 1
        for (uint32_t k = 0; k < nf; k++) { const uint64_t w = q[k * GLANES]; WCS64_DO(w) }
        if (nf * GLANES + ln < n) { const uint64_t w = q[nf * GLANES]; WCS64_DO(w) }
        r -= n; }
}
#endif
static __attribute__((noinline)) uint32_t serve_wc64(const uint64_t* __UNIFORM__ ts, const uint32_t* __UNIFORM__ ttails,
                                                     const int* __UNIFORM__ table, uint32_t ln,
                                                     uint32_t __UNIFORM__ sub, uint32_t __UNIFORM__ blk
#if OVF
                                                     , uint64_t __UNIFORM__ ctab_
#endif
                                                     ) {
    constexpr uint32_t NB = (uint32_t)(NASSUME / GLANES);
    const uint32_t* tl0 = ttails + blk * (uint32_t)GLANES;
#pragma GCC unroll 1
    for (uint32_t c = 0; c < ((uint32_t)(TILE_AQ) + (uint32_t)(SPLIT) - 1) / (uint32_t)(SPLIT) * GLANES; c++) {
        const uint32_t t  = sub + (c / GLANES) * (uint32_t)(SPLIT);
        if (t >= (uint32_t)(TILE_AQ)) break;   // fewer tiles than SPLIT (ablation, piece B off): warp uniform
        const uint32_t j  = c % GLANES;
        const uint32_t q2 = t * (uint32_t)NASSUME + blk * (uint32_t)GLANES + j;   // tqid_of, QMAJOR
        uint32_t tl = tl0[t * (uint32_t)NASSUME + j];
#if OVF
        const uint32_t tov = tl > (uint32_t)TQ_CAP ? tl - (uint32_t)TQ_CAP : 0u;
#endif
        tl = tl > (uint32_t)TQ_CAP ? (uint32_t)TQ_CAP : tl;
        const uint64_t* q = QPTR_(ts, TQ_STRIDE_W, q2) + QIX_(ts, ln);
        const int* tb = table + (size_t)(blk * (uint32_t)GLANES + j) * TABLE_SIZE;
        const uint32_t nf = tl / GLANES;
#pragma GCC unroll 1
        for (uint32_t f = 0; f < (uint32_t)(FINE_T); f++) {
            const uint32_t base = t * (uint32_t)TILE_SZ + f * (uint32_t)(FTILE_SZ);
            (void)base;
            uint32_t k = 0;
            for (; k + (WC_U) <= nf; k += (WC_U)) {
                uint64_t w[WC_U];
#pragma GCC unroll 8
                for (uint32_t u = 0; u < (uint32_t)(WC_U); u++) w[u] = q[(k + u) * QSTEP_(ts)];
#if SERVE_MLP
                uint32_t v[WC_U];
#pragma GCC unroll 8
                for (uint32_t u = 0; u < (uint32_t)(WC_U); u++) {
                    const uint32_t x_ = TK_LIX((uint32_t)w[u]);
                    v[u] = ((FINE_T) == 1 || x_ - base < (uint32_t)(FTILE_SZ)) ? (uint32_t)tb[x_] : 0u; }
#pragma GCC unroll 8
                for (uint32_t u = 0; u < (uint32_t)(WC_U); u++) {
                    const uint32_t p_ = (uint32_t)w[u]; const uint32_t x_ = TK_LIX(p_);
                    if ((FINE_T) == 1 || x_ - base < (uint32_t)(FTILE_SZ)) { const uint32_t que_ = TK_QUE(p_);
                        aq_append64(RQ_ID((uint32_t)(w[u] >> 32), que_, NB, blk), ((uint64_t)que_ << 32) | (uint64_t)v[u]); } }
#else
#pragma GCC unroll 8
                for (uint32_t u = 0; u < (uint32_t)(WC_U); u++) WCS64_DO(w[u])
#endif
            }
            for (; k < nf; k++) { const uint64_t w = q[k * QSTEP_(ts)]; WCS64_DO(w) }
            if (nf * GLANES + ln < tl) { const uint64_t w = q[nf * QSTEP_(ts)]; WCS64_DO(w) }
#if OVF
            if (tov) serve_ovf64((const uint64_t*)(uintptr_t)ctab_ + (size_t)q2 * (OVF_MAXC), tov, tb, base, ln, blk);
#endif
        }
    }
    return 0;
}
#endif
#if WC_SERVE && REQ32
#if OVF
static __attribute__((noinline, cold)) void serve_ovf(const uint64_t* cr, uint32_t r, const int* tp, uint32_t ln, uint32_t blk) {
#pragma GCC unroll 1
    for (uint32_t cc = 0; r != 0; cc++) {
        const uint32_t n = r < (uint32_t)(OVF_CH) ? r : (uint32_t)(OVF_CH);
        serve_q_wc((const uint32_t*)(uintptr_t)cr[cc] + ln, tp, n, ln, (uint32_t)(NASSUME / GLANES), blk, GLANES);
        r -= n; }
}
#endif
static __attribute__((noinline)) uint32_t serve_wc(const uint32_t* __UNIFORM__ ts32, const uint32_t* __UNIFORM__ ttails,
                                                   const int* __UNIFORM__ table, uint32_t ln, uint32_t __UNIFORM__ sub,
                                                   uint32_t __UNIFORM__ NB, uint32_t __UNIFORM__ blk
#if OVF
                                                   , uint64_t __UNIFORM__ ctab_
#endif
                                                   ) {
    // few values live across iterations: over-long queues are clamped, not counted
    // (the host audits every tail), and NB is the compile-time block count
    (void)NB;
    const uint32_t* tl0 = ttails + blk * (uint32_t)GLANES;
    const int* tb0 = table + (size_t)blk * GLANES * TABLE_SIZE;
    for (uint32_t c = 0; c < ((uint32_t)(TILE_AQ) + (uint32_t)(SPLIT) - 1) / (uint32_t)(SPLIT) * GLANES; c++) {
        const uint32_t t  = sub + (c / GLANES) * (uint32_t)(SPLIT);
        if (t >= (uint32_t)(TILE_AQ)) break;   // fewer tiles than SPLIT (ablation, piece B off): warp uniform
        const uint32_t j  = c % GLANES;
        const uint32_t q2 = t * (uint32_t)NASSUME + blk * (uint32_t)GLANES + j;   // tqid_of, QMAJOR
        uint32_t tl = tl0[t * (uint32_t)NASSUME + j];
#if OVF
        const int* tp_ = tb0 + (size_t)j * TABLE_SIZE + (size_t)t * (uint32_t)TILE_SZ;
        serve_q_wc(QPTR_(ts32, TQ_STRIDE32, q2) + QIX_(ts32, ln), tp_, tl < (uint32_t)TQ_CAP ? tl : (uint32_t)TQ_CAP,
                   ln, (uint32_t)(NASSUME / GLANES), blk, QSTEP_(ts32));
        if (tl > (uint32_t)TQ_CAP)     // the queue's overflow chunks (warp-uniform, rare: kept out of line)
            serve_ovf((const uint64_t*)(uintptr_t)ctab_ + (size_t)q2 * (OVF_MAXC), tl - (uint32_t)TQ_CAP, tp_, ln, blk);
#else
        tl = tl > (uint32_t)TQ_CAP ? (uint32_t)TQ_CAP : tl;
        serve_q_wc(QPTR_(ts32, TQ_STRIDE32, q2) + QIX_(ts32, ln), tb0 + (size_t)j * TABLE_SIZE + (size_t)t * (uint32_t)TILE_SZ,
                   tl, ln, (uint32_t)(NASSUME / GLANES), blk, QSTEP_(ts32));
#endif
    }
    return 0;
}
#endif
#if DRAIN_DIRECT
static_assert((REPLY_TILE) && !(TGT_T) && !(APPLY), "DRAIN_DIRECT: reply tiles, natural tgt layout, no engine");
// the drain round in one leaf: the warp takes reply tiles t = sub, sub+SPLIT, ... and for each
// the block's 16 actors one at a time; 16 lanes read 16 consecutive replies (WC_U loads first)
// and store each value at tgt[actor*M + question]: one actor's tile is a contiguous span
#define DD_DO(w) row[(uint32_t)((w) >> 32)] = (int)(uint32_t)(w)
#if LANE_RECV
// piece C off: lane ln drains its own actor's reply tiles t = sub, sub + SPLIT, ... alone, a reply per load
static __attribute__((noinline)) void drain_direct(const uint64_t* __UNIFORM__ rq, int* __UNIFORM__ tgt,
                                                   uint32_t ln, uint32_t __UNIFORM__ sub, uint32_t __UNIFORM__ blk) {
    const uint32_t ac = blk * (uint32_t)GLANES + ln;
    int* row = tgt + (size_t)ac * (M);
#pragma GCC unroll 1
    for (uint32_t t = sub; t < (uint32_t)(REPLY_TILE); t += (uint32_t)(SPLIT)) {
        const uint64_t* r = QPTR_(rq, RQ_STRIDE, ac * (uint32_t)REPLY_TILE + t);
#pragma GCC unroll 1
        for (uint32_t k = 0; k < (uint32_t)(RTILE_SZ); k++) { const uint64_t w = r[QIX_(r, k)]; DD_DO(w); }
    }
}
#else
static __attribute__((noinline)) void drain_direct(const uint64_t* __UNIFORM__ rq, int* __UNIFORM__ tgt,
                                                   uint32_t ln, uint32_t __UNIFORM__ sub, uint32_t __UNIFORM__ blk) {
    constexpr uint32_t NF = (uint32_t)(RTILE_SZ) / GLANES, REM = (uint32_t)(RTILE_SZ) % GLANES;
#pragma GCC unroll 1
    for (uint32_t c = 0; c < ((uint32_t)(REPLY_TILE) + (uint32_t)(SPLIT) - 1) / (uint32_t)(SPLIT) * GLANES; c++) {
        const uint32_t t = sub + (c / GLANES) * (uint32_t)(SPLIT);
        if (t >= (uint32_t)(REPLY_TILE)) break;   // fewer tiles than SPLIT (ablation, piece B off): warp uniform
        const uint32_t ac = blk * (uint32_t)GLANES + c % GLANES;
        const uint64_t* r = QPTR_(rq, RQ_STRIDE, ac * (uint32_t)REPLY_TILE + t) + QIX_(rq, ln);
        int* row = tgt + (size_t)ac * (M);
        uint32_t k = 0;
#pragma GCC unroll 1
        for (; k + (WC_U) <= NF; k += (WC_U)) {
            uint64_t w[WC_U];
#pragma GCC unroll 8
            for (uint32_t u = 0; u < (uint32_t)(WC_U); u++) w[u] = r[(k + u) * QSTEP_(rq)];
#pragma GCC unroll 8
            for (uint32_t u = 0; u < (uint32_t)(WC_U); u++) DD_DO(w[u]);
        }
        for (; k < NF; k++) { const uint64_t w = r[k * QSTEP_(rq)]; DD_DO(w); }
        if (REM && ln < REM) { const uint64_t w = r[NF * QSTEP_(rq)]; DD_DO(w); }
    }
}
#endif
#endif
#if WC_DRAIN && REPLY_TILE
// warp-cooperative drain, the whole round in one leaf: per reply tile the warp fills
// the rows of its 16 threads one actor at a time (16 consecutive replies per load, all
// loads of an actor's tile issued before any store), then every lane writes its own
// actor's row to tgt exactly as before (16 actors = one tgt line per step).
static __attribute__((noinline)) void drain_wc(const uint64_t* __UNIFORM__ rq, int* __UNIFORM__ rows, int* row,
                                               int* tgt, uint32_t ln, uint32_t me, uint32_t __UNIFORM__ sub,
                                               uint32_t __UNIFORM__ blk, uint32_t __UNIFORM__ N) {
    constexpr uint32_t NF = (uint32_t)(RTILE_SZ) / GLANES, REM = (uint32_t)(RTILE_SZ) % GLANES;
    for (uint32_t tt = 0; tt < (uint32_t)(REPLY_TILE) / (uint32_t)(SPLIT); tt++) {
        const uint32_t t = sub + tt * (uint32_t)(SPLIT);
        const uint32_t base = t * (uint32_t)RTILE_SZ;
        for (uint32_t j = 0; j < (uint32_t)GLANES; j++) {
            const uint64_t* r = rq + (size_t)((blk * (uint32_t)GLANES + j) * (uint32_t)REPLY_TILE + t) * RQ_STRIDE + ln;
            int* rj = rows + j * (uint32_t)RTILE_SZ - base;
            uint64_t w[NF];
#pragma GCC unroll 16
            for (uint32_t k = 0; k < NF; k++) w[k] = r[k * GLANES];
#pragma GCC unroll 16
            for (uint32_t k = 0; k < NF; k++) rj[(uint32_t)(w[k] >> 32)] = (int)(uint32_t)w[k];
            if (REM && ln < REM) { const uint64_t x = r[NF * GLANES]; rj[(uint32_t)(x >> 32)] = (int)(uint32_t)x; }
        }
        vx_fence();
        for (uint32_t jj = 0; jj < (uint32_t)RTILE_SZ; jj++)
            tgt[(size_t)(base + jj) * N + me] = row[jj];
        vx_fence();
    }
}
#endif
static __attribute__((noinline)) uint32_t serve_round(args_t* __UNIFORM__ a, int me, int tid, int blk,
                                                      int N, int NB, int* buf, uint32_t* toolong_out,
                                                      int sub) {
    int*  table  = (int*)a->table_addr;
    const req_t* qs = (const req_t*)a->qs_addr;
    const volatile uint32_t* tails = (const volatile uint32_t*)a->tails_addr;
    int*  rp_tag = (int*)a->rp_tag_addr;
    int*  rp_val = (int*)a->rp_val_addr;
    int*  ptbl   = (int*)a->ptbl_addr;
    uint32_t over = 0, toolong = 0;
    uint64_t a0 = 0; (void)a0;
#if SWQ
    swq_phase(a->rp_tag_addr, (uint32_t)RQ_STRIDE * 8u, a->rp_val_addr, 8, RQ_STRIDE);   // the reply geometry
#endif
#if PROFILE_SUB
    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];
#endif
#if TILE_AQ
    // Ladder step 4. The reorder pass ran before this call and switched the
    // geometry to the reply queues, so here we only serve tile by tile.
    (void)rp_tag; (void)rp_val; (void)ptbl; (void)buf; (void)tid; (void)qs; (void)tails;
    { const uint64_t* ts = (const uint64_t*)a->ts_addr;
      const volatile uint32_t* ttails = (const volatile uint32_t*)a->ttails_addr;
      // SPLIT: the owner's table is read only, so its sub-lanes can serve
      // different tiles at once; each reads and replies, none writes the table.
      for (uint32_t tt = 0; tt < ((uint32_t)(TILE_AQ) + (uint32_t)(SPLIT) - 1) / (uint32_t)(SPLIT); tt++) {   // constant trip count (T42)
          const uint32_t t = (uint32_t)sub + tt * (uint32_t)(SPLIT);
          if (t >= (uint32_t)(TILE_AQ)) break;   // fewer tiles than SPLIT (ablation, piece B off): warp uniform
#if WC_SERVE
          break;   // the whole round is served by serve_wc / serve_wc64 below
#else
          uint32_t tid2 = tqid_of((uint32_t)me, t, (uint32_t)N);
          uint32_t tail = ttails[tid2];
          if (tail > (uint32_t)TQ_CAP) { toolong++; tail = TQ_CAP; }
          SUB_T(a0);
#if REQ32
          serve_tile32(QPTR_((const uint32_t*)a->ts_addr, TQ_STRIDE32, tid2), tail,
                       table + (size_t)me * TABLE_SIZE, t, (uint32_t)NB, (uint32_t)blk);
#else
          serve_tile(QPTR_(ts, TQ_STRIDE_W, tid2), tail, table + (size_t)me * TABLE_SIZE,
                     (uint32_t)NB, (uint32_t)blk);
#endif
          SUB_ACC(v_place, a0);
#endif
      }
#if WC_SERVE && REQ32
      toolong += serve_wc((const uint32_t*)a->ts_addr, (const uint32_t*)a->ttails_addr, table,
                          (uint32_t)tid % (uint32_t)GLANES, (uint32_t)sub, (uint32_t)NB, (uint32_t)blk
#if OVF
                          , a->ctab_addr
#endif
                          );
#elif WC_SERVE
      toolong += serve_wc64(ts, (const uint32_t*)a->ttails_addr, table,
                            (uint32_t)tid % (uint32_t)GLANES, (uint32_t)sub, (uint32_t)blk
#if OVF
                            , a->ctab_addr
#endif
                            );
#endif
    }
    *toolong_out += toolong;
    return over;
#elif REPLY_AQ
    // Ladder step 3: no pool, no pieces. Stream my whole inbox and append each
    // reply straight to the asker's queue.
    (void)rp_tag; (void)rp_val; (void)ptbl; (void)buf; (void)tid;
    for (int ab = 0; ab < NB; ab++) {
        uint32_t qid  = (uint32_t)me * (uint32_t)NB + (uint32_t)ab;
        uint32_t tail = tails[qid];
        if (tail > (uint32_t)AQ_QCAP) { toolong++; tail = AQ_QCAP; }
        SUB_T(a0);
        serve_aq(qs + (size_t)qid * AQ_QSTRIDE_W, tail, table + (size_t)me * TABLE_SIZE,
                 (uint32_t)ab * GLANES, (uint32_t)NB, (uint32_t)blk);
        SUB_ACC(v_place, a0);
    }
    *toolong_out += toolong;
    return over;
#else
    // One asker block at a time: lane tid streams its own queue (me, ab) in
    // pieces of PIECE words; the sixteen pieces of a block fill at most CAP2.
    {
        int* cnt  = buf;                 // GLANES replies to each asker
        int* cur  = cnt + GLANES;        // GLANES cursor
        int* srtq = cur + GLANES;        // CAP2   question ids
        int* srtv = srtq + CAP2;         // CAP2   values
#if STAGE_PIECE
        int* st   = srtv + CAP2;         // CAP2   this lane's piece, at st[tid*PIECE]
#endif
        int  o    = me;

        for (int ab = 0; ab < NB; ab++) {
            uint32_t qid  = (uint32_t)me * (uint32_t)NB + (uint32_t)ab;
            uint32_t tail = tails[qid];
            if (tail > (uint32_t)AQ_QCAP) { toolong++; tail = AQ_QCAP; }
            const int* myq = qs + (size_t)qid * AQ_QSTRIDE_W;

            for (uint32_t ch = 0; ch < (uint32_t)NCHUNK; ch++) {
                uint32_t lo = ch * PIECE;
                uint32_t hi = lo + PIECE; if (hi > tail) hi = tail;
                if (lo > hi) lo = hi;
                cnt[tid] = 0;
                POOL_SYNC();
                SUB_T(a0);

                // pass A: count MY OWN piece by asker
#if STAGE_PIECE == 2
                // PROBE for a hardware stream-in: the block loads each lane's
                // piece with lockstep line loads (one line per instruction),
                // which in software has lanes touching neighbours' mail.
                stage_pieces(qs, tails, st, (uint32_t)blk * GLANES * (uint32_t)NB + (uint32_t)ab,
                             (uint32_t)NB, ch, tid);
                POOL_SYNC();
                pass_a(st + tid * PIECE, 0, hi - lo, cnt);
#elif STAGE_PIECE
                pass_a_stage(myq, lo, hi, cnt, st + tid * PIECE);
#else
                pass_a(myq, lo, hi, cnt);
#endif
                POOL_SYNC();
                SUB_ACC(v_count, a0); SUB_T(a0);

                { int pre = 0;
#if SHFL
                  int x = cnt[tid];
                  for (int dd = 1; dd < GLANES; dd <<= 1) {
                      int y = vx_shfl_up(x, dd, GLANES - 1, 0);
                      x += (tid >= dd) ? y : 0;   // see shfl_probe: never branch
                  }
                  pre = x - cnt[tid];
#else
                  for (int k = 0; k < GLANES; k++) if (k < tid) pre += cnt[k];
#endif
                  cur[tid] = pre; }
                POOL_SYNC();
                SUB_ACC(v_scan, a0); SUB_T(a0);

                // pass B: look up MY OWN table and place the reply by asker.
                // Position ascending in my queue is the asker's program order,
                // so replies per (owner, asker) pair stay FIFO.
#if STAGE_PIECE
                over += pass_b(st + tid * PIECE, 0, hi - lo, table + (size_t)o * TABLE_SIZE, cur, srtq, srtv);
#else
                over += pass_b(myq, lo, hi, table + (size_t)o * TABLE_SIZE, cur, srtq, srtv);
#endif
                POOL_SYNC();
                SUB_ACC(v_place, a0); SUB_T(a0);

                // publish DENSELY, and record where each asker's run begins
                { size_t rb = ((size_t)(blk * NB + ab) * NCHUNK + ch) * CAP2;
                  int total = cur[GLANES - 1];
                  if (total > CAP2) total = CAP2;
                  size_t po = ((size_t)(blk * NB + ab) * NCHUNK + ch) * TSTRIDE;
                  for (int k = tid; k <= GLANES; k += GLANES) {
                      int v = (k == 0) ? 0 : cur[k - 1];
                      ptbl[po + k] = (v > CAP2) ? CAP2 : v;
                  }
                  publish(rp_tag + rb, rp_val + rb, srtq, srtv, total, tid);
                }
                POOL_SYNC();
                SUB_ACC(v_pub, a0);
            }
        }
    }
    *toolong_out += toolong;
    return over;
#endif
}

static __attribute__((noinline)) void drain_round(args_t* __UNIFORM__ a, int me, int tid, int blk,
                                                  int N, int NB, int* buf, int sub) {
    const int* rp_tag = (const int*)a->rp_tag_addr;
    const int* rp_val = (const int*)a->rp_val_addr;
    const int* ptbl   = (const int*)a->ptbl_addr;
    int*  tgt    = (int*)a->tgt_addr;
    (void)N;
#if REPLY_TILE && APPLY
    // one engine command per reply tile: STORE value at tgt[question*N + me].
    // A tile is RTILE_SZ questions, one tgt line each for this block's 16
    // lanes, so its lines fit the engine's combining cache and are written once.
    (void)ptbl; (void)NB; (void)buf; (void)rp_tag; (void)rp_val; (void)tgt;
#if DRAIN_SPLIT
    // sub-lanes take disjoint reply tiles, i.e. disjoint tgt entries
    for (uint32_t tt = 0; tt < (uint32_t)(REPLY_TILE) / (uint32_t)(SPLIT); tt++) {   // constant trip count (T42)
        const uint32_t t = (uint32_t)sub + tt * (uint32_t)(SPLIT);
        const uint32_t nq = 1u;
#else
    // sub-lane 0 drains; the others issue EMPTY commands rather than branching,
    // so no lane diverges upstream of ap_wait's vote (the SIMT collective rule)
    for (uint32_t t = 0; t < (uint32_t)REPLY_TILE; t++) {
        const uint32_t nq = (sub == 0) ? 1u : 0u;
#endif
        uint32_t rid = (uint32_t)me * (uint32_t)REPLY_TILE + t;
        ap_issue(ap_slot((uint32_t)tid % (uint32_t)(NUM_THREADS)), a->rp_tag_addr, (uint64_t)RQ_STRIDE * 8ull,
                 a->rp_val_addr, rid, nq, a->tgt_addr + 4ull * (uint64_t)me, 4ull * (uint64_t)N,
                 AP_SPEC(32, 32, 0, 32, AP_STORE, 8, 0));
        ap_wait();
    }
#elif REPLY_TILE && DRAIN_DIRECT
    (void)ptbl; (void)NB; (void)buf; (void)me; (void)N;
    drain_direct((const uint64_t*)a->rp_tag_addr, tgt, (uint32_t)tid % GLANES, (uint32_t)sub, (uint32_t)blk);
#elif REPLY_TILE
    (void)ptbl; (void)NB;
    { const uint64_t* rq = (const uint64_t*)a->rp_tag_addr;
      int* row = buf + tid * RTILE_SZ;
      // SPLIT: sub-lanes take disjoint reply tiles, i.e. disjoint tgt entries
      for (uint32_t tt = 0; tt < (uint32_t)(REPLY_TILE) / (uint32_t)(SPLIT); tt++) {   // constant trip count (T42)
          const uint32_t t = (uint32_t)sub + tt * (uint32_t)(SPLIT);
          uint32_t rid  = (uint32_t)me * (uint32_t)REPLY_TILE + t;
          uint32_t base = t * (uint32_t)RTILE_SZ;
#if WC_DRAIN
          // warp-cooperative: the warp fills the rows of its 16 threads one actor at a
          // time, 16 consecutive replies per load (2 lines), not 16 streams per load
          break;   // the whole round is drained by drain_wc below
          (void)rid; (void)base;
#elif DRAIN_PROBE < 2                       // DRAIN_PROBE: PROBE only (fails the check). 1 = no tgt writes, 2 = no drain
          drain_tile_in(rq + (size_t)rid * RQ_STRIDE, row, base);
#endif
          POOL_SYNC();
#if DRAIN_PROBE < 1
          drain_tile_out(row, tgt, base, (uint32_t)me, (uint32_t)N);
#endif
          POOL_SYNC();
      }
#if WC_DRAIN
      drain_wc(rq, buf + (uint32_t)sub * (uint32_t)GLANES * (uint32_t)RTILE_SZ, row, tgt,
               (uint32_t)tid % (uint32_t)GLANES, (uint32_t)me, (uint32_t)sub, (uint32_t)blk, (uint32_t)N);
#endif
    }
#elif REPLY_AQ
    // my own reply queues, one contiguous stream each, one load per reply
    (void)ptbl; (void)buf; (void)tid;
    { const uint64_t* rq = (const uint64_t*)a->rp_tag_addr;
      const volatile uint32_t* rtails = (const volatile uint32_t*)a->rp_val_addr;
      for (int ob = 0; ob < NB; ob++) {
          uint32_t rid  = (uint32_t)me * (uint32_t)NB + (uint32_t)ob;
          uint32_t tail = rtails[rid];
          if (tail > (uint32_t)RQ_CAP) tail = RQ_CAP;
          drain_aq(rq + (size_t)rid * RQ_CAP, tail, tgt, (uint32_t)me, (uint32_t)N);
      }
    }
#else
    {
        int* poffs = buf;                            // NCHUNK*TSTRIDE
        for (int ob = 0; ob < NB; ob++) {
            { size_t tb = ((size_t)(ob * NB + blk) * NCHUNK) * TSTRIDE;
              for (int i = tid; i < (int)NCHUNK * TSTRIDE; i += GLANES) poffs[i] = ptbl[tb + i]; }
            POOL_SYNC();
            for (uint32_t ch = 0; ch < (uint32_t)NCHUNK; ch++) {
                const int* oo = poffs + ch * TSTRIDE;
                size_t rb = ((size_t)(ob * NB + blk) * NCHUNK + ch) * CAP2;
                for (int i = oo[tid]; i < oo[tid + 1]; i++)
                    tgt[TGT_IDX(me, (uint32_t)rp_tag[rb + i], N)] = rp_val[rb + i];
            }
            POOL_SYNC();
        }
    }
#endif
}

#if TILE_AQ
// Round 1b: my own inbox into my own tile queues.
static __attribute__((noinline)) void reorder_round(args_t* __UNIFORM__ a, int me, int NB, uint32_t* toolong_out) {
    const req_t* qs = (const req_t*)a->qs_addr;
    const volatile uint32_t* tails = (const volatile uint32_t*)a->tails_addr;
    uint32_t toolong = 0;
    for (int ab = 0; ab < NB; ab++) {
        uint32_t qid  = (uint32_t)me * (uint32_t)NB + (uint32_t)ab;
        uint32_t tail = tails[qid];
        if (tail > (uint32_t)AQ_QCAP) { toolong++; tail = AQ_QCAP; }
        tile_reorder(qs + (size_t)qid * AQ_QSTRIDE_W, tail, (uint32_t)ab,
                     (uint32_t)me, (uint32_t)NB * GLANES);
    }
    *toolong_out += toolong;
}
#endif

void kernel_body(args_t* __UNIFORM__ a) {
    volatile uint32_t* bar = (volatile uint32_t*)a->bar_addr;
    int   N   = a->N;
    int   blk = (int)blockIdx.x;
    int   tid = (int)threadIdx.x;
    // A block is SPLIT warps wide, so the block count, and therefore every barrier
    // count and every blk-keyed index, is exactly what it was at one warp. The
    // lane's ACTOR is g / SPLIT and its slice of that actor's work is g % SPLIT.
    int   sub = ((SPLIT) == 1) ? 0 : tid / GLANES;   // WARP-major: warp w of the block is sub-lane w
    int   me  = blk * GLANES + tid % GLANES;   // of the block's 16 actors, so every warp is a
                                     // faithful copy of the one-warp kernel on its own
                                     // slice: 16 consecutive owners per warp (R8e's
                                     // layout) and 16 consecutive actors per tgt line.
    int   NB  = N / GLANES;
    uint32_t over = 0, toolong = 0;

    int* buf = (int*)__local_mem((size_t)BUF_WORDS * sizeof(int));
    uint64_t t0 = csr_read(VX_CSR_MCYCLE);

    // ================= round 1: ask =================
    ask_round(a, me, tid, blk, N, NB, buf, sub);
    vx_fence();
    uint64_t t1 = csr_read(VX_CSR_MCYCLE);
    gbarrier(bar, 0, tid, NB);
#if TILE_AQ && DIRECT_TILE
#if AQ_ON
    // ISA2: the L2 keeps one queue set's state, so before the switch to the reply queues each warp reads the counts
    // of the request tile queues it will serve (tiles sub, sub + SPLIT, ... of its block's 16 owners) into its own
    // ttails entries; a barrier follows before the switch
    { volatile uint32_t* tt = (volatile uint32_t*)a->ttails_addr;
      const uint32_t nt = (uint32_t)sub < (uint32_t)(TILE_AQ) ? ((uint32_t)(TILE_AQ) - (uint32_t)sub + (SPLIT) - 1) / (uint32_t)(SPLIT) : 0u;
#pragma GCC unroll 1
      for (uint32_t i = (uint32_t)tid % GLANES; i < nt * GLANES; i += GLANES) {
          const uint32_t t = (uint32_t)sub + (i / GLANES) * (uint32_t)(SPLIT);
          const uint32_t q = tqid_of((uint32_t)blk * GLANES + i % GLANES, t, (uint32_t)N);
          tt[q] = vx_aq_count(q);
      } }
    gbarrier(bar, 1, tid, NB);   // slot 1: unused on this path (DIRECT_TILE)
#endif
    // The requests already sit in the tile queues. Straight to the reply geometry.
    if (blk == 0) aq_flush();          // tail charge for the round just ended
#if OVF
    if (blk == 0) ((volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE)[7] = (2ull << 60);   // replies never overflow
#endif
    if (blk == 0)
        aq_geometry(a->rp_tag_addr, (uint64_t)RQ_STRIDE * 8ull, a->rp_val_addr, 8, RQ_STRIDE);
    vx_fence();
    gbarrier(bar, 6, tid, NB);
#elif TILE_AQ
    // Reconfigure for the tile queues, reorder my own inbox, then reconfigure
    // for the reply queues. Each switch resets the engine's internal lengths;
    // the lengths already written to memory are untouched and still readable.
#if TILE_NOGEO != 1
    if (blk == 0) aq_flush();          // tail charge for the round just ended
    if (blk == 0)
        aq_geometry(a->ts_addr, (uint64_t)TQ_STRIDE_W * 8ull, a->ttails_addr, 8, TQ_CAP);
#endif
    vx_fence();
    gbarrier(bar, 1, tid, NB);
    reorder_round(a, me, NB, &toolong);
    vx_fence();
    gbarrier(bar, 5, tid, NB);
#if TILE_NOGEO != 1 && TILE_NOGEO != 2
    if (blk == 0) aq_flush();          // tail charge for the round just ended
    if (blk == 0)
        aq_geometry(a->rp_tag_addr, (uint64_t)RQ_STRIDE * 8ull, a->rp_val_addr, 8, RQ_STRIDE);
#endif
    vx_fence();
#if TILE_NOGEO != 3
    gbarrier(bar, 6, tid, NB);
#endif
#elif REPLY_AQ
    // Every request has landed, so the queue engine can be reconfigured for the
    // reply round: wider words, a different storage array and a different length
    // array. The request lengths already written to memory are not touched.
    if (blk == 0) aq_flush();          // tail charge for the round just ended
    if (blk == 0)
        aq_geometry(a->rp_tag_addr, (uint64_t)RQ_STRIDE * 8ull, a->rp_val_addr, 8, RQ_STRIDE);
    vx_fence();
    gbarrier(bar, 1, tid, NB);
#endif
    uint64_t t2 = csr_read(VX_CSR_MCYCLE);

    // ================= round 2: serve, and reply =================
    over = serve_round(a, me, tid, blk, N, NB, buf, &toolong, sub);
    POOL_SYNC();
    vx_fence();
    uint64_t t3 = csr_read(VX_CSR_MCYCLE);
    gbarrier(bar, 4, tid, NB);
    if (blk == 0) aq_flush();          // reply tiles are exact: charges only a mid line tail
    vx_fence();
#if LANE_TS
    uint64_t t3b = csr_read(VX_CSR_MCYCLE);
#endif

    // ================= round 3: drain my replies =================
    drain_round(a, me, tid, blk, N, NB, buf, sub);
    vx_fence();
    uint64_t t4 = csr_read(VX_CSR_MCYCLE);

    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];
    if (sub == 0) {
    ph->request = t1 - t0;
    ph->barrier = t2 - t1;
    ph->serve   = t3 - t2;
    ph->drain   = t4 - t3;
    ph->total   = t4 - t0;
    ph->t_start = t0; ph->t_end = t4;
    }
#if LANE_TS
    { uint64_t* lt = (uint64_t*)(a->phase_addr + (uint64_t)N * sizeof(phase_cycles_t))
                     + ((uint64_t)me * (SPLIT) + (uint64_t)sub) * 6;
      lt[0] = t0; lt[1] = t1; lt[2] = t2; lt[3] = t3; lt[4] = t3b; lt[5] = t4; }
#endif
    if (over)    __atomic_fetch_add((uint32_t*)&bar[2], over, __ATOMIC_RELAXED);
    if (toolong) __atomic_fetch_add((uint32_t*)&bar[3], toolong, __ATOMIC_RELAXED);
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N / GLANES;
    uint32_t block_dim = GLANES * (SPLIT);
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
