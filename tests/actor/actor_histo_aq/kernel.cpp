// See common.h. One round, transported by appending.

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"
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
#ifndef ACTOR
#define ACTOR 1           // this kernel is always the actor version (the non-actor histogram is baseline_histo)
#endif
#include "../isa2.h"
// ISA2 converts the paths the paper's histogram runs use (tiles queued directly by the sender)
static_assert(!AQ_ON || ((DIRECT_BT) && (BUCKET_TILE) && !(APPLY) && !(CNT_T) && !(HYB_OUT) && !(OVF) && !(WC_RECV) && !(VPATH)),
              "ISA2: this histogram path still reads the old queue layout");
#if AQ_ON
#define BT_COUNT(BT, I) vx_aq_count((uint32_t)(I))
#else
#define BT_COUNT(BT, I) (BT)[I]
#endif

static inline uint32_t mix(uint32_t a, uint32_t b) {
    uint32_t x = a * 0x9E3779B9u + b * 0x85EBCA77u;
    x ^= (x >> 13);
    x *= 0xC2B2AE3Du;
    x ^= (x >> 16);
    return x;
}

// One arrival per WARP: a block is SPLIT warps wide, and counting only warp 0
// let a sibling warp through before the other blocks' sibling warps were done
// (the index gather lost replies this way at 4 warps). Same as before at SPLIT=1.
static inline void gbarrier(volatile uint32_t* bar, int slot, int tid, int NB) {
#if ISA2
    vx_fence();                                           // ISA2: this warp's writes are acknowledged before it arrives
#endif
    if (tid % GLANES == 0) __atomic_fetch_add((uint32_t*)&bar[slot], 1, __ATOMIC_RELAXED);
    vx_fence();
#if ISA2
    // the arrival count is kept by L2 atomics: poll it with an atomic read, then drop the lines cached before
    while (!vx_vote_all(__atomic_fetch_add((uint32_t*)&bar[slot], 0u, __ATOMIC_RELAXED) >= (uint32_t)NB * (uint32_t)(SPLIT))) { }
    ISA2_L1INV();
#else
    while (!vx_vote_all(bar[slot] >= (uint32_t)NB * (uint32_t)(SPLIT))) { }
#endif
}

// The same barrier, reusable: one slot counts arrivals for every chunk, so the
// threshold rises by NB each time instead of needing a slot per chunk. The
// single lane increment stays upstream of the vote exactly as above, which the
// SIMT collective rule requires.
static inline void gbarrier_ep(volatile uint32_t* bar, int slot, int tid, int NB, uint32_t ep) {
#if ISA2
    vx_fence();                                           // ISA2: this warp's writes are acknowledged before it arrives
#endif
    if (tid % GLANES == 0) __atomic_fetch_add((uint32_t*)&bar[slot], 1, __ATOMIC_RELAXED);
    vx_fence();
#if ISA2
    while (!vx_vote_all(__atomic_fetch_add((uint32_t*)&bar[slot], 0u, __ATOMIC_RELAXED) >= (uint32_t)NB * (uint32_t)(SPLIT) * ep)) { }
    ISA2_L1INV();
#else
    while (!vx_vote_all(bar[slot] >= (uint32_t)NB * (uint32_t)(SPLIT) * ep)) { }
#endif
}

#if SWQ
// software queue: the storage, tail array, stride and capacity of the current geometry;
// every lane stores the same values at kernel entry (see swq_set)
static int* g_swq_st; static uint32_t* g_swq_tl; static uint32_t g_swq_stride, g_swq_cap;
static inline void swq_set(uint64_t st, uint64_t tl, uint32_t stride, uint32_t cap) {
    g_swq_st = (int*)st; g_swq_tl = (uint32_t*)tl; g_swq_stride = stride; g_swq_cap = cap;
}
static inline void aq_append(uint32_t q, int word) {
    const uint32_t s = __atomic_fetch_add(&g_swq_tl[q], 1u, __ATOMIC_RELAXED);
    if (s < g_swq_cap) g_swq_st[(size_t)q * g_swq_stride + s] = word;
}
#else
static inline void aq_append(uint32_t q, int word) {
#if ISA2
    vx_aq_append32(q, (uint32_t)word);
#else
    *(volatile int*)(uintptr_t)((uint64_t)AQ_BASE + 64ull * ((uint64_t)q + 1ull)) = word;
#endif
}
#endif
// Warp uniform on purpose. Writing these from one lane is a divergent region,
// and a collective downstream of one reconverges wrongly: see aq_mechanisms.md.
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

#if APPLY
// The apply engine: a lane hands its OWN queues and its OWN counts to the
// engine and waits. Each lane writes its own command line (a 16-lane store to
// 16 lines, no divergence); the last word triggers. The done word is one
// address for the whole core, so the poll is one coalesced load per iteration.
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

#if BUCKET_TILE
// Tile queue index of (tile, owner). Tile major under QMAJOR for the same L1
// set spreading as the inbox queues.
static inline uint32_t bqid_of(uint32_t tile, uint32_t owner, uint32_t N) {
#if QMAJOR
    return tile * (N + (uint32_t)(QROWPAD)) + owner;
#else
    (void)N; return owner * (uint32_t)BUCKET_TILE + tile;
#endif
}
#endif
// Queue index of (sender block, owner). See QMAJOR in common.h.
static inline uint32_t qid_of(uint32_t sblk, uint32_t owner, uint32_t NB, uint32_t N) {
#if QMAJOR
    (void)NB; return sblk * (N + (uint32_t)(QROWPAD)) + owner;   // QROWPAD: see common.h
#else
    (void)N;  return owner * NB + sblk;
#endif
}

// SEND. One instruction per update and no destination address computed.
static __attribute__((noinline)) void send_round(uint32_t me, uint32_t NB, uint32_t blk,
                                                 uint32_t N, uint32_t G,
                                                 uint32_t lo, uint32_t hi) {
#if NCONST
    (void)N; (void)G; (void)NB;
    // ILP. Forming one message is an eighteen deep dependent chain (hash,
    // multiply high modulus, two field extracts), and this machine is in order
    // with one warp per core, so every step waits for the previous one to write
    // back and the loop runs at about eleven cycles an instruction. The messages
    // are INDEPENDENT of one another, so UNROLL of them are formed at once and
    // their chains interleave in the issue stream. Nothing about the actor model
    // changes: an actor is still preparing its own outgoing mail, in the same
    // order, one append per message.
    // The trip count is a CONSTANT and lo only offsets the index. lo depends on
    // the sub-lane, i.e. on the thread id, so a loop bounded by [lo, hi) is
    // divergent to the compiler even though every lane of a warp shares it, and
    // it wrapped the loop in split/join and predicated EVERY iteration: two
    // warp control instructions per message, 12% of the send round (T42).
    (void)hi;
    for (uint32_t k = 0; k < (uint32_t)((M) / (CHUNKS) / (SPLIT)); k += (uint32_t)(UNROLL)) {
        const uint32_t i = lo + k;
        uint32_t g[UNROLL];
        for (uint32_t u = 0; u < (uint32_t)(UNROLL); u++)
            g[u] = skew_key(mix(me, i + u), (uint32_t)((L) * (NASSUME)));
#if SWQ && DIRECT_BT
        // software queue: claim all UNROLL slots first (UNROLL atomics in flight), then store
        { uint32_t qq[UNROLL], sl[UNROLL];
          for (uint32_t u = 0; u < (uint32_t)(UNROLL); u++) {
              const uint32_t b = g[u] / (uint32_t)(NASSUME);
              qq[u] = bqid_of(b / (uint32_t)BTILE_SZ, g[u] % (uint32_t)(NASSUME), (uint32_t)(NASSUME));
              sl[u] = __atomic_fetch_add(&g_swq_tl[qq[u]], 1u, __ATOMIC_RELAXED);
          }
          for (uint32_t u = 0; u < (uint32_t)(UNROLL); u++)
              if (sl[u] < g_swq_cap) g_swq_st[(size_t)qq[u] * g_swq_stride + sl[u]] = (int)(g[u] / (uint32_t)(NASSUME));
        }
        if (0)
#endif
        for (uint32_t u = 0; u < (uint32_t)(UNROLL); u++) {
#if DIRECT_BT
            uint32_t b = g[u] / (uint32_t)(NASSUME);
            aq_append(bqid_of(b / (uint32_t)BTILE_SZ, g[u] % (uint32_t)(NASSUME), (uint32_t)(NASSUME)), (int)b);
#else
            aq_append(qid_of(blk, g[u] % (uint32_t)(NASSUME), (uint32_t)(NBA), (uint32_t)(NASSUME)),
                      (int)(g[u] / (uint32_t)(NASSUME)));
#endif
        }
    }
#else
    for (uint32_t i = lo; i < hi; i++) {
        uint32_t g = skew_key(mix(me, i), G);
#if DIRECT_BT
        uint32_t b = g / N;
        aq_append(bqid_of(b / (uint32_t)BTILE_SZ, g % N, N), (int)b);
#else
        aq_append(qid_of(blk, g % N, NB, N), (int)(g / N));
#endif
    }
#endif
}
#if WC_TILE
static_assert((DIRECT_BT) && (QMAJOR) && !(CNT_T) && !(APPLY), "WC_TILE: DIRECT_BT, QMAJOR, natural counts, no engine");
// warp-cooperative tile apply, the whole round in one leaf. The warp takes tiles
// t = sub, sub+SPLIT, ... and for each the block's 16 actors one at a time: 16 lanes read 16
// consecutive words of the actor's tile queue (WC_U lines loaded before use) and add into the
// warp's scratchpad tile with scratchpad atomics; then the warp writes the actor's tile of
// counts as whole lines (natural layout) and clears the tile.
#if WC_NOAMO   // PROBE ONLY: plain increments race within a warp, counts come out wrong
#define WC_ADD(x) ((x)++)
#else
#define WC_ADD(x) __atomic_fetch_add(&(x), 1, __ATOMIC_RELAXED)
#endif
#if OVF
// one segment of a queue, warp cooperative: lane ln takes entries ln, ln+16, ...; WC_U lines loaded first
#define WC_SEG(QP, N_, OP) { const uint32_t nf_ = (N_) / GLANES; uint32_t k_ = 0; \
    for (; k_ + (WC_U) <= nf_; k_ += (WC_U)) { int w_[WC_U]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(WC_U); u_++) w_[u_] = (QP)[(k_ + u_) * GLANES]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(WC_U); u_++) OP(w_[u_]); } \
    for (; k_ < nf_; k_++) OP((QP)[k_ * GLANES]); \
    if (nf_ * GLANES + ln < (N_)) OP((QP)[nf_ * GLANES]); }
// the whole queue: its region, then its overflow chunks (warp-uniform trip counts)
#define WC_ALL(OP) { const uint32_t n0_ = tl < (uint32_t)(BQ_CAP) ? tl : (uint32_t)(BQ_CAP); WC_SEG(q, n0_, OP) \
    if (tl > (uint32_t)(BQ_CAP)) { const uint64_t* cr_ = ctab + (size_t)bid * (OVF_MAXC); uint32_t r_ = tl - (uint32_t)(BQ_CAP); \
        _Pragma("GCC unroll 1") for (uint32_t c_ = 0; r_ != 0; c_++) { \
            const uint32_t n_ = r_ < (uint32_t)(OVF_CH) ? r_ : (uint32_t)(OVF_CH); \
            const int* qc_ = (const int*)(uintptr_t)cr_[c_] + ln; WC_SEG(qc_, n_, OP) r_ -= n_; } } }
#endif
static __attribute__((noinline)) void tile_wc(const int* __UNIFORM__ bs, const uint32_t* __UNIFORM__ bt,
                                              int* __UNIFORM__ wacc, int* __UNIFORM__ counts, uint32_t ln,
                                              uint32_t __UNIFORM__ sub, uint32_t __UNIFORM__ blk
#if OVF
                                              , const uint64_t* __UNIFORM__ ctab
#endif
                                              ) {
    // ceil(BUCKET_TILE / SPLIT) tiles per warp; the last may not exist (a warp-uniform skip),
    // so the tile count need not divide by SPLIT and BTILE_SZ can be a whole number of lines
#pragma GCC unroll 1
    for (uint32_t c = 0; c < ((uint32_t)(BUCKET_TILE) + (SPLIT) - 1) / (uint32_t)(SPLIT) * GLANES; c++) {
        const uint32_t t = sub + (c / GLANES) * (uint32_t)(SPLIT);
        if (t >= (uint32_t)(BUCKET_TILE)) break;
        const uint32_t ow = blk * (uint32_t)GLANES + c % GLANES;
        const uint32_t bid = t * ((uint32_t)NASSUME + (uint32_t)(QROWPAD)) + ow;
        uint32_t tl = BT_COUNT(bt, bid);
#if !OVF
        tl = tl > (uint32_t)BQ_CAP ? (uint32_t)BQ_CAP : tl;
#endif
        const int* q = QPTR_(bs, BQ_STRIDE_W, bid) + QIX_(bs, ln);   // ISA2 layout: lane ln's entries QSTEP_(bs) apart
        const uint32_t nf = tl / GLANES;
#if HYB_APPLY && OVF
        if (tl * (uint32_t)(HYB_K) < (uint32_t)BTILE_SZ) {
            int* cw = counts + (size_t)ow * (L);
#define HYB_OP(x) __atomic_fetch_add(&cw[x], 1, __ATOMIC_RELAXED)
            WC_ALL(HYB_OP)
#undef HYB_OP
            continue;
        }
#elif HYB_APPLY
        if (tl * (uint32_t)(HYB_K) < (uint32_t)BTILE_SZ) {
            // sparse queue: too few messages per owned line to pay for a tile pass, so the
            // owner applies each message straight to its own counts (warp-uniform branch)
            int* cw = counts + (size_t)ow * (L);
#pragma GCC unroll 1
            for (uint32_t k = 0; k < nf; k++) __atomic_fetch_add(&cw[q[k * QSTEP_(bs)]], 1, __ATOMIC_RELAXED);
            if (nf * GLANES + ln < tl) __atomic_fetch_add(&cw[q[nf * QSTEP_(bs)]], 1, __ATOMIC_RELAXED);
            continue;
        }
#endif
        // FINE_T > 1: one pass per scratchpad-sized slice of the coarse tile; the
        // coarse queue is small and re-read from cache, the counts slice is resident
#pragma GCC unroll 1
        for (uint32_t f = 0; f < (uint32_t)(FINE_T); f++) {
        const uint32_t base = t * (uint32_t)BTILE_SZ + f * (uint32_t)FTILE_SZ;
#if FINE_T > 1
#define WC_PUT(x) { const uint32_t r_ = (uint32_t)(x) - base; if (r_ < (uint32_t)FTILE_SZ) WC_ADD(wacc[r_]); }
#else
        int* wa = wacc - (int)base;
#define WC_PUT(x) WC_ADD(wa[x])
#endif
#if OVF
        (void)nf;
        WC_ALL(WC_PUT)
#else
        uint32_t k = 0;
        for (; k + (WC_U) <= nf; k += (WC_U)) {
            int w[WC_U];
#pragma GCC unroll 8
            for (uint32_t u = 0; u < (uint32_t)(WC_U); u++) w[u] = q[(k + u) * QSTEP_(bs)];
#pragma GCC unroll 8
            for (uint32_t u = 0; u < (uint32_t)(WC_U); u++) WC_PUT(w[u]);
        }
        for (; k < nf; k++) WC_PUT(q[k * QSTEP_(bs)]);
        if (nf * GLANES + ln < tl) WC_PUT(q[nf * QSTEP_(bs)]);
#endif
#undef WC_PUT
        vx_fence();
        int* out = counts + (size_t)ow * (L) + (size_t)base;
#pragma GCC unroll 1
        for (uint32_t m = 0; m < ((uint32_t)FTILE_SZ + GLANES - 1) / GLANES; m++) {
            const uint32_t jj = ln + m * (uint32_t)GLANES;
            if (jj < (uint32_t)FTILE_SZ) {
#if !TILE_NOWB
                out[jj] = wacc[jj];
#endif
                wacc[jj] = 0; }
        }
        vx_fence();
        }
    }
}
#endif
#if HYB_OUT
static_assert((DIRECT_BT) && (TILE_ACC) && !(CNT_T) && !(APPLY) && !(WC_TILE), "HYB_OUT: DIRECT_BT + TILE_ACC, natural counts");
// the whole receive round in one leaf: per tile, every lane applies ITS actor's tile queue into
// its own scratchpad row (per-lane mapping, 16 streams in flight), then the warp writes the 16
// rows out one actor at a time, 16 consecutive counts per store (sequential lines), and clears them
static __attribute__((noinline)) void tile_hyb(const int* __UNIFORM__ bs, const uint32_t* __UNIFORM__ bt,
                                               int* __UNIFORM__ rows, int* __UNIFORM__ counts, uint32_t ln,
                                               uint32_t __UNIFORM__ sub, uint32_t __UNIFORM__ blk) {
    int* acc = rows + ln * (uint32_t)(BACC_STRIDE);
    for (uint32_t tt = 0; tt < (uint32_t)(BUCKET_TILE) / (uint32_t)(SPLIT); tt++) {
        const uint32_t t = sub + tt * (uint32_t)(SPLIT);
        const uint32_t base = t * (uint32_t)BTILE_SZ;
        const uint32_t bid = t * ((uint32_t)NASSUME + (uint32_t)(QROWPAD)) + blk * (uint32_t)GLANES + ln;
        uint32_t tl = bt[bid];
        tl = tl > (uint32_t)BQ_CAP ? (uint32_t)BQ_CAP : tl;
        const int* q = bs + (size_t)bid * BQ_STRIDE_W;
        int* a = acc - (int)base;
        for (uint32_t i = 0; i < tl; i++) a[q[i]]++;
        vx_fence();
        for (uint32_t j = 0; j < (uint32_t)GLANES; j++) {
            int* row = rows + j * (uint32_t)(BACC_STRIDE);
            int* out = counts + (size_t)(blk * (uint32_t)GLANES + j) * (L) + base;
            for (uint32_t m = 0; m < ((uint32_t)BTILE_SZ + GLANES - 1) / GLANES; m++) {
                const uint32_t jj = ln + m * (uint32_t)GLANES;
                if (jj < (uint32_t)BTILE_SZ) {
#if !TILE_NOWB
                    out[jj] = row[jj];
#endif
                    row[jj] = 0; }
            }
        }
        vx_fence();
    }
}
#endif
#if RECV_IL == 2
// two of my queues at once into two rows: two independent load -> RMW chains per lane
static __attribute__((noinline)) void recv_run2(const int* q0, uint32_t t0, const int* q1, uint32_t t1,
                                                int* a0, int* a1) {
    const uint32_t n = t0 < t1 ? t0 : t1;
    uint32_t i = 0;
    for (; i < n; i++) { const int x = q0[i], y = q1[i]; a0[x]++; a1[y]++; }
    for (uint32_t j = i; j < t0; j++) a0[q0[j]]++;
    for (uint32_t j = i; j < t1; j++) a1[q1[j]]++;
}
#endif
#if WC_RECV
// warp-cooperative receive (a whole leaf: no spills). The warp takes the block's 16
// actors one at a time; lane ln counts entries ln, ln+16, ... of each of the actor's
// queues for this sub-lane into its private row, then the warp sums the 16 private
// rows into that actor's row of this sub-lane.
static __attribute__((noinline)) void recv_wc(const int* __UNIFORM__ qs, const uint32_t* __UNIFORM__ tails,
                                              int* __UNIFORM__ rows, int* __UNIFORM__ pb, uint32_t ln,
                                              uint32_t __UNIFORM__ sub, uint32_t __UNIFORM__ blk) {
    constexpr uint32_t NQ = (uint32_t)(NASSUME / GLANES) / (uint32_t)(SPLIT);   // my queues per actor
    int* priv = pb + ln * (uint32_t)(L);
    // one flat loop over (actor, queue); bounds are warp uniform, full 16-word steps run
    // unguarded with WC_U queue lines loaded before any is used, one guarded remainder
    for (uint32_t c = 0; c < (uint32_t)GLANES * NQ; c++) {
        const uint32_t j = c / NQ, k = c % NQ;
        if (k == 0) for (uint32_t b = 0; b < (uint32_t)L; b++) ((volatile int*)priv)[b] = 0;
        const uint32_t qid = qid_of(sub + k * (uint32_t)(SPLIT), blk * (uint32_t)GLANES + j,
                                    (uint32_t)(NASSUME / GLANES), (uint32_t)NASSUME);
        uint32_t tl = tails[qid];
        if (tl > (uint32_t)AQ_QCAP) tl = AQ_QCAP;
        const int* q = qs + (size_t)qid * QSTRIDE_W + ln;
        const uint32_t nf = tl / GLANES;
        uint32_t m = 0;
        for (; m + (WC_U) <= nf; m += (WC_U)) {
            int w[WC_U];
#pragma GCC unroll 8
            for (uint32_t u = 0; u < (uint32_t)(WC_U); u++) w[u] = q[(m + u) * GLANES];
#pragma GCC unroll 8
            for (uint32_t u = 0; u < (uint32_t)(WC_U); u++) priv[w[u]]++;
        }
        for (; m < nf; m++) priv[q[m * GLANES]]++;
        if (nf * GLANES + ln < tl) priv[q[nf * GLANES]]++;
        if (k == NQ - 1) {
            vx_fence();
            int* dst = rows + j * (uint32_t)(L);
            for (uint32_t b0 = 0; b0 < ((uint32_t)L + GLANES - 1) / GLANES; b0++) {
                const uint32_t b = ln + b0 * (uint32_t)GLANES;
                if (b < (uint32_t)L) {
                    int v = 0;
                    for (uint32_t l = 0; l < (uint32_t)GLANES; l++) v += pb[l * (uint32_t)(L) + b];
                    dst[b] = v;
                }
            }
            vx_fence();
        }
    }
}
#endif
// RECEIVE. My own queues, contiguous, accumulated in my own scratchpad row.
static __attribute__((noinline)) void recv_run(const int* q, uint32_t tail, int* acc) {
#if RECV_U > 1
    // fewer loop instructions per message: RECV_U loads, then RECV_U updates in order
    uint32_t i = 0;
    for (; i + (RECV_U) <= tail; i += (RECV_U)) {
        int w[RECV_U];
#pragma GCC unroll 8
        for (uint32_t u = 0; u < (uint32_t)(RECV_U); u++) w[u] = q[i + u];
#pragma GCC unroll 8
        for (uint32_t u = 0; u < (uint32_t)(RECV_U); u++) acc[w[u]]++;
    }
    for (; i < tail; i++) acc[q[i]]++;
#else
    for (uint32_t i = 0; i < tail; i++) acc[q[i]]++;
#endif
}
#if MEMACC
// Same, but straight into my own counts in memory, which is the only option
// once L * GLANES exceeds the scratchpad.
static __attribute__((noinline)) void recv_mem(const int* q, uint32_t tail, int* mine) {
    for (uint32_t i = 0; i < tail; i++) mine[q[QIX_(q, i)]]++;
}
#endif
#if BUCKET_TILE
// Mechanism 2b for the histogram: my own inbox into my own per bucket-tile
// queues, so each apply pass touches only L / BUCKET_TILE of my counts.
static __attribute__((noinline)) void btile_reorder(const int* q, uint32_t tail, uint32_t me, uint32_t N) {
    for (uint32_t i = 0; i < tail; i++)
        aq_append(bqid_of((uint32_t)q[i] / (uint32_t)BTILE_SZ, me, N), q[i]);
}
#endif
#if TILE_ACC
// The point of the tile. Once the inbox is grouped by tile, the tile's counts
// are small enough to live in scratchpad, so the whole pass costs BTILE_SZ
// stores instead of one read-modify-write per message. Writes go from O(M) to
// O(L). One call per tile, never one per message: a call inside a hot loop pays
// the aliased stack traffic (see the v2 regression in the log).
static __attribute__((noinline)) void tile_apply(const int* q, uint32_t tail, int* acc,
                                                 uint32_t base, int* out) {
    for (uint32_t j = 0; j < (uint32_t)BTILE_SZ; j++) acc[j] = 0;
    for (uint32_t i = 0; i < tail; i++) acc[(uint32_t)q[QIX_(q, i)] - base]++;
    for (uint32_t j = 0; j < (uint32_t)BTILE_SZ; j++) out[j] = acc[j];
}
#if CNT_T
// Same, but the writeback is bucket-major so all 16 lanes land on one line.
static __attribute__((noinline)) void tile_apply_t(const int* q, uint32_t tail, int* acc,
                                                   uint32_t base, int* counts,
                                                   uint32_t me, uint32_t N) {
    for (uint32_t j = 0; j < (uint32_t)BTILE_SZ; j++) acc[j] = 0;
    for (uint32_t i = 0; i < tail; i++) acc[(uint32_t)q[i] - base]++;
    for (uint32_t j = 0; j < (uint32_t)BTILE_SZ; j++)
        counts[(size_t)(base + j) * N + me] = acc[j];
}
#if TILE_LEAF
// Same, as a LEAF: no zeroing loop (clang turns it into a memset call, which
// makes the function non-leaf and costs 9 callee-saved register spills per call,
// each a 16-lane scattered store to the lane stacks). The caller zeroes acc once;
// the write-back clears each word as it goes, ready for the next tile.
static __attribute__((noinline)) void tile_apply_tl(const int* q, uint32_t tail, int* acc,
                                                    uint32_t base, int* counts,
                                                    uint32_t me, uint32_t N) {
    for (uint32_t i = 0; i < tail; i++) acc[(uint32_t)q[i] - base]++;
    for (uint32_t j = 0; j < (uint32_t)BTILE_SZ; j++) {
#if !TILE_NOWB                            // TILE_NOWB=1: PROBE only, drops the write-back (fails the check)
        counts[(size_t)(base + j) * N + me] = acc[j];
#endif
        acc[j] = 0;
    }
}
#endif
#endif
#endif

#if VPATH
static_assert((SPLIT) == 1 && (QMAJOR) && (NCONST) && !(BUCKET_TILE) && !(MEMACC) && !(APPLY) && !(WC_RECV) && (CHUNKS) == 1,
              "VPATH: plain path, one lane per actor share");
// one lane runs VACT actors in turn; queues are keyed (sender key, owner), a sender key
// being QSG launched blocks, so the queue count is NBA * N whatever VACT is
static __attribute__((noinline)) void send_v(uint32_t me, uint32_t skey) {
    for (uint32_t i = 0; i < (uint32_t)(M); i++) {
        const uint32_t g = skew_key(mix(me, i), (uint32_t)((L) * (NASSUME)));
        aq_append(skey * ((uint32_t)NASSUME + (uint32_t)(QROWPAD)) + g % (uint32_t)(NASSUME), (int)(g / (uint32_t)(NASSUME)));
    }
}
void kernel_body(args_t* __UNIFORM__ a) {
    volatile uint32_t* bar = (volatile uint32_t*)a->bar_addr;
    const int* qs = (const int*)a->qs_addr;
    const volatile uint32_t* tails = (const volatile uint32_t*)a->tails_addr;
    int* counts = (int*)a->counts_addr;
    const uint32_t N = a->N;
    int blk = (int)blockIdx.x, tid = (int)threadIdx.x;
    const uint32_t skey = (uint32_t)blk / (uint32_t)(QSG);
    const uint32_t NBLKV = N / (GLANES * (VACT));
    int* acc = (int*)__local_mem((size_t)BUF_WORDS * sizeof(int)) + tid * (ACC_LS);
    uint64_t t0 = csr_read(VX_CSR_MCYCLE);
    aq_geometry(a->qs_addr, (uint64_t)QSTRIDE_W * sizeof(int), a->tails_addr, 4, AQ_QCAP);
#if SWQ
    swq_set(a->qs_addr, a->tails_addr, QSTRIDE_W, AQ_QCAP);
#endif
    vx_fence();
    for (uint32_t v = 0; v < (uint32_t)(VACT); v++)
        send_v(((uint32_t)blk * (VACT) + v) * GLANES + (uint32_t)tid, skey);
    vx_fence();
    uint64_t t1 = csr_read(VX_CSR_MCYCLE);
    gbarrier_ep(bar, 0, tid, (int)NBLKV, 1);
    if (blk == 0) aq_flush();
    vx_fence();
    uint64_t t2 = csr_read(VX_CSR_MCYCLE);
    for (uint32_t v = 0; v < (uint32_t)(VACT); v++) {
        const uint32_t me = ((uint32_t)blk * (VACT) + v) * GLANES + (uint32_t)tid;
        for (int b = 0; b < L; b++) acc[b] = 0;
        vx_fence();
        for (uint32_t sk = 0; sk < (uint32_t)NBA; sk++) {
            const uint32_t qid = sk * (N + (uint32_t)(QROWPAD)) + me;
            uint32_t tail = tails[qid];
            if (tail > (uint32_t)AQ_QCAP) tail = AQ_QCAP;
            recv_run(qs + (size_t)qid * QSTRIDE_W, tail, acc);
        }
        for (int b = 0; b < L; b++) counts[CNT_IDX(me, b)] = acc[b];
    }
    vx_fence();
    uint64_t t3 = csr_read(VX_CSR_MCYCLE);
    for (uint32_t v = 0; v < (uint32_t)(VACT); v++) {
        phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[((uint32_t)blk * (VACT) + v) * GLANES + (uint32_t)tid];
        ph->send = t1 - t0; ph->barrier = t2 - t1; ph->recv = t3 - t2; ph->total = t3 - t0;
        ph->t_start = t0; ph->t_end = t3;
    }
}
#else
void kernel_body(args_t* __UNIFORM__ a) {
    const int* qs = (const int*)a->qs_addr;
    const volatile uint32_t* tails = (const volatile uint32_t*)a->tails_addr;
    int* counts = (int*)a->counts_addr;
    volatile uint32_t* bar = (volatile uint32_t*)a->bar_addr;
    int N   = a->N;
    int blk = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    // SPLIT lanes share one actor. The lane's ACTOR is g / SPLIT and its slice of
    // that actor's mail is g % SPLIT, so the warp count no longer sets N.
    // A BLOCK is SPLIT warps wide, so the block count stays N/GLANES however many
    // warps the machine has. One warp per block meant the launch had to dispatch
    // SPLIT times as many blocks, and that overhead grew faster than the warps
    // saved (143K cycles at one warp, 1,167K at eight).
    int sub = ((SPLIT) == 1) ? 0 : tid / GLANES;   // WARP-major: warp w of the block is sub-lane w (a constant at SPLIT=1)
    int me  = blk * GLANES + tid % GLANES;   // of the block's 16 actors, so every warp is a
                                     // faithful copy of the one-warp kernel on its own
                                     // slice: 16 consecutive owners per warp (R8e's
                                     // layout) and 16 consecutive actors per tgt line.
    int NB  = N / GLANES;                 // sender blocks, the queue geometry
    int NBLK = N / GLANES;                // blocks launched, the barrier count
    int sblk = me / GLANES;               // this ACTOR's block, the queue's sender key
    uint32_t G = (uint32_t)(L * N);

    int* buf = (int*)__local_mem((size_t)BUF_WORDS * sizeof(int));
#if TILE_ACC
    int* acc = buf + tid * BACC_STRIDE;
#elif MEMACC
    int* acc = buf;
#else
    int* acc = buf + tid * (ACC_LS);
#endif

    uint64_t t0 = csr_read(VX_CSR_MCYCLE);
#if DIRECT_BT
    aq_geometry(a->bs_addr, (uint64_t)BQ_STRIDE_W * sizeof(int), a->btails_addr, 4, BQ_CAP);
#if OVF
    { volatile uint64_t* c = (volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE;   // overflow pool (control word 7)
      c[7] = (1ull << 60) | a->pool_addr;
      c[7] = (3ull << 60) | ((uint64_t)(OVF_CH) << 32) | (uint64_t)(OVF_MAXC);
      c[7] = (4ull << 60) | (uint64_t)(OVF_NCH);
      c[7] = (2ull << 60) | a->ctab_addr; }
#endif
#if SWQ
    swq_set(a->bs_addr, a->btails_addr, BQ_STRIDE_W, BQ_CAP);
#endif
#else
    aq_geometry(a->qs_addr, (uint64_t)QSTRIDE_W * sizeof(int), a->tails_addr, 4, AQ_QCAP);
#if SWQ
    swq_set(a->qs_addr, a->tails_addr, QSTRIDE_W, AQ_QCAP);
#endif
#endif
    vx_fence();
    // P3, the chunked round. The actor sends only its own queries [lo, hi) and
    // then receives, CHUNKS times, so the mail IN FLIGHT is the whole round
    // divided by CHUNKS and the same queue storage is refilled each time. The
    // destination keys do not change, so every queue still gets its whole share
    // of a chunk and combining is untouched; what changes is only how much is
    // outstanding between the append that fills a line and the read that
    // consumes it. CHUNKS = 1 is exactly the unchunked kernel.
    uint64_t c_send = 0, c_bar = 0, c_recv = 0, l_t1 = 0, l_t2 = 0;
    for (uint32_t ck = 0; ck < (uint32_t)(CHUNKS); ck++) {
    uint64_t t0c = csr_read(VX_CSR_MCYCLE);
    { const uint32_t span = (uint32_t)((M) / (CHUNKS) / (SPLIT));
      const uint32_t base = ck * (uint32_t)((M) / (CHUNKS)) + (uint32_t)sub * span;
      send_round((uint32_t)me, (uint32_t)NB, (uint32_t)sblk, (uint32_t)N, G, base, base + span); }
    vx_fence();
    uint64_t t1 = csr_read(VX_CSR_MCYCLE);
    gbarrier_ep(bar, 0, tid, NBLK, 3 * ck + 1);
    if (blk == 0) aq_flush();          // tail charge for the send round
    vx_fence();
    uint64_t t2 = csr_read(VX_CSR_MCYCLE);

#if BUCKET_TILE
#if !DIRECT_BT
    // reconfigure the queue engine for my bucket-tile queues, reorder, apply
    if (blk == 0)
        aq_geometry(a->bs_addr, (uint64_t)BQ_STRIDE_W * sizeof(int), a->btails_addr, 4, BQ_CAP);
    vx_fence();
    gbarrier(bar, 1, tid, NBLK);
    for (int sb = 0; sb < NB; sb++) {
        uint32_t qid  = qid_of((uint32_t)sb, (uint32_t)me, (uint32_t)NB, (uint32_t)N);
        uint32_t tail = tails[qid];
        if (tail > (uint32_t)AQ_QCAP) tail = AQ_QCAP;
        btile_reorder(qs + (size_t)qid * QSTRIDE_W, tail, (uint32_t)me, (uint32_t)N);
    }
    vx_fence();
    gbarrier(bar, 2, tid, NBLK);
    if (blk == 0) aq_flush();          // tail charge for the reorder round
    vx_fence();
#else
    (void)qs; (void)tails;   // the senders keyed by tile already; nothing to reorder
#endif
    { const int* bs = (const int*)a->bs_addr;
      const volatile uint32_t* bt = (const volatile uint32_t*)a->btails_addr;
#if TILE_LEAF && TILE_ACC
      for (uint32_t j = 0; j < (uint32_t)BTILE_SZ; j++) acc[j] = 0;   // once; tile_apply_tl clears as it writes
#endif
#if HYB_OUT
      for (uint32_t j = 0; j < (uint32_t)BTILE_SZ; j++) acc[j] = 0;
      vx_fence();
      tile_hyb(bs, (const uint32_t*)bt, buf + (size_t)sub * GLANES * BACC_STRIDE, counts, (uint32_t)tid % GLANES,
               (uint32_t)sub, (uint32_t)blk);
      if (0)
#endif
#if WC_TILE
      for (uint32_t j = (uint32_t)tid % GLANES; j < (uint32_t)FTILE_SZ; j += GLANES) buf[(size_t)sub * FTILE_SZ + j] = 0;
      vx_fence();
      tile_wc(bs, (const uint32_t*)bt, buf + (size_t)sub * FTILE_SZ, counts, (uint32_t)tid % GLANES, (uint32_t)sub, (uint32_t)blk
#if OVF
              , (const uint64_t*)a->ctab_addr
#endif
              );
      if (0)
#endif
      // constant trip count, sub only offsets the tile (see send_round). Ablation (piece B and C off,
      // BUCKET_TILE=1 with SPLIT > 1): one owner queue, drained by the warp sub = 0 lane of each actor
      for (uint32_t tt = 0; tt < ((uint32_t)(BUCKET_TILE) + (uint32_t)(SPLIT) - 1) / (uint32_t)(SPLIT); tt++) {
          const uint32_t t = (uint32_t)sub + tt * (uint32_t)(SPLIT);
          if (t >= (uint32_t)(BUCKET_TILE)) break;     // warp uniform (sub is the warp of the block)
          uint32_t bid = bqid_of(t, (uint32_t)me, (uint32_t)N);
          uint32_t tail = BT_COUNT(bt, bid);
          if (tail > (uint32_t)BQ_CAP) tail = BQ_CAP;
#if APPLY
          (void)tail; (void)bs;
          ap_issue(ap_slot((uint32_t)tid % (uint32_t)(NUM_THREADS)), a->bs_addr, (uint64_t)BQ_STRIDE_W * sizeof(int), a->btails_addr,
                   bid, 1, a->counts_addr + 4ull * (uint64_t)me, 4ull * (uint64_t)N,
                   AP_SPEC(0, 32, 0, 0, AP_ADD, 4, 1));
          ap_wait();
#elif CNT_T && TILE_LEAF
          tile_apply_tl(bs + (size_t)bid * BQ_STRIDE_W, tail, acc, t * (uint32_t)BTILE_SZ,
                        counts, (uint32_t)me, (uint32_t)N);
#elif CNT_T
          tile_apply_t(bs + (size_t)bid * BQ_STRIDE_W, tail, acc, t * (uint32_t)BTILE_SZ,
                       counts, (uint32_t)me, (uint32_t)N);
#elif TILE_ACC
          tile_apply(QPTR_(bs, BQ_STRIDE_W, bid), tail, acc, t * (uint32_t)BTILE_SZ,
                     counts + (size_t)me * L + (size_t)t * BTILE_SZ);
#else
          recv_mem(QPTR_(bs, BQ_STRIDE_W, bid), tail, counts + (size_t)me * L);
#endif
      } }
    (void)acc;
#elif APPLY
    // one command for the whole inbox: my NB queues, ADD 1 at counts[bucket*N + me]
    (void)qs; (void)tails; (void)counts;
#if QMAJOR
    ap_issue(ap_slot((uint32_t)tid % (uint32_t)(NUM_THREADS)), a->qs_addr,
             ((uint64_t)QSTRIDE_W * sizeof(int)) | ((uint64_t)(uint32_t)N << 32),   // step N queues
             a->tails_addr, (uint32_t)me, (sub == 0) ? (uint32_t)NB : 0u,
             a->counts_addr + 4ull * (uint64_t)me, 4ull * (uint64_t)N,
             AP_SPEC(0, 32, 0, 0, AP_ADD, 4, 1));
#else
    ap_issue(ap_slot((uint32_t)tid % (uint32_t)(NUM_THREADS)), a->qs_addr, (uint64_t)QSTRIDE_W * sizeof(int), a->tails_addr,
             (uint32_t)me * (uint32_t)NB, (sub == 0) ? (uint32_t)NB : 0u,
             a->counts_addr + 4ull * (uint64_t)me, 4ull * (uint64_t)N,
             AP_SPEC(0, 32, 0, 0, AP_ADD, 4, 1));
#endif
    ap_wait();
    (void)acc;
#elif MEMACC
    for (int sb = 0; sb < NB; sb++) {
        uint32_t qid  = qid_of((uint32_t)sb, (uint32_t)me, (uint32_t)NB, (uint32_t)N);
        uint32_t tail = tails[qid];
        if (tail > (uint32_t)AQ_QCAP) tail = AQ_QCAP;
        recv_mem(qs + (size_t)qid * QSTRIDE_W, tail, counts + (size_t)me * L);
    }
    (void)acc;
#else
#if WC_RECV
    // warp-cooperative: the warp takes the block's 16 actors one at a time and its
    // 16 lanes read 16 consecutive words of one queue (a line per load); each lane
    // counts into a private row, then the warp sums the 16 rows into that actor's row.
    recv_wc(qs, (const uint32_t*)tails, buf + (size_t)sub * GLANES * (L),
            buf + (size_t)(GLANES) * (SPLIT) * (L) + (size_t)sub * GLANES * (L),
            (uint32_t)tid % (uint32_t)GLANES, (uint32_t)sub, (uint32_t)blk);
    (void)acc;
#else
#if RECV_IL == 2
    { int* acc2 = buf + (size_t)(GLANES) * (SPLIT) * (ACC_LS) + (size_t)tid * (ACC_LS);
      for (int b = 0; b < L; b++) { acc[b] = 0; acc2[b] = 0; }
      vx_fence();
      for (uint32_t k = 0; k < (uint32_t)NB / (uint32_t)(SPLIT); k += 2) {
          const uint32_t qa = qid_of((uint32_t)sub + k * (uint32_t)(SPLIT), (uint32_t)me, (uint32_t)NB, (uint32_t)N);
          const uint32_t qb = qid_of((uint32_t)sub + (k + 1) * (uint32_t)(SPLIT), (uint32_t)me, (uint32_t)NB, (uint32_t)N);
          uint32_t ta = tails[qa], tb = tails[qb];
          if (ta > (uint32_t)AQ_QCAP) ta = AQ_QCAP;
          if (tb > (uint32_t)AQ_QCAP) tb = AQ_QCAP;
          recv_run2(qs + (size_t)qa * QSTRIDE_W, ta, qs + (size_t)qb * QSTRIDE_W, tb, acc, acc2);
      }
      for (int b = 0; b < L; b++) acc[b] += acc2[b]; }
#else
    for (int b = 0; b < L; b++) acc[b] = 0;
    vx_fence();
    // SPLIT: sub-lane s drains queues s, s+SPLIT, ... into its own row
    // (constant trip count, see send_round); sub-lane 0 then sums the rows.
    for (uint32_t k = 0; k < (uint32_t)NB / (uint32_t)(SPLIT); k++) {
        const uint32_t sb = (uint32_t)sub + k * (uint32_t)(SPLIT);
        uint32_t qid  = qid_of(sb, (uint32_t)me, (uint32_t)NB, (uint32_t)N);
        uint32_t tail = tails[qid];
        if (tail > (uint32_t)AQ_QCAP) tail = AQ_QCAP;
        recv_run(qs + (size_t)qid * QSTRIDE_W, tail, acc);
    }
#endif
#endif
#if (SPLIT) > 1
    __syncthreads();
    if (sub == 0) {                       // warp uniform: sub is the warp of the block
        for (int b = 0; b < L; b++) {
            int v = 0;
            for (int s = 0; s < (SPLIT); s++) v += buf[(s * GLANES + tid) * (ACC_LS) + b];
            counts[CNT_IDX(me, b)] = v;
        }
    }
#else
    for (int b = 0; b < L; b++) counts[CNT_IDX(me, b)] = acc[b];
#endif
#endif
    vx_fence();
    uint64_t t3 = csr_read(VX_CSR_MCYCLE);
    c_send += t1 - t0c; c_bar += t2 - t1; c_recv += t3 - t2; l_t1 = t1; l_t2 = t2;
#if (CHUNKS) > 1
    // Everyone has drained this chunk, so the queues may be emptied for the
    // next one. Two barriers: the reset must not race a straggler's drain, and
    // the next chunk's appends must not race the reset. The LAST chunk is not
    // reset, so the tails the host audits still hold that chunk's appends and
    // the capacity check keeps working. The condition is a loop counter, so it
    // is warp uniform and every block takes the same branch.
    if (ck + 1 < (uint32_t)(CHUNKS)) {
        gbarrier_ep(bar, 0, tid, NBLK, 3 * ck + 2);
        if (blk == 0) aq_round();
        vx_fence();
        gbarrier_ep(bar, 0, tid, NBLK, 3 * ck + 3);
    }
#endif
    }
    uint64_t t4 = csr_read(VX_CSR_MCYCLE);

    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];
    ph->send = c_send; ph->barrier = c_bar; ph->recv = c_recv; ph->total = t4 - t0;
    ph->t_start = t0; ph->t_end = t4;
#if LANE_TS
    { uint64_t* lt = (uint64_t*)(a->phase_addr + (uint64_t)N * sizeof(phase_cycles_t))
                     + ((uint64_t)me * (SPLIT) + (uint64_t)sub) * 4;
      lt[0] = t0; lt[1] = l_t1; lt[2] = l_t2; lt[3] = t4; }
#endif
}

#endif
int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N / GLANES / (VPATH ? (VACT) : 1);
    uint32_t block_dim = GLANES * (SPLIT);
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
