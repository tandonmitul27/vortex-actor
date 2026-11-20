// Sparse matrix transpose kernel (see common.h).
// ACTOR=1: one block = 16 actors on SPLIT warps; warp `sub` serves actors o = blk*16 + sub + k*SPLIT, k < OW,
// so an actor's rows, queue and arrays are only touched by its own warp. The warp's 16 lanes are the PE's threads.
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
#define TILE_A     ((NOTILE) ? 0 : (TILE_W))   // scratchpad words per warp: NOTILE applies in memory and allocates no tile
#if !defined(ACTOR) || ACTOR == 1
LMEM_CHECK((TILE_A) * (SPLIT) * 4, (NASSUME) / (GLANES));
#endif
#include "../abl.h"
#include "../probe_apply.h"
#define PH_W ((phase_cycles_t*)a->phase_addr + (blk * (SPLIT) + sub))   // this warp's phase record (probes)

static inline void gbarrier_ep(volatile uint32_t* bar, int slot, int tid, int NB, uint32_t ep) {
    if (tid % GLANES == 0) __atomic_fetch_add((uint32_t*)&bar[slot], 1, __ATOMIC_RELAXED);
    vx_fence();
#if ISA2
    // the arrival count is kept by L2 atomics: poll it with an atomic read (an L1 copy would be stale), then drop
    // every line this core cached before the barrier
    while (!vx_vote_all(__atomic_fetch_add((uint32_t*)&bar[slot], 0u, __ATOMIC_RELAXED) >= (uint32_t)NB * (uint32_t)(SPLIT) * ep)) { }
    ISA2_L1INV();
#else
    while (!vx_vote_all(bar[slot] >= (uint32_t)NB * (uint32_t)(SPLIT) * ep)) { }
#endif
}

#if ACTOR == 1
// warp-uniform control stores (never from one lane: SIMT collective rule)
static inline void aq_geometry(uint64_t qbase, uint64_t qstride, uint64_t tailbase, uint64_t wsize, uint64_t cap) {
#if !BASELINE && !(ISA2 && SWQ)   // ISA2: the append off (software queues) writes no L2 control register
    volatile uint64_t* c = (volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE;
    c[0] = qbase; c[1] = qstride; c[2] = tailbase; c[3] = wsize; c[4] = cap;
#endif
}
static inline void aq_round(void) {
#if !BASELINE && !(ISA2 && SWQ)
 ((volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE)[6] = 1; 
#endif
}
#include "../swq.h"
static inline void aq_append32(uint32_t q, uint32_t word) {
#if SWQ
    swq_put(q, word);
#elif ISA2
    vx_aq_append32(q, word);
#else
    *(volatile uint32_t*)(uintptr_t)((uint64_t)AQ_BASE + 64ull * ((uint64_t)q + 1ull)) = word;
#endif
}
#if MSG64
static_assert(NOTILE, "MSG64: only B off needs the 64-bit phase-2 message");
static inline void aq_append64(uint32_t q, uint64_t word) {
#if SWQ
    swq_put(q, word);
#elif ISA2
    vx_aq_append64(q, word);
#else
    *(volatile uint64_t*)(uintptr_t)((uint64_t)AQ_BASE + 64ull * ((uint64_t)q + 1ull)) = word;
#endif
}
#endif
// one queue segment, warp cooperative: lane ln takes entries ln, ln+16, ...; WC_U lines loaded first
// (ISA2: S_ = elements between lane ln's consecutive entries: QSTEP in a queue's region, GLANES in a chunk)
#define WC_SEG(QP, N_, OP, S_) { const uint32_t nf_ = (N_) / GLANES; uint32_t k_ = 0; \
    for (; k_ + (WC_U) <= nf_; k_ += (WC_U)) { uint32_t w_[WC_U]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(WC_U); u_++) w_[u_] = (QP)[(k_ + u_) * S_]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(WC_U); u_++) OP(w_[u_]); } \
    for (; k_ < nf_; k_++) OP((QP)[k_ * S_]); \
    if (nf_ * GLANES + ln < (N_)) OP((QP)[nf_ * S_]); }
// one whole queue: its region, then its overflow chunks (warp-uniform trip counts)
#define WC_QUEUE(QI, OP) { const uint32_t tl_ = QCOUNT(QI); msgs += tl_; \
    const uint32_t* q_ = QPTR(QI) + QIX(ln); \
    const uint32_t n0_ = tl_ < (uint32_t)(Q_CAP) ? tl_ : (uint32_t)(Q_CAP); \
    WC_SEG(q_, n0_, OP, QSTEP) \
    uint32_t r_ = tl_ - n0_; const uint64_t* cr_ = ctab + (size_t)(QI) * maxc; \
    _Pragma("GCC unroll 1") for (uint32_t c_ = 0; r_ != 0; c_++) { \
        const uint32_t n_ = r_ < (uint32_t)(OVF_CH) ? r_ : (uint32_t)(OVF_CH); \
        const uint32_t* qc_ = (const uint32_t*)(uintptr_t)cr_[c_] + ln; \
        WC_SEG(qc_, n_, OP, GLANES) \
        r_ -= n_; } }
#define QS_W QS32
#if MSG64
// phase 2 with 64-bit messages: the same walks on 8-byte entries (stride QS32 / 2 entries)
#define WC_SEG64(QP, N_, OP, S_) { const uint32_t nf_ = (N_) / GLANES; uint32_t k_ = 0; \
    for (; k_ + (WC_U) <= nf_; k_ += (WC_U)) { uint64_t w_[WC_U]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(WC_U); u_++) w_[u_] = (QP)[(k_ + u_) * S_]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(WC_U); u_++) OP(w_[u_]); } \
    for (; k_ < nf_; k_++) OP((QP)[k_ * S_]); \
    if (nf_ * GLANES + ln < (N_)) OP((QP)[nf_ * S_]); }
#define WC_QUEUE64(QI, OP) { const uint32_t tl_ = QCOUNT(QI); msgs += tl_; \
    const uint64_t* q_ = QPTR(QI) + QIX(ln); \
    const uint32_t n0_ = tl_ < (uint32_t)(Q_CAP) ? tl_ : (uint32_t)(Q_CAP); \
    WC_SEG64(q_, n0_, OP, QSTEP) \
    uint32_t r_ = tl_ - n0_; const uint64_t* cr_ = ctab + (size_t)(QI) * maxc; \
    _Pragma("GCC unroll 1") for (uint32_t c_ = 0; r_ != 0; c_++) { \
        const uint32_t n_ = r_ < (uint32_t)(OVF_CH) ? r_ : (uint32_t)(OVF_CH); \
        const uint64_t* qc_ = (const uint64_t*)(uintptr_t)cr_[c_] + ln; \
        WC_SEG64(qc_, n_, OP, GLANES) \
        r_ -= n_; } }
#define QUEUE_STATE64 const uint64_t* qs = (const uint64_t*)a->qs_addr; \
    const volatile uint32_t* tails = (const volatile uint32_t*)a->tails_addr; \
    const uint64_t* ctab = (const uint64_t*)a->ctab_addr; const uint32_t maxc = a->ovf_maxc;
#endif
#define FOR_MY_ACTORS(o) for (uint32_t ka_ = 0, o = blk * GLANES + sub; ka_ < (uint32_t)(OW); ka_++, o += (SPLIT))
#define QUEUE_STATE const uint32_t* qs = (const uint32_t*)a->qs_addr; \
    const volatile uint32_t* tails = (const volatile uint32_t*)a->tails_addr; \
    const uint64_t* ctab = (const uint64_t*)a->ctab_addr; const uint32_t maxc = a->ovf_maxc;

// Each round is one leaf over the warp's OW actors: one prologue per round (every callee save is a
// scattered stack write on this machine). Queue t * N + owner holds the owner's column tile t.

// phase 1 send: every local nonzero, col / P to owner col % P (32-bit word: the in-tile column)
static __attribute__((noinline)) void send_count(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln) {
    const int* aoff = (const int*)a->aoff_addr; const int* anz = (const int*)a->anz_addr;
    const uint32_t lr = a->lrows, lrs = lr + 1;
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        const int i1 = aoff[(size_t)o * lrs + lr];
#pragma GCC unroll 1
        for (int i = aoff[(size_t)o * lrs] + (int)ln; i < i1; i += GLANES) {
            const uint32_t col = (uint32_t)anz[i], cl = col / (NASSUME);
            aq_append32((cl >> (TB)) * (uint32_t)(NASSUME) + col % (NASSUME), cl & ((KW) - 1));
        }
    }
}
// phase 1 handler (lcounts[col]++): per tile queue, counted into the warp's scratchpad tile with scratchpad
// atomics, then the tile written out to lcounts as whole lines and cleared
static __attribute__((noinline)) uint32_t recv_count(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln,
                                                     uint32_t* tile) {
    QUEUE_STATE
    uint32_t* cnt = (uint32_t*)a->cnt_addr;
    const uint32_t lnc = a->lnc, T = a->ntile;
    uint32_t msgs = 0;
#if NOTILE
    // the owner's lcounts in memory, an L2 atomic per message
#if PROBE_NOAPPLY
    // measurement only: read every message, touch no state (the apply's cost = the normal run minus this one)
    uint32_t s_ = 0;
#define GCOUNT(x) { s_ ^= (uint32_t)(x); (void)lc; }
#else
#define GCOUNT(x) __atomic_fetch_add(&lc[x], 1u, __ATOMIC_RELAXED);
#endif
#if LANE_RECV
    PA_DECL
#define GCOUNT_P(x) PA_H(GCOUNT(x))
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES(o, t, 1u) { uint32_t* lc = cnt + (size_t)o * lnc; LANE_QUEUE(o, GCOUNT_P) }
#undef GCOUNT_P
    PA_FLUSH(PH_W)
#else
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) { uint32_t* lc = cnt + (size_t)o * lnc; WC_QUEUE(o, GCOUNT) }
#endif
#undef GCOUNT
#if PROBE_NOAPPLY
    if (s_ == 0x12345678u) cnt[0] = s_;
#endif
    (void)T; (void)tile;
#elif LANE_RECV
    // a lane per tile queue, counted into the lane's own KW-word scratchpad tile, written out by that lane
    uint32_t* lt = tile + ln * (uint32_t)(KW);
#define TCOUNT(x) lt[x] += 1u;
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES(o, t, T) {
        LANE_QUEUE(t * (uint32_t)(NASSUME) + o, TCOUNT)
        uint32_t* lc = cnt + (size_t)o * lnc;
        const uint32_t base = t * (uint32_t)(KW), w = lnc - base < (uint32_t)(KW) ? lnc - base : (uint32_t)(KW);
#pragma GCC unroll 1
        for (uint32_t j = 0; j < w; j++) { lc[base + j] = lt[j]; lt[j] = 0; }
    }
#else
#define TCOUNT(x) __atomic_fetch_add(&tile[x], 1u, __ATOMIC_RELAXED);
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        uint32_t* lc = cnt + (size_t)o * lnc;
#pragma GCC unroll 1
        for (uint32_t t = 0; t < T; t++) {
            WC_QUEUE(t * (uint32_t)(NASSUME) + o, TCOUNT)
            vx_fence();
            const uint32_t base = t * (uint32_t)(TILE_W), w = lnc - base < (uint32_t)(TILE_W) ? lnc - base : (uint32_t)(TILE_W);
#pragma GCC unroll 1
            for (uint32_t m = 0; m < (w + GLANES - 1) / GLANES; m++) {
                const uint32_t j = ln + m * GLANES;
                if (j < w) { lc[base + j] = tile[j]; tile[j] = 0; }
            }
            vx_fence();
        }
    }
#endif
#undef TCOUNT
    return msgs;
}
// lcounts -> At offsets: per actor, lane ln scans block ln of the counts, the lanes' block sums give the bases
static __attribute__((noinline)) void scan_round(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln,
                                                 volatile uint32_t* ws) {
    const uint32_t lnc = a->lnc;
    const uint32_t B = (lnc + GLANES - 1) / GLANES;
    const uint32_t b0 = ln * B < lnc ? ln * B : lnc, b1 = b0 + B < lnc ? b0 + B : lnc;
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        const uint32_t* lc = (const uint32_t*)a->cnt_addr + (size_t)o * lnc;
        uint32_t* lo = (uint32_t*)a->toff_addr + (size_t)o * (lnc + 1);
        uint32_t s = 0;
#pragma GCC unroll 1
        for (uint32_t i = b0; i < b1; i++) s += lc[i];
        ws[ln] = s;
        vx_fence();
        uint32_t base = 0, tot = 0;
#pragma GCC unroll 1
        for (uint32_t k = 0; k < (uint32_t)(GLANES); k++) { const uint32_t x = ws[k]; tot += x; base += k < ln ? x : 0; }
#if NOTILE
        uint32_t* wk = (uint32_t*)a->wrk_addr + (size_t)o * lnc;   // the handler's next position per column
#pragma GCC unroll 1
        for (uint32_t i = b0; i < b1; i++) { lo[i] = base; wk[i] = base; base += lc[i]; }
#else
#pragma GCC unroll 1
        for (uint32_t i = b0; i < b1; i++) { lo[i] = base; base += lc[i]; }
#endif
        lo[lnc] = tot;
        vx_fence();
    }
}
// phase 2 send: a lane per local row, {row, col / P} to owner col % P (32-bit word: row, in-tile column)
static __attribute__((noinline)) void send_place(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln) {
    const int* anz = (const int*)a->anz_addr;
    const uint32_t lr = a->lrows, lrs = lr + 1;
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        const int* aoff = (const int*)a->aoff_addr + (size_t)o * lrs;
#pragma GCC unroll 1
        for (uint32_t r = ln; r < lr; r += GLANES) {
#if MSG64
            const uint64_t hi = (uint64_t)(r * (uint32_t)(NASSUME) + o) << (TB);
#else
            const uint32_t hi = (r * (uint32_t)(NASSUME) + o) << (TB);
#endif
#pragma GCC unroll 1
            for (int j = aoff[r]; j < aoff[r + 1]; j++) {
                const uint32_t col = (uint32_t)anz[j], cl = col / (NASSUME);
#if MSG64
                aq_append64((cl >> (TB)) * (uint32_t)(NASSUME) + col % (NASSUME), hi | (cl & ((KW) - 1)));
#else
                aq_append32((cl >> (TB)) * (uint32_t)(NASSUME) + col % (NASSUME), hi | (cl & ((KW) - 1)));
#endif
            }
        }
    }
}
// phase 2 handler (At.lnonzero[loffset[col] + wrkoff[col]++] = row): per tile queue, the scratchpad tile loaded
// with the offsets, then each message's position taken from it with a scratchpad atomic
#if MSG64
#undef QS_W
#define QS_W ((QS32) / 2)
#endif
static __attribute__((noinline)) uint32_t recv_place(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln,
                                                     uint32_t* tile) {
#if MSG64
    QUEUE_STATE64
#else
    QUEUE_STATE
#endif
    const uint32_t lnc = a->lnc, T = a->ntile, tcap = a->tcap;
    uint32_t msgs = 0;
#if NOTILE
    // the owner's wrkoff in memory: an L2 atomic per message takes the position
    PA_DECL
#if PROBE_NOAPPLY
    uint64_t s_ = 0;   // measurement only, as recv_count
#define GPLACE(x) { s_ ^= (uint64_t)(x); (void)wk; (void)lnz; }
#else
#define GPLACE(x) { lnz[__atomic_fetch_add(&wk[(x) & ((KW) - 1)], 1u, __ATOMIC_RELAXED)] = (int)((x) >> (TB)); }
#endif
#if LANE_RECV
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES(o, t, 1u) {
#else
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
#endif
        uint32_t* wk = (uint32_t*)a->wrk_addr + (size_t)o * lnc;
        int* lnz = (int*)a->tnz_addr + (size_t)o * tcap;
#if LANE_RECV
#define GPLACE_P(x) PA_H(GPLACE(x))
        LANE_QUEUE(o, GPLACE_P)
#undef GPLACE_P
#elif MSG64
        WC_QUEUE64(o, GPLACE)
#else
        WC_QUEUE(o, GPLACE)
#endif
    }
#undef GPLACE
    PA_FLUSH(PH_W)
#if PROBE_NOAPPLY
    if (s_ == 0x123456789abcdefull) ((int*)a->tnz_addr)[0] = (int)s_;
#endif
    (void)T; (void)tile;
#elif LANE_RECV
    // a lane per tile queue: its own KW-word tile loaded with the offsets, positions taken from it
    uint32_t* lt = tile + ln * (uint32_t)(KW);
#define TPLACE(x) { lnz[lt[(x) & ((KW) - 1)]++] = (int)((x) >> (TB)); }
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES(o, t, T) {
        const uint32_t* lo = (const uint32_t*)a->toff_addr + (size_t)o * (lnc + 1);
        int* lnz = (int*)a->tnz_addr + (size_t)o * tcap;
        const uint32_t base = t * (uint32_t)(KW), w = lnc - base < (uint32_t)(KW) ? lnc - base : (uint32_t)(KW);
#pragma GCC unroll 1
        for (uint32_t j = 0; j < w; j++) lt[j] = lo[base + j];
        LANE_QUEUE(t * (uint32_t)(NASSUME) + o, TPLACE)
    }
#undef TPLACE
#else
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        const uint32_t* lo = (const uint32_t*)a->toff_addr + (size_t)o * (lnc + 1);
        int* lnz = (int*)a->tnz_addr + (size_t)o * tcap;
#define TPLACE(x) { lnz[__atomic_fetch_add(&tile[(x) & ((TILE_W) - 1)], 1u, __ATOMIC_RELAXED)] = (int)((x) >> (TB)); }
#pragma GCC unroll 1
        for (uint32_t t = 0; t < T; t++) {
            const uint32_t base = t * (uint32_t)(TILE_W), w = lnc - base < (uint32_t)(TILE_W) ? lnc - base : (uint32_t)(TILE_W);
#pragma GCC unroll 1
            for (uint32_t m = 0; m < (w + GLANES - 1) / GLANES; m++) {
                const uint32_t j = ln + m * GLANES;
                if (j < w) tile[j] = lo[base + j];
            }
            vx_fence();
            WC_QUEUE(t * (uint32_t)(NASSUME) + o, TPLACE)
            vx_fence();
        }
#undef TPLACE
    }
#endif
    return msgs;
}
#if MSG64
#undef QS_W
#define QS_W QS32
#endif
#endif

void kernel_body(args_t* __UNIFORM__ a) {
    const int* aoff = (const int*)a->aoff_addr;
    const int* anz = (const int*)a->anz_addr;
    uint32_t* cnt = (uint32_t*)a->cnt_addr;
    uint32_t* wrk = (uint32_t*)a->wrk_addr;
    uint32_t* toff = (uint32_t*)a->toff_addr;
    int* tnz = (int*)a->tnz_addr;
    volatile uint32_t* bar = (volatile uint32_t*)a->bar_addr;
    const int blk = (int)blockIdx.x, tid = (int)threadIdx.x;
    const uint32_t sub = (uint32_t)tid / GLANES, ln = (uint32_t)tid % GLANES;
    const uint32_t wi = (uint32_t)blk * (SPLIT) + sub;
    const int NB = (int)((uint32_t)(NASSUME) / GLANES);
    uint32_t msgs = 0;
#if ACTOR == 1
    aq_geometry(a->qs_addr, (uint64_t)(QS32) * 4ull, a->tails_addr, 4, Q_CAP);
#if !BASELINE && !(ISA2 && SWQ)
    { volatile uint64_t* c = (volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE;   // overflow pool
      c[7] = (1ull << 60) | a->pool_addr;
      c[7] = (3ull << 60) | ((uint64_t)(OVF_CH) << 32) | (uint64_t)a->ovf_maxc;
      c[7] = (4ull << 60) | (uint64_t)a->ovf_nch;
      c[7] = (2ull << 60) | a->ctab_addr; }
#endif
#if SWQ
    swq_geometry(a->qs_addr, (uint64_t)(QS32) * 4ull, a->tails_addr, 4, Q_CAP);
    swq_pool(a->pool_addr, a->ctab_addr, a->ovf_maxc, a->ovf_nch, a->swq_addr);
#endif
    volatile uint32_t* ws = (volatile uint32_t*)a->sums_addr + (size_t)wi * GLANES;
    uint32_t* tile = (uint32_t*)__local_mem((size_t)(TILE_A) * (SPLIT) * 4) + sub * (uint32_t)(TILE_W);
#pragma GCC unroll 1
    for (uint32_t m = 0; m < (uint32_t)(TILE_A) / GLANES; m++) tile[ln + m * GLANES] = 0;
#endif
    vx_fence();
    gbarrier_ep(bar, 0, tid, NB, 1);                      // every warp has set the geometry
    const uint64_t t0 = csr_read(VX_CSR_MCYCLE);
#if PROBE_TIMERS
    uint64_t pt_s = 0, pt_r = 0, pt_ = 0;   // measurement only
#define PT_S0 pt_ = csr_read(VX_CSR_MCYCLE);
#define PT_S1 pt_s += csr_read(VX_CSR_MCYCLE) - pt_;
#define PT_R0 pt_ = csr_read(VX_CSR_MCYCLE);
#define PT_R1 pt_r += csr_read(VX_CSR_MCYCLE) - pt_;
#else
#define PT_S0
#define PT_S1
#define PT_R0
#define PT_R1
#endif
#if ACTOR == 1
    // ---- phase 1: count the columns
    PT_S0 send_count(a, (uint32_t)blk, sub, ln); PT_S1
    vx_fence();
    gbarrier_ep(bar, 1, tid, NB, 1);
    PT_R0 msgs += recv_count(a, (uint32_t)blk, sub, ln, tile); PT_R1
    vx_fence();
    gbarrier_ep(bar, 2, tid, NB, 1);
#if SWQ
    swq_round((uint32_t)a->N * a->ntile, wi * GLANES + ln, (uint32_t)(NASSUME) * (SPLIT));
#else
    if (blk == 0) aq_round();                             // block-uniform: empty the queues for phase 2
#endif
    vx_fence();
    gbarrier_ep(bar, 3, tid, NB, 1);
#if MSG64
    aq_geometry(a->qs_addr, (uint64_t)(QS32) * 4ull, a->tails_addr, 8, Q_CAP);   // phase 2: 8-byte messages
#if SWQ
    swq_geometry(a->qs_addr, (uint64_t)(QS32) * 4ull, a->tails_addr, 8, Q_CAP);
#endif
#endif
    const uint64_t ta = csr_read(VX_CSR_MCYCLE);
    // ---- the offsets of my actors' rows of At
    scan_round(a, (uint32_t)blk, sub, ln, ws);
    const uint64_t tb = csr_read(VX_CSR_MCYCLE);
    // ---- phase 2: place the nonzeros
    PT_S0 send_place(a, (uint32_t)blk, sub, ln); PT_S1
    vx_fence();
    gbarrier_ep(bar, 4, tid, NB, 1);
    PT_R0 msgs += recv_place(a, (uint32_t)blk, sub, ln, tile); PT_R1
    vx_fence();
    gbarrier_ep(bar, 5, tid, NB, 1);
#if SWQ
    swq_round((uint32_t)a->N * a->ntile, wi * GLANES + ln, (uint32_t)(NASSUME) * (SPLIT));
#else
    if (blk == 0) aq_round();                             // block-uniform: charge the partial lines
#endif
    vx_fence();
    gbarrier_ep(bar, 6, tid, NB, 1);
#else
    // ---- transpose_matrix_agi, one PE: a thread per nonzero, then per row, grid stride
    const uint32_t gtid = wi * GLANES + ln;
    const uint32_t stride = (uint32_t)(NASSUME) * (SPLIT);
    const uint32_t nw = stride / GLANES;
    const uint32_t n = a->numrows, nnz = (uint32_t)aoff[n];
#pragma GCC unroll 1
    for (uint32_t i = gtid; i < nnz; i += stride)          // histogram the column counts of A
        __atomic_fetch_add(&cnt[anz[i]], 1u, __ATOMIC_RELAXED);
    vx_fence();
    gbarrier_ep(bar, 1, tid, NB, 1);
    const uint64_t ta = csr_read(VX_CSR_MCYCLE);
    // the offsets: thread gtid scans block gtid of the columns; block sums, then warp sums, give the bases
    volatile uint32_t* sums = (volatile uint32_t*)a->sums_addr;
    volatile uint32_t* wsum = sums + stride;
    const uint32_t B = (n + stride - 1) / stride;
    const uint32_t b0 = gtid * B < n ? gtid * B : n, b1 = b0 + B < n ? b0 + B : n;
    uint32_t s = 0;
#pragma GCC unroll 1
    for (uint32_t i = b0; i < b1; i++) s += cnt[i];
    sums[gtid] = s;
    vx_fence();
    gbarrier_ep(bar, 2, tid, NB, 1);
    uint32_t wt = 0, lb = 0;
#pragma GCC unroll 1
    for (uint32_t k = 0; k < (uint32_t)(GLANES); k++) { const uint32_t x = sums[wi * GLANES + k]; wt += x; lb += k < ln ? x : 0; }
    wsum[wi] = wt;
    vx_fence();
    gbarrier_ep(bar, 3, tid, NB, 1);
    uint32_t wb = 0;
#pragma GCC unroll 1
    for (uint32_t k = 0; k < wi; k++) wb += wsum[k];
    uint32_t base = wb + lb;
#pragma GCC unroll 1
    for (uint32_t i = b0; i < b1; i++) { toff[i] = base; wrk[i] = base; base += cnt[i]; }
    if (wi == nw - 1) toff[n] = wb + wt;                   // warp-uniform
    vx_fence();
    gbarrier_ep(bar, 4, tid, NB, 1);
    const uint64_t tb = csr_read(VX_CSR_MCYCLE);
    // redistribute the nonzeros
#pragma GCC unroll 1
    for (uint32_t r = gtid; r < n; r += stride) {
#pragma GCC unroll 1
        for (int j = aoff[r]; j < aoff[r + 1]; j++) {
            const uint32_t pos = __atomic_fetch_add(&wrk[anz[j]], 1u, __ATOMIC_RELAXED);
            tnz[pos] = (int)r;
        }
    }
    vx_fence();
    gbarrier_ep(bar, 5, tid, NB, 1);
#endif
    const uint64_t t1 = csr_read(VX_CSR_MCYCLE);
    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[wi];
    ph->p1 = ta - t0; ph->scan = tb - ta; ph->p2 = t1 - tb; ph->total = t1 - t0;
    PH_MSGS(ph, msgs);
    ph->t0 = t0; ph->t1 = t1;
#if PROBE_TIMERS && ACTOR == 1
    ph->snd = pt_s; ph->rcv = pt_r;
#endif
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N / GLANES;
    uint32_t block_dim = GLANES * (SPLIT);
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
