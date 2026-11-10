// PageRank (pull) kernel (see common.h).
// ACTOR=1: one block = 16 actors on SPLIT warps; warp `sub` serves actors o = blk*16 + sub + k*SPLIT, k < OW,
// so an actor's rows, queues and arrays are only touched by its own warp. The warp's 16 lanes are the PE's threads.
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
LMEM_CHECK((TILE_A) * (SPLIT) * 8, (NASSUME) / (GLANES));
#endif
#include "../abl.h"

static inline void gbarrier_ep(volatile uint32_t* bar, int slot, int tid, int NB, uint32_t ep) {
#if ISA2
    vx_fence();                                           // ISA2: this warp's writes are acknowledged before it arrives
#endif
    if (tid % GLANES == 0) __atomic_fetch_add((uint32_t*)&bar[slot], 1, __ATOMIC_RELAXED);
    vx_fence();
#if ISA2
    // the arrival count is kept by L2 atomics: poll it with an atomic read (an L1 copy would be stale),
    // then drop every line this core cached before the barrier
    while (!vx_vote_all(__atomic_fetch_add((uint32_t*)&bar[slot], 0u, __ATOMIC_RELAXED) >= (uint32_t)NB * (uint32_t)(SPLIT) * ep)) { }
    ISA2_L1INV();
#else
    while (!vx_vote_all(bar[slot] >= (uint32_t)NB * (uint32_t)(SPLIT) * ep)) { }
#endif
}
#if HWBAR
// the global barrier counts cores (simx Cluster::barrier), not blocks: D off puts NBLK = 64 blocks on 16 cores
#ifndef NCORES
#define NCORES     NUM_CORES
#endif
#define BARRIER() { vx_fence(); vx_barrier(0x80000000, NCORES); ISA2_L1INV(); }
#else
#define BARRIER() { vx_fence(); gbarrier_ep(bar, 0, tid, NB, ++ep); }
#endif

// Zacas amocas.d: if (*p == expected) *p = desired; returns the old *p
static inline uint64_t amocas64(volatile uint64_t* p, uint64_t expected, uint64_t desired) {
    uint64_t rd = expected;
    asm volatile(".insn r 0x2f, 0x3, 0x14, %0, %1, %2" : "+r"(rd) : "r"(p), "r"(desired) : "memory");
    return rd;
}
static inline double as_f64(uint64_t b) { double d; __builtin_memcpy(&d, &b, 8); return d; }
static inline uint64_t as_u64(double d) { uint64_t b; __builtin_memcpy(&b, &d, 8); return b; }
// *p += v on a double: the amocas loop
static inline void atomic_add_f64(volatile uint64_t* p, double v) {
    uint64_t old = *p;
#pragma GCC unroll 1
    for (;;) {
        const uint64_t prev = amocas64(p, old, as_u64(as_f64(old) + v));
        if (prev == old) break;
        old = prev;
    }
}
// lane partials of a warp -> the warp's sum (same order in every lane)
static inline double warp_sum(volatile double* ws, uint32_t ln, double e) {
    ws[ln] = e;
    vx_fence();
    double s = 0;
#pragma GCC unroll 1
    for (uint32_t k = 0; k < (uint32_t)(GLANES); k++) s += ws[k];
    return s;
}
// rows of PE o (of P)
static inline uint32_t pe_rows(uint32_t n, uint32_t o, uint32_t P) { return n > o ? (n - 1 - o) / P + 1 : 0; }

#if ACTOR == 1
// warp-uniform control stores (never from one lane: SIMT collective rule)
static inline void aq_geometry(uint64_t qbase, uint64_t qstride, uint64_t tailbase, uint64_t wsize, uint64_t cap) {
#if !BASELINE && !(ISA2 && SWQ)   // ISA2: the append off writes no L2 control register
    volatile uint64_t* c = (volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE;
    c[0] = qbase; c[1] = qstride; c[2] = tailbase; c[3] = wsize; c[4] = cap;
#endif
}
static inline void aq_flush(void) {
#if !BASELINE && !ISA2   // ISA2: no flush (the barrier waits for every append\'s acknowledgement)
 ((volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE)[5] = 1; 
#endif
}
static inline void aq_pool(uint64_t base, uint64_t ctab, uint32_t maxc, uint32_t nch) {
    volatile uint64_t* c = (volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE;
    c[7] = (1ull << 60) | base;
    c[7] = (3ull << 60) | ((uint64_t)(OVF_CH) << 32) | (uint64_t)maxc;
    c[7] = (4ull << 60) | (uint64_t)nch;
    c[7] = (2ull << 60) | ctab;
}
// the REQUEST geometry (32-bit words) and the RESPONSE geometry (64-bit words), each with its own pool; a switch
// resets the model's tails (the ones in memory stay readable) and flushes the old geometry's partial lines
static inline void geo_req(const args_t* a) {
    aq_pool(a->rqpool_addr, a->rqctab_addr, a->rq_maxc, a->rq_nch);
    aq_geometry(a->rqs_addr, (uint64_t)(RQS32) * 4ull, a->rqtails_addr, 4, RQ_CAP);
}
static inline void geo_rep(const args_t* a) {
    aq_pool(a->rppool_addr, a->rpctab_addr, a->rp_maxc, a->rp_nch);
    aq_geometry(a->rps_addr, (uint64_t)(RPS64) * 8ull, a->rptails_addr, 8, RP_CAP);
}
#include "../swq.h"
static inline void aq_append32(uint32_t q, uint32_t word) {
#if SWQ
    swq_put(q, word);
#elif ISA2
    aq_isa2_put(q, word);
#else
    *(volatile uint32_t*)(uintptr_t)((uint64_t)AQ_BASE + 64ull * ((uint64_t)q + 1ull)) = word;
#endif
}
static inline void aq_append64(uint32_t q, uint64_t word) {
#if SWQ
    swq_put(q, word);
#elif ISA2
    aq_isa2_put(q, word);
#else
    *(volatile uint64_t*)(uintptr_t)((uint64_t)AQ_BASE + 64ull * ((uint64_t)q + 1ull)) = word;
#endif
}
#if SWQ
// a switch starts the new geometry's queues empty (the unit resets its tails and reuses the pool): every thread sets
// the geometry, then empties what the last iteration left in it (swq_round); a barrier follows before any append
static inline void swq_req(const args_t* a, uint32_t gt, uint32_t gn) {
    swq_geometry(a->rqs_addr, (uint64_t)(RQS32) * 4ull, a->rqtails_addr, 4, RQ_CAP);
    swq_pool(a->rqpool_addr, a->rqctab_addr, a->rq_maxc, a->rq_nch, a->swq_addr);
    swq_round(a->N * a->nrq, gt, gn);
}
static inline void swq_rep(const args_t* a, uint32_t gt, uint32_t gn) {
    swq_geometry(a->rps_addr, (uint64_t)(RPS64) * 8ull, a->rptails_addr, 8, RP_CAP);
    swq_pool(a->rppool_addr, a->rpctab_addr, a->rp_maxc, a->rp_nch, a->swq_addr + 64);
    swq_round(a->N * a->nrp, gt, gn);
}
#endif
// one queue segment, warp cooperative: lane ln takes entries ln, ln+16, ...; WC_U lines loaded first
// (ISA2: S_ = elements between lane ln's consecutive entries: QSTEP_ in a queue's region, GLANES in a chunk)
#define WC_SEG(QP, N_, OP, S_) { const uint32_t nf_ = (N_) / GLANES; uint32_t k_ = 0; \
    for (; k_ + (WC_U) <= nf_; k_ += (WC_U)) { __typeof__(+(QP)[0]) w_[WC_U]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(WC_U); u_++) w_[u_] = (QP)[(k_ + u_) * S_]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(WC_U); u_++) OP(w_[u_]); } \
    for (; k_ < nf_; k_++) OP((QP)[k_ * S_]); \
    if (nf_ * GLANES + ln < (N_)) OP((QP)[nf_ * S_]); }
// one whole queue: its region, then its overflow chunks (warp-uniform trip counts); then the consumer zeroes its tail
// (SWQ: swq_round zeroes it with the chunk list; a zeroed tail would hide the list from it and leave stale chunk pointers)
#define WC_QUEUE(QS, STR, CAP, TL, CT, MAXC, QI, OP) { const uint32_t qi_ = (QI); const uint32_t tl_ = (TL)[qi_]; msgs += tl_; \
    const __typeof__(QS) q_ = QPTR_(QS, STR, qi_) + QIX_(QS, ln); \
    const uint32_t n0_ = tl_ < (uint32_t)(CAP) ? tl_ : (uint32_t)(CAP); \
    WC_SEG(q_, n0_, OP, QSTEP_(QS)) \
    uint32_t r_ = tl_ - n0_; const uint64_t* cr_ = (CT) + (size_t)qi_ * (MAXC); \
    _Pragma("GCC unroll 1") for (uint32_t c_ = 0; r_ != 0; c_++) { \
        const uint32_t n_ = r_ < (uint32_t)(OVF_CH) ? r_ : (uint32_t)(OVF_CH); \
        const __typeof__(QS) qc_ = (__typeof__(QS))(uintptr_t)cr_[c_] + ln; \
        WC_SEG(qc_, n_, OP, GLANES) \
        r_ -= n_; } \
    if (!(SWQ)) (TL)[qi_] = 0; }
// LANE_RECV (piece C off): one lane drains one whole queue alone, a word per load; then zeroes its tail
#define LANE_QUEUE_P(QS, STR, CAP, TL, CT, MAXC, QI, OP) { const uint32_t qi_ = (QI); const uint32_t tl_ = (TL)[qi_]; msgs += tl_; \
    const __typeof__(QS) q_ = QPTR_(QS, STR, qi_); \
    const uint32_t n0_ = tl_ < (uint32_t)(CAP) ? tl_ : (uint32_t)(CAP); \
    _Pragma("GCC unroll 1") for (uint32_t k_ = 0; k_ < n0_; k_++) OP(q_[QIX_(QS, k_)]); \
    uint32_t r_ = tl_ - n0_; const uint64_t* cr_ = (CT) + (size_t)qi_ * (MAXC); \
    _Pragma("GCC unroll 1") for (uint32_t c_ = 0; r_ != 0; c_++) { \
        const uint32_t n_ = r_ < (uint32_t)(OVF_CH) ? r_ : (uint32_t)(OVF_CH); \
        const __typeof__(QS) qc_ = (__typeof__(QS))(uintptr_t)cr_[c_]; \
        _Pragma("GCC unroll 1") for (uint32_t k_ = 0; k_ < n_; k_++) OP(qc_[k_]); \
        r_ -= n_; } \
    if (!(SWQ)) (TL)[qi_] = 0; }
#define FOR_MY_ACTORS(o) for (uint32_t ka_ = 0, o = blk * GLANES + sub; ka_ < (uint32_t)(OW); ka_++, o += (SPLIT))
#if AQ_ON
// ISA2: the L2 keeps one queue set's state, and the program switches sets between requests and responses, so after
// the send barrier each warp reads the counts of the queues it will drain (t * NASSUME + o, its actors x T tiles) and
// keeps them in its own tails array; a barrier follows before the switch
static __attribute__((noinline)) void snap_counts(volatile uint32_t* tl, uint32_t T, uint32_t blk, uint32_t sub, uint32_t ln) {
#pragma GCC unroll 1
    for (uint32_t i = ln; i < (uint32_t)(OW) * T; i += GLANES) {
        const uint32_t o = blk * GLANES + sub + (i / T) * (SPLIT), t = i % T, q = t * (uint32_t)(NASSUME) + o;
        tl[q] = vx_aq_count(q);
    }
}
#endif

// Each round is one leaf over the warp's OW actors: one prologue per round (every callee save is a
// scattered stack write on this machine).

// REQUEST: a lane per local row y; every nonzero x asks owner x % P, queue ((x / P) / RQ_W) * N + owner,
// word {asker | y | (x / P) % RQ_W}
static __attribute__((noinline)) void req_round(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln) {
    const int* anz = (const int*)a->anz_addr;
    const uint32_t lrs = a->lrows + 1, n = a->numrows;
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        const int* aoff = (const int*)a->aoff_addr + (size_t)o * lrs;
        const uint32_t lro = pe_rows(n, o, (uint32_t)(NASSUME));
#pragma GCC unroll 1
        for (uint32_t r = ln; r < lro; r += GLANES) {
            const int j0 = aoff[r], j1 = aoff[r + 1];
            const uint32_t hi = (o << (32 - (LNB))) | (r << (RB));
#pragma GCC unroll 1
            for (int j = j0; j < j1; j++) {
                const uint32_t col = (uint32_t)anz[j], xl = col / (NASSUME);
                aq_append32((xl >> (RB)) * (uint32_t)(NASSUME) + col % (NASSUME), hi | (xl & ((RQ_W) - 1)));
            }
        }
    }
}
// REQUEST handler: per source tile queue, read pr[x] and deg(x), reply {prf rounded to its top 64 - TB bits |
// y % TILE_W} to the asker's queue (y / TILE_W) * N + asker
static __attribute__((noinline)) uint32_t srv_round(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln) {
    const uint32_t* rqs = (const uint32_t*)a->rqs_addr;
    volatile uint32_t* tails = (volatile uint32_t*)a->rqtails_addr;
    const uint64_t* ctab = (const uint64_t*)a->rqctab_addr;
    const uint32_t maxc = a->rq_maxc, lr = a->lrows, lrs = lr + 1, T = a->nrq;
    uint32_t msgs = 0;
#define SERVE(w) { const uint32_t sw_ = (uint32_t)(w), sx_ = xb + (sw_ & ((RQ_W) - 1)); \
    const double sf_ = lpr[sx_] / (double)(ao[sx_ + 1] - ao[sx_]); \
    const uint32_t sy_ = (sw_ >> (RB)) & ((1u << (YB)) - 1u); \
    aq_append64((sy_ >> (TB)) * (uint32_t)(NASSUME) + (sw_ >> (32 - (LNB))), \
                ((as_u64(sf_) + (1ull << ((TB) - 1))) & ~(uint64_t)((KW) - 1)) | (sy_ & ((KW) - 1))); }
#if LANE_RECV
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES(o, t, T) {
        const int* ao = (const int*)a->aoff_addr + (size_t)o * lrs;
        const double* lpr = (const double*)a->pr_addr + (size_t)o * lr;
        const uint32_t xb = t * (uint32_t)(RQ_W);
        LANE_QUEUE_P(rqs, RQS32, RQ_CAP, tails, ctab, maxc, t * (uint32_t)(NASSUME) + o, SERVE)
    }
#else
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        const int* ao = (const int*)a->aoff_addr + (size_t)o * lrs;
        const double* lpr = (const double*)a->pr_addr + (size_t)o * lr;
#pragma GCC unroll 1
        for (uint32_t t = 0; t < T; t++) {
            const uint32_t xb = t * (uint32_t)(RQ_W);
            WC_QUEUE(rqs, RQS32, RQ_CAP, tails, ctab, maxc, t * (uint32_t)(NASSUME) + o, SERVE)
        }
    }
#endif
#undef SERVE
    return msgs;
}
// RESPONSE handler (curr[y] += prf): per row tile queue, added into the warp's scratchpad tile (amocas loops),
// then written out as whole lines with the update fused (curr = (1 - d) / n + d curr) and cleared; the lane's part of
// sum |curr - pr| goes to ws[ln]
static __attribute__((noinline)) uint32_t rep_round(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln,
                                                    volatile uint64_t* tile, volatile double* ws) {
    const uint64_t* rps = (const uint64_t*)a->rps_addr;
    volatile uint32_t* tails = (volatile uint32_t*)a->rptails_addr;
    const uint64_t* ctab = (const uint64_t*)a->rpctab_addr;
    const uint32_t maxc = a->rp_maxc, lr = a->lrows, n = a->numrows, T = a->nrp;
    const double c0 = (1 - (DAMPING)) / (double)n;
    uint32_t msgs = 0;
    double e = 0;
#if NOTILE
    // curr in memory, an L2 amocas loop per message; then the update over the actor's rows
#define GADD(x) atomic_add_f64((volatile uint64_t*)&lc[(x) & ((KW) - 1)], as_f64((x) & ~(uint64_t)((KW) - 1)));
#if LANE_RECV
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES(o, t, 1u) {
        double* lc = (double*)a->cur_addr + (size_t)o * lr;
        const double* lpr = (const double*)a->pr_addr + (size_t)o * lr;
        const uint32_t lro = pe_rows(n, o, (uint32_t)(NASSUME));
        LANE_QUEUE_P(rps, RPS64, RP_CAP, tails, ctab, maxc, o, GADD)
#pragma GCC unroll 1
        for (uint32_t r = 0; r < lro; r++) { const double c = c0 + (DAMPING) * lc[r]; lc[r] = c; e += __builtin_fabs(c - lpr[r]); }
    }
#else
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        double* lc = (double*)a->cur_addr + (size_t)o * lr;
        const double* lpr = (const double*)a->pr_addr + (size_t)o * lr;
        const uint32_t lro = pe_rows(n, o, (uint32_t)(NASSUME));
        WC_QUEUE(rps, RPS64, RP_CAP, tails, ctab, maxc, o, GADD)
        vx_fence();
#pragma GCC unroll 1
        for (uint32_t r = ln; r < lro; r += GLANES) { const double c = c0 + (DAMPING) * lc[r]; lc[r] = c; e += __builtin_fabs(c - lpr[r]); }
        vx_fence();
    }
#endif
#undef GADD
    (void)T; (void)tile;
#elif LANE_RECV
    // a lane per row tile queue, added into the lane's own KW-double tile, written out by that lane (update fused)
    volatile uint64_t* lt = tile + ln * (uint32_t)(KW);
#define TADD(x) { const uint32_t i_ = (uint32_t)((x) & ((KW) - 1)); lt[i_] = as_u64(as_f64(lt[i_]) + as_f64((x) & ~(uint64_t)((KW) - 1))); }
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES(o, t, T) {
        double* lc = (double*)a->cur_addr + (size_t)o * lr;
        const double* lpr = (const double*)a->pr_addr + (size_t)o * lr;
        const uint32_t lro = pe_rows(n, o, (uint32_t)(NASSUME));
        LANE_QUEUE_P(rps, RPS64, RP_CAP, tails, ctab, maxc, t * (uint32_t)(NASSUME) + o, TADD)
        const uint32_t base = t * (uint32_t)(KW), w = lro > base ? (lro - base < (uint32_t)(KW) ? lro - base : (uint32_t)(KW)) : 0;
#pragma GCC unroll 1
        for (uint32_t j = 0; j < w; j++) { const double c = c0 + (DAMPING) * as_f64(lt[j]); lc[base + j] = c; e += __builtin_fabs(c - lpr[base + j]); lt[j] = 0; }
    }
#undef TADD
#else
#define TADD(x) atomic_add_f64(&tile[(x) & ((TILE_W) - 1)], as_f64((x) & ~(uint64_t)((TILE_W) - 1)));
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        double* lc = (double*)a->cur_addr + (size_t)o * lr;
        const double* lpr = (const double*)a->pr_addr + (size_t)o * lr;
        const uint32_t lro = pe_rows(n, o, (uint32_t)(NASSUME));
#pragma GCC unroll 1
        for (uint32_t t = 0; t < T; t++) {
            WC_QUEUE(rps, RPS64, RP_CAP, tails, ctab, maxc, t * (uint32_t)(NASSUME) + o, TADD)
            vx_fence();
            const uint32_t base = t * (uint32_t)(TILE_W), w = lro - base < (uint32_t)(TILE_W) ? lro - base : (uint32_t)(TILE_W);
#pragma GCC unroll 1
            for (uint32_t m = 0; m < (w + GLANES - 1) / GLANES; m++) {
                const uint32_t j = ln + m * GLANES;
                if (j < w) { const double c = c0 + (DAMPING) * as_f64(tile[j]); lc[base + j] = c; e += __builtin_fabs(c - lpr[base + j]); tile[j] = 0; }
            }
            vx_fence();
        }
    }
#undef TADD
#endif
    ws[ln] = e;
    return msgs;
}
// pr = curr over my actors' rows
static __attribute__((noinline)) void copy_round(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln) {
    const uint32_t lr = a->lrows, n = a->numrows;
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        double* lc = (double*)a->cur_addr + (size_t)o * lr;
        double* lpr = (double*)a->pr_addr + (size_t)o * lr;
        const uint32_t lro = pe_rows(n, o, (uint32_t)(NASSUME));
#pragma GCC unroll 1
        for (uint32_t r = ln; r < lro; r += GLANES) { lpr[r] = lc[r]; if (NOTILE) lc[r] = 0; }   // NOTILE: curr restarts at 0
    }
}
#endif

void kernel_body(args_t* __UNIFORM__ a) {
    volatile uint32_t* bar = (volatile uint32_t*)a->bar_addr;
    const int blk = (int)blockIdx.x, tid = (int)threadIdx.x;
    const uint32_t sub = (uint32_t)tid / GLANES, ln = (uint32_t)tid % GLANES;
    const uint32_t wi = (uint32_t)blk * (SPLIT) + sub;
    const int NB = (int)((uint32_t)(NASSUME) / GLANES);
    const uint32_t nw = (uint32_t)NB * (SPLIT);
    volatile double* ws = (volatile double*)a->err_addr + (size_t)wi * GLANES;
    volatile double* wsum = (volatile double*)a->err_addr + (size_t)nw * GLANES;
    uint32_t ep = 0, it = 0;
    uint64_t msgs = 0, cq = 0, cs = 0, cr = 0;
#if ACTOR == 1
#if SWQ
    swq_req(a, wi * GLANES + ln, nw * GLANES);
#else
    if (blk == 0) geo_req(a);
#endif
    volatile uint64_t* tile = (volatile uint64_t*)__local_mem((size_t)(TILE_A) * (SPLIT) * 8) + sub * (uint32_t)(TILE_W);
#pragma GCC unroll 1
    for (uint32_t m = 0; m < (uint32_t)(TILE_A) / GLANES; m++) tile[ln + m * GLANES] = 0;
#endif
    BARRIER();                                            // the REQUEST geometry is set
    const uint64_t t0 = csr_read(VX_CSR_MCYCLE);
#if ACTOR == 1
#pragma GCC unroll 1
    while (it < (uint32_t)(MAXIT)) {
        it++;
        const uint64_t ts = csr_read(VX_CSR_MCYCLE);
        req_round(a, (uint32_t)blk, sub, ln);
        BARRIER();
#if AQ_ON
        snap_counts((volatile uint32_t*)a->rqtails_addr, a->nrq, (uint32_t)blk, sub, ln);
        BARRIER();
#endif
#if SWQ
        swq_rep(a, wi * GLANES + ln, nw * GLANES);
#else
        if (blk == 0) { aq_flush(); geo_rep(a); }         // block-uniform: tail charge, then the RESPONSE geometry
#endif
        BARRIER();
        const uint64_t ta = csr_read(VX_CSR_MCYCLE);
        msgs += srv_round(a, (uint32_t)blk, sub, ln);
        BARRIER();
#if AQ_ON
        snap_counts((volatile uint32_t*)a->rptails_addr, (NOTILE) ? 1u : a->nrp, (uint32_t)blk, sub, ln);
        BARRIER();
#endif
#if SWQ
        swq_req(a, wi * GLANES + ln, nw * GLANES);
#else
        if (blk == 0) { aq_flush(); geo_req(a); }         // next iteration's REQUEST geometry (no appends until then)
#endif
        const uint64_t tb = csr_read(VX_CSR_MCYCLE);
        msgs += rep_round(a, (uint32_t)blk, sub, ln, tile, ws);
        vx_fence();
        double ew = 0;
#pragma GCC unroll 1
        for (uint32_t k = 0; k < (uint32_t)(GLANES); k++) ew += ws[k];
        wsum[wi] = ew;
        BARRIER();
        double err = 0;
#pragma GCC unroll 1
        for (uint32_t k = 0; k < nw; k++) err += wsum[k];
        cq += ta - ts; cs += tb - ta;
        if (err < (ERR_DELTA)) { cr += csr_read(VX_CSR_MCYCLE) - tb; break; }
        copy_round(a, (uint32_t)blk, sub, ln);            // my actors' rows only: the next serve (after a barrier) reads them
        cr += csr_read(VX_CSR_MCYCLE) - tb;
    }
#else
    // ---- pagerank_agi (pull), one PE: a thread per row, grid stride; the update fused (same thread, same row)
    const int* aoff = (const int*)a->aoff_addr;
    const int* anz = (const int*)a->anz_addr;
    double* pr = (double*)a->pr_addr;
    double* cur = (double*)a->cur_addr;
    const uint32_t gtid = wi * GLANES + ln;
    const uint32_t stride = (uint32_t)(NASSUME) * (SPLIT);
    const uint32_t n = a->numrows;
    const double c0 = (1 - (DAMPING)) / (double)n;
#pragma GCC unroll 1
    while (it < (uint32_t)(MAXIT)) {
        it++;
        const uint64_t ts = csr_read(VX_CSR_MCYCLE);
        double e = 0;
#pragma GCC unroll 1
        for (uint32_t y = gtid; y < n; y += stride) {
            const int j0 = aoff[y], j1 = aoff[y + 1];
            double s = 0;
#pragma GCC unroll 1
            for (int j = j0; j < j1; j++) { const int x = anz[j]; s += pr[x] / (double)(aoff[x + 1] - aoff[x]); }
            const double c = c0 + (DAMPING) * s;
            cur[y] = c; e += __builtin_fabs(c - pr[y]);
        }
        const double ew = warp_sum(ws, ln, e);
        wsum[wi] = ew;
        BARRIER();
        const uint64_t tb = csr_read(VX_CSR_MCYCLE);
        double err = 0;
#pragma GCC unroll 1
        for (uint32_t k = 0; k < nw; k++) err += wsum[k];
        cq += tb - ts;
        if (err < (ERR_DELTA)) { cr += csr_read(VX_CSR_MCYCLE) - tb; break; }
#pragma GCC unroll 1
        for (uint32_t i = gtid; i < n; i += stride) pr[i] = cur[i];
        BARRIER();
        cr += csr_read(VX_CSR_MCYCLE) - tb;
    }
#endif
    const uint64_t t1 = csr_read(VX_CSR_MCYCLE);
    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[wi];
    ph->req = cq; ph->srv = cs; ph->rep = cr; ph->total = t1 - t0;
    PH_MSGS(ph, (uint32_t)msgs); ph->iters = it; ph->t0 = t0; ph->t1 = t1;
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N / GLANES;
    uint32_t block_dim = GLANES * (SPLIT);
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
