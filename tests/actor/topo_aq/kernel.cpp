// Toposort kernel (see common.h).
// ACTOR=1: one block = 16 actors on SPLIT warps; warp `sub` serves actors o = blk*16 + sub + k*SPLIT, k < OW,
// so an actor's rows, columns and queues are only touched by its own warp. The warp's 16 lanes are the PE's threads.
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
LMEM_CHECK((SPW) * (SPLIT) * 4, (NASSUME) / (GLANES));
#endif
#include "../abl.h"
#include "../probe_apply.h"
#define PH_W ((phase_cycles_t*)a->phase_addr + (blk * (SPLIT) + sub))   // this warp's phase record (probes)

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
static inline void aq_round(void) {
#if !BASELINE && !(ISA2 && SWQ)   // ISA2: the append off writes no L2 control register
 ((volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE)[6] = 1; 
#endif
}
#include "../swq.h"
#if MSG64
static_assert(NOTILE, "MSG64: only B off needs the 64-bit message");
#endif
static inline void aq_append32(uint32_t q, msg_t word) {   // msg_t: 32-bit, or 64-bit under MSG64
#if SWQ
    swq_put(q, word);
#elif ISA2
    aq_isa2_put(q, word);
#else
    *(volatile msg_t*)(uintptr_t)((uint64_t)AQ_BASE + 64ull * ((uint64_t)q + 1ull)) = word;
#endif
}
// one queue segment, warp cooperative: lane ln takes entries ln, ln+16, ...; U_ lines loaded first (the remainder
// loop exists only for U_ > 1)
// (ISA2: S_ = elements between lane ln's consecutive entries: QSTEP in a queue's region, GLANES in a chunk)
#define WC_SEGU_S(QP, N_, OP, U_, S_) { const uint32_t nf_ = (N_) / GLANES; uint32_t k_ = 0; \
    _Pragma("GCC unroll 1") for (; k_ + (U_) <= nf_; k_ += (U_)) { msg_t w_[U_]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(U_); u_++) w_[u_] = (QP)[(k_ + u_) * S_]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(U_); u_++) OP(w_[u_]); } \
    if ((U_) > 1) { _Pragma("GCC unroll 1") for (; k_ < nf_; k_++) OP((QP)[k_ * S_]); } \
    if (nf_ * GLANES + ln < (N_)) OP((QP)[nf_ * S_]); }
#define WC_SEGU(QP, N_, OP, U_) WC_SEGU_S(QP, N_, OP, U_, GLANES)
// entries [H, TL) of queue QI, segment by segment (the region, then each overflow chunk): one WC_SEGU site
#if AQ_ON
// ISA2 layout: a region segment steps line by line (QSTEP), a chunk segment word by word
#define WC_RANGE(QI, H, TL, OP, U) { uint32_t p_ = (H); const uint32_t tl_ = (TL); msgs += tl_ - p_; \
    const msg_t* q0_ = QPTR(QI); const uint64_t* cr_ = ctab + (size_t)(QI) * maxc; \
    _Pragma("GCC unroll 1") while (p_ < tl_) { \
        const msg_t* bq_; uint32_t e_; size_t s_; \
        if (p_ < (uint32_t)(Q_CAP)) { bq_ = q0_ + QIX(p_ + ln); e_ = (uint32_t)(Q_CAP); s_ = QSTEP; } \
        else { const uint32_t x_ = p_ - (uint32_t)(Q_CAP); \
            bq_ = (const msg_t*)(uintptr_t)cr_[x_ / (uint32_t)(OVF_CH)] + x_ % (uint32_t)(OVF_CH) + ln; \
            e_ = p_ + ((uint32_t)(OVF_CH) - x_ % (uint32_t)(OVF_CH)); s_ = GLANES; } \
        if (e_ > tl_) e_ = tl_; \
        WC_SEGU_S(bq_, e_ - p_, OP, U, s_) \
        p_ = e_; } }
#else
#define WC_RANGE(QI, H, TL, OP, U) { uint32_t p_ = (H); const uint32_t tl_ = (TL); msgs += tl_ - p_; \
    const msg_t* q0_ = qs + (size_t)(QI) * (QS_W); const uint64_t* cr_ = ctab + (size_t)(QI) * maxc; \
    _Pragma("GCC unroll 1") while (p_ < tl_) { \
        const msg_t* b_; uint32_t e_; \
        if (p_ < (uint32_t)(Q_CAP)) { b_ = q0_ + p_; e_ = (uint32_t)(Q_CAP); } \
        else { const uint32_t x_ = p_ - (uint32_t)(Q_CAP); \
            b_ = (const msg_t*)(uintptr_t)cr_[x_ / (uint32_t)(OVF_CH)] + x_ % (uint32_t)(OVF_CH); \
            e_ = p_ + ((uint32_t)(OVF_CH) - x_ % (uint32_t)(OVF_CH)); } \
        if (e_ > tl_) e_ = tl_; \
        const msg_t* bq_ = b_ + ln; \
        WC_SEGU(bq_, e_ - p_, OP, U) \
        p_ = e_; } }
#endif
#define FOR_MY_ACTORS(o) for (uint32_t ka_ = 0, o = blk * GLANES + sub; ka_ < (uint32_t)(OW); ka_++, o += (SPLIT))
#define QS_W ((QS32) * 4 / (uint32_t)sizeof(msg_t))   // queue stride in entries
#define QUEUE_STATE const msg_t* qs = (const msg_t*)a->qs_addr; \
    const volatile uint32_t* tails = (const volatile uint32_t*)a->tails_addr; \
    const uint64_t* ctab = (const uint64_t*)a->ctab_addr; const uint32_t maxc = a->ovf_maxc;

// Phase 1: round 0 sends the MATCH of every row with rowcnt 1 at the start; odd rounds drain MATCH and append DEC,
// even rounds drain DEC and append MATCH. Each round type is ONE leaf over the warp's actors (pointers reloaded from
// args, sizes compile time: no spilled state in the round path). Each round ends with one barrier; block 0 then
// flushes the round's partial lines. A round that appended nothing ends the levels.
// returns (msgs drained << 32) | this lane's appends
#if PROF
#define PT0 const uint64_t q0_ = csr_read(VX_CSR_MCYCLE);
#define PT1(K, D) { volatile uint32_t d_ = (D); (void)d_; } const uint64_t q1_ = csr_read(VX_CSR_MCYCLE); pf[K] += q1_ - q0_;
#define PT2(K) { const uint64_t q2_ = csr_read(VX_CSR_MCYCLE); pf[K] += q2_ - q1_; }
#else
#define PT0
#define PT1(K, D)
#define PT2(K)
#endif
static __attribute__((noinline)) uint64_t match_round(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln, uint64_t* pf) {
    QUEUE_STATE
    constexpr uint32_t N = NASSUME, tb = TB, tmask = (1u << TB) - 1u, lnb = LNB, lr = LR;
    uint32_t* heads = (uint32_t*)a->heads_addr;
    const uint32_t* tnz = (const uint32_t*)a->tnz_addr;
    const uint32_t* toff0 = (const uint32_t*)a->toff_addr;
    uint32_t msgs = 0, sent = 0;
    PA_DECL
    // MATCH handler: expand local column j of tmat, one DEC per nonzero row
#if LANE_RECV
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES(o, t, 1u) {
        const uint32_t* toff = toff0 + (size_t)o * (lr + 1);
        const uint32_t h = heads[o], tl = QCOUNT(o);
#else
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        const uint32_t* toff = toff0 + (size_t)o * (lr + 1);
        PT0
        const uint32_t h = heads[o], tl = QCOUNT(o);
        PT1(0, tl - h)
#endif
#define MATCH(w) { const uint32_t c_ = ((w) << lnb) | o, e1_ = toff[(w) + 1]; \
    _Pragma("GCC unroll 1") for (uint32_t e_ = toff[w]; e_ < e1_; e_++) { const uint32_t rr_ = tnz[e_], i_ = rr_ >> lnb; \
        PA_S(aq_append32(N + (i_ >> tb) * N + (rr_ & (N - 1)), (msg_t)(i_ & tmask) | ((msg_t)c_ << tb))) sent++; } }
#if LANE_RECV
#define MATCH_P(w) PA_H(MATCH(w))
        LANE_RANGE(o, h, tl, MATCH_P)
#undef MATCH_P
#else
        WC_RANGE(o, h, tl, MATCH, 1)
        PT2(1)
#endif
#undef MATCH
        heads[o] = tl;
    }
    PA_FLUSH(PH_W)
    return ((uint64_t)msgs << 32) | sent;
}
static __attribute__((noinline)) uint64_t dec_round(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln, uint32_t lv, uint64_t* pf) {
    QUEUE_STATE
    constexpr uint32_t N = NASSUME, tb = TB, tmask = (1u << TB) - 1u, lnb = LNB, lr = LR, ts = NT;
    uint32_t* heads = (uint32_t*)a->heads_addr;
    uint32_t msgs = 0, sent = 0;
    PA_DECL
    // DEC handler: rowsum -= c, rowcnt--; a row reaching rowcnt 1 takes this level and sends its MATCH
#if LANE_RECV
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES(o, t, ts) {
        uint32_t* cn = (uint32_t*)a->cnt_addr + (size_t)o * lr;
        uint32_t* sm = (uint32_t*)a->sum_addr + (size_t)o * lr;
        {
            const uint32_t qi = N + t * N + o, h = heads[qi], tl = QCOUNT(qi);
#else
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        uint32_t* cn = (uint32_t*)a->cnt_addr + (size_t)o * lr;
        uint32_t* sm = (uint32_t*)a->sum_addr + (size_t)o * lr;
#pragma GCC unroll 1
        for (uint32_t t = 0; t < ts; t++) {
            PT0
            const uint32_t qi = N + t * N + o, h = heads[qi], tl = QCOUNT(qi);
            PT1(2, tl - h)
#endif
#define DEC(w) { const uint32_t i_ = (t << tb) | ((w) & tmask), c_ = (w) >> tb; \
    const uint32_t old_ = __atomic_fetch_add(&cn[i_], 0xffffffffu, __ATOMIC_RELAXED); \
    __atomic_fetch_sub(&sm[i_], c_, __ATOMIC_RELAXED); \
    if (old_ == 2) { const uint32_t s_ = ((volatile uint32_t*)sm)[i_]; const size_t x_ = (size_t)o * lr + i_; \
        ((uint32_t*)a->mat_addr)[x_] = s_; ((uint32_t*)a->lev_addr)[x_] = lv; PA_S(aq_append32(s_ & (N - 1), s_ >> lnb)) sent++; } }
#if LANE_RECV
#define DEC_P(w) PA_H(DEC(w))
            LANE_RANGE(qi, h, tl, DEC_P)
#undef DEC_P
#else
            WC_RANGE(qi, h, tl, DEC, WC_U)
            PT2(3)
#endif
#undef DEC
            heads[qi] = tl;
        }
    }
    PA_FLUSH(PH_W)
    return ((uint64_t)msgs << 32) | sent;
}
// round 0: the rows with rowcnt 1 (level 0) send their MATCH
static __attribute__((noinline)) uint32_t match0(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln) {
    constexpr uint32_t N = NASSUME, lnb = LNB, lr = LR;
    const uint32_t* iq = (const uint32_t*)a->iq_addr;
    const uint32_t* iqo = (const uint32_t*)a->iqoff_addr;
    const uint32_t* sum = (const uint32_t*)a->sum_addr;
    uint32_t* lev = (uint32_t*)a->lev_addr;
    uint32_t* mat = (uint32_t*)a->mat_addr;
    uint32_t sent = 0;
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        const uint32_t k1 = iqo[o + 1];
#pragma GCC unroll 1
        for (uint32_t k = iqo[o] + ln; k < k1; k += GLANES) {
            const uint32_t i = o * lr + iq[k], c = sum[i];
            mat[i] = c; lev[i] = 0;
            aq_append32(c & (N - 1), c >> lnb); sent++;
        }
    }
    return sent;
}
static __attribute__((noinline)) uint64_t phase1(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln,
                                                 int tid, int NB, volatile uint32_t* bar, uint32_t* epp, uint32_t* nrounds,
                                                 uint64_t* prof) {
    volatile uint32_t* flag = (volatile uint32_t*)a->flag_addr;
    uint32_t ep = *epp;
    uint64_t msgs = 0;
#if PROF
    uint64_t pm = 0, pd = 0, pb = 0, tp = csr_read(VX_CSR_MCYCLE);
#define TICK(v) { const uint64_t n_ = csr_read(VX_CSR_MCYCLE); v += n_ - tp; tp = n_; }
#else
#define TICK(v)
#endif
    PA_W0
#if ROUND1
    const uint64_t r10_ = csr_read(VX_CSR_MCYCLE);
#endif
    uint32_t sent = match0(a, blk, sub, ln);
    if (vx_vote_any((int)(sent != 0))) flag[0] = 1;
    vx_fence();
    PA_W1(PH_W, pa_x)
#if ROUND1
    PH_W->r1 = csr_read(VX_CSR_MCYCLE) - r10_;           // probe: level 0's MATCH sends only
    *epp = ep; *nrounds = 1; (void)bar; (void)tid; (void)NB; (void)prof;
    return 0;
#endif
    TICK(pd)
    BARRIER();
    TICK(pb)
    uint32_t r = 1;
#pragma GCC unroll 1
    for (; r < (uint32_t)(MAXR); r++) {
        if (flag[r - 1] == 0) break;                      // uniform: every thread reads the same flag
        if (blk == 0) aq_flush();                         // block-uniform: round r-1's partial lines
        PA_W0
        const uint64_t rv = (r & 1) ? match_round(a, blk, sub, ln, prof + 3) : dec_round(a, blk, sub, ln, r >> 1, prof + 3);
        PA_W1(PH_W, pa_r)
        msgs += rv >> 32;
        if (r & 1) { TICK(pm) } else { TICK(pd) }
        if (vx_vote_any((int)((uint32_t)rv != 0))) flag[r] = 1;
        vx_fence();
        BARRIER();
        TICK(pb)
    }
#if PROF
    prof[0] = pm; prof[1] = pd; prof[2] = pb;
#else
    (void)prof;
#endif
#undef TICK
    *epp = ep; *nrounds = r;
    return msgs;
}
// phase 2: level counts of my actors' rows (scratchpad atomics) -> hist[L][o]
static __attribute__((noinline)) void level_hist(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln,
                                                 uint32_t* ctrs, uint32_t nl) {
    constexpr uint32_t N = NASSUME;
    const uint32_t lrows = a->lrows, n = a->numrows;
    const uint32_t* lev = (const uint32_t*)a->lev_addr;
    uint32_t* hist = (uint32_t*)a->hist_addr;
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        const uint32_t nr = pe_rows(n, o, N);
        const uint32_t* lv = lev + (size_t)o * lrows;
#pragma GCC unroll 1
        for (uint32_t L = ln; L < nl; L += GLANES) ctrs[L] = 0;
        vx_fence();
#pragma GCC unroll 1
        for (uint32_t i = ln; i < nr; i += GLANES) __atomic_fetch_add(&ctrs[lv[i]], 1u, __ATOMIC_RELAXED);
        vx_fence();
#pragma GCC unroll 1
        for (uint32_t L = ln; L < nl; L += GLANES) hist[L * N + o] = ctrs[L];
        vx_fence();
    }
}
// phase 2: warp wi scans levels wi, wi + warps, ...: hist[L][*] -> its exclusive prefix over PEs, ltot[L] = total
static __attribute__((noinline)) void level_prefix(const args_t* __UNIFORM__ a, uint32_t wi, uint32_t ln,
                                                   volatile uint32_t* ws, uint32_t nl) {
    constexpr uint32_t N = NASSUME, PER = (NASSUME) / GLANES, NW = (NBLK) * (SPLIT);
    uint32_t* hist = (uint32_t*)a->hist_addr;
    volatile uint32_t* ltot = (volatile uint32_t*)a->ltot_addr;
#pragma GCC unroll 1
    for (uint32_t L = wi; L < nl; L += NW) {
        uint32_t* h = hist + L * N + ln * PER;
        uint32_t s = 0;
#pragma GCC unroll 1
        for (uint32_t x = 0; x < PER; x++) s += h[x];
        ws[ln] = s;
        vx_fence();
        uint32_t base = 0, tot = 0;
#pragma GCC unroll 1
        for (uint32_t x = 0; x < (uint32_t)(GLANES); x++) { const uint32_t v = ws[x]; tot += v; base += x < ln ? v : 0; }
#pragma GCC unroll 1
        for (uint32_t x = 0; x < PER; x++) { const uint32_t v = h[x]; h[x] = base; base += v; }
        ltot[L] = tot;
        vx_fence();
    }
}
// phase 2: rperm = (n - 1) - level_start[level]++ (level_start: the levels below, then the PEs before me), and the
// CPERM {in-tile column | rperm} to the owner of the matched column
static __attribute__((noinline)) void send_cperm(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln,
                                                 uint32_t* ctrs, uint32_t nl) {
    constexpr uint32_t N = NASSUME, tb = TB, tmask = (1u << TB) - 1u, lnb = LNB;
    const uint32_t lrows = a->lrows, n = a->numrows;
    const uint32_t* lev = (const uint32_t*)a->lev_addr;
    const uint32_t* mat = (const uint32_t*)a->mat_addr;
    uint32_t* rperm = (uint32_t*)a->rperm_addr;
    const uint32_t* hist = (const uint32_t*)a->hist_addr;
    const volatile uint32_t* ltot = (const volatile uint32_t*)a->ltot_addr;
    uint32_t* lstart = ctrs + (MAXL);
    uint32_t run = 0;
#pragma GCC unroll 1
    for (uint32_t L = 0; L < nl; L++) { lstart[L] = run; run += ltot[L]; }   // uniform: every lane the same
    vx_fence();
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        const uint32_t nr = pe_rows(n, o, N);
#pragma GCC unroll 1
        for (uint32_t L = ln; L < nl; L += GLANES) ctrs[L] = lstart[L] + hist[L * N + o];
        vx_fence();
#pragma GCC unroll 1
        for (uint32_t i = ln; i < nr; i += GLANES) {
            const uint32_t x = o * lrows + i;
            const uint32_t pos = (n - 1) - __atomic_fetch_add(&ctrs[lev[x]], 1u, __ATOMIC_RELAXED);
            rperm[x] = pos;
            const uint32_t c = mat[x], j = c >> lnb;
            aq_append32(N + (j >> tb) * N + (c & (N - 1)), (msg_t)(j & tmask) | ((msg_t)pos << tb));
        }
        vx_fence();
    }
}
// phase 2 handler: cperm[j] = rperm, applied in a scratchpad tile and written out as whole lines
static __attribute__((noinline)) uint64_t recv_cperm(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln,
                                                     uint32_t* tile) {
    QUEUE_STATE
    constexpr uint32_t N = NASSUME, tb = TB, tmask = (1u << TB) - 1u;
    const uint32_t lrows = a->lrows, n = a->numrows, ts = a->ntile;
    uint32_t* cperm = (uint32_t*)a->cperm_addr;
    uint64_t msgs = 0;
    PA_DECL
#if NOTILE
    // cperm[j] = rperm in memory (a plain store per message: each column is placed once)
#if LANE_RECV
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES(o, t, 1u) {
#else
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
#endif
        uint32_t* cp = cperm + (size_t)o * lrows;
        const uint32_t qi = N + o;
#define CPERM(x) { cp[(x) & tmask] = (x) >> tb; }
#if LANE_RECV
#define CPERM_P(x) PA_H(CPERM(x))
        LANE_RANGE(qi, 0u, QCOUNT(qi), CPERM_P)
#undef CPERM_P
#else
        WC_RANGE(qi, 0u, QCOUNT(qi), CPERM, WC_U)
#endif
#undef CPERM
    }
    (void)tile; (void)ts;
    PA_FLUSH(PH_W)
#elif LANE_RECV
    // a lane per column tile queue, its own KW-word tile, written out by that lane
    uint32_t* lt = tile + ln * (uint32_t)(KW);
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES(o, t, ts) {
        const uint32_t nc = pe_rows(n, o, N);
        const uint32_t qi = N + t * N + o;
        const uint32_t w = nc > (t << tb) ? (nc - (t << tb) < (uint32_t)(KW) ? nc - (t << tb) : (uint32_t)(KW)) : 0;
#define CPERM(x) { lt[(x) & tmask] = (x) >> tb; }
        LANE_RANGE(qi, 0u, QCOUNT(qi), CPERM)
#undef CPERM
        uint32_t* cp = cperm + (size_t)o * lrows + (t << tb);
#pragma GCC unroll 1
        for (uint32_t m = 0; m < w; m++) cp[m] = lt[m];
    }
#else
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        const uint32_t nc = pe_rows(n, o, N);
#pragma GCC unroll 1
        for (uint32_t t = 0; t < ts; t++) {
            const uint32_t qi = N + t * N + o;
            const uint32_t w = nc - (t << tb) < (uint32_t)(TILE_W) ? nc - (t << tb) : (uint32_t)(TILE_W);
#define CPERM(x) { tile[(x) & tmask] = (x) >> tb; }
            WC_RANGE(qi, 0u, QCOUNT(qi), CPERM, WC_U)
#undef CPERM
            vx_fence();
            uint32_t* cp = cperm + (size_t)o * lrows + (t << tb);
#pragma GCC unroll 1
            for (uint32_t m = ln; m < w; m += GLANES) cp[m] = tile[m];
            vx_fence();
        }
    }
#endif
    return msgs;
}
#endif

void kernel_body(args_t* __UNIFORM__ a) {
    volatile uint32_t* bar = (volatile uint32_t*)a->bar_addr;
    const int blk = (int)blockIdx.x, tid = (int)threadIdx.x;
    const uint32_t sub = (uint32_t)tid / GLANES, ln = (uint32_t)tid % GLANES;
    const uint32_t wi = (uint32_t)blk * (SPLIT) + sub;
    const int NB = NBLK;
    uint32_t ep = 1;
    uint64_t msgs = 0;
    uint32_t rounds = 0, levels = 0;
    uint64_t prof[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#if ACTOR == 1
    aq_geometry(a->qs_addr, (uint64_t)(QS32) * 4ull, a->tails_addr, sizeof(msg_t), Q_CAP);
#if !BASELINE && !(ISA2 && SWQ)   // ISA2: the append off writes no L2 control register
    { volatile uint64_t* c = (volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE;   // overflow pool
      c[7] = (1ull << 60) | a->pool_addr;
      c[7] = (3ull << 60) | ((uint64_t)(OVF_CH) << 32) | (uint64_t)a->ovf_maxc;
      c[7] = (4ull << 60) | (uint64_t)a->ovf_nch;
      c[7] = (2ull << 60) | a->ctab_addr; }
#endif
#if SWQ
    swq_geometry(a->qs_addr, (uint64_t)(QS32) * 4ull, a->tails_addr, sizeof(msg_t), Q_CAP);
    swq_pool(a->pool_addr, a->ctab_addr, a->ovf_maxc, a->ovf_nch, a->swq_addr);
#endif
    volatile uint32_t* ws = (volatile uint32_t*)a->sums_addr + (size_t)wi * GLANES;
    const uint32_t ub = (uint32_t)blk;
    uint32_t* tile = (uint32_t*)__local_mem((size_t)(SPW) * (SPLIT) * 4) + sub * (uint32_t)(SPW);
#endif
    vx_fence();
    ep--; BARRIER();                                      // every warp has set the geometry
    const uint64_t t0 = csr_read(VX_CSR_MCYCLE);
#if ACTOR == 1
    // ---- phase 1: the levels
    msgs += phase1(a, ub, sub, ln, tid, NB, bar, &ep, &rounds, prof);
    levels = (rounds - 1) / 2;                            // the last round matched nothing, the one before it expanded
#if SWQ
    swq_round(a->N * (1u + a->ntile), wi * GLANES + ln, (uint32_t)NB * (SPLIT) * GLANES);
#else
    if (blk == 0) aq_round();                             // block-uniform: empty the queues for phase 2
#endif
    const uint64_t ta = csr_read(VX_CSR_MCYCLE);
#if !ROUND1
    // ---- phase 2: positions by level, then cperm by message
    level_hist(a, ub, sub, ln, tile, levels);
    vx_fence();
    BARRIER();
    level_prefix(a, wi, ln, ws, levels);
    vx_fence();
    BARRIER();
    { PA_W0
    send_cperm(a, ub, sub, ln, tile, levels);
    vx_fence();
    PA_W1(PH_W, pa_x) }
    BARRIER();
    if (blk == 0) aq_flush();                             // block-uniform: the CPERM partial lines
    { PA_W0
    msgs += recv_cperm(a, ub, sub, ln, tile);
    vx_fence();
    PA_W1(PH_W, pa_r) }
    BARRIER();
#endif
#else
    // ---- toposort_matrix_agi, one PE: a thread per queued row, grid stride, level by level
    const uint32_t n = a->numrows;
    const uint32_t* toff = (const uint32_t*)a->toff_addr;
    const uint32_t* tnz = (const uint32_t*)a->tnz_addr;
    uint32_t* cnt = (uint32_t*)a->cnt_addr;
    uint32_t* sum = (uint32_t*)a->sum_addr;
    uint32_t* rperm = (uint32_t*)a->rperm_addr;
    uint32_t* cperm = (uint32_t*)a->cperm_addr;
    volatile uint32_t* queue = (volatile uint32_t*)a->iq_addr;
    volatile uint32_t* ctr = (volatile uint32_t*)a->ctr_addr;   // [0] pivots, [1] queue end
    volatile uint32_t* vsum = sum;
    const uint32_t gtid = wi * GLANES + ln;
    const uint32_t stride = (uint32_t)(NBLK) * (SPLIT) * GLANES;
    uint32_t start = 0, end = a->iq0;
#pragma GCC unroll 1
    while (start != end && levels < (uint32_t)(MAXR)) {
        levels++;
#pragma GCC unroll 1
        for (uint32_t k = start + gtid; k < end; k += stride) {
            const uint32_t l_row = queue[k], S_col = vsum[l_row];
            const uint32_t pos = __atomic_fetch_add((uint32_t*)&ctr[0], 1u, __ATOMIC_RELAXED);   // claim our spot
            rperm[l_row] = n - 1 - pos;
            cperm[S_col] = n - 1 - pos;
            const uint32_t e1 = toff[S_col + 1];
#pragma GCC unroll 1
            for (uint32_t e = toff[S_col]; e < e1; e++) {
                const uint32_t S_row = tnz[e];
                const uint32_t old = __atomic_fetch_add(&cnt[S_row], 0xffffffffu, __ATOMIC_RELAXED);
                __atomic_fetch_sub(&sum[S_row], S_col, __ATOMIC_RELAXED);
                if (old == 2) { const uint32_t lp = __atomic_fetch_add((uint32_t*)&ctr[1], 1u, __ATOMIC_RELAXED); queue[lp] = S_row; }
            }
        }
        vx_fence();
        BARRIER();                                        // lgp_barrier
        start = end; end = ctr[1];
        BARRIER();                                        // lgp_reduce_add_l(end - start): every thread has read end
    }
    rounds = levels;
    const uint64_t ta = csr_read(VX_CSR_MCYCLE);
#endif
    const uint64_t t1 = csr_read(VX_CSR_MCYCLE);
    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[wi];
    ph->p1 = ta - t0; ph->p2 = t1 - ta; ph->total = t1 - t0; ph->rounds = rounds; ph->levels = levels;
    ph->t0 = t0; ph->t1 = t1; PH_MSGS(ph, (uint32_t)msgs);
    for (int x = 0; x < 8; x++) ph->prof[x] = prof[x];
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = NBLK;
    uint32_t block_dim = GLANES * (SPLIT);
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
