// Random permutation kernel (see common.h).
// ACTOR=1: one block = APB actors on SPLIT warps; warp `sub` serves actors o = blk*APB + sub + k*SPLIT, k < OW,
// so an actor's slots, darts and queues are only touched by its own warp. The warp's 16 lanes are the PE's threads.
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
LMEM_CHECK((TILE0) ? (SPW) * (SPLIT) * 4 : 0, NBLK);   // TILE0=0 allocates no tile
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
// HWBAR=1: the machine's global barrier (vx_barrier with the global bit: every warp of every core); 0: the
// software barrier above (an atomic counter the warps poll)
#if HWBAR
#define BARRIER() { vx_fence(); vx_barrier(0x80000000, NBLK); ISA2_L1INV(); }
#else
#define BARRIER() gbarrier_ep(bar, 0, tid, NB, ++ep)
#endif
// Zacas amocas.w: if (*p == expected) *p = desired; returns the old *p
static inline uint32_t amocas32(volatile uint32_t* p, uint32_t expected, uint32_t desired) {
    uint32_t rd = expected;
    asm volatile(".insn r 0x2f, 0x2, 0x14, %0, %1, %2" : "+r"(rd) : "r"(p), "r"(desired) : "memory");
    return rd;
}

#if ACTOR == 1
// warp-uniform control stores (never from one lane: SIMT collective rule)
static inline void aq_geometry(uint64_t qbase, uint64_t qstride, uint64_t tailbase, uint64_t wsize, uint64_t cap) {
#if !BASELINE && !(ISA2 && SWQ)   // ISA2: the append off writes no L2 control register
    volatile uint64_t* c = (volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE;
    c[0] = qbase; c[1] = qstride; c[2] = tailbase; c[3] = wsize; c[4] = cap;
#endif
}
static inline void aq_round(void) {
#if !BASELINE && !(ISA2 && SWQ)   // ISA2: the append off writes no L2 control register
 ((volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE)[6] = 1; 
#endif
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
#if MSG64
static_assert(NOTILE, "MSG64: only B off needs the 64-bit phase-1 message");
static inline void aq_appendm(uint32_t q, msg_t word) {   // phase 1 (THROW, REPLY): 8-byte messages
#if SWQ
    swq_put(q, word);
#elif ISA2
    aq_isa2_put(q, word);
#else
    *(volatile uint64_t*)(uintptr_t)((uint64_t)AQ_BASE + 64ull * ((uint64_t)q + 1ull)) = word;
#endif
}
#else
#define aq_appendm aq_append32
#endif
// one queue segment, warp cooperative: lane ln takes entries ln, ln+16, ...; U_ lines loaded first
// (ISA2: S_ = elements between lane ln's consecutive entries: QSTEP in a queue's region, GLANES in a chunk)
#define WC_SEGU(QP, N_, OP, U_, S_) { const uint32_t nf_ = (N_) / GLANES; uint32_t k_ = 0; \
    for (; k_ + (U_) <= nf_; k_ += (U_)) { msg_t w_[U_]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(U_); u_++) w_[u_] = (QP)[(k_ + u_) * S_]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(U_); u_++) OP(w_[u_]); } \
    for (; k_ < nf_; k_++) OP((QP)[k_ * S_]); \
    if (nf_ * GLANES + ln < (N_)) OP((QP)[nf_ * S_]); }
#define WC_SEG(QP, N_, OP, S_) WC_SEGU(QP, N_, OP, WC_U, S_)
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
#define FOR_MY_ACTORS(o) for (uint32_t ka_ = 0, o = blk * (uint32_t)(APB) + sub; ka_ < (uint32_t)(OW); ka_++, o += (SPLIT))
#define QUEUE_STATE32 const uint32_t* qs = (const uint32_t*)a->qs_addr; \
    const volatile uint32_t* tails = (const volatile uint32_t*)a->tails_addr; \
    const uint64_t* ctab = (const uint64_t*)a->ctab_addr; const uint32_t maxc = a->ovf_maxc;
#define QUEUE_STATE const msg_t* qs = (const msg_t*)a->qs_addr; \
    const volatile uint32_t* tails = (const volatile uint32_t*)a->tails_addr; \
    const uint64_t* ctab = (const uint64_t*)a->ctab_addr; const uint32_t maxc = a->ovf_maxc;

#define QS_W ((QS32) * 4 / (uint32_t)sizeof(*qs))   // queue stride in entries of the local queue view
// LANE_RECV: lane ln takes the warp's queues ln, ln + 16, ... (OW actors x T tiles); o = actor, t = tile
#define FOR_MY_LANE_QUEUES_RP(o, t, T) for (uint32_t qi_ = ln; qi_ < (uint32_t)(OW) * (T); qi_ += GLANES) \
    for (uint32_t o = blk * (uint32_t)(APB) + sub + (qi_ / (T)) * (SPLIT), t = qi_ % (T), once_ = 1; once_; once_ = 0)

// Phase 1 is ONE leaf (one prologue for all rounds). The queues are never emptied during phase 1: tails only
// grow, and each queue's consumer keeps a head (heads[q], written by its own warp), so a round costs two
// barriers and no aq_round. A REPLY of failure re-throws the dart at once (the selector's reply handler puts it
// back, its throw loop sends it again), so the drain of round k and the throws of round k+1 are one step.

// entries [H, TL) of queue QI: the region, then the overflow chunks (warp-uniform trip counts)
#define WC_RANGE(QI, H, TL, OP, U) { const uint32_t h_ = (H), tl_ = (TL); msgs += tl_ - h_; \
    if (h_ < (uint32_t)(Q_CAP)) { const uint32_t e_ = tl_ < (uint32_t)(Q_CAP) ? tl_ : (uint32_t)(Q_CAP); \
        const msg_t* q_ = QPTR(QI) + QIX(h_ + ln); WC_SEGU(q_, e_ - h_, OP, U, QSTEP) } \
    if (tl_ > (uint32_t)(Q_CAP)) { uint32_t p_ = (h_ > (uint32_t)(Q_CAP) ? h_ : (uint32_t)(Q_CAP)) - (uint32_t)(Q_CAP); \
        const uint32_t pe_ = tl_ - (uint32_t)(Q_CAP); const uint64_t* cr_ = ctab + (size_t)(QI) * maxc; \
        _Pragma("GCC unroll 1") while (p_ < pe_) { const uint32_t o_ = p_ % (uint32_t)(OVF_CH); \
            const uint32_t n_ = ((uint32_t)(OVF_CH) - o_) < (pe_ - p_) ? ((uint32_t)(OVF_CH) - o_) : (pe_ - p_); \
            const msg_t* qc_ = (const msg_t*)(uintptr_t)cr_[p_ / (uint32_t)(OVF_CH)] + o_ + ln; \
            WC_SEGU(qc_, n_, OP, U, GLANES) p_ += n_; } } }

static inline uint32_t pc32(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u); x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    return (((x + (x >> 4)) & 0x0f0f0f0fu) * 0x01010101u) >> 24;
}
// WC_RANGE with a segment macro SEG(QP, N_) in place of WC_SEGU
#define WC_RANGEX(QI, H, TL, SEG) { const uint32_t h_ = (H), tl_ = (TL); msgs += tl_ - h_; \
    if (h_ < (uint32_t)(Q_CAP)) { const uint32_t e_ = tl_ < (uint32_t)(Q_CAP) ? tl_ : (uint32_t)(Q_CAP); \
        const msg_t* q_ = QPTR(QI) + QIX(h_ + ln); SEG(q_, e_ - h_, QSTEP) } \
    if (tl_ > (uint32_t)(Q_CAP)) { uint32_t p_ = (h_ > (uint32_t)(Q_CAP) ? h_ : (uint32_t)(Q_CAP)) - (uint32_t)(Q_CAP); \
        const uint32_t pe_ = tl_ - (uint32_t)(Q_CAP); const uint64_t* cr_ = ctab + (size_t)(QI) * maxc; \
        _Pragma("GCC unroll 1") while (p_ < pe_) { const uint32_t o_ = p_ % (uint32_t)(OVF_CH); \
            const uint32_t n_ = ((uint32_t)(OVF_CH) - o_) < (pe_ - p_) ? ((uint32_t)(OVF_CH) - o_) : (pe_ - p_); \
            const msg_t* qc_ = (const msg_t*)(uintptr_t)cr_[p_ / (uint32_t)(OVF_CH)] + o_ + ln; \
            SEG(qc_, n_, GLANES) p_ += n_; } } }

static __attribute__((noinline)) uint64_t phase1(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln,
                                                 int tid, int NB, volatile uint32_t* bar, uint32_t* epp,
                                                 uint64_t* thrown, uint64_t* rounds, uint32_t* tile, uint64_t* prof) {
    QUEUE_STATE
    constexpr uint32_t lN = LN, lM = LMS, M = MM, seed = RP_SEED, ts = TS, jb = JB, tbs = TBS, rbase = RBASE;
    constexpr uint32_t tmask = (1u << tbs) - 1u, jmask = (1u << jb) - 1u, ssh = jb + tbs;
    uint32_t* heads = (uint32_t*)a->heads_addr;
    volatile uint32_t* fails = (volatile uint32_t*)a->fails_addr;
    uint32_t ep = *epp, nthrown = 0;
    uint64_t msgs = 0;
#if PROF
    uint64_t pr0 = 0, pr1 = 0, pr2 = 0, pr3 = 0, pr4 = 0, tp = csr_read(VX_CSR_MCYCLE);
#define TICK(v) { const uint64_t n_ = csr_read(VX_CSR_MCYCLE); v += n_ - tp; tp = n_; }
#else
#define TICK(v)
#endif
// THROW send of dart j of actor o, its k-th throw: {sender | dart | in-tile slot} to the queue of (owner, slot tile)
#define THROW(o, j, k) { const uint32_t r_ = rp_slot((j) * (uint32_t)(NASSUME) + (o), (k), seed, M), sl_ = r_ >> (AB); \
    aq_appendm((sl_ >> tbs) * (uint32_t)(NASSUME) + (r_ & ((NASSUME) - 1)), ((msg_t)(o) << ssh) | ((msg_t)(j) << tbs) | (sl_ & tmask)); \
    nthrown++; }
    PA_DECL
#if PROBE_APPLYT || ROUND1
    const uint64_t r10_ = csr_read(VX_CSR_MCYCLE);
#endif
    // round 0's throws: every dart
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
#pragma GCC unroll 1
        for (uint32_t j = ln; j < lN; j += GLANES) THROW(o, j, 0u)
    }
    vx_fence();
#if PROBE_APPLYT
    PH_W->pa_x += csr_read(VX_CSR_MCYCLE) - r10_;
#endif
#if ROUND1
    PH_W->r1 = csr_read(VX_CSR_MCYCLE) - r10_;            // probe: round 0's throws only
    *epp = ep; *thrown += nthrown; (void)bar; (void)tid; (void)NB; (void)rounds; (void)tile; (void)prof; (void)heads; (void)fails;
    return msgs;
#endif
    TICK(pr0)
    BARRIER();
    TICK(pr4)
#pragma GCC unroll 1
    for (uint32_t k = 0; k < (uint32_t)(MAXR); k++) {
#if PROBE_APPLYT
        const uint64_t rs0_ = csr_read(VX_CSR_MCYCLE);
#endif
        // THROW handler: claim the slot if it is empty, REPLY {landed | dart} to the sender
#if LANE_RECV
        // a lane per (owner, slot tile) queue
#if TILE0
        if (k < (uint32_t)(TILEK)) {
            // the lane's own TILE_W-slot tile: loaded, claimed in (lane-private: plain), written back by that lane
            uint32_t* lt = tile + ln * (uint32_t)(TILE_W);
#pragma GCC unroll 1
            FOR_MY_LANE_QUEUES_RP(o, t, ts) {
                const uint32_t qi = t * (uint32_t)(NASSUME) + o, h = heads[qi], tl = QCOUNT(qi);
                uint32_t* sl = (uint32_t*)a->tgt_addr + (size_t)o * lM + (t << tbs);
                const uint32_t w = lM - (t << tbs) < (uint32_t)(TILE_W) ? lM - (t << tbs) : (uint32_t)(TILE_W);
#pragma GCC unroll 1
                for (uint32_t m = 0; m < w; m++) lt[m] = sl[m];
#define SERVE(w) { const uint32_t j_ = ((w) >> tbs) & jmask, s_ = (w) >> ssh; uint32_t* p_ = &lt[(w) & tmask]; \
    const uint32_t won_ = *p_ == EMPTY; if (won_) *p_ = j_ * (uint32_t)(NASSUME) + s_; \
    aq_appendm(rbase + s_, (won_ << 31) | j_); }
                LANE_RANGE(qi, h, tl, SERVE)
#undef SERVE
                heads[qi] = tl;
#pragma GCC unroll 1
                for (uint32_t m = 0; m < w; m++) sl[m] = lt[m];
            }
        } else
#endif
        {
#pragma GCC unroll 1
            FOR_MY_LANE_QUEUES_RP(o, t, ts) {
                const uint32_t qi = t * (uint32_t)(NASSUME) + o, h = heads[qi], tl = QCOUNT(qi);
                volatile uint32_t* sl = (volatile uint32_t*)a->tgt_addr + (size_t)o * lM + (t << tbs);
#define SERVE(w) { const uint32_t j_ = ((w) >> tbs) & jmask, s_ = (w) >> ssh; \
    const uint32_t won_ = amocas32(&sl[(w) & tmask], EMPTY, j_ * (uint32_t)(NASSUME) + s_) == EMPTY; \
    PA_S(aq_appendm(rbase + s_, (won_ << 31) | j_)) }
#define SERVE_P(w) PA_H(SERVE(w))
                LANE_RANGE(qi, h, tl, SERVE_P)
#undef SERVE_P
#undef SERVE
                heads[qi] = tl;
            }
        }
#else
#pragma GCC unroll 1
        FOR_MY_ACTORS(o) {
            uint32_t nw = 0;                              // this lane's wins for actor o (CNTW)
#if TILE0
            if (k < (uint32_t)(TILEK)) {
                // rounds k < TILEK: the owner's slot tile is loaded into the scratchpad, takes the round's claims
                // with scratchpad atomics and is written out once (whole lines)
#pragma GCC unroll 1
                for (uint32_t t = 0; t < ts; t++) {
                    const uint32_t qi = t * (uint32_t)(NASSUME) + o, h = heads[qi], tl = QCOUNT(qi);
                    uint32_t* sl = (uint32_t*)a->tgt_addr + (size_t)o * lM + (t << tbs);
                    const uint32_t w = lM - (t << tbs) < (uint32_t)(TILE_W) ? lM - (t << tbs) : (uint32_t)(TILE_W);
#pragma GCC unroll 1
                    for (uint32_t m = 0; m < ((uint32_t)(TILE_W) + GLANES - 1) / GLANES; m++) { const uint32_t jj = ln + m * GLANES;
                        if ((uint32_t)(TILE_W) >= GLANES || jj < (uint32_t)(TILE_W)) tile[jj] = jj < w ? sl[jj] : EMPTY; }   // a tile may be narrower than the warp
                    vx_fence();
#define SERVE(w) { const uint32_t j_ = ((w) >> tbs) & jmask, s_ = (w) >> ssh; \
    const uint32_t won_ = amocas32(&tile[(w) & tmask], EMPTY, j_ * (uint32_t)(NASSUME) + s_) == EMPTY; \
    nw += won_; aq_appendm(rbase + s_, (won_ << 31) | j_); }
                    WC_RANGE(qi, h, tl, SERVE, WC_U)
#undef SERVE
                    heads[qi] = tl;
                    vx_fence();
#pragma GCC unroll 1
                    for (uint32_t m = 0; m < (w + GLANES - 1) / GLANES; m++) { const uint32_t jj = ln + m * GLANES; if (jj < w) sl[jj] = tile[jj]; }
                    vx_fence();
                }
            } else
#endif
#if SKIPE
#pragma GCC unroll 1
            for (uint32_t t0 = 0; t0 < ts; t0 += GLANES) {
                // lane b fetches tile queue t0+b's head and tail (no branch); the ballot names the non-empty ones
                const uint32_t tq = t0 + ln < ts ? t0 + ln : ts - 1, qb = tq * (uint32_t)(NASSUME) + o;
                const uint32_t ne = (uint32_t)vx_vote_ballot((int)(t0 + ln < ts && QCOUNT(qb) != heads[qb]));
#pragma GCC unroll 1
                for (uint32_t m = ne; m != 0; m &= m - 1) {      // the set bits only (uniform)
                const uint32_t t = t0 + pc32((m & (0u - m)) - 1u);
#else
#pragma GCC unroll 1
            for (uint32_t t = 0; t < ts; t++) {
#endif
                const uint32_t qi = t * (uint32_t)(NASSUME) + o, h = heads[qi], tl = QCOUNT(qi);
                volatile uint32_t* sl = (volatile uint32_t*)a->tgt_addr + (size_t)o * lM + (t << tbs);
#if PIPE
#define SV_CLAIM(w, r) { r = amocas32(&sl[(w) & tmask], EMPTY, (((w) >> tbs) & jmask) * (uint32_t)(NASSUME) + ((w) >> ssh)); }
#define SV_REPLY(w, r) { const uint32_t won_ = (r) == EMPTY; nw += won_; aq_appendm(rbase + ((w) >> ssh), (won_ << 31) | (((w) >> tbs) & jmask)); }
#define SV_SEG(QP, N_, S_) { const uint32_t nf_ = (N_) / GLANES; uint32_t k_ = 0; \
    for (; k_ + (WC_U) <= nf_; k_ += (WC_U)) { uint32_t w_[WC_U], r_[WC_U]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(WC_U); u_++) w_[u_] = (QP)[(k_ + u_) * S_]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(WC_U); u_++) SV_CLAIM(w_[u_], r_[u_]) \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(WC_U); u_++) SV_REPLY(w_[u_], r_[u_]) } \
    for (; k_ < nf_; k_++) { const uint32_t w0_ = (QP)[k_ * S_]; uint32_t r0_; SV_CLAIM(w0_, r0_) SV_REPLY(w0_, r0_) } \
    if (nf_ * GLANES + ln < (N_)) { const uint32_t w0_ = (QP)[nf_ * S_]; uint32_t r0_; SV_CLAIM(w0_, r0_) SV_REPLY(w0_, r0_) } }
                WC_RANGEX(qi, h, tl, SV_SEG)
#undef SV_SEG
#undef SV_REPLY
#undef SV_CLAIM
#else
#if CLAIM
#define SERVE(w) { const uint32_t j_ = ((w) >> tbs) & jmask, s_ = (w) >> ssh; \
    const uint32_t won_ = amocas32(&sl[(w) & tmask], EMPTY, j_ * (uint32_t)(NASSUME) + s_) == EMPTY; \
    nw += won_; aq_appendm(rbase + s_, (won_ << 31) | j_); }
#else
#define SERVE(w) { const uint32_t j_ = ((w) >> tbs) & jmask, s_ = (w) >> ssh, key_ = j_ * (uint32_t)(NASSUME) + s_; \
    volatile uint32_t* p_ = &sl[(w) & tmask]; \
    if (*p_ == EMPTY) *p_ = key_; \
    const uint32_t won_ = *p_ == key_; \
    nw += won_; aq_appendm(rbase + s_, (won_ << 31) | j_); }
#endif
                WC_RANGE(qi, h, tl, SERVE, WC_U)
#undef SERVE
#endif
                heads[qi] = tl;
            }
#if SKIPE
            }
#endif
#if CNTW
            if (nw) __atomic_fetch_add((uint32_t*)a->cnts_addr + o, nw, __ATOMIC_RELAXED);
#else
            (void)nw;
#endif
        }
#endif
#if PROBE_APPLYT
        PA_FLUSH(PH_W)
        PH_W->pa_r += csr_read(VX_CSR_MCYCLE) - rs0_;
#endif
        vx_fence();
#if PROF
        if (k == 0) TICK(pr1) else TICK(pr2)
#endif
        BARRIER();
        TICK(pr4)
        // REPLY handler: a landed dart is done; a failed one is thrown again at once (its (k+1)-th throw)
        uint32_t nf = 0;
#if PROBE_APPLYT
        const uint64_t rr0_ = csr_read(VX_CSR_MCYCLE);
#endif
#if LANE_RECV
#pragma GCC unroll 1
        FOR_MY_LANE_QUEUES_RP(o, t, 1u) {
            const uint32_t qi = rbase + o, h = heads[qi], tl = QCOUNT(qi);
#define REPLY(w) { if (!((w) >> 31)) { const uint32_t j_ = (w) & jmask; THROW(o, j_, k + 1) nf++; } }
#define REPLY_P(w) PA_H(PA_S(REPLY(w)))   // the reply handler applies nothing: a failed dart is thrown again
            LANE_RANGE(qi, h, tl, REPLY_P)
#undef REPLY_P
#else
#pragma GCC unroll 1
        FOR_MY_ACTORS(o) {
            const uint32_t qi = rbase + o, h = heads[qi], tl = QCOUNT(qi);
#define REPLY(w) { if (!((w) >> 31)) { const uint32_t j_ = (w) & jmask; THROW(o, j_, k + 1) nf++; } }
            WC_RANGE(qi, h, tl, REPLY, 1)
#endif
#undef REPLY
            heads[qi] = tl;
        }
        if (nf) __atomic_fetch_add((uint32_t*)&fails[k], nf, __ATOMIC_RELAXED);
#if PROBE_APPLYT
        PA_FLUSH(PH_W)
        PH_W->pa_r += csr_read(VX_CSR_MCYCLE) - rr0_;
#endif
        vx_fence();
        TICK(pr3)
        BARRIER();
        TICK(pr4)
        (*rounds)++;
        if (fails[k] == 0) break;                         // uniform: every thread reads the same count
    }
#undef THROW
#if PROF
    prof[0] = pr0; prof[1] = pr1; prof[2] = pr2; prof[3] = pr3; prof[4] = pr4;
#else
    (void)prof;
#endif
#undef TICK
    *epp = ep; *thrown += nthrown;
    return msgs;
}
// phase 2: my actors' filled slots, counted (cnts[o])
static __attribute__((noinline)) void count_round(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln,
                                                  volatile uint32_t* ws) {
    constexpr uint32_t lM = LMS, B = (lM + GLANES - 1) / GLANES;
    const uint32_t b0 = ln * B < lM ? ln * B : lM, b1 = b0 + B < lM ? b0 + B : lM;
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        const uint32_t* sl = (const uint32_t*)a->tgt_addr + (size_t)o * lM;
        uint32_t c = 0;
#pragma GCC unroll 1
        for (uint32_t i = b0; i < b1; i++) c += sl[i] != EMPTY;
        ws[ln] = c;
        vx_fence();
        uint32_t tot = 0;
#pragma GCC unroll 1
        for (uint32_t x = 0; x < (uint32_t)(GLANES); x++) tot += ws[x];
        ((uint32_t*)a->cnts_addr)[o] = tot;
        vx_fence();
    }
}
// phase 2 send: offset = the counts of the actors before me; my i-th filled slot to PE (offset + i) % P
static __attribute__((noinline)) void send_perm(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln,
                                                volatile uint32_t* ws) {
    constexpr uint32_t lM = LMS, B = (lM + GLANES - 1) / GLANES;
    const uint32_t b0 = ln * B < lM ? ln * B : lM, b1 = b0 + B < lM ? b0 + B : lM;
    const volatile uint32_t* cnts = (const volatile uint32_t*)a->cnts_addr;
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        // lane ln sums actors ln * PX .. ln * PX + PX - 1 (PX = 16 up to 256 actors, N / 16 above)
        constexpr uint32_t PX = (uint32_t)(NASSUME) / GLANES > GLANES ? (uint32_t)(NASSUME) / GLANES : GLANES;
        uint32_t part = 0;
#pragma GCC unroll 1
        for (uint32_t x = 0; x < PX; x++) { const uint32_t p = ln * PX + x; part += p < o ? cnts[p] : 0; }
        ws[ln] = part;
        vx_fence();
        uint32_t offset = 0;
#pragma GCC unroll 1
        for (uint32_t x = 0; x < (uint32_t)(GLANES); x++) offset += ws[x];
        vx_fence();
        const uint32_t* sl = (const uint32_t*)a->tgt_addr + (size_t)o * lM;
#if SEND1
        // one coalesced pass: lane ln takes slot m*16+ln; the ballot of filled slots gives each lane its rank
        uint32_t base = offset;
        const uint32_t below = (1u << ln) - 1u;
#pragma GCC unroll 1
        for (uint32_t m = 0; m < (lM + GLANES - 1) / GLANES; m++) {
            const uint32_t i = m * GLANES + ln, v = i < lM ? sl[i] : EMPTY, f = v != EMPTY;
            const uint32_t b = (uint32_t)vx_vote_ballot((int)f);
            if (f) aq_append32((base + pc32(b & below)) & ((NASSUME) - 1), v);
            base += pc32(b);
        }
        (void)b0; (void)b1;
#else
        uint32_t c = 0;
#pragma GCC unroll 1
        for (uint32_t i = b0; i < b1; i++) c += sl[i] != EMPTY;
        ws[ln] = c;
        vx_fence();
        uint32_t base = offset;
#pragma GCC unroll 1
        for (uint32_t x = 0; x < (uint32_t)(GLANES); x++) base += x < ln ? ws[x] : 0;
#pragma GCC unroll 1
        for (uint32_t i = b0; i < b1; i++) {
            const uint32_t v = sl[i];
            if (v != EMPTY) { aq_append32(base & ((NASSUME) - 1), v); base++; }
        }
#endif
        vx_fence();
    }
}
// phase 2 handler: lperm[pos++] = val, pos = the message's place in my queue (arrival order)
static __attribute__((noinline)) uint32_t recv_perm(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln) {
    QUEUE_STATE32
    constexpr uint32_t lN = LN;
    uint32_t msgs = 0;
    PA_DECL
#if LANE_RECV
    // a lane per queue: the whole queue copied by that lane, in arrival order
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES_RP(o, t, 1u) {
        uint32_t* lp = (uint32_t*)a->perm_addr + (size_t)o * lN;
        uint32_t pos = 0;
#define PUT(x) { lp[pos++] = (x); }
#define PUT_P(x) PA_H(PUT(x))
        LANE_RANGE(o, 0u, QCOUNT(o), PUT_P)
#undef PUT_P
#undef PUT
    }
    PA_FLUSH(PH_W)
    (void)ctab; (void)maxc;
    return msgs;
#endif
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        uint32_t* lp = (uint32_t*)a->perm_addr + (size_t)o * lN;
        const uint32_t tl = QCOUNT(o);
        msgs += tl;
        const uint32_t n0 = tl < (uint32_t)(Q_CAP) ? tl : (uint32_t)(Q_CAP);
        const uint32_t* q = QPTR(o);
#pragma GCC unroll 1
        for (uint32_t i = ln; i < n0; i += GLANES) lp[i] = q[QIX(i)];
        uint32_t r = tl - n0, pos = n0;
        const uint64_t* cr = ctab + (size_t)o * maxc;
#pragma GCC unroll 1
        for (uint32_t c = 0; r != 0; c++) {
            const uint32_t n = r < (uint32_t)(OVF_CH) ? r : (uint32_t)(OVF_CH);
            const uint32_t* qc = (const uint32_t*)(uintptr_t)cr[c];
#pragma GCC unroll 1
            for (uint32_t i = ln; i < n; i += GLANES) lp[pos + i] = qc[i];
            pos += n; r -= n;
        }
    }
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
    uint64_t msgs = 0, thrown = 0, rounds = 0;
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
#if TILE0
    uint32_t* tile = (uint32_t*)__local_mem((size_t)(SPW) * (SPLIT) * 4) + sub * (uint32_t)(SPW);
#else
    uint32_t* tile = nullptr;
#endif
#endif
    vx_fence();
    ep--; BARRIER();                                      // every warp has set the geometry
    const uint64_t t0 = csr_read(VX_CSR_MCYCLE);
#if ACTOR == 1
    // ---- phase 1: throw the darts in rounds until a round has no failure
    uint64_t prof[5] = {0, 0, 0, 0, 0};
    msgs += phase1(a, ub, sub, ln, tid, NB, bar, &ep, &thrown, &rounds, tile, prof);
#if SWQ
    swq_round(a->rbase + a->N, wi * GLANES + ln, (uint32_t)NB * (SPLIT) * GLANES);
#else
    if (blk == 0) aq_round();                             // block-uniform: empty the queues for phase 2
#endif
    vx_fence();
    BARRIER();
#if MSG64
    aq_geometry(a->qs_addr, (uint64_t)(QS32) * 4ull, a->tails_addr, 4, Q_CAP);   // phase 2: 4-byte messages
#if SWQ
    swq_geometry(a->qs_addr, (uint64_t)(QS32) * 4ull, a->tails_addr, 4, Q_CAP);
#endif
#endif
    const uint64_t ta = csr_read(VX_CSR_MCYCLE);
#if !ROUND1
    // ---- phase 2: pack my filled slots, send the i-th to PE (offset + i) % P
#if !CNTW
    count_round(a, ub, sub, ln, ws);
    vx_fence();
    BARRIER();
#endif
    { PA_W0
    send_perm(a, ub, sub, ln, ws);
    vx_fence();
    PA_W1(PH_W, pa_x) }
    BARRIER();
    { PA_W0
    msgs += recv_perm(a, ub, sub, ln);
    vx_fence();
    PA_W1(PH_W, pa_r) }
    BARRIER();
#if SWQ
    swq_round(a->rbase + a->N, wi * GLANES + ln, (uint32_t)NB * (SPLIT) * GLANES);
#else
    if (blk == 0) aq_round();                             // block-uniform: charge the partial lines
#endif
    vx_fence();
    BARRIER();
#endif
#else
    // ---- copied_rand_permp_agi, one PE: a thread per item, grid stride
    volatile uint32_t* tgt = (volatile uint32_t*)a->tgt_addr;
    uint32_t* perm = (uint32_t*)a->perm_addr;
    const uint32_t gtid = wi * GLANES + ln;
    const uint32_t stride = (uint32_t)(NBLK) * (SPLIT) * GLANES;
    const uint32_t nw = stride / GLANES;
    constexpr uint32_t n = NN, M = MM, seed = RP_SEED;
#pragma GCC unroll 1
    for (uint32_t i = gtid; i < n; i += stride) {         // throw the darts until each hits
#pragma GCC unroll 1
        for (uint32_t k = 0;; k++) {
            thrown++;
            if (amocas32(&tgt[rp_slot(i, k, seed, M)], EMPTY, i) == EMPTY) break;
        }
    }
    vx_fence();
    BARRIER();
    const uint64_t ta = csr_read(VX_CSR_MCYCLE);
    // pack the filled slots in slot order: thread gtid takes block gtid of the slots; block sums, then warp
    // sums, give the bases
    volatile uint32_t* sums = (volatile uint32_t*)a->sums_addr;
    volatile uint32_t* wsum = sums + stride;
    const uint32_t B = (M + stride - 1) / stride;
    const uint32_t b0 = gtid * B < M ? gtid * B : M, b1 = b0 + B < M ? b0 + B : M;
    uint32_t s = 0;
#pragma GCC unroll 1
    for (uint32_t i = b0; i < b1; i++) s += tgt[i] != EMPTY;
    sums[gtid] = s;
    vx_fence();
    BARRIER();
    uint32_t wt = 0, lb = 0;
#pragma GCC unroll 1
    for (uint32_t x = 0; x < (uint32_t)(GLANES); x++) { const uint32_t v = sums[wi * GLANES + x]; wt += v; lb += x < ln ? v : 0; }
    wsum[wi] = wt;
    vx_fence();
    BARRIER();
    uint32_t pos = lb;
#pragma GCC unroll 1
    for (uint32_t x = 0; x < wi; x++) pos += wsum[x];
#pragma GCC unroll 1
    for (uint32_t i = b0; i < b1; i++) { const uint32_t v = tgt[i]; if (v != EMPTY) perm[pos++] = v; }
    (void)nw;
    vx_fence();
    BARRIER();
#endif
    const uint64_t t1 = csr_read(VX_CSR_MCYCLE);
    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[wi];
    // msgs (queue tails) is the same in every lane; thrown is each lane's own, summed into the record
    ph->p1 = ta - t0; ph->p2 = t1 - ta; ph->total = t1 - t0; ph->rounds = rounds; ph->t0 = t0; ph->t1 = t1;
    PH_MSGS(ph, (uint32_t)msgs);
#if ACTOR == 1 && PROF
    for (int x = 0; x < 5; x++) ph->prof[x] = prof[x];
#endif
    __atomic_fetch_add(&ph->throws, thrown, __ATOMIC_RELAXED);
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = NBLK;
    uint32_t block_dim = GLANES * (SPLIT);
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
