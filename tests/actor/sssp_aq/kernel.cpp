// SSSP (Bellman-Ford) kernel (see common.h).
// ACTOR=1: one block = 16 actors on SPLIT warps; warp `sub` serves actors o = blk*16 + sub + k*SPLIT, k < OW,
// so an actor's vertices, weights, senders and queues are only touched by its own warp. The warp's 16 lanes are the
// PE's threads.
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
#include "../probe_apply.h"
static_assert(!ROUND1 || NOTILE || LANE_RECV, "ROUND1: the baseline's drain path only");

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
static inline double as_f64(uint64_t b) { double d; __builtin_memcpy(&d, &b, 8); return d; }
static inline uint64_t as_u64(double d) { uint64_t b; __builtin_memcpy(&b, &d, 8); return b; }

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
#include "../swq.h"
static inline void aq_append64(uint32_t q, uint64_t word) {
#if SWQ
    swq_put(q, word);
#elif ISA2
    aq_isa2_put(q, word);
#else
    *(volatile uint64_t*)(uintptr_t)((uint64_t)AQ_BASE + 64ull * ((uint64_t)q + 1ull)) = word;
#endif
}
// one queue segment, warp cooperative: lane ln takes entries ln, ln+16, ...; U_ lines loaded first (the remainder
// loop exists only for U_ > 1)
// (ISA2: S_ = elements between lane ln's consecutive entries: QSTEP in a queue's region, GLANES in a chunk)
#define WC_SEGU_S(QP, N_, OP, U_, S_) { const uint32_t nf_ = (N_) / GLANES; uint32_t k_ = 0; \
    _Pragma("GCC unroll 1") for (; k_ + (U_) <= nf_; k_ += (U_)) { uint64_t w_[U_]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(U_); u_++) w_[u_] = (QP)[(k_ + u_) * S_]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(U_); u_++) OP(w_[u_]); } \
    if ((U_) > 1) { _Pragma("GCC unroll 1") for (; k_ < nf_; k_++) OP((QP)[k_ * S_]); } \
    if (nf_ * GLANES + ln < (N_)) OP((QP)[nf_ * S_]); }
#define WC_SEGU(QP, N_, OP, U_) WC_SEGU_S(QP, N_, OP, U_, GLANES)
// entries [H, TL) of queue QI, segment by segment (the region, then each overflow chunk): one WC_SEGU site
#if AQ_ON
// ISA2 layout: a region segment steps line by line (QSTEP), a chunk segment word by word
#define WC_RANGE(QI, H, TL, OP, U) { uint32_t p_ = (H); const uint32_t tl_ = (TL); msgs += tl_ - p_; \
    const uint64_t* q0_ = QPTR(QI); const uint64_t* cr_ = ctab + (size_t)(QI) * maxc; \
    _Pragma("GCC unroll 1") while (p_ < tl_) { \
        const uint64_t* bq_; uint32_t e_; size_t s_; \
        if (p_ < (uint32_t)(Q_CAP)) { bq_ = q0_ + QIX(p_ + ln); e_ = (uint32_t)(Q_CAP); s_ = QSTEP; } \
        else { const uint32_t x_ = p_ - (uint32_t)(Q_CAP); \
            bq_ = (const uint64_t*)(uintptr_t)cr_[x_ / (uint32_t)(OVF_CH)] + x_ % (uint32_t)(OVF_CH) + ln; \
            e_ = p_ + ((uint32_t)(OVF_CH) - x_ % (uint32_t)(OVF_CH)); s_ = GLANES; } \
        if (e_ > tl_) e_ = tl_; \
        WC_SEGU_S(bq_, e_ - p_, OP, U, s_) \
        p_ = e_; } }
#else
#define WC_RANGE(QI, H, TL, OP, U) { uint32_t p_ = (H); const uint32_t tl_ = (TL); msgs += tl_ - p_; \
    const uint64_t* q0_ = qs + (size_t)(QI) * (QS64); const uint64_t* cr_ = ctab + (size_t)(QI) * maxc; \
    _Pragma("GCC unroll 1") while (p_ < tl_) { \
        const uint64_t* b_; uint32_t e_; \
        if (p_ < (uint32_t)(Q_CAP)) { b_ = q0_ + p_; e_ = (uint32_t)(Q_CAP); } \
        else { const uint32_t x_ = p_ - (uint32_t)(Q_CAP); \
            b_ = (const uint64_t*)(uintptr_t)cr_[x_ / (uint32_t)(OVF_CH)] + x_ % (uint32_t)(OVF_CH); \
            e_ = p_ + ((uint32_t)(OVF_CH) - x_ % (uint32_t)(OVF_CH)); } \
        if (e_ > tl_) e_ = tl_; \
        const uint64_t* bq_ = b_ + ln; \
        WC_SEGU(bq_, e_ - p_, OP, U) \
        p_ = e_; } }
#endif
// entries [H, TL) of queue QI drained by the calling lane alone (LANE_RECV)
#define LANE_RANGE(QI, H, TL, OP) { uint32_t p_ = (H); const uint32_t tl_ = (TL); msgs += tl_ - p_; \
    const uint64_t* q0_ = QPTR(QI); const uint64_t* cr_ = ctab + (size_t)(QI) * maxc; \
    _Pragma("GCC unroll 1") for (; p_ < tl_; p_++) { \
        const uint64_t* b_; \
        if (p_ < (uint32_t)(Q_CAP)) b_ = q0_ + QIX(p_); \
        else { const uint32_t x_ = p_ - (uint32_t)(Q_CAP); \
            b_ = (const uint64_t*)(uintptr_t)cr_[x_ / (uint32_t)(OVF_CH)] + x_ % (uint32_t)(OVF_CH); } \
        OP(*b_); } }
#define QS_W QS64   // queue stride in entries (QPTR)
#define FOR_MY_ACTORS(o) for (uint32_t ka_ = 0, o = blk * GLANES + sub; ka_ < (uint32_t)(OW); ka_++, o += (SPLIT))

// Round r, ONE leaf over the warp's actors (pointers from args, sizes compile time). Per actor: drain parity r & 1,
// tile by tile (a tile with no messages is skipped): load the tile of tent into the scratchpad, the handler's min on
// it, then compare and write back as whole lines; the vertices that dropped (warp ballot) are this round's senders,
// which relax their out-edges into parity (r & 1) ^ 1. Round 0's sender is the source.
// returns (msgs drained << 32) | this lane's appends
static __attribute__((noinline)) uint64_t ss_round(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln,
                                                   uint32_t r, volatile uint64_t* tile, uint64_t* pf) {
    const uint64_t* qs = (const uint64_t*)a->qs_addr;
    const volatile uint32_t* tails = (const volatile uint32_t*)a->tails_addr;
    const uint64_t* ctab = (const uint64_t*)a->ctab_addr;
    const uint32_t maxc = a->ovf_maxc;
    uint32_t* heads = (uint32_t*)a->heads_addr;
    const uint32_t* nbr = (const uint32_t*)a->nbr_addr;
    const double* wt = (const double*)a->wt_addr;
    constexpr uint32_t N = NASSUME, vl = VL, nt = NT, tb = TB, tw = KW;
    constexpr uint64_t tmask = (uint64_t)(KW) - 1;
    const uint32_t p = r & 1u, nb = (p ^ 1u) * nt;
    uint32_t msgs = 0, sent = 0;
#if NOTILE || LANE_RECV
    // ablation drain: every queue of the warp first, the dropped vertices counted into fcnt[o] (then the send below)
    uint32_t* fcnt = (uint32_t*)a->cnt_addr;
    uint32_t* stamp = (uint32_t*)a->stamp_addr;
    const uint64_t td = csr_read(VX_CSR_MCYCLE);
    PA_DECL
    if (r != 0) {
#if NOTILE
        // handler: an L2 atomic min on tent; a vertex that dropped is queued once a round (round stamp)
#define GMIN(x) { const uint32_t l_ = (uint32_t)((x) & tmask); const uint64_t v_ = (x) & ~tmask; \
        if (v_ < __atomic_fetch_min(&te_[l_], v_, __ATOMIC_RELAXED) && __atomic_exchange_n(&st_[l_], r, __ATOMIC_RELAXED) != r) \
            fr_[__atomic_fetch_add(&fcnt[o], 1u, __ATOMIC_RELAXED)] = l_; }
#if LANE_RECV
#pragma GCC unroll 1
        FOR_MY_LANE_QUEUES(o, t, 1u) {
#else
#pragma GCC unroll 1
        FOR_MY_ACTORS(o) {
#endif
            uint64_t* te_ = (uint64_t*)a->tent_addr + (size_t)o * vl;
            uint32_t* st_ = stamp + (size_t)o * vl;
            uint32_t* fr_ = (uint32_t*)a->fr_addr + (size_t)o * vl;
            const uint32_t qi = p * N + o, h = heads[qi], tl = QCOUNT(qi);
#if LANE_RECV
#define GMIN_P(x) PA_H(GMIN(x))
            LANE_RANGE(qi, h, tl, GMIN_P)
#undef GMIN_P
#else
            WC_RANGE(qi, h, tl, GMIN, WC_U)
#endif
            heads[qi] = tl;
        }
#undef GMIN
#else
        // LANE_RECV with tiles: a lane per tile queue, its own KW-weight tile loaded from tent, min applied, compared
        // and written back by that lane
        volatile uint64_t* lt = tile + ln * (uint32_t)(KW);
#define LMIN(x) { const uint32_t i_ = (uint32_t)((x) & tmask); const uint64_t v_ = (x) & ~tmask; if (v_ < lt[i_]) lt[i_] = v_; }
#pragma GCC unroll 1
        FOR_MY_LANE_QUEUES(o, t, nt) {
            const uint32_t qi = (p * nt + t) * N + o, h = heads[qi], tl = QCOUNT(qi);
            if (h != tl) {
                uint64_t* te_ = (uint64_t*)a->tent_addr + (size_t)o * vl;
                uint32_t* fr_ = (uint32_t*)a->fr_addr + (size_t)o * vl;
                const uint32_t base = t * (uint32_t)(KW), w = vl - base < (uint32_t)(KW) ? vl - base : (uint32_t)(KW);
#pragma GCC unroll 1
                for (uint32_t j = 0; j < w; j++) lt[j] = te_[base + j];
                LANE_RANGE(qi, h, tl, LMIN)
                heads[qi] = tl;
#pragma GCC unroll 1
                for (uint32_t j = 0; j < w; j++) { const uint64_t nv = lt[j];
                    if (nv < te_[base + j]) { te_[base + j] = nv; fr_[__atomic_fetch_add(&fcnt[o], 1u, __ATOMIC_RELAXED)] = base + j; } }
            }
        }
#undef LMIN
#endif
    }
#if PROBE_APPLYT
    { phase_cycles_t* ph_ = &((phase_cycles_t*)a->phase_addr)[blk * (SPLIT) + sub]; PA_FLUSH(ph_) }
#endif
    vx_fence();
    pf[0] += csr_read(VX_CSR_MCYCLE) - td;
#endif
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        uint64_t* tent = (uint64_t*)a->tent_addr + (size_t)o * vl;
        uint32_t* fr = (uint32_t*)a->fr_addr + (size_t)o * vl;
        const uint64_t t0 = csr_read(VX_CSR_MCYCLE);
        uint32_t fc = 0;
#if NOTILE || LANE_RECV
#if ROUND1
        if (r == 0) fc = vl;                               // probe: every vertex sends in round 0
#else
        if (r == 0) fc = o == a->src % N ? 1u : 0u;       // uniform: the source (placed in fr by the host)
#endif
        else fc = fcnt[o];                                 // the drain above counted this actor's dropped vertices
        vx_fence();
        fcnt[o] = 0;                                       // uniform store, after every lane read it
        (void)tile; (void)heads; (void)tw;
#else
        if (r == 0) fc = o == a->src % N ? 1u : 0u;       // uniform: the source (placed in fr by the host)
        else {
#pragma GCC unroll 1
            for (uint32_t t = 0; t < nt; t++) {
                const uint32_t qi = (p * nt + t) * N + o, h = heads[qi], tl = QCOUNT(qi);
                if (h == tl) continue;                    // uniform
                const uint32_t base = t * tw, w = vl - base < tw ? vl - base : tw;
#pragma GCC unroll 1
                for (uint32_t m = 0; m < (w + GLANES - 1) / GLANES; m++) { const uint32_t j = ln + m * GLANES; if (j < w) tile[j] = tent[base + j]; }
                vx_fence();
                // handler: tent(w) = min(tent(w), tw)
#define TMIN(x) __atomic_fetch_min((uint64_t*)&tile[(x) & tmask], (x) & ~tmask, __ATOMIC_RELAXED);
                WC_RANGE(qi, h, tl, TMIN, WC_U)
#undef TMIN
                heads[qi] = tl;
                vx_fence();
                // compare and write back; the dropped vertices are compacted into fr with a ballot (no divergence before it)
#pragma GCC unroll 1
                for (uint32_t m = 0; m < (w + GLANES - 1) / GLANES; m++) {
                    const uint32_t j = ln + m * GLANES, jj = j < w ? j : w - 1;
                    const uint64_t nv = tile[jj], ov = tent[base + jj];
                    const int ch = (j < w) & (nv < ov);
                    const uint32_t mask = (uint32_t)vx_vote_ballot(ch);
                    if (ch) { tent[base + j] = nv; fr[fc + (uint32_t)__builtin_popcount(mask & ((1u << ln) - 1u))] = base + j; }
                    fc += (uint32_t)__builtin_popcount(mask);
                }
                vx_fence();
            }
        }
#endif
        const uint64_t t1 = csr_read(VX_CSR_MCYCLE);
        // the senders: a warp per vertex, lanes over its out-edges, {tw rounded | in-tile index} to (owner, tile)
        const uint32_t* off = (const uint32_t*)a->off_addr + (size_t)o * (vl + 1);
#if SEND_LANE
#pragma GCC unroll 1
        for (uint32_t i = ln; i < fc; i += GLANES) {
            const uint32_t l = ROUND1 ? i : fr[i], e1 = off[l + 1];
            const double d = as_f64(tent[l]);
#pragma GCC unroll 1
            for (uint32_t e = off[l]; e < e1; e++) {
#else
#pragma GCC unroll 1
        for (uint32_t i = 0; i < fc; i++) {
            const uint32_t l = ROUND1 ? i : fr[i], e1 = off[l + 1];
            const double d = as_f64(tent[l]);
#pragma GCC unroll 1
            for (uint32_t e = off[l] + ln; e < e1; e += GLANES) {
#endif
                const uint32_t v = nbr[e], vlc = v / N;
                const uint64_t x = (as_u64(d + wt[e]) + (1ull << (tb - 1))) & ~tmask;
                aq_append64((nb + (vlc >> tb)) * N + (v & (N - 1)), x | (vlc & tmask)); sent++;
            }
        }
        pf[0] += t1 - t0; pf[1] += csr_read(VX_CSR_MCYCLE) - t1;
    }
    return ((uint64_t)msgs << 32) | sent;
}
#endif

void kernel_body(args_t* __UNIFORM__ a) {
    volatile uint32_t* bar = (volatile uint32_t*)a->bar_addr;
    const int blk = (int)blockIdx.x, tid = (int)threadIdx.x;
    const uint32_t sub = (uint32_t)tid / GLANES, ln = (uint32_t)tid % GLANES;
    const uint32_t wi = (uint32_t)blk * (SPLIT) + sub;
    const int NB = (int)((uint32_t)(NASSUME) / GLANES);
    uint32_t ep = 0, r = 0;
    uint64_t msgs = 0, pf[2] = {0, 0};
    (void)ep;
#if ACTOR == 1
    aq_geometry(a->qs_addr, (uint64_t)(QS64) * 8ull, a->tails_addr, 8, Q_CAP);
#if !BASELINE && !(ISA2 && SWQ)   // ISA2: the append off writes no L2 control register
    { volatile uint64_t* c = (volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE;   // overflow pool
      c[7] = (1ull << 60) | a->pool_addr;
      c[7] = (3ull << 60) | ((uint64_t)(OVF_CH) << 32) | (uint64_t)a->ovf_maxc;
      c[7] = (4ull << 60) | (uint64_t)a->ovf_nch;
      c[7] = (2ull << 60) | a->ctab_addr; }
#endif
#if SWQ
    swq_geometry(a->qs_addr, (uint64_t)(QS64) * 8ull, a->tails_addr, 8, Q_CAP);
    swq_pool(a->pool_addr, a->ctab_addr, a->ovf_maxc, a->ovf_nch, a->swq_addr);
#endif
    volatile uint64_t* tile = (volatile uint64_t*)__local_mem((size_t)(TILE_A) * (SPLIT) * 8) + sub * (uint32_t)(TILE_W);
#endif
    BARRIER();                                            // every warp has set the geometry
    const uint64_t t0 = csr_read(VX_CSR_MCYCLE);
#if ACTOR == 1
    volatile uint32_t* flag = (volatile uint32_t*)a->flag_addr;
#pragma GCC unroll 1
    for (; r < (uint32_t)(MAXR); r++) {
        if (r != 0 && flag[r - 1] == 0) break;            // uniform: every thread reads the same flag
        if (r != 0 && blk == 0) aq_flush();               // block-uniform: round r-1's partial lines
        const uint64_t rv = ss_round(a, (uint32_t)blk, sub, ln, r, tile, pf);
        msgs += rv >> 32;
        if (vx_vote_any((int)((uint32_t)rv != 0))) flag[r] = 1;
        vx_fence();
        BARRIER();
        if (ROUND1) { r += 2; break; }                    // probe: round 0 only
    }
    const uint32_t rounds = r - 1;                        // rounds that sent (the last one found nothing)
#else
    // ---- Bellman-Ford on one PE: a warp per queued vertex (grid stride), lanes over its out-edges
    const uint32_t* off = (const uint32_t*)a->off_addr;
    const uint32_t* nbr = (const uint32_t*)a->nbr_addr;
    const double* wt = (const double*)a->wt_addr;
    uint64_t* tent = (uint64_t*)a->tent_addr;
    uint32_t* fq = (uint32_t*)a->fr_addr;
    uint32_t* stamp = (uint32_t*)a->stamp_addr;
    volatile uint32_t* cnt = (volatile uint32_t*)a->cnt_addr;
    const uint32_t nw = (uint32_t)NB * (SPLIT), n = a->n;
#pragma GCC unroll 1
    for (; r < (uint32_t)(MAXR); r++) {
        const uint32_t fsz = cnt[r % 3];
        if (fsz == 0) break;                              // uniform
        if (wi == 0) cnt[(r + 2) % 3] = 0;                // warp-uniform: round r - 1's size, read by nobody now
        const uint32_t* f = fq + (size_t)(r & 1u) * n;
        uint32_t* nf = fq + (size_t)((r & 1u) ^ 1u) * n;
        uint32_t* nc = (uint32_t*)&cnt[(r + 1) % 3];
        const uint64_t t1 = csr_read(VX_CSR_MCYCLE);
#if SEND_LANE
#pragma GCC unroll 1
        for (uint32_t i = wi * GLANES + ln; i < fsz; i += nw * GLANES) {
            const uint32_t v = f[i], e1 = off[v + 1];
            const double d = as_f64(((volatile uint64_t*)tent)[v]);
#pragma GCC unroll 1
            for (uint32_t e = off[v]; e < e1; e++) {
#else
#pragma GCC unroll 1
        for (uint32_t i = wi; i < fsz; i += nw) {
            const uint32_t v = f[i], e1 = off[v + 1];
            const double d = as_f64(((volatile uint64_t*)tent)[v]);
#pragma GCC unroll 1
            for (uint32_t e = off[v] + ln; e < e1; e += GLANES) {
#endif
                const uint32_t u = nbr[e];
                const uint64_t x = as_u64(d + wt[e]);
                if (x < ((volatile uint64_t*)tent)[u] && x < __atomic_fetch_min(&tent[u], x, __ATOMIC_RELAXED)
                    && __atomic_exchange_n(&stamp[u], r + 1, __ATOMIC_RELAXED) != r + 1)
                    nf[__atomic_fetch_add(nc, 1u, __ATOMIC_RELAXED)] = u;
            }
        }
        pf[1] += csr_read(VX_CSR_MCYCLE) - t1;
        BARRIER();
    }
    const uint32_t rounds = r;
#endif
    const uint64_t t1 = csr_read(VX_CSR_MCYCLE);
    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[wi];
    ph->drain = pf[0]; ph->send = pf[1]; ph->total = t1 - t0;
    PH_MSGS(ph, (uint32_t)msgs); ph->rounds = rounds; ph->t0 = t0; ph->t1 = t1;
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N / GLANES;
    uint32_t block_dim = GLANES * (SPLIT);
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
