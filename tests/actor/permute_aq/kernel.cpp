// Sparse matrix permutation kernel (see common.h).
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
#define TILE_A     ((NOTILE) ? 0 : (TILE_W))   // scratchpad words per warp: NOTILE works in memory and allocates no tile
#if !defined(ACTOR) || ACTOR == 1
LMEM_CHECK((TILE_A) * (SPLIT) * 4, (NASSUME) / (GLANES));
#endif
#include "../abl.h"

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
#define FOR_MY_ACTORS(o) for (uint32_t ka_ = 0, o = blk * GLANES + sub; ka_ < (uint32_t)(OW); ka_++, o += (SPLIT))
#define QUEUE_STATE const uint32_t* qs = (const uint32_t*)a->qs_addr; \
    const volatile uint32_t* tails = (const volatile uint32_t*)a->tails_addr; (void)tails; \
    const uint64_t* ctab = (const uint64_t*)a->ctab_addr; const uint32_t maxc = a->ovf_maxc;

// Each round is one leaf over the warp's OW actors: one prologue per round (every callee save is a
// scattered stack write on this machine).

// round 1 send: every local row, {rperminv / P << 16 | length} to owner rperminv % P
static __attribute__((noinline)) void send_rowcnt(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln) {
    const uint32_t lr = a->lrows, lrs = lr + 1;
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        const int* aoff = (const int*)a->aoff_addr + (size_t)o * lrs;
        const int* rpi = (const int*)a->rpi_addr + (size_t)o * lr;
#pragma GCC unroll 1
        for (uint32_t r = ln; r < lr; r += GLANES) {
            const uint32_t nr = (uint32_t)rpi[r];
            aq_append32(nr % (NASSUME), ((nr / (NASSUME)) << 16) | (uint32_t)(aoff[r + 1] - aoff[r]));
        }
    }
}
// round 1 handler: rowcnt[row] = length
static __attribute__((noinline)) uint32_t recv_rowcnt(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln) {
    QUEUE_STATE
    const uint32_t lr = a->lrows;
    uint32_t msgs = 0;
#if PROBE_NOAPPLY && NOTILE
    // measurement only: read every message, touch no state (the apply's cost = the normal run minus this one)
    uint32_t s_ = 0;
#define ROWCNT(w) { s_ ^= (w); (void)rc; }
#else
#define ROWCNT(w) rc[(w) >> 16] = (w) & 0xffffu;
#endif
#if LANE_RECV
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES(o, t, 1u) {
        uint32_t* rc = (uint32_t*)a->cnt_addr + (size_t)o * lr;
        LANE_QUEUE(o, ROWCNT)
#else
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        uint32_t* rc = (uint32_t*)a->cnt_addr + (size_t)o * lr;
        WC_QUEUE(o, ROWCNT)
#endif
    }
#undef ROWCNT
#if PROBE_NOAPPLY && NOTILE
    if (s_ == 0x12345678u) ((uint32_t*)a->cnt_addr)[0] = s_;
#endif
    return msgs;
}
// lengths -> Ap offsets: per actor, lane ln scans block ln of the lengths, the lanes' block sums give the bases
static __attribute__((noinline)) void scan_round(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln,
                                                 volatile uint32_t* ws) {
    const uint32_t lr = a->lrows;
    const uint32_t B = (lr + GLANES - 1) / GLANES;
    const uint32_t b0 = ln * B < lr ? ln * B : lr, b1 = b0 + B < lr ? b0 + B : lr;
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        const uint32_t* lc = (const uint32_t*)a->cnt_addr + (size_t)o * lr;
        uint32_t* lo = (uint32_t*)a->poff_addr + (size_t)o * (lr + 1);
        uint32_t s = 0;
#pragma GCC unroll 1
        for (uint32_t i = b0; i < b1; i++) s += lc[i];
        ws[ln] = s;
        vx_fence();
        uint32_t base = 0, tot = 0;
#pragma GCC unroll 1
        for (uint32_t k = 0; k < (uint32_t)(GLANES); k++) { const uint32_t x = ws[k]; tot += x; base += k < ln ? x : 0; }
#if NOTILE
        uint32_t* wk = (uint32_t*)a->wrk_addr + (size_t)o * lr;   // the handler's next position per row
#pragma GCC unroll 1
        for (uint32_t i = b0; i < b1; i++) { lo[i] = base; wk[i] = base; base += lc[i]; }
#else
#pragma GCC unroll 1
        for (uint32_t i = b0; i < b1; i++) { lo[i] = base; base += lc[i]; }
#endif
        lo[lr] = tot;
        vx_fence();
    }
}
// round 2 send: a lane per local row r, every nonzero (r, c): {rperminv[r] << TB | c's index in its table tile} to the
// queue of (owner c % P, table tile (c / P) / TILE_W)
static __attribute__((noinline)) void send_req(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln) {
    const int* anz = (const int*)a->anz_addr;
    const uint32_t lr = a->lrows, lrs = lr + 1;
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        const int* aoff = (const int*)a->aoff_addr + (size_t)o * lrs;
        const int* rpi = (const int*)a->rpi_addr + (size_t)o * lr;
#pragma GCC unroll 1
        for (uint32_t r = ln; r < lr; r += GLANES) {
            const uint32_t hi = (uint32_t)rpi[r] << (TB);
#pragma GCC unroll 1
            for (int j = aoff[r]; j < aoff[r + 1]; j++) {
                const uint32_t c = (uint32_t)anz[j], cl = c / (NASSUME);
                aq_append32((cl >> (TB)) * (uint32_t)(NASSUME) + c % (NASSUME), hi | (cl & ((KW) - 1)));
            }
        }
    }
}
// round 2 handler, the column owner: per table tile queue, that tile of its cperminv in the scratchpad; each request
// is forwarded as {new row's index in its row tile << CB | cperminv[c]} to the queue of (new row's owner, row tile),
// keys after the request tiles
#define FWD(x, V) { const uint32_t nr_ = (x) >> (TB), nl_ = nr_ / (NASSUME); \
    aq_append32((T3 + (nl_ >> (RB))) * (uint32_t)(NASSUME) + nr_ % (NASSUME), ((nl_ & ((RK) - 1)) << (CB)) | (V)); }
static __attribute__((noinline)) uint32_t serve_fwd(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln,
                                                    uint32_t* tile) {
    QUEUE_STATE
    const uint32_t lr = a->lrows, T3 = a->ntile3;
    uint32_t msgs = 0;
#if NOTILE
    // one request queue per owner: the owner's cperminv read in memory
#if PROBE_NOAPPLY
#define GFWD(x) { FWD(x, (x) & ((1u << (CB)) - 1u)) (void)cp; }   // measurement only: forward without reading cperminv
#else
#define GFWD(x) FWD(x, (uint32_t)cp[(x) & ((KW) - 1)])
#endif
#if LANE_RECV
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES(o, t, 1u) { const int* cp = (const int*)a->cpi_addr + (size_t)o * lr; LANE_QUEUE(o, GFWD) }
#else
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) { const int* cp = (const int*)a->cpi_addr + (size_t)o * lr; WC_QUEUE(o, GFWD) }
#endif
#undef GFWD
    (void)tile;
#elif LANE_RECV
    // a lane per table tile queue, with that tile of cperminv in the lane's own KW-word scratchpad tile
    uint32_t* lt = tile + ln * (uint32_t)(KW);
#define LFWD(x) FWD(x, lt[(x) & ((KW) - 1)])
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES(o, t, T3) {
        const int* cp = (const int*)a->cpi_addr + (size_t)o * lr;
        const uint32_t base = t * (uint32_t)(KW), w = lr - base < (uint32_t)(KW) ? lr - base : (uint32_t)(KW);
#pragma GCC unroll 1
        for (uint32_t j = 0; j < w; j++) lt[j] = (uint32_t)cp[base + j];
        LANE_QUEUE(t * (uint32_t)(NASSUME) + o, LFWD)
    }
#undef LFWD
#else
#define TFWD(x) FWD(x, tile[(x) & ((KW) - 1)])
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        const int* cp = (const int*)a->cpi_addr + (size_t)o * lr;
#pragma GCC unroll 1
        for (uint32_t t = 0; t < T3; t++) {
            const uint32_t base = t * (uint32_t)(KW), w = lr - base < (uint32_t)(KW) ? lr - base : (uint32_t)(KW);
#pragma GCC unroll 1
            for (uint32_t m = 0; m < (w + GLANES - 1) / GLANES; m++) {
                const uint32_t j = ln + m * GLANES;
                if (j < w) tile[j] = (uint32_t)cp[base + j];
            }
            vx_fence();
            WC_QUEUE(t * (uint32_t)(NASSUME) + o, TFWD)
            vx_fence();
        }
    }
#undef TFWD
#endif
    return msgs;
}
#undef FWD
// round 3 handler (Ap.lnonzero[loffset[row] + wrkoff[row]++] = col): per row tile queue, the scratchpad tile loaded
// with the offsets, then each message's position taken from it with a scratchpad atomic
static __attribute__((noinline)) uint32_t recv_place(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln,
                                                     uint32_t* tile) {
    QUEUE_STATE
    const uint32_t lr = a->lrows, T = a->ntile, T3 = a->ntile3, tcap = a->tcap;
    uint32_t msgs = 0;
#if NOTILE
    // one forward queue per owner: the owner's wrkoff in memory, an L2 atomic per message takes the position
#if PROBE_NOAPPLY
    uint32_t s_ = 0;   // measurement only, as recv_rowcnt
#define GPLACE(x) { s_ ^= (x); (void)wk; (void)lnz; }
#else
#define GPLACE(x) { lnz[__atomic_fetch_add(&wk[(x) >> (CB)], 1u, __ATOMIC_RELAXED)] = (int)((x) & ((1u << (CB)) - 1u)); }
#endif
#if LANE_RECV
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES(o, t, 1u) {
#else
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
#endif
        uint32_t* wk = (uint32_t*)a->wrk_addr + (size_t)o * lr;
        int* lnz = (int*)a->pnz_addr + (size_t)o * tcap;
#if LANE_RECV
        LANE_QUEUE(T3 * (uint32_t)(NASSUME) + o, GPLACE)
#else
        WC_QUEUE(T3 * (uint32_t)(NASSUME) + o, GPLACE)
#endif
    }
#undef GPLACE
#if PROBE_NOAPPLY
    if (s_ == 0x12345678u) ((int*)a->pnz_addr)[0] = (int)s_;
#endif
    (void)T; (void)tile;
#elif LANE_RECV
    // a lane per row tile queue: its own tile loaded with the offsets, positions taken from it
    uint32_t* lt = tile + ln * (uint32_t)(KW);
#define LPLACE(x) { lnz[lt[(x) >> (CB)]++] = (int)((x) & ((1u << (CB)) - 1u)); }
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES(o, t, T) {
        const uint32_t* lo = (const uint32_t*)a->poff_addr + (size_t)o * (lr + 1);
        int* lnz = (int*)a->pnz_addr + (size_t)o * tcap;
        const uint32_t base = t * (uint32_t)(RK), w = lr - base < (uint32_t)(RK) ? lr - base : (uint32_t)(RK);
#pragma GCC unroll 1
        for (uint32_t j = 0; j < w; j++) lt[j] = lo[base + j];
        LANE_QUEUE((T3 + t) * (uint32_t)(NASSUME) + o, LPLACE)
    }
#undef LPLACE
#else
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        const uint32_t* lo = (const uint32_t*)a->poff_addr + (size_t)o * (lr + 1);
        int* lnz = (int*)a->pnz_addr + (size_t)o * tcap;
#define TPLACE(x) { lnz[__atomic_fetch_add(&tile[(x) >> (CB)], 1u, __ATOMIC_RELAXED)] = (int)((x) & ((1u << (CB)) - 1u)); }
#pragma GCC unroll 1
        for (uint32_t t = 0; t < T; t++) {
            const uint32_t base = t * (uint32_t)(RK), w = lr - base < (uint32_t)(RK) ? lr - base : (uint32_t)(RK);
#pragma GCC unroll 1
            for (uint32_t m = 0; m < (w + GLANES - 1) / GLANES; m++) {
                const uint32_t j = ln + m * GLANES;
                if (j < w) tile[j] = lo[base + j];
            }
            vx_fence();
            WC_QUEUE((T3 + t) * (uint32_t)(NASSUME) + o, TPLACE)
            vx_fence();
        }
#undef TPLACE
    }
#endif
    return msgs;
}
#endif

void kernel_body(args_t* __UNIFORM__ a) {
    const int* aoff = (const int*)a->aoff_addr;
    const int* anz = (const int*)a->anz_addr;
    const int* rpi = (const int*)a->rpi_addr;
    const int* cpi = (const int*)a->cpi_addr;
    uint32_t* poff = (uint32_t*)a->poff_addr;
    int* pnz = (int*)a->pnz_addr;
    volatile uint32_t* bar = (volatile uint32_t*)a->bar_addr;
    const int blk = (int)blockIdx.x, tid = (int)threadIdx.x;
    const uint32_t sub = (uint32_t)tid / GLANES, ln = (uint32_t)tid % GLANES;
    const uint32_t wi = (uint32_t)blk * (SPLIT) + sub;
    const int NB = (int)((uint32_t)(NASSUME) / GLANES);
    uint32_t msgs = 0;
    (void)aoff; (void)anz; (void)rpi; (void)cpi; (void)poff; (void)pnz;
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
    const uint32_t ub = (uint32_t)blk;
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
    // ---- round 1: row lengths to the rows' new owners
    PT_S0 send_rowcnt(a, ub, sub, ln); PT_S1
    vx_fence();
    gbarrier_ep(bar, 1, tid, NB, 1);
    PT_R0 msgs += recv_rowcnt(a, ub, sub, ln); PT_R1
    vx_fence();
    gbarrier_ep(bar, 2, tid, NB, 1);
#if SWQ
    swq_round((uint32_t)a->N * (a->ntile3 + a->ntile), wi * GLANES + ln, (uint32_t)(NASSUME) * (SPLIT));
#else
    if (blk == 0) aq_round();                             // block-uniform: empty the queues for round 2
#endif
    vx_fence();
    gbarrier_ep(bar, 3, tid, NB, 1);
    const uint64_t ta = csr_read(VX_CSR_MCYCLE);
    // ---- the offsets of my actors' rows of Ap
    scan_round(a, ub, sub, ln, ws);
    const uint64_t tb = csr_read(VX_CSR_MCYCLE);
    // ---- round 2: requests to the columns' owners, who forward the renamed columns to the rows' new owners
    PT_S0 send_req(a, ub, sub, ln); PT_S1
    vx_fence();
    gbarrier_ep(bar, 4, tid, NB, 1);
    PT_R0 msgs += serve_fwd(a, ub, sub, ln, tile); PT_R1
    vx_fence();
    gbarrier_ep(bar, 5, tid, NB, 1);
    const uint64_t tc = csr_read(VX_CSR_MCYCLE);
    // ---- round 3: place the nonzeros
    PT_R0 msgs += recv_place(a, ub, sub, ln, tile); PT_R1
    vx_fence();
    gbarrier_ep(bar, 6, tid, NB, 1);
#if SWQ
    swq_round((uint32_t)a->N * (a->ntile3 + a->ntile), wi * GLANES + ln, (uint32_t)(NASSUME) * (SPLIT));
#else
    if (blk == 0) aq_round();                             // block-uniform: charge the partial lines
#endif
    vx_fence();
    gbarrier_ep(bar, 7, tid, NB, 1);
#else
    // ---- permute_matrix_agi, one PE, grid stride
    int* rperm = (int*)a->cnt_addr;
    const uint32_t gtid = wi * GLANES + ln;
    const uint32_t stride = (uint32_t)(NASSUME) * (SPLIT);
    const uint32_t nw = stride / GLANES;
    const uint32_t n = a->numrows;
#pragma GCC unroll 1
    for (uint32_t i = gtid; i < n; i += stride)          // compute rperm from rperminv
        rperm[rpi[i]] = (int)i;
    vx_fence();
    gbarrier_ep(bar, 1, tid, NB, 1);
    const uint64_t ta = csr_read(VX_CSR_MCYCLE);
    // the offsets of Ap: thread gtid scans block gtid of Ap's rows (row i has A's row rperm[i]'s length);
    // block sums, then warp sums, give the bases
    volatile uint32_t* sums = (volatile uint32_t*)a->sums_addr;
    volatile uint32_t* wsum = sums + stride;
    const uint32_t B = (n + stride - 1) / stride;
    const uint32_t b0 = gtid * B < n ? gtid * B : n, b1 = b0 + B < n ? b0 + B : n;
    uint32_t s = 0;
#pragma GCC unroll 1
    for (uint32_t i = b0; i < b1; i++) { const int row = rperm[i]; s += (uint32_t)(aoff[row + 1] - aoff[row]); }
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
    for (uint32_t i = b0; i < b1; i++) { const int row = rperm[i]; poff[i] = base; base += (uint32_t)(aoff[row + 1] - aoff[row]); }
    if (wi == nw - 1) poff[n] = wb + wt;                   // warp-uniform
    vx_fence();
    gbarrier_ep(bar, 4, tid, NB, 1);
    const uint64_t tb = csr_read(VX_CSR_MCYCLE);
    // fill in permuted rows
#pragma GCC unroll 1
    for (uint32_t i = gtid; i < n; i += stride) {
        const int row = rperm[i];
        uint32_t pos = poff[i];
#pragma GCC unroll 1
        for (int j = aoff[row]; j < aoff[row + 1]; j++) pnz[pos++] = anz[j];
    }
    vx_fence();
    gbarrier_ep(bar, 5, tid, NB, 1);
    const uint64_t tc = csr_read(VX_CSR_MCYCLE);
    // finally permute column indices
#pragma GCC unroll 1
    for (uint32_t i = gtid; i < n; i += stride) {
#pragma GCC unroll 1
        for (uint32_t j = poff[i]; j < poff[i + 1]; j++) pnz[j] = cpi[pnz[j]];
    }
    vx_fence();
    gbarrier_ep(bar, 6, tid, NB, 1);
#endif
    const uint64_t t1 = csr_read(VX_CSR_MCYCLE);
    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[wi];
    ph->p1 = ta - t0; ph->scan = tb - ta; ph->p2 = tc - tb; ph->p3 = t1 - tc; ph->total = t1 - t0;
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
