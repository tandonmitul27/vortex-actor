// PageRank (push) kernel (see common.h).
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
static inline void aq_round(void) {
#if !BASELINE && !(ISA2 && SWQ)   // ISA2: the append off writes no L2 control register
 ((volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE)[6] = 1; 
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
// one queue segment, warp cooperative: lane ln takes entries ln, ln+16, ...; WC_U lines loaded first
// (ISA2: S_ = elements between lane ln's consecutive entries: QSTEP in a queue's region, GLANES in a chunk)
#define WC_SEG(QP, N_, OP, S_) { const uint32_t nf_ = (N_) / GLANES; uint32_t k_ = 0; \
    for (; k_ + (WC_U) <= nf_; k_ += (WC_U)) { uint64_t w_[WC_U]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(WC_U); u_++) w_[u_] = (QP)[(k_ + u_) * S_]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(WC_U); u_++) OP(w_[u_]); } \
    for (; k_ < nf_; k_++) OP((QP)[k_ * S_]); \
    if (nf_ * GLANES + ln < (N_)) OP((QP)[nf_ * S_]); }
// one whole queue: its region, then its overflow chunks (warp-uniform trip counts)
#define WC_QUEUE(QI, OP) { const uint32_t tl_ = QCOUNT(QI); msgs += tl_; \
    const uint64_t* q_ = QPTR(QI) + QIX(ln); \
    const uint32_t n0_ = tl_ < (uint32_t)(Q_CAP) ? tl_ : (uint32_t)(Q_CAP); \
    WC_SEG(q_, n0_, OP, QSTEP) \
    uint32_t r_ = tl_ - n0_; const uint64_t* cr_ = ctab + (size_t)(QI) * maxc; \
    _Pragma("GCC unroll 1") for (uint32_t c_ = 0; r_ != 0; c_++) { \
        const uint32_t n_ = r_ < (uint32_t)(OVF_CH) ? r_ : (uint32_t)(OVF_CH); \
        const uint64_t* qc_ = (const uint64_t*)(uintptr_t)cr_[c_] + ln; \
        WC_SEG(qc_, n_, OP, GLANES) \
        r_ -= n_; } }
#define QS_W QS64
#define FOR_MY_ACTORS(o) for (uint32_t ka_ = 0, o = blk * GLANES + sub; ka_ < (uint32_t)(OW); ka_++, o += (SPLIT))
#define QUEUE_STATE const uint64_t* qs = (const uint64_t*)a->qs_addr; \
    const volatile uint32_t* tails = (const volatile uint32_t*)a->tails_addr; \
    const uint64_t* ctab = (const uint64_t*)a->ctab_addr; const uint32_t maxc = a->ovf_maxc;

// Each round is one leaf over the warp's OW actors: one prologue per round (every callee save is a
// scattered stack write on this machine). Queue t * N + owner holds the owner's row tile t.

// send: a lane per local row y, prf = pr[y] / deg(y) to owner col % P of every nonzero (64-bit word: prf
// rounded to its top 64 - TB bits | the in-tile row)
static __attribute__((noinline)) void send_round(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln) {
    const int* anz = (const int*)a->anz_addr;
    const uint32_t lr = a->lrows, lrs = lr + 1, n = a->numrows;
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        const int* aoff = (const int*)a->aoff_addr + (size_t)o * lrs;
        const double* lpr = (const double*)a->pr_addr + (size_t)o * lr;
        const uint32_t lro = pe_rows(n, o, (uint32_t)(NASSUME));
#pragma GCC unroll 1
        for (uint32_t r = ln; r < lro; r += GLANES) {
            const int j0 = aoff[r], j1 = aoff[r + 1];
            const uint64_t hi = (as_u64(lpr[r] / (double)(j1 - j0)) + (1ull << ((TB) - 1))) & ~(uint64_t)((KW) - 1);
#pragma GCC unroll 1
            for (int j = j0; j < j1; j++) {
                const uint32_t col = (uint32_t)anz[j], cl = col / (NASSUME);
                aq_append64((cl >> (TB)) * (uint32_t)(NASSUME) + col % (NASSUME), hi | (cl & ((KW) - 1)));
            }
        }
    }
}
// handler (curr[col / P] += prf): per tile queue, added into the warp's scratchpad tile (amocas loops), then
// the tile written out to curr as whole lines and cleared
static __attribute__((noinline)) uint32_t recv_round(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln,
                                                     volatile uint64_t* tile) {
    QUEUE_STATE
    const uint32_t lr = a->lrows, T = a->ntile;
    uint32_t msgs = 0;
#if NOTILE
    // curr in memory, an L2 amocas loop per message
#define GADD(x) atomic_add_f64((volatile uint64_t*)&lc[(x) & ((KW) - 1)], as_f64((x) & ~(uint64_t)((KW) - 1)));
#if LANE_RECV
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES(o, t, 1u) { double* lc = (double*)a->cur_addr + (size_t)o * lr; LANE_QUEUE(o, GADD) }
#else
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) { double* lc = (double*)a->cur_addr + (size_t)o * lr; WC_QUEUE(o, GADD) }
#endif
#undef GADD
    (void)T; (void)tile;
#elif LANE_RECV
    // a lane per tile queue, added into the lane's own KW-double tile, written out by that lane
    volatile uint64_t* lt = tile + ln * (uint32_t)(KW);
#define TADD(x) { const uint32_t i_ = (uint32_t)((x) & ((KW) - 1)); lt[i_] = as_u64(as_f64(lt[i_]) + as_f64((x) & ~(uint64_t)((KW) - 1))); }
#pragma GCC unroll 1
    FOR_MY_LANE_QUEUES(o, t, T) {
        LANE_QUEUE(t * (uint32_t)(NASSUME) + o, TADD)
        double* lc = (double*)a->cur_addr + (size_t)o * lr;
        const uint32_t base = t * (uint32_t)(KW), w = lr - base < (uint32_t)(KW) ? lr - base : (uint32_t)(KW);
#pragma GCC unroll 1
        for (uint32_t j = 0; j < w; j++) { lc[base + j] = as_f64(lt[j]); lt[j] = 0; }
    }
#else
#define TADD(x) atomic_add_f64(&tile[(x) & ((TILE_W) - 1)], as_f64((x) & ~(uint64_t)((TILE_W) - 1)));
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        double* lc = (double*)a->cur_addr + (size_t)o * lr;
#pragma GCC unroll 1
        for (uint32_t t = 0; t < T; t++) {
            WC_QUEUE(t * (uint32_t)(NASSUME) + o, TADD)
            vx_fence();
            const uint32_t base = t * (uint32_t)(TILE_W), w = lr - base < (uint32_t)(TILE_W) ? lr - base : (uint32_t)(TILE_W);
#pragma GCC unroll 1
            for (uint32_t m = 0; m < (w + GLANES - 1) / GLANES; m++) {
                const uint32_t j = ln + m * GLANES;
                if (j < w) { lc[base + j] = as_f64(tile[j]); tile[j] = 0; }
            }
            vx_fence();
        }
    }
#endif
#if !NOTILE
#undef TADD
#endif
    return msgs;
}
// curr = (1 - d) / n + d curr over my actors' rows; returns the lane's part of sum |curr - pr|
static __attribute__((noinline)) double upd_round(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln) {
    const uint32_t lr = a->lrows, n = a->numrows;
    const double c0 = (1 - (DAMPING)) / (double)n;
    double e = 0;
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
        double* lc = (double*)a->cur_addr + (size_t)o * lr;
        const double* lpr = (const double*)a->pr_addr + (size_t)o * lr;
        const uint32_t lro = pe_rows(n, o, (uint32_t)(NASSUME));
#pragma GCC unroll 1
        for (uint32_t r = ln; r < lro; r += GLANES) { const double c = c0 + (DAMPING) * lc[r]; lc[r] = c; e += __builtin_fabs(c - lpr[r]); }
    }
    return e;
}
// pr = curr over my actors' rows
static __attribute__((noinline)) void copy_round(const args_t* __UNIFORM__ a, uint32_t blk, uint32_t sub, uint32_t ln) {
    const uint32_t lr = a->lrows, n = a->numrows;
#pragma GCC unroll 1
    FOR_MY_ACTORS(o) {
#if NOTILE
        double* lc = (double*)a->cur_addr + (size_t)o * lr;
#else
        const double* lc = (const double*)a->cur_addr + (size_t)o * lr;
#endif
        double* lpr = (double*)a->pr_addr + (size_t)o * lr;
        const uint32_t lro = pe_rows(n, o, (uint32_t)(NASSUME));
#pragma GCC unroll 1
#if NOTILE
        for (uint32_t r = ln; r < lro; r += GLANES) { lpr[r] = lc[r]; lc[r] = 0; }
#else
        for (uint32_t r = ln; r < lro; r += GLANES) lpr[r] = lc[r];
#endif
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
    uint64_t msgs = 0, cs = 0, cr = 0, cu = 0;
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
#pragma GCC unroll 1
    for (uint32_t m = 0; m < (uint32_t)(TILE_A) / GLANES; m++) tile[ln + m * GLANES] = 0;
#endif
    BARRIER();                                            // every warp has set the geometry
    const uint64_t t0 = csr_read(VX_CSR_MCYCLE);
#if ACTOR == 1
#pragma GCC unroll 1
    while (it < (uint32_t)(MAXIT)) {
        it++;
        const uint64_t ts = csr_read(VX_CSR_MCYCLE);
        send_round(a, (uint32_t)blk, sub, ln);
        BARRIER();
        const uint64_t ta = csr_read(VX_CSR_MCYCLE);
        msgs += recv_round(a, (uint32_t)blk, sub, ln, tile);
        BARRIER();
#if SWQ
        swq_round(a->N * a->ntile, wi * GLANES + ln, nw * GLANES);
#else
        if (blk == 0) aq_round();                         // block-uniform: empty the queues (charges the partial lines)
#endif
        BARRIER();
        const uint64_t tb = csr_read(VX_CSR_MCYCLE);
        const double ew = warp_sum(ws, ln, upd_round(a, (uint32_t)blk, sub, ln));
        wsum[wi] = ew;
        BARRIER();
        double err = 0;
#pragma GCC unroll 1
        for (uint32_t k = 0; k < nw; k++) err += wsum[k];
        cs += ta - ts; cr += tb - ta;
        if (err < (ERR_DELTA)) { cu += csr_read(VX_CSR_MCYCLE) - tb; break; }
        copy_round(a, (uint32_t)blk, sub, ln);            // my actors' rows only: the next send reads only them
        cu += csr_read(VX_CSR_MCYCLE) - tb;
    }
#else
    // ---- pagerank_agi, one PE: a thread per row, grid stride
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
#pragma GCC unroll 1
        for (uint32_t i = gtid; i < n; i += stride) cur[i] = 0;
        BARRIER();
#pragma GCC unroll 1
        for (uint32_t y = gtid; y < n; y += stride) {
            const int j0 = aoff[y], j1 = aoff[y + 1];
            const double prf = pr[y] / (double)(j1 - j0);
#pragma GCC unroll 1
            for (int j = j0; j < j1; j++) atomic_add_f64((volatile uint64_t*)&cur[anz[j]], prf);
        }
        BARRIER();
        const uint64_t tb = csr_read(VX_CSR_MCYCLE);
        double e = 0;
#pragma GCC unroll 1
        for (uint32_t i = gtid; i < n; i += stride) { const double c = c0 + (DAMPING) * cur[i]; cur[i] = c; e += __builtin_fabs(c - pr[i]); }
        const double ew = warp_sum(ws, ln, e);
        wsum[wi] = ew;
        BARRIER();
        double err = 0;
#pragma GCC unroll 1
        for (uint32_t k = 0; k < nw; k++) err += wsum[k];
        cs += tb - ts;
        if (err < (ERR_DELTA)) { cu += csr_read(VX_CSR_MCYCLE) - tb; break; }
#pragma GCC unroll 1
        for (uint32_t i = gtid; i < n; i += stride) pr[i] = cur[i];
        BARRIER();
        cu += csr_read(VX_CSR_MCYCLE) - tb;
    }
#endif
    const uint64_t t1 = csr_read(VX_CSR_MCYCLE);
    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[wi];
    ph->send = cs; ph->recv = cr; ph->upd = cu; ph->total = t1 - t0;
    PH_MSGS(ph, (uint32_t)msgs); ph->iters = it; ph->t0 = t0; ph->t1 = t1;
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N / GLANES;
    uint32_t block_dim = GLANES * (SPLIT);
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
