// Triangle counting kernel (see common.h). binary_search, intersect and the loops are bale-gpu's.
// ACTOR=1: one block = 16 actors on SPLIT warps; warp `sub` serves actors o = blk*16 + sub + k*SPLIT,
// k < OW, so an actor's rows, queue and counter are only touched by its own warp. The warp's 16 lanes
// are the PE's threads: grid stride over the actor's local rows (send), over its messages (receive).
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
LMEM_CHECK((SPLIT) * (TS_W) * 4, (NASSUME) / (GLANES));
#endif
#include "../abl.h"

// BAL: the per-row pass ends at rend[row] (a split row's start), so split rows are left to the chunks
#if BAL
#define RE_PARM     , const int* rend
#define RE_ARG(p)   , (p)
#define ROW_END(u)  rend[u]
#else
#define RE_PARM
#define RE_ARG(p)
#define ROW_END(u)  loffset[(u) + 1]
#endif

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

#if ACTOR
static inline int binary_search(const int* arr, int l, int r, int key) {
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (arr[m] == key) {
            return 1;
        } else if (arr[m] > key) {
            r = m - 1;
        } else {
            l = m + 1;
        }
    }
    return 0;
}
#endif
#if ACTOR == 7
// TRUST's linear_search (TC-Compare approach/TRUST/tc.cu), verbatim; block_bucketnum = TR_BB, shared_BUCKET_SIZE = 6
static inline int trust_linear_search(int neighbor, const int* shared_partition, const int* partition, const int* bin_count,
                                      int bin, int BIN_START) {
    for (;;) {
        int i = bin;
        int len = bin_count[i];
        int step = 0;
        int nowlen;
        if (len < 6)
            nowlen = len;
        else
            nowlen = 6;
        while (step < nowlen) {
            if (shared_partition[i] == neighbor)
                return 1;
            i += TR_BB;
            step += 1;
        }

        len -= 6;
        i = bin + BIN_START;
        step = 0;
        while (step < len) {
            if (partition[i] == neighbor)
                return 1;
            i += TR_BB;
            step += 1;
        }
        if (len + 6 < 99)
            break;
        bin++;
    }
    return 0;
}
#endif
#if ACTOR == 6
// GroupTC's bin_search (TC-Compare approach/GroupTC/tc.cu), verbatim; vertex_t = int
static inline int gtc_bin_search(const int* arr, int len, int val, int& offset) {
    int ret = 0;
    int halfsize;
    int candidate;
    int temp = len;
    while (temp > 1) {
        halfsize = temp / 2;
        candidate = arr[ret + halfsize];
        ret += (candidate < val) ? halfsize : 0;
        temp -= halfsize;
    }
    ret += (arr[ret] < val);
    offset = ret;
    return ret < len && arr[ret] == val;
}
#endif

#if MSG32 && BAL
#error "MSG32 is not wired for BAL"
#endif
#if ACTOR == 1 || ACTOR == 4
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
static inline void aq_append32(uint32_t q, uint32_t word) {
#if SWQ
    swq_put(q, word);
#elif ISA2
    aq_isa2_put(q, word);
#else
    *(volatile uint32_t*)(uintptr_t)((uint64_t)AQ_BASE + 64ull * ((uint64_t)q + 1ull)) = word;
#endif
}
// one queue segment, warp cooperative: lane ln takes entries ln, ln+16, ...; WC_U lines loaded first
// (ISA2: S_ = elements between lane ln's consecutive entries: QSTEP_ in a queue's region, GLANES in a chunk)
#define WC_SEG(QP, N_, OP, S_) { const uint32_t nf_ = (N_) / GLANES; uint32_t k_ = 0; \
    for (; k_ + (WC_U) <= nf_; k_ += (WC_U)) { msg_t w_[WC_U]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(WC_U); u_++) w_[u_] = (QP)[(k_ + u_) * S_]; \
        _Pragma("GCC unroll 8") for (uint32_t u_ = 0; u_ < (uint32_t)(WC_U); u_++) OP(w_[u_]); } \
    for (; k_ < nf_; k_++) OP((QP)[k_ * S_]); \
    if (nf_ * GLANES + ln < (N_)) OP((QP)[nf_ * S_]); }
#endif

#if ACTOR == 1

// tc_process for one actor: its queue region, then its overflow chunks; returns its lane's triangles
#if MSG32
#define VT_PARM     , uint32_t vt, uint32_t wbits
#define TC_DECODE(w) const int v_local = (int)(vt | ((w) >> wbits)); const int w_global = (int)((w) & wmask);
#else
#define VT_PARM
#define TC_DECODE(w) const int v_local = (int)((w) >> 32); const int w_global = (int)(uint32_t)(w);
#endif
static __attribute__((noinline)) uint32_t recv_actor(const msg_t* q, uint32_t tl, const uint64_t* cr,
                                                     const int* loffset, const int* lnonzero, uint32_t ln VT_PARM) {
    uint32_t c = 0;
#if MSG32
    const uint32_t wmask = (1u << wbits) - 1u;
#endif
#define TC_PROCESS(w) { TC_DECODE(w) \
    const int* nbrV = &lnonzero[loffset[v_local]]; \
    const int nbrV_size = loffset[v_local + 1] - loffset[v_local]; \
    c += (uint32_t)binary_search(nbrV, 0, nbrV_size - 1, w_global); }
    const uint32_t n0 = tl < (uint32_t)(Q_CAP) ? tl : (uint32_t)(Q_CAP);
    WC_SEG(q, n0, TC_PROCESS, QSTEP_(q))
    uint32_t r = tl - n0;
#pragma GCC unroll 1
    for (uint32_t ch = 0; r != 0; ch++) {
        const uint32_t n = r < (uint32_t)(OVF_CH) ? r : (uint32_t)(OVF_CH);
        const msg_t* qc = (const msg_t*)(uintptr_t)cr[ch] + ln;
        WC_SEG(qc, n, TC_PROCESS, GLANES)
        r -= n; }
#undef TC_PROCESS
    return c;
}

// generate_pairs for one actor (mype = o, npes = N): every pair (v, w) of a local row u, w before v
static __attribute__((noinline)) void send_actor(const int* loffset, const int* lnonzero, uint32_t lnumrows,
                                                 uint32_t mype, uint32_t ln RE_PARM
#if MSG32
                                                 , uint32_t wbits, uint32_t vbits, uint32_t tbits
#endif
                                                 ) {
    const int npes = (int)(NASSUME);
#pragma GCC unroll 1
    for (uint32_t u_local = ln; u_local < lnumrows; u_local += GLANES) {
        const int u_global = (int)u_local * npes + (int)mype;
#pragma GCC unroll 1
        for (int j = loffset[u_local]; j < ROW_END(u_local); j++) {
            const int v_global = lnonzero[j];
            if (v_global >= u_global) break;
            const int v_local = v_global / npes;
            const int v_owner = v_global % npes;
#if MSG32
            // queue (owner, tile of v_local); v_local's low bits above w
            const uint32_t q = ((uint32_t)v_owner << tbits) + ((uint32_t)v_local >> vbits);
            const uint32_t hi = ((uint32_t)v_local & ((1u << vbits) - 1u)) << wbits;
#pragma GCC unroll 1
            for (int k = loffset[u_local]; k < j; k++)
                aq_append32(q, hi | (uint32_t)lnonzero[k]);
#else
#pragma GCC unroll 1
            for (int k = loffset[u_local]; k < j; k++) {
                const int w_global = lnonzero[k];
                aq_append64((uint32_t)v_owner, ((uint64_t)(uint32_t)v_local << 32) | (uint64_t)(uint32_t)w_global);
            }
#endif
        }
    }
}

#if SEDGE && !RTILE
// generate_pairs, edge-centric without tiles (piece B off): lanes take my edges j; j's row starts at j - pos (pos in
// lmap[j]); message {v_local, w} to v's owner, as send_actor
static __attribute__((noinline)) void send_edge0(const int* loffset, const int* lnonzero, uint32_t lnumrows,
                                                 uint32_t ln, const uint32_t* lmap) {
    const int npes = (int)(NASSUME), e1 = loffset[lnumrows];
#pragma GCC unroll 1
    for (int j = loffset[0] + (int)ln; j < e1; j += GLANES) {
        uint32_t pos = lmap[j] >> 24;
        if (pos == (SEDGE_PMAX)) {                // a row longer than SEDGE_PMAX: loffset[lo_] <= j < loffset[hi_]
            uint32_t lo_ = 0, hi_ = lnumrows;
#pragma GCC unroll 1
            while (hi_ - lo_ > 1) { const uint32_t m_ = (lo_ + hi_) >> 1; if (loffset[m_] <= j) lo_ = m_; else hi_ = m_; }
            pos = (uint32_t)(j - loffset[lo_]);
        }
        const int v_global = lnonzero[j];         // L is strictly lower (host checked): every v < u
        const uint64_t hi = (uint64_t)(uint32_t)(v_global / npes) << 32;
        const uint32_t q = (uint32_t)(v_global % npes);
#pragma GCC unroll 1
        for (int k = j - (int)pos; k < j; k++)
            aq_append64(q, hi | (uint64_t)(uint32_t)lnonzero[k]);
    }
}
#endif

#if LANE_RECV
// piece C off: one lane drains one whole queue alone (its region, then its chunks), a message per load
#define LANE_Q(Q, TL, CR, OP) { const uint32_t n0_ = (TL) < (uint32_t)(Q_CAP) ? (TL) : (uint32_t)(Q_CAP); \
    _Pragma("GCC unroll 1") for (uint32_t k_ = 0; k_ < n0_; k_++) OP((Q)[QIX_(Q, k_)]); \
    uint32_t r_ = (TL) - n0_; \
    _Pragma("GCC unroll 1") for (uint32_t c_ = 0; r_ != 0; c_++) { \
        const uint32_t n_ = r_ < (uint32_t)(OVF_CH) ? r_ : (uint32_t)(OVF_CH); \
        const msg_t* qc_ = (const msg_t*)(uintptr_t)(CR)[c_]; \
        _Pragma("GCC unroll 1") for (uint32_t k_ = 0; k_ < n_; k_++) OP(qc_[k_]); \
        r_ -= n_; } }
#if RTILE
static __attribute__((noinline)) uint32_t recv_tile_l(const msg_t* q, uint32_t tl, const uint64_t* cr,
                                                      const int* offs, const int* cols, uint32_t wbits) {
    uint32_t c = 0;
    const uint32_t wmask = (1u << wbits) - 1u;
#define TC_PROCESS(w) { const uint32_t r_ = (w) >> wbits; const int s_ = offs[r_]; \
    c += (uint32_t)binary_search(cols + s_, 0, offs[r_ + 1] - s_ - 1, (int)((w) & wmask)); }
    LANE_Q(q, tl, cr, TC_PROCESS)
#undef TC_PROCESS
    return c;
}
static __attribute__((noinline)) uint32_t recv_hash_l(const msg_t* q, uint32_t tl, const uint64_t* cr, const uint32_t* T) {
    const uint32_t hm = (HS) - 1;
    uint32_t c = 0;
#define TC_PROCESS(w) { const uint32_t key_ = (w); uint32_t h_ = HS_HASH(key_), t_; \
    while ((t_ = T[h_]) != key_ && t_ != HS_EMPTY) h_ = (h_ + 1) & hm; \
    c += (uint32_t)(t_ == key_); }
    LANE_Q(q, tl, cr, TC_PROCESS)
#undef TC_PROCESS
    return c;
}
#else
static __attribute__((noinline)) uint32_t recv_actor_l(const msg_t* q, uint32_t tl, const uint64_t* cr,
                                                       const int* loffset, const int* lnonzero) {
    uint32_t c = 0;
#define TC_PROCESS(w) { TC_DECODE(w) \
    const int* nbrV = &lnonzero[loffset[v_local]]; \
    const int nbrV_size = loffset[v_local + 1] - loffset[v_local]; \
    c += (uint32_t)binary_search(nbrV, 0, nbrV_size - 1, w_global); }
    LANE_Q(q, tl, cr, TC_PROCESS)
#undef TC_PROCESS
    return c;
}
#endif
#endif

#if RTILE
// generate_pairs, RTILE: v's queue (owner, tile) and in-tile row from lmap[j]; message {in-tile row, w}
static __attribute__((noinline)) void send_tile(const int* loffset, const int* lnonzero, uint32_t lnumrows,
                                                uint32_t ln, const uint32_t* lmap, uint32_t wbits) {
#pragma GCC unroll 1
    for (uint32_t u_local = ln; u_local < lnumrows; u_local += GLANES) {
#pragma GCC unroll 1
        for (int j = loffset[u_local] + 1; j < loffset[u_local + 1]; j++) {   // j = the first: no k < j
            const uint32_t rm = lmap[j];          // L is strictly lower (host checked): every v < u
            const uint32_t q = rm >> (LMB), hi = (rm & ((1u << (LMB)) - 1u)) << wbits;
#pragma GCC unroll 1
            for (int k = loffset[u_local]; k < j; k++)
                aq_append32(q, hi | (uint32_t)lnonzero[k]);
        }
    }
}

#if SEDGE
// generate_pairs, edge-centric: lanes take my edges j; j's row starts at j - pos (pos in lmap[j])
static __attribute__((noinline)) void send_edge(const int* loffset, const int* lnonzero, uint32_t lnumrows,
                                                uint32_t ln, const uint32_t* lmap, uint32_t wbits) {
    const int e1 = loffset[lnumrows];
#pragma GCC unroll 1
    for (int j = loffset[0] + (int)ln; j < e1; j += GLANES) {
        const uint32_t rm = lmap[j];
        uint32_t pos = rm >> 24;
        if (pos == (SEDGE_PMAX)) {                // a row longer than SEDGE_PMAX: loffset[lo_] <= j < loffset[hi_]
            uint32_t lo_ = 0, hi_ = lnumrows;
#pragma GCC unroll 1
            while (hi_ - lo_ > 1) { const uint32_t m_ = (lo_ + hi_) >> 1; if (loffset[m_] <= j) lo_ = m_; else hi_ = m_; }
            pos = (uint32_t)(j - loffset[lo_]);
        }
        const uint32_t q = (rm >> (LMB)) & ((1u << (24 - (LMB))) - 1u), hi = (rm & ((1u << (LMB)) - 1u)) << wbits;
#pragma GCC unroll 1
        for (int k = j - (int)pos; k < j; k++)
            aq_append32(q, hi | (uint32_t)lnonzero[k]);
    }
}
#endif

#if SHEAVY
// generate_pairs, heavy rows shared: rows of > SHEAVY entries are left by their lane (a ballot names them) and done by
// the whole warp, each j's pairs k < j split over the lanes
static __attribute__((noinline)) void send_tile_sh(const int* loffset, const int* lnonzero, uint32_t lnumrows,
                                                   uint32_t ln, const uint32_t* lmap, uint32_t wbits) {
#pragma GCC unroll 1
    for (uint32_t s = 0; s < lnumrows; s += GLANES) {
        const uint32_t u_local = s + ln;
        const int a = loffset[u_local < lnumrows ? u_local : lnumrows];      // past the end: an empty row
        const int b = loffset[u_local + 1 < lnumrows ? u_local + 1 : lnumrows];
        const int heavy = b - a > (int)(SHEAVY);
        uint32_t hm = (uint32_t)vx_vote_ballot(heavy);                        // uniform: no divergence above it
        if (!heavy) {
#pragma GCC unroll 1
            for (int j = a + 1; j < b; j++) {
                const uint32_t rm = lmap[j];
                const uint32_t q = rm >> 10, hi = (rm & 1023u) << wbits;
#pragma GCC unroll 1
                for (int k = a; k < j; k++)
                    aq_append32(q, hi | (uint32_t)lnonzero[k]);
            }
        }
#pragma GCC unroll 1
        while (hm != 0) {
            const uint32_t i = (uint32_t)__builtin_ctz(hm);
            hm &= hm - 1;
            const int a2 = loffset[s + i], b2 = loffset[s + i + 1];
#pragma GCC unroll 1
            for (int j = a2 + 1; j < b2; j++) {
                const uint32_t rm = lmap[j];
                const uint32_t q = rm >> 10, hi = (rm & 1023u) << wbits;
#pragma GCC unroll 1
                for (int k = a2 + (int)ln; k < j; k += GLANES)
                    aq_append32(q, hi | (uint32_t)lnonzero[k]);
            }
        }
    }
}
#endif

// tc_process for one tile queue: offs = the tile's row offsets into cols (the scratchpad tile, or global memory)
static __attribute__((noinline)) uint32_t recv_tile(const msg_t* q, uint32_t tl, const uint64_t* cr,
                                                    const int* offs, const int* cols, uint32_t ln, uint32_t wbits) {
    uint32_t c = 0;
    const uint32_t wmask = (1u << wbits) - 1u;
#define TC_PROCESS(w) { const uint32_t r_ = (w) >> wbits; const int s_ = offs[r_]; \
    c += (uint32_t)binary_search(cols + s_, 0, offs[r_ + 1] - s_ - 1, (int)((w) & wmask)); }
    const uint32_t n0 = tl < (uint32_t)(Q_CAP) ? tl : (uint32_t)(Q_CAP);
    WC_SEG(q, n0, TC_PROCESS, QSTEP_(q))
    uint32_t r = tl - n0;
#pragma GCC unroll 1
    for (uint32_t ch = 0; r != 0; ch++) {
        const uint32_t n = r < (uint32_t)(OVF_CH) ? r : (uint32_t)(OVF_CH);
        const msg_t* qc = (const msg_t*)(uintptr_t)cr[ch] + ln;
        WC_SEG(qc, n, TC_PROCESS, GLANES)
        r -= n; }
#undef TC_PROCESS
    return c;
}

#if RTILE >= 2
// tc_process for one tile queue, RTILE 2/3: probe the tile's hash set T for the message word
#if HSIZE
// hb = log2 of the tile's slots
static __attribute__((noinline)) uint32_t recv_hash(const msg_t* q, uint32_t tl, const uint64_t* cr, const uint32_t* T, uint32_t ln,
                                                    uint32_t hb) {
    const uint32_t hm = (1u << hb) - 1u;
#define TC_HASH(k) (((uint32_t)(k) * 0x9E3779B1u) >> (32 - hb))
#else
static __attribute__((noinline)) uint32_t recv_hash(const msg_t* q, uint32_t tl, const uint64_t* cr, const uint32_t* T, uint32_t ln) {
    const uint32_t hm = (HS) - 1;
#define TC_HASH(k) HS_HASH(k)
#endif
    uint32_t c = 0;
#define TC_PROCESS(w) { const uint32_t key_ = (w); uint32_t h_ = TC_HASH(key_), t_; \
    while ((t_ = T[h_]) != key_ && t_ != HS_EMPTY) h_ = (h_ + 1) & hm; \
    c += (uint32_t)(t_ == key_); }
    const uint32_t n0 = tl < (uint32_t)(Q_CAP) ? tl : (uint32_t)(Q_CAP);
    WC_SEG(q, n0, TC_PROCESS, QSTEP_(q))
    uint32_t r = tl - n0;
#pragma GCC unroll 1
    for (uint32_t ch = 0; r != 0; ch++) {
        const uint32_t n = r < (uint32_t)(OVF_CH) ? r : (uint32_t)(OVF_CH);
        const msg_t* qc = (const msg_t*)(uintptr_t)cr[ch] + ln;
        WC_SEG(qc, n, TC_PROCESS, GLANES)
        r -= n; }
#undef TC_PROCESS
#undef TC_HASH
    return c;
}
#endif
#endif

#if BAL
// the split rows: chunk {base, ja, jb} is the pairs (v, w) of j in [ja, jb), k < j, of the row prefix copied
// at rep[base]; the warp's lanes share the chunk, lane ln taking j = ja + ln, ja + ln + 16, ...
static __attribute__((noinline)) void send_chunks(const uint32_t* ch, uint32_t c0, uint32_t c1, const int* rep, uint32_t ln) {
    const int npes = (int)(NASSUME);
#pragma GCC unroll 1
    for (uint32_t c = c0; c < c1; c++) {
        const int* row = rep + ch[3 * c];
        const uint32_t jb = ch[3 * c + 2];
#pragma GCC unroll 1
        for (uint32_t j = ch[3 * c + 1] + ln; j < jb; j += GLANES) {
            const int v_global = row[j];
            const int v_local = v_global / npes;
            const int v_owner = v_global % npes;
#pragma GCC unroll 1
            for (uint32_t k = 0; k < j; k++)
                aq_append64((uint32_t)v_owner, ((uint64_t)(uint32_t)v_local << 32) | (uint64_t)(uint32_t)row[k]);
        }
    }
}
#endif
#endif

#if ACTOR == 0 || ACTOR == 4
static inline int intersect(const int* arr1, int arr1_size, const int* arr2, int arr2_size) {
    int count = 0;
    int i1 = 0;
    int i2 = 0;
    while ((i1 < arr1_size) && (i2 < arr2_size)) {
        int d = arr1[i1] - arr2[i2];
        if (d == 0) {
            count++;
            i1++;
            i2++;
        } else if (d < 0) {
            i1++;
        } else if (d > 0) {
            i2++;
        }
    }
    return count;
}
#endif

#if ACTOR == 4
// bulk-pull's generate_pairs for one actor: a request for L[v] per edge (u, v) of a local row u whose v another actor owns
static __attribute__((noinline)) void pull_request(const int* loffset, const int* lnonzero, uint32_t lnumrows,
                                                   uint32_t mype, uint32_t ln) {
    const int npes = (int)(NASSUME);
#pragma GCC unroll 1
    for (uint32_t u_local = ln; u_local < lnumrows; u_local += GLANES) {
        const int u_global = (int)u_local * npes + (int)mype;
#pragma GCC unroll 1
        for (int j = loffset[u_local]; j < loffset[u_local + 1]; j++) {
            const int v_global = lnonzero[j];
            const int v_owner = v_global % npes;
            if (v_global < u_global && v_owner != (int)mype)
                aq_append64((uint32_t)v_owner, ((uint64_t)mype << 48) | ((uint64_t)u_local << 24) | (uint64_t)(uint32_t)(v_global / npes));
        }
    }
}

// one queue segment of requests, GLANES at a time: lane ln takes request b + ln, reserves its slots (the k-slot
// append: an atomic add on the requester's data and header tails) and writes its header; then the warp copies
// the batch's rows L[v_local] one by one, its lanes on consecutive words
static inline void serve_seg(const uint64_t* qp, size_t qs_step, uint32_t n, const int* loffset, const int* lnonzero, uint32_t ln,
                             int* data, uint32_t* rtail, uint32_t* hdr) {
#pragma GCC unroll 1
    for (uint32_t b = 0; b < n; b += GLANES) {
        uint32_t len = 0, src = 0, dst = 0;
        if (b + ln < n) {
#if AQ_ON
            const uint64_t w = qp[(size_t)(b / GLANES) * qs_step];   // ISA2 layout: lane ln's b-th entry
#else
            const uint64_t w = qp[b]; (void)qs_step;
#endif
            const uint32_t req = (uint32_t)(w >> 48), u_local = (uint32_t)(w >> 24) & 0xffffffu, v_local = (uint32_t)w & 0xffffffu;
            src = (uint32_t)loffset[v_local]; len = (uint32_t)loffset[v_local + 1] - src;
            dst = __atomic_fetch_add(&rtail[2 * req], len, __ATOMIC_RELAXED);
            const uint32_t h = __atomic_fetch_add(&rtail[2 * req + 1], 1u, __ATOMIC_RELAXED);
            hdr[3 * h] = u_local; hdr[3 * h + 1] = dst; hdr[3 * h + 2] = len;
        }
#pragma GCC unroll 1
        for (uint32_t r = 0; r < (uint32_t)(GLANES); r++) {
            const uint32_t l = (uint32_t)vx_shfl_idx(len, (int)r, GLANES - 1, 0);
            const uint32_t s = (uint32_t)vx_shfl_idx(src, (int)r, GLANES - 1, 0);
            const uint32_t d = (uint32_t)vx_shfl_idx(dst, (int)r, GLANES - 1, 0);
#pragma GCC unroll 1
            for (uint32_t i = 0; i < l; i += GLANES) {
                const uint32_t x = i + ln < l ? i + ln : l - 1;   // past the row's end, rewrite its last word
                data[d + x] = lnonzero[s + x];
            }
        }
    }
}

// tc_process + tc_process2 for one actor: its request queue region, then its overflow chunks
static __attribute__((noinline)) void pull_serve(const uint64_t* q, uint32_t tl, const uint64_t* cr,
                                                 const int* loffset, const int* lnonzero, uint32_t ln, const args_t* a) {
    int* data = (int*)a->data_addr;
    uint32_t* rtail = (uint32_t*)a->rtail_addr;
    uint32_t* hdr = (uint32_t*)a->hdr_addr;
    uint32_t done = 0;
#pragma GCC unroll 1
    for (uint32_t ch = 0; done < tl; ch++) {          // one inlined copy: fewer live values, no spills
        const uint64_t* p = ch == 0 ? q : (const uint64_t*)(uintptr_t)cr[ch - 1] + ln;
        const uint32_t cap = ch == 0 ? (uint32_t)(Q_CAP) : (uint32_t)(OVF_CH);
        const uint32_t n = tl - done < cap ? tl - done : cap;
        serve_seg(p, ch == 0 ? (size_t)QSTEP_(q) : (size_t)(GLANES), n, loffset, lnonzero, ln, data, rtail, hdr);
        done += n; }
}

// tc_process3 for one actor: intersect L[u] with the L[v] it owns, then with every received L[v]
static __attribute__((noinline)) uint32_t pull_merge(const int* loffset, const int* lnonzero, uint32_t lnumrows,
                                                     uint32_t mype, uint32_t ln, const int* data, const uint32_t* hdr, uint32_t nh) {
    const int npes = (int)(NASSUME);
    uint32_t c = 0;
#pragma GCC unroll 1
    for (uint32_t u_local = ln; u_local < lnumrows; u_local += GLANES) {
        const int* nbrU = &lnonzero[loffset[u_local]];
        const int nbrU_size = loffset[u_local + 1] - loffset[u_local];
#pragma GCC unroll 1
        for (int j = loffset[u_local]; j < loffset[u_local + 1]; j++) {
            const int v = lnonzero[j];
            if (v % npes == (int)mype) {
                const int v_local = v / npes;
                c += (uint32_t)intersect(nbrU, nbrU_size, &lnonzero[loffset[v_local]], loffset[v_local + 1] - loffset[v_local]);
            }
        }
    }
#pragma GCC unroll 1
    for (uint32_t h = ln; h < nh; h += GLANES) {
        const uint32_t u_local = hdr[3 * h];
        c += (uint32_t)intersect(&lnonzero[loffset[u_local]], loffset[u_local + 1] - loffset[u_local], data + hdr[3 * h + 1], (int)hdr[3 * h + 2]);
    }
    return c;
}
#endif

#if ACTOR != 2
void kernel_body(args_t* __UNIFORM__ a) {
    const int* loffset = (const int*)a->loffset_addr;
    const int* lnonzero = (const int*)a->lnonzero_addr;
    volatile uint32_t* bar = (volatile uint32_t*)a->bar_addr;
    const int blk = (int)blockIdx.x, tid = (int)threadIdx.x;
    const uint32_t sub = (uint32_t)tid / GLANES, ln = (uint32_t)tid % GLANES;
    const int NB = (int)((uint32_t)(NASSUME) / GLANES);
    uint64_t tserve = 0;
#if ACTOR == 1 || ACTOR == 4
    aq_geometry(a->qs_addr, (uint64_t)(QSTRIDE_E) * sizeof(msg_t), a->tails_addr, sizeof(msg_t), Q_CAP);
#if !BASELINE && !(ISA2 && SWQ)   // ISA2: the append off writes no L2 control register
    { volatile uint64_t* c = (volatile uint64_t*)(uintptr_t)(uint64_t)AQ_BASE;   // overflow pool
      c[7] = (1ull << 60) | a->pool_addr;
      c[7] = (3ull << 60) | ((uint64_t)(OVF_CH) << 32) | (uint64_t)a->ovf_maxc;
      c[7] = (4ull << 60) | (uint64_t)a->ovf_nch;
      c[7] = (2ull << 60) | a->ctab_addr; }
#endif
#if SWQ
    swq_geometry(a->qs_addr, (uint64_t)(QSTRIDE_E) * sizeof(msg_t), a->tails_addr, sizeof(msg_t), Q_CAP);
    swq_pool(a->pool_addr, a->ctab_addr, a->ovf_maxc, a->ovf_nch, a->swq_addr);
#endif
    const msg_t* qs = (const msg_t*)a->qs_addr;
#if MSG32
    const uint32_t wbits = a->wbits, vbits = a->vbits, tbits = a->tbits;
#elif RTILE
    const uint32_t wbits = a->wbits;
    const uint32_t* lmap = (const uint32_t*)a->lmap_addr;
#elif SEDGE
    const uint32_t* lmap = (const uint32_t*)a->lmap_addr;
#endif
    const volatile uint32_t* tails = (const volatile uint32_t*)a->tails_addr;
    const uint64_t* ctab = (const uint64_t*)a->ctab_addr;
#endif
#if BAL
    const int* rend = (const int*)a->rend_addr;
    const uint32_t* ch = (const uint32_t*)a->ch_addr;
    const uint32_t* chofs = (const uint32_t*)a->chofs_addr;
#endif
    vx_fence();
    gbarrier_ep(bar, 0, tid, NB, 1);                      // every warp has set the geometry
    uint32_t tri = 0, msgs = 0;
    const uint64_t t0 = csr_read(VX_CSR_MCYCLE);
#if ACTOR == 1
    // ---- send: generate_pairs of my actors
    const uint32_t lr = a->lrows, lrs = lr + 1;
#pragma GCC unroll 1
    for (uint32_t k = 0; k < (uint32_t)(OW); k++) {
        const uint32_t o = (uint32_t)blk * GLANES + sub + k * (SPLIT);
#if SEDGE && RTILE
        send_edge(loffset + (size_t)o * lrs, lnonzero, lr, ln, lmap, wbits);
#elif SEDGE
        send_edge0(loffset + (size_t)o * lrs, lnonzero, lr, ln, lmap);
#elif SHEAVY
        send_tile_sh(loffset + (size_t)o * lrs, lnonzero, lr, ln, lmap, wbits);
#elif RTILE
        send_tile(loffset + (size_t)o * lrs, lnonzero, lr, ln, lmap, wbits);
#else
        send_actor(loffset + (size_t)o * lrs, lnonzero, lr, o, ln RE_ARG(rend + (size_t)o * lr)
#if MSG32
                   , wbits, vbits, tbits
#endif
                   );
#endif
#if BAL
        send_chunks(ch, chofs[o], chofs[o + 1], (const int*)a->rep_addr, ln);
#endif
    }
#elif ACTOR == 4
    // ---- bulk-pull: request the rows my actors' edges need
    const uint32_t lr = a->lrows, lrs = lr + 1;
#pragma GCC unroll 1
    for (uint32_t k = 0; k < (uint32_t)(OW); k++) {
        const uint32_t o = (uint32_t)blk * GLANES + sub + k * (SPLIT);
        pull_request(loffset + (size_t)o * lrs, lnonzero, lr, o, ln);
    }
#elif ACTOR == 3
    // ---- Polak's CalculateTriangles, one GPU: a thread per edge, grid stride
    {
        const int m = (int)a->m;
        const int* __restrict__ edges = lnonzero;
        const int* __restrict__ nodes = loffset;
        const int from = blk * (GLANES * (SPLIT)) + tid;   // blockDim.x * blockIdx.x + threadIdx.x
        const int step = (int)(NASSUME) * (SPLIT);         // gridDim.x * blockDim.x
        uint32_t count = 0;
#pragma GCC unroll 1
        for (int i = from; i < m; i += step) {
            int u = edges[i], v = edges[m + i];

            int u_it = nodes[u], u_end = nodes[u + 1];
            int v_it = nodes[v], v_end = nodes[v + 1];

            int a_ = edges[u_it], b_ = edges[v_it];
            while (u_it < u_end && v_it < v_end) {
                int d = a_ - b_;
                if (d <= 0)
                    a_ = edges[++u_it];
                if (d >= 0)
                    b_ = edges[++v_it];
                if (d == 0)
                    ++count;
            }
        }
        tri = count;
    }
#elif ACTOR == 5
    // ---- wedge lookup, one PE (no actors): a thread per edge (u, v) of L, grid stride; every w of L[u] before v
    // is looked up in L[v] (bale's tc_process, without the message)
    {
        const int m = (int)a->m;
        const int* erow = (const int*)a->erow_addr;
        const int from = blk * (GLANES * (SPLIT)) + tid;
        const int step = (int)(NASSUME) * (SPLIT);
        uint32_t count = 0;
#pragma GCC unroll 1
        for (int j = from; j < m; j += step) {
            const int u = erow[j], v = lnonzero[j];
            const int* nbrV = &lnonzero[loffset[v]];
            const int nbrV_size = loffset[v + 1] - loffset[v];
#pragma GCC unroll 1
            for (int k = loffset[u]; k < j; k++)
                count += (uint32_t)binary_search(nbrV, 0, nbrV_size - 1, lnonzero[k]);
        }
        tri = count;
    }
#elif ACTOR == 6
    // ---- GroupTC (TC-Compare approach/GroupTC/tc.cu grouptc), one GPU: block_bucketnum = blockDim, subwarp_size =
    // the warp, warp_step = warps per block; their table N(u) after v (u < v) is L[u] before v here
    {
        const int B = GLANES * (SPLIT);                    // block_bucketnum
        const int m = (int)a->m;
        const int* erow = (const int*)a->erow_addr;        // src_list
        int* sh_tb_start = (int*)__local_mem((size_t)4 * B * sizeof(int));
        int* sh_tb_len = sh_tb_start + B;
        int* sh_ele_start = sh_tb_start + 2 * B;
        int* sh_ele_len = sh_tb_start + 3 * B;
        uint32_t P_counter = 0;
#pragma GCC unroll 1
        for (int i = blk * B; i < m; i += NB * B) {
            if (i + tid < m) {
                int src = erow[i + tid];
                int dst = lnonzero[i + tid];
                int temp;

                int tb_start, tb_len, ele_start, ele_len;
                tb_start = loffset[src];
                tb_len = i + tid - tb_start;
                ele_start = loffset[dst];
                ele_len = loffset[dst + 1] - ele_start;

                // Flip search table with search element
                if (tb_len * 2 < ele_len) {
                    temp = tb_start; tb_start = ele_start; ele_start = temp;
                    temp = tb_len; tb_len = ele_len; ele_len = temp;
                }

                sh_tb_start[tid] = tb_start;
                sh_tb_len[tid] = tb_len;
                sh_ele_start[tid] = ele_start;
                sh_ele_len[tid] = ele_len;
            }

            __syncthreads();

            int now = (int)sub;
            int end = (m - i) < B ? (m - i) : B;
            int workid = (int)ln;
            int offset = 0;
            int last_now = -1;

            int neighbor_degree = sh_ele_len[now];
            // Get 2-hop neighbors
#pragma GCC unroll 1
            while (now < end) {
                // If the current 1-hop neighbor has been processed, find the next 1-hop neighbor to process.
                while (now < end && workid >= neighbor_degree) {
                    now += (SPLIT);
                    if (now < end) {
                        workid -= neighbor_degree;
                        neighbor_degree = sh_ele_len[now];
                    }
                }

                if (now < end) {
                    // If this time is the same 1-hop neighbor as last time, use the offset of the last search
                    offset = last_now == now ? offset : 0;

                    // 2-hop neighbor found, Perform the binary search
                    P_counter += (uint32_t)gtc_bin_search(lnonzero + (sh_tb_start[now] + offset), sh_tb_len[now] - offset,
                                                          lnonzero[sh_ele_start[now] + workid], offset);
                    last_now = now;
                }
                workid += GLANES;
            }
            __syncthreads();
        }
        tri = P_counter;
    }
#elif ACTOR == 7
    // ---- TRUST (TC-Compare approach/TRUST/tc.cu trust), one GPU: n_threads = blockDim, WARPSIZE = the warp, a
    // superwarp = two warps (theirs 64 of 1024 threads), block_bucketnum = TR_BB; G_counter's reduction is the
    // per-thread counter every kernel here writes
    {
        const int block_bucketnum = TR_BB, warp_bucketnum = 32, shared_BUCKET_SIZE = 6, BUCKET_SIZE = 100;
        const int USE_WARP = 2, CHUNK_SIZE = 1;
        const int BDIM = GLANES * (SPLIT), SUPERWARP = 2 * GLANES, NSUPER = BDIM / SUPERWARP;
        const int* adj_list = lnonzero;
        const int* beg_pos = loffset;
        const int vertex_count = (int)a->numrows, warpfirstvertex = (int)a->wfv;
        int* partition = (int*)a->part_addr;
        int* G_INDEX = (int*)a->gidx_addr;
        int* bin_count = (int*)__local_mem((size_t)(7 * TR_BB + 2) * sizeof(int));
        int* shared_partition = bin_count + block_bucketnum;     // block_bucketnum * shared_BUCKET_SIZE + 1
        volatile int* ver = bin_count + 7 * block_bucketnum + 1;

        int BIN_START = blk * block_bucketnum * BUCKET_SIZE;
        uint32_t P_counter = 0;

        // CTA for large degree vertex
        int vertex = blk * CHUNK_SIZE;
        int vertex_end = vertex + CHUNK_SIZE;
#pragma GCC unroll 1
        while (vertex < warpfirstvertex) {
            int start = beg_pos[vertex];
            int end = beg_pos[vertex + 1];
            int now = tid + start;
            int MODULO = block_bucketnum - 1;
            int BIN_OFFSET = 0;
            // clean bin_count
            for (int i = tid; i < block_bucketnum; i += BDIM)
                bin_count[i] = 0;
            __syncthreads();

            // count hash bin
            while (now < end) {
                int temp = adj_list[now];
                int bin = temp & MODULO;
                int index;
                index = __atomic_fetch_add(&bin_count[bin], 1, __ATOMIC_RELAXED);
                if (index < shared_BUCKET_SIZE) {
                    shared_partition[index * block_bucketnum + bin] = temp;
                } else if (index < BUCKET_SIZE) {
                    index = index - shared_BUCKET_SIZE;
                    partition[index * block_bucketnum + bin + BIN_START] = temp;
                }
                now += BDIM;
            }
            __syncthreads();

            // list intersection
            now = beg_pos[vertex];
            end = beg_pos[vertex + 1];
            int superwarp_ID = tid / SUPERWARP;
            int superwarp_TID = tid % SUPERWARP;
            int workid = superwarp_TID;
            now = now + superwarp_ID;
            int neighbor = adj_list[now];
            int neighbor_start = beg_pos[neighbor];
            int neighbor_degree = beg_pos[neighbor + 1] - neighbor_start;
            while (now < end) {
                while (now < end && workid >= neighbor_degree) {
                    now += NSUPER;
                    workid -= neighbor_degree;
                    neighbor = adj_list[now];
                    neighbor_start = beg_pos[neighbor];
                    neighbor_degree = beg_pos[neighbor + 1] - neighbor_start;
                }
                if (now < end) {
                    int temp = adj_list[neighbor_start + workid];
                    int bin = temp & MODULO;
                    P_counter += (uint32_t)trust_linear_search(temp, shared_partition, partition, bin_count, bin + BIN_OFFSET, BIN_START);
                }
                workid += SUPERWARP;
            }

            __syncthreads();
            vertex++;
            if (vertex == vertex_end) {
                if (tid == 0) {
                    *ver = __atomic_fetch_add(&G_INDEX[1], CHUNK_SIZE, __ATOMIC_RELAXED);
                }
                __syncthreads();
                vertex = *ver;
                vertex_end = vertex + CHUNK_SIZE;
            }
        }

        // warp method
        int WARPID = tid / GLANES;
        int WARP_TID = tid % GLANES;
        vertex = warpfirstvertex + ((WARPID + blk * BDIM / GLANES)) * CHUNK_SIZE;
        vertex_end = vertex + CHUNK_SIZE;
#pragma GCC unroll 1
        while (vertex < vertex_count) {
            int degree = beg_pos[vertex + 1] - beg_pos[vertex];
            if (degree < USE_WARP)
                break;
            int start = beg_pos[vertex];
            int end = beg_pos[vertex + 1];
            int now = WARP_TID + start;
            int MODULO = warp_bucketnum - 1;
            int BIN_OFFSET = WARPID * warp_bucketnum;
            // clean bin_count
            for (int i = BIN_OFFSET + WARP_TID; i < BIN_OFFSET + warp_bucketnum; i += GLANES)
                bin_count[i] = 0;

            // count hash bin
            while (now < end) {
                int temp = adj_list[now];
                int bin = temp & MODULO;
                bin += BIN_OFFSET;
                int index;
                index = __atomic_fetch_add(&bin_count[bin], 1, __ATOMIC_RELAXED);
                if (index < shared_BUCKET_SIZE) {
                    shared_partition[index * block_bucketnum + bin] = temp;
                } else if (index < BUCKET_SIZE) {
                    index = index - shared_BUCKET_SIZE;
                    partition[index * block_bucketnum + bin + BIN_START] = temp;
                }
                now += GLANES;
            }

            now = beg_pos[vertex];
            end = beg_pos[vertex + 1];

            int workid = WARP_TID;
            while (now < end) {
                int neighbor = adj_list[now];
                int neighbor_start = beg_pos[neighbor];
                int neighbor_degree = beg_pos[neighbor + 1] - neighbor_start;

                while (now < end && workid >= neighbor_degree) {
                    now++;
                    workid -= neighbor_degree;
                    neighbor = adj_list[now];
                    neighbor_start = beg_pos[neighbor];
                    neighbor_degree = beg_pos[neighbor + 1] - neighbor_start;
                }
                if (now < end) {
                    int temp = adj_list[neighbor_start + workid];
                    int bin = temp & MODULO;
                    P_counter += (uint32_t)trust_linear_search(temp, shared_partition, partition, bin_count, bin + BIN_OFFSET, BIN_START);
                }
                now = vx_shfl_idx(now, GLANES - 1, GLANES - 1, 0);         // __shfl_sync(.., now, 31)
                workid = vx_shfl_idx(workid, GLANES - 1, GLANES - 1, 0);
                workid += WARP_TID + 1;
            }
            vertex++;
            if (vertex == vertex_end) {
                if (WARP_TID == 0) {
                    vertex = __atomic_fetch_add(&G_INDEX[2], CHUNK_SIZE, __ATOMIC_RELAXED);
                }
                vertex = vx_shfl_idx(vertex, 0, GLANES - 1, 0);
                vertex_end = vertex + CHUNK_SIZE;
            }
        }
        tri = P_counter;
    }
#else
    // ---- agp kernel, one PE: a thread per row, grid stride
    {
        const uint32_t gtid = ((uint32_t)blk * (SPLIT) + sub) * GLANES + ln;
        const uint32_t stride = (uint32_t)(NASSUME) * (SPLIT);
        int local_count = 0;
#pragma GCC unroll 1
        for (uint32_t u = gtid; u < a->numrows; u += stride) {
            const int* nbrU = &lnonzero[loffset[u]];
            const int nbrU_size = loffset[u + 1] - loffset[u];
#pragma GCC unroll 1
            for (int j = loffset[u]; j < ROW_END(u); j++) {
                const int v = lnonzero[j];
                const int* nbrV = &lnonzero[loffset[v]];
                const int nbrV_size = loffset[v + 1] - loffset[v];
                local_count += intersect(nbrU, nbrU_size, nbrV, nbrV_size);
            }
        }
#if BAL
        // the split rows: this warp's chunks {u, ja, jb}, lane ln taking j = ja + ln, ja + ln + 16, ...
        const uint32_t wi = (uint32_t)blk * (SPLIT) + sub;
#pragma GCC unroll 1
        for (uint32_t c = chofs[wi]; c < chofs[wi + 1]; c++) {
            const int u = (int)ch[3 * c];
            const int* nbrU = &lnonzero[loffset[u]];
            const int nbrU_size = loffset[u + 1] - loffset[u];
            const int jb = loffset[u] + (int)ch[3 * c + 2];
#pragma GCC unroll 1
            for (int j = loffset[u] + (int)(ch[3 * c + 1] + ln); j < jb; j += GLANES) {
                const int v = lnonzero[j];
                const int* nbrV = &lnonzero[loffset[v]];
                const int nbrV_size = loffset[v + 1] - loffset[v];
                local_count += intersect(nbrU, nbrU_size, nbrV, nbrV_size);
            }
        }
#endif
        tri = (uint32_t)local_count;
    }
#endif
    vx_fence();
    const uint64_t tb = csr_read(VX_CSR_MCYCLE);
    gbarrier_ep(bar, 1, tid, NB, 1);
    const uint64_t tc = csr_read(VX_CSR_MCYCLE);
#if ACTOR == 1
    // ---- receive: tc_process of my actors' queues, in my own rows
#if LANE_RECV
    // piece C off: lane ln takes queues qofs[o] + ln, + 16, ... of each of my actors (RTILE: its own HS-slot hash set
    // in the scratchpad, built by that lane), or (RTILE=0) the queues of actors ln, ln + 16, ... of my OW
#if RTILE
    uint32_t* lT = (uint32_t*)__local_mem((size_t)(SPLIT) * (TS_W) * 4) + sub * (TS_W) + ln * (HS);
    const uint32_t* tdesc = (const uint32_t*)a->tdesc_addr;
    const uint32_t* qofs = (const uint32_t*)a->qofs_addr;
#pragma GCC unroll 1
    for (uint32_t k = 0; k < (uint32_t)(OW); k++) {
        const uint32_t o = (uint32_t)blk * GLANES + sub + k * (SPLIT);
        const int* lo = loffset + (size_t)o * lrs;
#pragma GCC unroll 1
        for (uint32_t qi = qofs[o] + ln; qi < qofs[o + 1]; qi += GLANES) {
            const uint32_t tl = QCOUNT(qi);
            if (tl == 0) continue;
            msgs += tl;
            const uint32_t r0 = tdesc[2 * qi], nr = tdesc[2 * qi + 1];
            const msg_t* q = QPTR_(qs, QSTRIDE_E, qi);
            const uint64_t* cr = ctab + (size_t)qi * a->ovf_maxc;
            if (nr >> 31) { tri += recv_tile_l(q, tl, cr, lo + r0, lnonzero, wbits); continue; }   // a long row
#if HADAPT
            const uint32_t ne = (uint32_t)(lo[r0 + nr] - lo[r0]);
            if (4 * tl < (uint32_t)(HADAPT) * ne) { tri += recv_tile_l(q, tl, cr, lo + r0, lnonzero, wbits); continue; }
#endif
            const uint32_t hm = (HS) - 1;
#pragma GCC unroll 1
            for (uint32_t i = 0; i <= hm; i++) lT[i] = HS_EMPTY;
#pragma GCC unroll 1
            for (uint32_t r = 0; r < nr; r++) {
#pragma GCC unroll 1
                for (int j = lo[r0 + r]; j < lo[r0 + r + 1]; j++) {
                    const uint32_t key = r << wbits | (uint32_t)lnonzero[j];
                    uint32_t h = HS_HASH(key);
#pragma GCC unroll 1
                    while (lT[h] != HS_EMPTY) h = (h + 1) & hm;
                    lT[h] = key;
                }
            }
            tri += recv_hash_l(q, tl, cr, lT);
        }
    }
#else
#pragma GCC unroll 1
    for (uint32_t k = ln; k < (uint32_t)(OW); k += GLANES) {
        const uint32_t o = (uint32_t)blk * GLANES + sub + k * (SPLIT);
        const uint32_t tl = QCOUNT(o);
        msgs += tl;
        tri += recv_actor_l(QPTR_(qs, QSTRIDE_E, o), tl, ctab + (size_t)o * a->ovf_maxc, loffset + (size_t)o * lrs, lnonzero);
    }
#endif
    vx_fence();
#else
#if RTILE
    int* sp = (int*)__local_mem((size_t)(SPLIT) * (TS_W) * 4) + sub * (TS_W);
    const uint32_t* tdesc = (const uint32_t*)a->tdesc_addr;
    const uint32_t* qofs = (const uint32_t*)a->qofs_addr;
#endif
#pragma GCC unroll 1
    for (uint32_t k = 0; k < (uint32_t)(OW); k++) {
        const uint32_t o = (uint32_t)blk * GLANES + sub + k * (SPLIT);
#if RTILE
        // the owner's tiles: load the tile into my scratchpad region, drain its queue there
        const int* lo = loffset + (size_t)o * lrs;
#pragma GCC unroll 1
        for (uint32_t qi = qofs[o]; qi < qofs[o + 1]; qi++) {
            const uint32_t tl = QCOUNT(qi);
            if (tl == 0) continue;
            msgs += tl;
            const uint32_t r0 = tdesc[2 * qi], nr = tdesc[2 * qi + 1];
            const int* offs; const int* cols;
            if (nr >> 31) { offs = lo + r0; cols = lnonzero; }           // a row longer than a tile
#if RTILE >= 2
            else {
#if RTILE == 2
                const uint32_t* img = (const uint32_t*)a->timg_addr + (size_t)qi * (HS);
                for (uint32_t i = ln; i < (uint32_t)(HS); i += GLANES) sp[i] = (int)img[i];
#else
                // build the tile's hash set from my rows r0 .. r0 + nr: lanes over the rows; a key is written to an
                // empty slot and read back, and a lane whose key another lane overwrote moves on
#if HSIZE || HADAPT
                const uint32_t ne = (uint32_t)(lo[r0 + nr] - lo[r0]);    // the tile's edges
#endif
#if HADAPT
                if (4 * tl < (uint32_t)(HADAPT) * ne) {    // too few messages to pay for the build: search the rows in place
                    tri += recv_tile(QPTR_(qs, QSTRIDE_E, qi) + QIX_(qs, ln), tl, ctab + (size_t)qi * a->ovf_maxc, lo + r0, lnonzero, ln, wbits);
                    continue;
                }
#endif
#if HSIZE
                uint32_t hb = 4;
#pragma GCC unroll 1
                while ((1u << hb) < 2 * ne) hb++;
                const uint32_t hm = (1u << hb) - 1u;
#define TC_HASH(k) (((uint32_t)(k) * 0x9E3779B1u) >> (32 - hb))
#else
                const uint32_t hm = (HS) - 1;
#define TC_HASH(k) HS_HASH(k)
#endif
                volatile uint32_t* T = (volatile uint32_t*)sp;
                for (uint32_t i = ln; i <= hm; i += GLANES) T[i] = HS_EMPTY;
                vx_fence();
#pragma GCC unroll 1
                for (uint32_t r = ln; r < nr; r += GLANES) {
#pragma GCC unroll 1
                    for (int j = lo[r0 + r]; j < lo[r0 + r + 1]; j++) {
                        const uint32_t key = r << wbits | (uint32_t)lnonzero[j];
                        uint32_t h = TC_HASH(key);
#pragma GCC unroll 1
                        while (true) {
                            if (T[h] == HS_EMPTY) { T[h] = key; if (T[h] == key) break; }
                            h = (h + 1) & hm;
                        }
                    }
                }
#undef TC_HASH
#endif
                vx_fence();
#if HSIZE
                tri += recv_hash(QPTR_(qs, QSTRIDE_E, qi) + QIX_(qs, ln), tl, ctab + (size_t)qi * a->ovf_maxc, (const uint32_t*)sp, ln, hb);
#else
                tri += recv_hash(QPTR_(qs, QSTRIDE_E, qi) + QIX_(qs, ln), tl, ctab + (size_t)qi * a->ovf_maxc, (const uint32_t*)sp, ln);
#endif
                continue;
            }
#else
            else {
                const int base = lo[r0], nnz = lo[r0 + nr] - base;
                for (uint32_t i = ln; i <= nr; i += GLANES) sp[i] = lo[r0 + i] - base;
                for (int i = (int)ln; i < nnz; i += GLANES) sp[nr + 1 + i] = lnonzero[base + i];
                vx_fence();
                offs = sp; cols = sp + nr + 1;
            }
#endif
            tri += recv_tile(QPTR_(qs, QSTRIDE_E, qi) + QIX_(qs, ln), tl, ctab + (size_t)qi * a->ovf_maxc, offs, cols, ln, wbits);
        }
#elif MSG32
        // the owner's queues, tile by tile: queue t holds v_local = (t << vbits) | (message >> wbits)
#pragma GCC unroll 1
        for (uint32_t t = 0; t < (1u << tbits); t++) {
            const uint32_t qi = (o << tbits) + t;
            const uint32_t tl = QCOUNT(qi);
            msgs += tl;
            tri += recv_actor(QPTR_(qs, QSTRIDE_E, qi) + QIX_(qs, ln), tl, ctab + (size_t)qi * a->ovf_maxc,
                              loffset + (size_t)o * lrs, lnonzero, ln, t << vbits, wbits);
        }
#else
        const uint32_t tl = QCOUNT(o);
        msgs += tl;
        tri += recv_actor(QPTR_(qs, QSTRIDE_E, o) + QIX_(qs, ln), tl, ctab + (size_t)o * a->ovf_maxc,
                          loffset + (size_t)o * lrs, lnonzero, ln);
#endif
    }
    vx_fence();
#endif
#elif ACTOR == 4
    // ---- serve my actors' requests with their rows, then intersect what my actors received
#pragma GCC unroll 1
    for (uint32_t k = 0; k < (uint32_t)(OW); k++) {
        const uint32_t o = (uint32_t)blk * GLANES + sub + k * (SPLIT);
        const uint32_t tl = QCOUNT(o);
        msgs += tl;
        pull_serve(QPTR_(qs, QSTRIDE_E, o) + QIX_(qs, ln), tl, ctab + (size_t)o * a->ovf_maxc, loffset + (size_t)o * lrs, lnonzero, ln, a);
    }
    vx_fence();
    tserve = csr_read(VX_CSR_MCYCLE) - tc;
    gbarrier_ep(bar, 4, tid, NB, 1);
    {
        const int* data = (const int*)a->data_addr;
        const uint32_t* hdr = (const uint32_t*)a->hdr_addr;
        const uint32_t* hofs = (const uint32_t*)a->hofs_addr;
        const volatile uint32_t* rtail = (const volatile uint32_t*)a->rtail_addr;
#pragma GCC unroll 1
        for (uint32_t k = 0; k < (uint32_t)(OW); k++) {
            const uint32_t o = (uint32_t)blk * GLANES + sub + k * (SPLIT);
            tri += pull_merge(loffset + (size_t)o * lrs, lnonzero, lr, o, ln, data, hdr + 3 * (size_t)hofs[o], rtail[2 * o + 1] - hofs[o]);
        }
    }
    vx_fence();
#endif
    const uint64_t td = csr_read(VX_CSR_MCYCLE);
    gbarrier_ep(bar, 2, tid, NB, 1);
#if ACTOR == 1 || ACTOR == 4
#if !SWQ
    if (blk == 0) aq_round();                             // block-uniform: charge the partial lines
#endif
    vx_fence();
    gbarrier_ep(bar, 3, tid, NB, 1);
#endif
    const uint64_t t1 = csr_read(VX_CSR_MCYCLE);
    ((uint32_t*)a->cnt_addr)[((uint32_t)blk * (SPLIT) + sub) * GLANES + ln] = tri;
    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[(uint32_t)blk * (SPLIT) + sub];
    ph->send = tb - t0; ph->barrier = (tc - tb) + (t1 - td); ph->recv = td - tc; ph->total = t1 - t0;
    PH_MSGS(ph, msgs); ph->t0 = t0; ph->t1 = t1; ph->pad = tserve;
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N / GLANES;
    uint32_t block_dim = GLANES * (SPLIT);
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
#else
// ---- the plain actor: actor_ig's mailbox primitives, made non-blocking so no lane spins alone

// push a packet to dst; 0 if the channel is full
static inline int try_send(ring_t* grid, int N, int dst, int sender, uint32_t done, int idx_field, int val_field) {
    ring_t* out = &grid[sender * N + dst];
    if (out->tail - out->head >= CAP) return 0;
    out->slots[out->tail % CAP].sender    = sender;
    out->slots[out->tail % CAP].done_flag = done;
    out->slots[out->tail % CAP].idx       = idx_field;
    out->slots[out->tail % CAP].value     = val_field;
    vx_fence();                              // publish slot before bumping tail
    out->tail = out->tail + 1;
    return 1;
}

// non-blocking pop from src's channel; returns 0 if empty
static inline int try_recv(ring_t* grid, int N, int me, int src, packet_t* out) {
    ring_t* in = &grid[src * N + me];
    if (in->head == in->tail) return 0;
    vx_fence();                              // see payload before advancing head
    *out = in->slots[in->head % CAP];
    in->head = in->head + 1;
    return 1;
}

// generate_pairs as a resumable iterator: row u_local, v at j, w at k < j; false when exhausted
typedef struct { uint32_t u; int rs, je, ug, j, k; } gen_t;
static inline int gen_valid(gen_t* g, const int* loffset, const int* lnonzero, uint32_t lnumrows, int npes, int mype) {
    while (1) {
        if (g->j < g->je && lnonzero[g->j] < g->ug) {     // bale's "if (v_global >= u_global) break"
            if (g->k < g->j) return 1;
            g->j++; g->k = g->rs; continue;
        }
        if (++g->u >= lnumrows) return 0;
        g->rs = loffset[g->u]; g->je = loffset[g->u + 1]; g->ug = (int)g->u * npes + mype; g->j = g->rs; g->k = g->rs;
    }
}

void kernel_body(args_t* __UNIFORM__ a) {
    ring_t* grid = (ring_t*)a->grid_addr;
    const int N = (int)a->N, me = (int)blockIdx.x;
    const uint32_t lr = a->lrows;
    const int* loffset = (const int*)a->loffset_addr + (size_t)me * (lr + 1);
    const int* lnonzero = (const int*)a->lnonzero_addr;
    uint32_t tri = 0, handled = 0;
    int dones = 0, dsent = 0;
    uint64_t hcyc = 0, t_gen = 0;
    const uint64_t t0 = csr_read(VX_CSR_MCYCLE);
    gen_t g; g.u = 0; g.rs = loffset[0]; g.je = loffset[1]; g.ug = me; g.j = g.rs; g.k = g.rs;
    int more = lr > 0 && gen_valid(&g, loffset, lnonzero, lr, N, me);
    if (!more) t_gen = t0;
    while (!vx_vote_all(dones == N && dsent == N)) {
        int blocked = 0;
        if (more) {                          // send one message (Actor.send(owner(v), v, w))
            const int v = lnonzero[g.j];
            if (try_send(grid, N, v % N, me, 0, v / N, lnonzero[g.k])) {
                g.k++;
                more = gen_valid(&g, loffset, lnonzero, lr, N, me);
                if (!more) t_gen = csr_read(VX_CSR_MCYCLE);
            } else blocked = 1;
        } else if (dsent < N) {              // done to every actor
            if (try_send(grid, N, dsent, me, 1, 0, 0)) dsent++; else blocked = 1;
        }
        if (blocked || !more) {              // progress: drain my inbox, tc_process per message
            const uint64_t ts = csr_read(VX_CSR_MCYCLE);
            for (int src = 0; src < N; src++) {
                packet_t pkt;
                if (try_recv(grid, N, me, src, &pkt)) {
                    if (pkt.done_flag == 1) {
                        dones++;
                    } else {
                        const int* nbrV = &lnonzero[loffset[pkt.idx]];
                        const int nbrV_size = loffset[pkt.idx + 1] - loffset[pkt.idx];
                        tri += (uint32_t)binary_search(nbrV, 0, nbrV_size - 1, pkt.value);
                        handled++;
                    }
                }
            }
            hcyc += csr_read(VX_CSR_MCYCLE) - ts;
        }
    }
    const uint64_t t1 = csr_read(VX_CSR_MCYCLE);
    ((uint32_t*)a->cnt_addr)[me] = tri;
    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];
    ph->send = t_gen - t0; ph->recv = hcyc; ph->barrier = (t1 - t0) - (t_gen - t0);   // after my last send
    ph->total = t1 - t0; ph->msgs = handled; ph->t0 = t0; ph->t1 = t1;
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N;
    uint32_t block_dim = 1;
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
#endif
