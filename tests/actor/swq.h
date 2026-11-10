// SWQ=1: the L2 append ablation. The append unit's job is done by the sending lane with ordinary
// memory operations: slot = atomicAdd(&tails[q], 1), then a plain store of the word. Same storage
// layout as the unit (a region of `cap` words per queue, then OVF_CH-word chunks taken from the pool
// and listed in ctab[q * maxc + c]), so every receive side is unchanged. The unit's internal state
// lives in memory: tails[] (the unit writes them too) and one pool counter (args.swq_addr, zeroed by
// the host, as is ctab). Include after the kernel's own aq_* helpers; use SWQ_GEOMETRY / SWQ_POOL /
// swq_round in place of the unit's control stores.
#pragma once
#if SWQ
static char* g_sq_qb; static uint32_t* g_sq_tl; static uint32_t g_sq_qs, g_sq_ws, g_sq_cap;
static char* g_sq_pool; static uint64_t* g_sq_ctab; static uint32_t g_sq_maxc, g_sq_nch; static uint32_t* g_sq_next;
// every lane stores the same values (warp uniform, as the unit's control stores)
static inline void swq_geometry(uint64_t qbase, uint64_t qstride, uint64_t tailbase, uint64_t wsize, uint64_t cap) {
    g_sq_qb = (char*)qbase; g_sq_qs = (uint32_t)qstride; g_sq_tl = (uint32_t*)tailbase;
    g_sq_ws = (uint32_t)wsize; g_sq_cap = (uint32_t)cap;
}
static inline void swq_pool(uint64_t pool, uint64_t ctab, uint32_t maxc, uint32_t nch, uint64_t next) {
    g_sq_pool = (char*)pool; g_sq_ctab = (uint64_t*)ctab; g_sq_maxc = maxc; g_sq_nch = nch; g_sq_next = (uint32_t*)next;
}
static inline void swq_put(uint32_t q, uint64_t w) {
    const uint32_t t = __atomic_fetch_add(&g_sq_tl[q], 1u, __ATOMIC_RELAXED);
    char* p;
    if (t < g_sq_cap) {
        p = g_sq_qb + (size_t)q * g_sq_qs + (size_t)t * g_sq_ws;
    } else {
        // region full: the lane whose slot opens a chunk takes it from the pool and publishes it;
        // a lane landing later in the same chunk waits for that (1 = the pool or table ran out)
        const uint32_t e = t - g_sq_cap, c = e / (uint32_t)(OVF_CH), o = e % (uint32_t)(OVF_CH);
        if (c >= g_sq_maxc) return;                                   // dropped: the host check fails
        uint64_t* ce = g_sq_ctab + (size_t)q * g_sq_maxc + c;
        if (o == 0) {
            const uint32_t k = __atomic_fetch_add(g_sq_next, 1u, __ATOMIC_RELAXED);
            __atomic_exchange_n(ce, k < g_sq_nch ? (uint64_t)(uintptr_t)(g_sq_pool + (size_t)k * (OVF_CH) * g_sq_ws) : 1ull,
                                __ATOMIC_RELAXED);
        }
        uint64_t b;
        while ((b = __atomic_fetch_or(ce, 0ull, __ATOMIC_RELAXED)) == 0) { }
        if (b == 1) return;                                           // dropped
        p = (char*)(uintptr_t)b + (size_t)o * g_sq_ws;
    }
    if (g_sq_ws == 8) *(uint64_t*)p = w; else *(uint32_t*)p = (uint32_t)w;
}
// END OF ROUND, what the unit does: empty every queue that holds something (its tail and its chunk
// list) and reuse the pool. Every thread of the grid takes queues gt, gt + gn, ...; nq = the
// geometry's queue count. Callers put a barrier after it, as after the unit's round store.
static inline void swq_round(uint32_t nq, uint32_t gt, uint32_t gn) {
#pragma GCC unroll 1
    for (uint32_t q = gt; q < nq; q += gn) {
        const uint32_t t = g_sq_tl[q];
        if (t == 0) continue;
        if (t > g_sq_cap) {
            uint32_t nc = (t - g_sq_cap + (uint32_t)(OVF_CH) - 1) / (uint32_t)(OVF_CH);
            if (nc > g_sq_maxc) nc = g_sq_maxc;
#pragma GCC unroll 1
            for (uint32_t c = 0; c < nc; c++) g_sq_ctab[(size_t)q * g_sq_maxc + c] = 0;
        }
        g_sq_tl[q] = 0;
    }
    if (gt == 0) *g_sq_next = 0;
}
#endif
