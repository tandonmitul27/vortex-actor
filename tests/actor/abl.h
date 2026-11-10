// Ablation helpers (results/mapping/ABLATION.md), shared by the *_aq kernels.
// LANE_RECV=1 is piece C off: one lane drains one whole queue alone, a word per load, instead of the warp
// taking one queue together. Include after the kernel's queue state macros; needs `qs` (the kernel's
// queue word pointer), `tails`, `ctab`, `maxc`, `msgs`, `blk`, `sub`, `ln`, and QS_W (queue stride in words).
#pragma once
#include "isa2.h"   // QCOUNT / QPTR / QIX: the count instruction and the queue layout under the L2 append (ISA2)
#ifndef LANE_RECV
#define LANE_RECV 0
#endif
#ifndef NOTILE
#define NOTILE    0
#endif
// one whole queue drained by the calling lane: its region, then its overflow chunks
#define LANE_QUEUE(QI, OP) { const uint32_t tl_ = QCOUNT(QI); msgs += tl_; \
    const auto* q_ = QPTR(QI); \
    const uint32_t n0_ = tl_ < (uint32_t)(Q_CAP) ? tl_ : (uint32_t)(Q_CAP); \
    _Pragma("GCC unroll 1") for (uint32_t k_ = 0; k_ < n0_; k_++) OP(q_[QIX(k_)]); \
    uint32_t r_ = tl_ - n0_; const uint64_t* cr_ = ctab + (size_t)(QI) * maxc; \
    _Pragma("GCC unroll 1") for (uint32_t c_ = 0; r_ != 0; c_++) { \
        const uint32_t n_ = r_ < (uint32_t)(OVF_CH) ? r_ : (uint32_t)(OVF_CH); \
        const auto* qc_ = (decltype(q_))(uintptr_t)cr_[c_]; \
        _Pragma("GCC unroll 1") for (uint32_t k_ = 0; k_ < n_; k_++) OP(qc_[k_]); \
        r_ -= n_; } }
// lane ln takes the warp's queues ln, ln + 16, ... (actor-major: OW actors x T tiles); o = actor, t = tile
#define FOR_MY_LANE_QUEUES(o, t, T) for (uint32_t qi_ = ln; qi_ < (uint32_t)(OW) * (T); qi_ += GLANES) \
    for (uint32_t o = blk * GLANES + sub + (qi_ / (T)) * (SPLIT), t = qi_ % (T), once_ = 1; once_; once_ = 0)
// the warp's per-warp message count: lanes add their own under LANE_RECV (low word; the buffer is zeroed)
#if LANE_RECV
#define PH_MSGS(ph, m) __atomic_fetch_add((uint32_t*)&(ph)->msgs, (m), __ATOMIC_RELAXED)
#else
#define PH_MSGS(ph, m) ((ph)->msgs = (m))
#endif
// entries [H, TL) of queue QI drained by the calling lane alone (monotone queues with heads)
#define LANE_RANGE(QI, H, TL, OP) { uint32_t p_ = (H); const uint32_t tl_ = (TL); msgs += tl_ - p_; \
    const auto* q0_ = QPTR(QI); const uint64_t* cr_ = ctab + (size_t)(QI) * maxc; \
    _Pragma("GCC unroll 1") for (; p_ < tl_; p_++) { \
        decltype(q0_) b_; \
        if (p_ < (uint32_t)(Q_CAP)) b_ = q0_ + QIX(p_); \
        else { const uint32_t x_ = p_ - (uint32_t)(Q_CAP); \
            b_ = (decltype(q0_))(uintptr_t)cr_[x_ / (uint32_t)(OVF_CH)] + x_ % (uint32_t)(OVF_CH); } \
        OP(*b_); } }
