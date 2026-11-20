// Measurement-only probes for the motivation figure (fig:mot), shared by the *_aq kernels. Both default off, and then
// every macro expands to its statement alone (the binary is unchanged).
// PROBE_APPLYT=1: per-lane timers around each handler (PA_H) and around each send a handler makes (PA_S). Lanes of a
//   warp read the same cycle counter, so a lane's sum is the warp's time in those regions while that lane was active.
//   At the end of a receive call (warps converged) PA_FLUSH keeps the largest lane's sums (amomaxu into the warp's
//   slots, read back with atomic reads: ISA2 atomics skip the L1) and adds them to the warp's totals ph->pa_h / pa_s.
//   With one queue per lane per call, the largest lane is the one active in every iteration of the call.
//   The kernel times its receive calls (ph->pa_r) and its sends outside handlers (ph->pa_x) at the warp level.
// ROUND1=1: only the first send round runs (the kernel's own choice of senders), timed into ph->r1; nothing is
//   received and the check fails. Paired with PROBE_SEND=1 (../swq.h) for the one-way send of fig:mot (c).
#pragma once
#ifndef PROBE_APPLYT
#define PROBE_APPLYT 0
#endif
#ifndef ROUND1
#define ROUND1 0
#endif
#if PROBE_APPLYT
#define PA_DECL uint64_t pa_h_ = 0, pa_s_ = 0;
#define PA_H(S) { const uint64_t pa0_ = csr_read(VX_CSR_MCYCLE); S; pa_h_ += csr_read(VX_CSR_MCYCLE) - pa0_; }
#define PA_S(S) { const uint64_t ps0_ = csr_read(VX_CSR_MCYCLE); S; pa_s_ += csr_read(VX_CSR_MCYCLE) - ps0_; }
#define PA_FLUSH(PH) { \
    __atomic_fetch_max(&(PH)->pa_slot[0], pa_h_, __ATOMIC_RELAXED); __atomic_fetch_max(&(PH)->pa_slot[1], pa_s_, __ATOMIC_RELAXED); \
    vx_fence(); \
    const uint64_t fh_ = __atomic_fetch_add(&(PH)->pa_slot[0], 0ull, __ATOMIC_RELAXED); \
    const uint64_t fs_ = __atomic_fetch_add(&(PH)->pa_slot[1], 0ull, __ATOMIC_RELAXED); \
    vx_fence(); \
    __atomic_fetch_and(&(PH)->pa_slot[0], 0ull, __ATOMIC_RELAXED); __atomic_fetch_and(&(PH)->pa_slot[1], 0ull, __ATOMIC_RELAXED); \
    ((volatile uint64_t*)&(PH)->pa_h)[0] += fh_; ((volatile uint64_t*)&(PH)->pa_s)[0] += fs_; \
    vx_fence(); pa_h_ = 0; pa_s_ = 0; }
// warp-level region timer, added to ph field F (every lane stores the same value)
#define PA_W0 const uint64_t pw0_ = csr_read(VX_CSR_MCYCLE);
#define PA_W1(PH, F) { ((volatile uint64_t*)&(PH)->F)[0] += csr_read(VX_CSR_MCYCLE) - pw0_; }
#else
#define PA_DECL
#define PA_H(S) S;
#define PA_S(S) S;
#define PA_FLUSH(PH)
#define PA_W0
#define PA_W1(PH, F)
#endif
