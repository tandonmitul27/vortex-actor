// Baseline histogram (no actors): each PE does M atomic bumps to random buckets.
// Same workload as actor_histo, but direct atomic adds instead of messages.

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"
#ifndef SPLIT
#define SPLIT 1   // lanes per actor, as in the actor kernels (log T41): the same problem
                  // on the same machine. Block b is actor b % N, slice b / N, so
                  // consecutive lanes stay consecutive actors and every coalesced store
                  // the baseline had at one warp it keeps.
#endif
#if ((M) % (SPLIT)) != 0
#error "M must be a multiple of SPLIT, or the last M % SPLIT updates are never issued"
#endif

// deterministic hash (host replays it to verify)
static inline uint32_t mix(uint32_t a, uint32_t b) {
    uint32_t x = a * 0x9E3779B9u + b * 0x85EBCA77u;
    x ^= (x >> 13);
    x *= 0xC2B2AE3Du;
    x ^= (x >> 16);
    return x;
}

void kernel_body(args_t* __UNIFORM__ a) {
    int*     counts  = (int*)a->counts_addr;
    int      N       = a->N;
    int      me      = (int)(blockIdx.x % (uint32_t)N);
    int      sub     = ((SPLIT) == 1) ? 0 : (int)(blockIdx.x / (uint32_t)N);
    uint32_t G_total = (uint32_t)(L * N);
    const uint32_t ilo = (uint32_t)sub * (uint32_t)((M) / (SPLIT));

    uint64_t apply = 0;
    uint64_t t0 = csr_read(VX_CSR_MCYCLE);
    // Constant trip count, ilo only offsets the index: bounds derived from the
    // block id are divergent to the compiler, which then predicates every
    // iteration (T42). The same fix as the actor kernels, so both sides compile
    // the same way.
    for (uint32_t k = 0; k < (uint32_t)((M) / (SPLIT)); k++) {
        const uint32_t i = ilo + k;
        uint32_t g   = mix(me, i) % G_total;
        uint32_t bkt = g;
        uint64_t ta = csr_read(VX_CSR_MCYCLE);
        __atomic_fetch_add(&counts[bkt], 1, __ATOMIC_RELAXED);  // PEs may share a bucket
        apply += csr_read(VX_CSR_MCYCLE) - ta;
    }
    uint64_t t_loop = csr_read(VX_CSR_MCYCLE);
    vx_fence();             // wait out the posted-write backlog (the contention)
    uint64_t t1 = csr_read(VX_CSR_MCYCLE);

    if (sub == 0) {
    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];
    ph->apply = apply;              // atomic-add issue cost
    ph->drain = t1 - t_loop;        // contended backlog waited out at the fence
    ph->total = t1 - t0;
    ph->t_start = t0; ph->t_end = t1;
    }
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N * (SPLIT);
    uint32_t block_dim = 1;
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
