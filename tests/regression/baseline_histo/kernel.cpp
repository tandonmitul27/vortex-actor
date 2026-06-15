// Baseline histogram (no actors): each PE does M atomic bumps to random buckets.
// Same workload as actor_histo, but direct atomic adds instead of messages.

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

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
    int      me      = blockIdx.x;
    uint32_t G_total = (uint32_t)(L * N);

    for (uint32_t i = 0; i < M; i++) {
        uint32_t g   = mix(me, i) % G_total;
        uint32_t bkt = g;
        __atomic_fetch_add(&counts[bkt], 1, __ATOMIC_RELAXED);  // PEs may share a bucket
    }
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N;
    uint32_t block_dim = 1;
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
