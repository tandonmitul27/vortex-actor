// Baseline index-gather (no actors): each PE does M direct loads from a shared table.
// Same workload as actor_ig, but plain reads instead of request/reply messages.

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
    int* table = (int*)a->table_addr;
    int* tgt   = (int*)a->tgt_addr;
    int  N     = a->N;
    int  me    = (int)(blockIdx.x % (uint32_t)N);
    int  sub   = ((SPLIT) == 1) ? 0 : (int)(blockIdx.x / (uint32_t)N);
    const uint32_t ilo = (uint32_t)sub * (uint32_t)((M) / (SPLIT));

    uint64_t t0 = csr_read(VX_CSR_MCYCLE);
    // constant trip count, ilo only offsets the index (T42, as in the actor kernels)
    for (uint32_t k = 0; k < (uint32_t)((M) / (SPLIT)); k++) {
        const uint32_t i = ilo + k;
        uint32_t dst  = mix(me, 2*i + 1) % (uint32_t)N;            // which PE's table
        uint32_t lidx = mix(me, 2*i + 2) % (uint32_t)TABLE_SIZE;   // which entry
        tgt[TGT_IDX(me, i, N)] = table[dst * TABLE_SIZE + lidx];   // a[b[i]] gather
    }
    vx_fence();
    uint64_t t1 = csr_read(VX_CSR_MCYCLE);
    if (sub == 0) {
        phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];
        ph->t_start = t0; ph->t_end = t1;
    }
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N * (SPLIT);
    uint32_t block_dim = 1;
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
