// Baseline index-gather (no actors): each PE does M direct loads from a shared table.
// Same workload as actor_ig, but plain reads instead of request/reply messages.

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
    int* table = (int*)a->table_addr;
    int* tgt   = (int*)a->tgt_addr;
    int  N     = a->N;
    int  me    = blockIdx.x;

    for (uint32_t i = 0; i < M; i++) {
        uint32_t dst  = mix(me, 2*i + 1) % (uint32_t)N;            // which PE's table
        uint32_t lidx = mix(me, 2*i + 2) % (uint32_t)TABLE_SIZE;   // which entry
        tgt[me * M + i] = table[dst * TABLE_SIZE + lidx];          // a[b[i]] gather
    }
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N;
    uint32_t block_dim = 1;
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
