// Bale histogram, aggregating actor + tree barrier, core-level combining via LMEM.
// See common.h. Each block is one core (blockIdx.x = core, threadIdx.x = local id).
// The core's P actors share one LMEM buffer core_out[owner*L + bucket]; every actor
// amoadds its M bumps into it (combining the core's P actors into one contribution
// per owner). The core flushes that buffer to global cout[core][owner][bucket], the
// tree barrier ensures all cores have flushed, and each owner sums its slice over the
// C cores. Same result as the private-mailbox variants; the global footprint is
// C*N*L (not N*N*L) and the drain is O(C) (not O(N)).

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

static inline uint32_t mix(uint32_t a, uint32_t b) {
    uint32_t x = a * 0x9E3779B9u + b * 0x85EBCA77u;
    x ^= (x >> 13);
    x *= 0xC2B2AE3Du;
    x ^= (x >> 16);
    return x;
}

void kernel_body(args_t* __UNIFORM__ a) {
    int*     cout_g   = (int*)a->cout_addr;        // global C*N*L (flush target)
    int*     counts   = (int*)a->counts_addr;
    int      N        = a->N;
    volatile uint32_t* arrived  = (volatile uint32_t*)a->arrived_addr;
    volatile uint32_t* released = (volatile uint32_t*)a->released_addr;

    int C  = (int)gridDim.x;                        // cores (one block per core)
    int P  = (int)blockDim.x;                       // actors per core
    int mc = (int)blockIdx.x;                       // my core
    int ml = (int)threadIdx.x;                      // my local id within the core
    int me = mc * P + ml;                           // global actor id
    (void)C;
    int*     lcounts  = &counts[me * L];
    uint32_t G_total  = (uint32_t)(L * N);
    int      NL       = N * L;

    // per-core shared combined buffer in LMEM: core_out[owner*L + bucket]
    int* cout = (int*)__local_mem((size_t)NL * sizeof(int));

    int Lc = 2 * me + 1, Rc = 2 * me + 2;
    int haveL = (Lc < N), haveR = (Rc < N);

    uint64_t c_sd = 0, c_sn = 0, c_h = 0, c_fl = 0;
    uint64_t t0 = csr_read(VX_CSR_MCYCLE);

    // 0. zero the shared LMEM buffer cooperatively (P actors split the N*L entries).
    for (int idx = ml; idx < NL; idx += P) cout[idx] = 0;
    __syncthreads();

    // 1. COMBINE: every local actor amoadds its M bumps into the shared LMEM buffer.
    for (uint32_t i = 0; i < M; i++) {
        uint32_t g     = mix(me, i) % G_total;
        uint32_t owner = g % (uint32_t)N;
        uint32_t lidx  = g / (uint32_t)N;
        __atomic_fetch_add(&cout[owner * L + (int)lidx], 1, __ATOMIC_RELAXED);
    }
    __syncthreads();
    uint64_t t1 = csr_read(VX_CSR_MCYCLE);
    c_sd = t1 - t0;

    // 2. FLUSH: copy this core's combined buffer to its global slice (cooperative).
    int base = mc * NL;
    for (int idx = ml; idx < NL; idx += P) cout_g[base + idx] = cout[idx];
    vx_fence();
    uint64_t t2 = csr_read(VX_CSR_MCYCLE);
    c_fl = t2 - t1;

    // 3. tree barrier (flags only): make sure every core has flushed.
    int phase = 0;   // 0 = wait children, 1 = wait release, 2 = done
    while (!vx_vote_all(phase == 2)) {
        if (phase == 0) {
            if ((!haveL || arrived[Lc]) && (!haveR || arrived[Rc])) {
                arrived[me] = 1; vx_fence();
                phase = 1;
            }
        } else if (phase == 1) {
            if (me == 0 || released[me]) {
                if (haveL) released[Lc] = 1;
                if (haveR) released[Rc] = 1;
                vx_fence();
                phase = 2;
            }
        }
    }
    uint64_t t3 = csr_read(VX_CSR_MCYCLE);
    c_sn = t3 - t2;

    // 4. DRAIN: owner me sums its L buckets across the C cores' flushed slices (O(C)).
    vx_fence();
    for (int b = 0; b < L; b++) {
        int s = 0;
        for (int c = 0; c < C; c++) s += cout_g[c * NL + me * L + b];
        lcounts[b] = s;
    }
    uint64_t t4 = csr_read(VX_CSR_MCYCLE);
    c_h = t4 - t3;

    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];
    ph->send_data  = c_sd;          // combine
    ph->send_done  = c_sn;          // tree-barrier flag work
    ph->handle_req = c_fl;          // flush
    ph->handle_rep = c_h;           // drain
    ph->empty_poll = 0;
    ph->tail_wait  = 0;
    ph->total      = t4 - t0;
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = vx_num_cores();
    uint32_t block_dim = vx_num_warps() * vx_num_threads();
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
