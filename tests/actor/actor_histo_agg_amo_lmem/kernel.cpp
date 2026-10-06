// Bale histogram, aggregating actor + tree barrier, two-level messaging via LMEM,
// INT slots + amoadd accumulate. Same routing as actor_histo_agg_lmem (same-core
// messages in LMEM, cross-core in global), but each message slot is an int[L] and
// the accumulate is a single-instruction amoadd into the actor's own slot (still
// single-writer -- no contention -- just a cheaper add than the byte RMW). The 4B
// slots are 4x the byte version's footprint, so this only fits LMEM at small P.

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
    msg_t*   gmsg     = (msg_t*)a->msg_addr;       // global grid (cross-core only)
    int*     counts   = (int*)a->counts_addr;
    int      N        = a->N;
    volatile uint32_t* arrived  = (volatile uint32_t*)a->arrived_addr;
    volatile uint32_t* released = (volatile uint32_t*)a->released_addr;

    int C  = (int)gridDim.x;                       // cores (one block per core)
    int P  = (int)blockDim.x;                      // actors per core
    int mc = (int)blockIdx.x;                      // my core
    int ml = (int)threadIdx.x;                     // my local id within the core
    int me = mc * P + ml;                          // global actor id
    (void)C;
    int*     lcounts  = &counts[me * L];
    uint32_t G_total  = (uint32_t)(L * N);

    // per-core LMEM message grid: lmsg[local_sender * P + local_recv]
    msg_t* lmsg = (msg_t*)__local_mem((size_t)P * P * sizeof(msg_t));

    int Lc = 2 * me + 1, Rc = 2 * me + 2;
    int haveL = (Lc < N), haveR = (Rc < N);

    uint64_t c_sd = 0, c_sn = 0, c_h = 0;
    uint64_t bd_acc_on = 0, bd_acc_off = 0;        // BREAKDOWN accumulators
    uint64_t t0 = csr_read(VX_CSR_MCYCLE);

    // 0. LMEM is not zero-initialized: each actor clears its own sender row.
    for (int j = 0; j < P; j++)
        for (int b = 0; b < L; b++) lmsg[ml * P + j].c[b] = 0;
    __syncthreads();

    // 1. ACCUMULATE my M bumps with amoadd, routing each by the owner's core:
    //    same core -> LMEM (single-writer ml), other core -> global (single-writer me).
    for (uint32_t i = 0; i < M; i++) {
        uint32_t g     = mix(me, i) % G_total;
        uint32_t owner = g % (uint32_t)N;
        uint32_t lidx  = g / (uint32_t)N;
        int oc = (int)owner / P;                   // owner's core
#if BD_FINE
        // BREAKDOWN (fine run only): time the two routing paths separately.
        // The in-loop csr reads perturb timing (~9% at N=256), so this build is
        // used ONLY for the on/off-chip ratio, never for absolute totals.
        uint64_t bs = csr_read(VX_CSR_MCYCLE);
        if (oc == mc) {
            int ol = (int)owner - mc * P;          // owner's local id
            __atomic_fetch_add(&lmsg[ml * P + ol].c[lidx], 1, __ATOMIC_RELAXED);
            bd_acc_on += csr_read(VX_CSR_MCYCLE) - bs;
        } else {
            __atomic_fetch_add(&gmsg[me * N + (int)owner].c[lidx], 1, __ATOMIC_RELAXED);
            bd_acc_off += csr_read(VX_CSR_MCYCLE) - bs;
        }
#else
        if (oc == mc) {
            int ol = (int)owner - mc * P;          // owner's local id
            __atomic_fetch_add(&lmsg[ml * P + ol].c[lidx], 1, __ATOMIC_RELAXED);
        } else {
            __atomic_fetch_add(&gmsg[me * N + (int)owner].c[lidx], 1, __ATOMIC_RELAXED);
        }
#endif
    }
    // 2. PUBLISH: fence the global writes; barrier makes the LMEM writes visible.
    vx_fence();
    __syncthreads();
    uint64_t t1 = csr_read(VX_CSR_MCYCLE);
    c_sd = t1 - t0;

    // 3. tree barrier (flags only): combine arrived[] up, broadcast released[] down.
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
    uint64_t t2 = csr_read(VX_CSR_MCYCLE);
    c_sn = t2 - t1;

    // 4. DRAIN: same-core senders from LMEM, cross-core senders from global.
    vx_fence();
    uint64_t t_dr0 = csr_read(VX_CSR_MCYCLE);
    for (int sl = 0; sl < P; sl++) {               // same-core: read my LMEM column
        int* c = lmsg[sl * P + ml].c;
        for (int b = 0; b < L; b++) lcounts[b] += c[b];
    }
    uint64_t t_dr1 = csr_read(VX_CSR_MCYCLE);      // BREAKDOWN: split the two drains
    for (int src = 0; src < N; src++) {            // cross-core: read my global column
        if (src / P == mc) continue;               // same-core already handled via LMEM
        int* c = gmsg[src * N + me].c;
        for (int b = 0; b < L; b++) lcounts[b] += c[b];
    }
    uint64_t t3 = csr_read(VX_CSR_MCYCLE);
    c_h = t3 - t2;

    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];
    ph->send_data  = c_sd;
    ph->send_done  = c_sn;
    ph->handle_req = 0;
    ph->handle_rep = c_h;
    ph->empty_poll = 0;
    ph->tail_wait  = 0;
    ph->total      = t3 - t0;
    ph->bd_acc_on   = bd_acc_on;      // BREAKDOWN: folds routed to the scratchpad
    ph->bd_acc_off  = bd_acc_off;     // BREAKDOWN: folds routed to global memory
    ph->bd_drain_on  = t_dr1 - t_dr0; // BREAKDOWN: drain of same-core scratchpad slots
    ph->bd_drain_off = t3 - t_dr1;    // BREAKDOWN: drain of cross-core global slots
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    // one block per core; block = the core's actors (they share LMEM)
    uint32_t grid_dim  = vx_num_cores();
    uint32_t block_dim = vx_num_warps() * vx_num_threads();
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
