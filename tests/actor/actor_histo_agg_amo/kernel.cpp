// actor_histo, AGGREGATING SPSC actor + tree barrier (Conveyor-style combining).
// See common.h. Three phases:
//   1. ACCUMULATE: walk my M bumps into my outgoing ring slots -- one slot per
//      owner, grid[me][owner].counts[bucket]++. Single-writer (me), so no atomics,
//      no contention: M cheap local read-modify-writes.
//   2. PUBLISH: for each owner I hit, fence + set tail=1 -> ONE combined message
//      carrying that owner's L counts. <= N sends, independent of M.
//   3. tree barrier, then a single drain pass adding each incoming message's counts.
// So both send and recv are O(N), not O(M): at high M the messaging amortizes away
// and cost -> the M local adds, chasing the atomicAdd baseline.

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
    ring_t*  grid     = (ring_t*)a->grid_addr;
    int*     counts   = (int*)a->counts_addr;
    int      N        = a->N;
    int      me       = blockIdx.x;
    volatile uint32_t* arrived  = (volatile uint32_t*)a->arrived_addr;
    volatile uint32_t* released = (volatile uint32_t*)a->released_addr;
    int*     lcounts  = &counts[me * L];
    uint32_t G_total  = (uint32_t)(L * N);

    int Lc = 2 * me + 1, Rc = 2 * me + 2;
    int haveL = (Lc < N), haveR = (Rc < N);

    uint64_t c_sd = 0, c_sn = 0, c_h = 0;
    uint64_t t0 = csr_read(VX_CSR_MCYCLE);

    // 1. ACCUMULATE my M bumps into my outgoing slots. Each PE writes ONLY its own
    //    slots (grid[me][*]) so it's contention-free -- but use amoadd anyway: it
    //    collapses the 3-instruction RMW (load+add+store) into ONE instruction,
    //    matching baseline's per-update instruction count with zero contention.
    //    This is the residual the plain-RMW agg pays (~2.7x at high M).
    for (uint32_t i = 0; i < M; i++) {
        uint32_t g     = mix(me, i) % G_total;
        uint32_t owner = g % (uint32_t)N;
        uint32_t lidx  = g / (uint32_t)N;
        __atomic_fetch_add(&grid[me * N + (int)owner].slots[0].counts[lidx], 1, __ATOMIC_RELAXED);
    }
    uint64_t t_acc = csr_read(VX_CSR_MCYCLE);    // end of the O(M) accumulate
    // 2. PUBLISH one combined message per owner that received >=1 bump
    for (int owner = 0; owner < N; owner++) {
        ring_t* out = &grid[me * N + owner];
        int any = 0;
        for (int b = 0; b < L; b++) any |= out->slots[0].counts[b];
        if (any) {
            out->slots[0].sender = (uint32_t)me;
            vx_fence();                          // publish counts before the tail bump
            out->tail = 1;
        }
    }
    uint64_t t1 = csr_read(VX_CSR_MCYCLE);
    c_sd = t1 - t0;                              // accumulate + publish (the O(M)+O(N) send)
    uint64_t c_acc = t_acc - t0;                 // BREAKDOWN: O(M) folds into the global grid
    uint64_t c_pub = t1 - t_acc;                 // BREAKDOWN: O(N) combined-message publish

    // 3. tree barrier (flags only): combine arrived[] up, broadcast released[] down
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

    // 4. final drain: release => everyone sent => add each incoming message's counts.
    //    Each src->me channel holds at most ONE combined message: a single pass.
    for (int src = 0; src < N; src++) {
        ring_t* in = &grid[src * N + me];
        if (in->head != in->tail) {              // a message is present
            vx_fence();
            for (int b = 0; b < L; b++) lcounts[b] += in->slots[0].counts[b];
            in->head = in->tail;
        }
    }
    uint64_t t3 = csr_read(VX_CSR_MCYCLE);
    c_h = t3 - t2;

    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];
    ph->send_data  = c_sd;          // accumulate + publish
    ph->send_done  = c_sn;          // tree-barrier flag work
    ph->handle_req = 0;
    ph->handle_rep = c_h;           // drain (add combined counts)
    ph->empty_poll = 0;
    ph->tail_wait  = 0;
    ph->total      = t3 - t0;
    ph->bd_acc_off = c_acc;         // BREAKDOWN: folds into the global grid
    ph->bd_pub_off = c_pub;         // BREAKDOWN: combined-message publish
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N;
    uint32_t block_dim = 1;
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
