// Bale histogram, two threads per actor (MAIN + RECV, split by physical warp id as
// in actor_histo_2t) with a tree barrier for termination. MAIN sends its M bumps
// then runs the combining tree barrier (arrived[] up, released[] down); the root
// self-releases so RECV[0] can see it. RECV waits for its release flag -- which
// means every actor has finished sending -- then drains its rings once and applies
// the bumps. A ring holds at most M < CAP messages, so nothing overflows while RECV
// waits. MAIN's phases overlap the RECV span, so the wall-time metric is `total`.

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

static inline void send_data(ring_t* grid, int N, int dst, int sender, int bucket) {
    ring_t* out = &grid[sender * N + dst];
    while (out->tail - out->head >= CAP) { }
    out->slots[out->tail % CAP].sender = sender;
    out->slots[out->tail % CAP].bucket = bucket;
    vx_fence();
    out->tail = out->tail + 1;
}

static inline int try_recv(ring_t* grid, int N, int me, int src, int* bucket) {
    ring_t* in = &grid[src * N + me];
    if (in->head == in->tail) return 0;
    vx_fence();
    *bucket = in->slots[in->head % CAP].bucket;
    in->head = in->head + 1;
    return 1;
}

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
    volatile uint32_t* arrived  = (volatile uint32_t*)a->arrived_addr;
    volatile uint32_t* released = (volatile uint32_t*)a->released_addr;

    // role + actor id from the physical warp (role-homogeneous warps, no divergence)
    uint32_t TPW  = (uint32_t)vx_num_threads();
    uint32_t WPC  = (uint32_t)vx_num_warps();
    uint32_t w    = (uint32_t)vx_warp_id();
    uint32_t t    = (uint32_t)vx_thread_id();
    uint32_t core = (uint32_t)vx_core_id();
    int      is_main = (w < WPC / 2);
    uint32_t local_warp = is_main ? w : (w - WPC / 2);
    int      me   = (int)(core * (WPC / 2) * TPW + local_warp * TPW + t);
    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];

    int Lc = 2 * me + 1, Rc = 2 * me + 2;
    int haveL = (Lc < N), haveR = (Rc < N);

    if (is_main) {
        // MAIN: send M bumps, then tree barrier (no draining).
        uint32_t G_total = (uint32_t)(L * N);
        uint64_t s0 = csr_read(VX_CSR_MCYCLE);
        for (uint32_t i = 0; i < M; i++) {
            uint32_t g = mix(me, i) % G_total;
            send_data(grid, N, (int)(g % (uint32_t)N), me, (int)(g / (uint32_t)N));
        }
        uint64_t s1 = csr_read(VX_CSR_MCYCLE);

        int phase = 0;  // 0 = wait children, 1 = wait release, 2 = done
        while (!vx_vote_all(phase == 2)) {
            if (phase == 0) {
                if ((!haveL || arrived[Lc]) && (!haveR || arrived[Rc])) {
                    arrived[me] = 1; vx_fence();
                    phase = 1;
                }
            } else if (phase == 1) {
                if (me == 0 || released[me]) {
                    if (me == 0) released[me] = 1;     // root self-release for RECV[0]
                    if (haveL) released[Lc] = 1;
                    if (haveR) released[Rc] = 1;
                    vx_fence();
                    phase = 2;
                }
            }
        }
        uint64_t s2 = csr_read(VX_CSR_MCYCLE);
        ph->send_data = s1 - s0;
        ph->send_done = s2 - s1;       // tree-barrier work
    } else {
        // RECV: wait for my release (= all sent), then drain once.
        int* lcounts = &counts[me * L];
        uint64_t t0 = csr_read(VX_CSR_MCYCLE);
        while (!vx_vote_all(released[me] != 0)) { /* spin on flag, cheap */ }
        uint64_t t1 = csr_read(VX_CSR_MCYCLE);
        int bk;
        for (int src = 0; src < N; src++) {
            while (try_recv(grid, N, me, src, &bk)) lcounts[bk]++;
        }
        uint64_t t2 = csr_read(VX_CSR_MCYCLE);
        ph->handle_req = 0;
        ph->handle_rep = t2 - t1;      // drain + apply
        ph->empty_poll = t1 - t0;      // idle waiting for the barrier to release me
        ph->tail_wait  = 0;
        ph->total      = t2 - t0;
    }
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = 2 * a->N;   // N main threads + N recv threads
    uint32_t block_dim = 1;
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
