// Bale histogram, SPSC grid + tree barrier, with send/receive multiplexing: send a
// burst of K bumps, then make one bounded drain pass over the N rings, and repeat
// until all M are sent; then the tree barrier and a final full drain. K = M gives
// the rigid "send all, then drain" schedule (same as actor_histo_tree); K = 1 is
// the finest interleave. CAP is large enough that a send never blocks, so the loop
// stays a non-spinning state machine.

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

static inline void send_data(ring_t* grid, int N, int dst, int sender, int bucket) {
    ring_t* out = &grid[sender * N + dst];
    while (out->tail - out->head >= CAP) { }          // never spins when CAP is sized right
    out->slots[out->tail % CAP].sender = sender;
    out->slots[out->tail % CAP].bucket = bucket;
    vx_fence();                                        // publish slot before bumping tail
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
    int      me       = blockIdx.x;
    volatile uint32_t* arrived  = (volatile uint32_t*)a->arrived_addr;
    volatile uint32_t* released = (volatile uint32_t*)a->released_addr;
    int*     lcounts  = &counts[me * L];
    uint32_t G_total  = (uint32_t)(L * N);

    int Lc = 2 * me + 1, Rc = 2 * me + 2;
    int haveL = (Lc < N), haveR = (Rc < N);

    uint32_t send_idx = 0;
    int phase = 0;   // 0 send+interleaved-drain, 1 combine up, 2 broadcast down, 3 final drain, 4 done
    uint64_t c_sd = 0, c_sn = 0, c_h = 0, c_e = 0;
    uint64_t t0 = csr_read(VX_CSR_MCYCLE);

    while (!vx_vote_all(phase == 4)) {
        if (phase == 0) {
            // --- send a burst of up to K bumps ---
            uint64_t s0 = csr_read(VX_CSR_MCYCLE);
            uint32_t sent_this = 0;
            while (sent_this < (uint32_t)K && send_idx < (uint32_t)M) {
                uint32_t g = mix(me, send_idx) % G_total;
                send_data(grid, N, (int)(g % (uint32_t)N), me, (int)(g / (uint32_t)N));
                send_idx++; sent_this++;
            }
            c_sd += csr_read(VX_CSR_MCYCLE) - s0;
            // --- one bounded drain pass over my N rings (<=1 pop per ring) ---
            uint64_t r0 = csr_read(VX_CSR_MCYCLE);
            int bk, hit = 0;
            for (int src = 0; src < N; src++)
                if (try_recv(grid, N, me, src, &bk)) { lcounts[bk]++; hit = 1; }
            uint64_t r1 = csr_read(VX_CSR_MCYCLE);
            if (hit) c_h += r1 - r0; else c_e += r1 - r0;
            if (send_idx == (uint32_t)M) phase = 1;
        } else if (phase == 1) {                       // combine up
            if ((!haveL || arrived[Lc]) && (!haveR || arrived[Rc])) {
                uint64_t b0 = csr_read(VX_CSR_MCYCLE);
                arrived[me] = 1; vx_fence();
                c_sn += csr_read(VX_CSR_MCYCLE) - b0;
                phase = 2;
            }
        } else if (phase == 2) {                       // broadcast down
            if (me == 0 || released[me]) {
                uint64_t b0 = csr_read(VX_CSR_MCYCLE);
                if (haveL) released[Lc] = 1;
                if (haveR) released[Rc] = 1;
                vx_fence();
                c_sn += csr_read(VX_CSR_MCYCLE) - b0;
                phase = 3;
            }
        } else if (phase == 3) {                       // final drain: everything is present now
            uint64_t r0 = csr_read(VX_CSR_MCYCLE);
            int bk;
            for (int src = 0; src < N; src++)
                while (try_recv(grid, N, me, src, &bk)) lcounts[bk]++;
            c_h += csr_read(VX_CSR_MCYCLE) - r0;
            phase = 4;
        }
    }
    uint64_t te = csr_read(VX_CSR_MCYCLE);

    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];
    ph->send_data  = c_sd;
    ph->send_done  = c_sn;
    ph->handle_req = 0;
    ph->handle_rep = c_h;
    ph->empty_poll = c_e;
    ph->tail_wait  = 0;
    ph->total      = te - t0;
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N;
    uint32_t block_dim = 1;
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
