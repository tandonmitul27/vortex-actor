// Bale histogram, one thread per actor, with a tree barrier for termination.
// Each actor sends its M bumps over per-pair SPSC rings, then all actors meet at a
// combining tree barrier (arrived[] combines up, released[] broadcasts down) rather
// than an all-to-all "done" broadcast. A ring holds at most M < CAP messages, so it
// can never overflow; we therefore skip draining during the run and drain every
// ring once, after the barrier.

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

// blocking push of a data bump (spins while the ring is full)
static inline void send_data(ring_t* grid, int N, int dst, int sender, int bucket) {
    ring_t* out = &grid[sender * N + dst];
    while (out->tail - out->head >= CAP) { }
    out->slots[out->tail % CAP].sender = sender;
    out->slots[out->tail % CAP].bucket = bucket;
    vx_fence();                              // publish slot before bumping tail
    out->tail = out->tail + 1;
}

// non-blocking pop of a bump; returns 0 if empty
static inline int try_recv(ring_t* grid, int N, int me, int src, int* bucket) {
    ring_t* in = &grid[src * N + me];
    if (in->head == in->tail) return 0;
    vx_fence();                              // see payload before advancing head
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

    // binary heap tree: children 2i+1, 2i+2; parent (i-1)/2; root 0.
    int Lc = 2 * me + 1, Rc = 2 * me + 2;
    int haveL = (Lc < N), haveR = (Rc < N);

    uint64_t c_sd = 0, c_sn = 0, c_h = 0, c_e = 0;
    uint64_t t0 = csr_read(VX_CSR_MCYCLE);

    // 1. send M bumps (no done messages anymore)
    for (uint32_t i = 0; i < M; i++) {
        uint32_t g     = mix(me, i) % G_total;
        uint32_t owner = g % (uint32_t)N;
        uint32_t lidx  = g / (uint32_t)N;
        send_data(grid, N, (int)owner, me, (int)lidx);
    }
    uint64_t t1 = csr_read(VX_CSR_MCYCLE);
    c_sd = t1 - t0;

    // 2. tree barrier, flags only (no draining -- rings can't overflow, M < CAP)
    int phase = 0;   // 0 = wait children, 1 = wait release, 2 = done
    while (!vx_vote_all(phase == 2)) {
        if (phase == 0) {                                  // combine up
            if ((!haveL || arrived[Lc]) && (!haveR || arrived[Rc])) {
                arrived[me] = 1; vx_fence();
                phase = 1;
            }
        } else if (phase == 1) {                           // broadcast down
            if (me == 0 || released[me]) {
                if (haveL) released[Lc] = 1;
                if (haveR) released[Rc] = 1;
                vx_fence();
                phase = 2;
            }
        }
    }
    uint64_t t2 = csr_read(VX_CSR_MCYCLE);
    c_sn = t2 - t1;                  // tree-barrier flag work

    // 3. final drain: release => everyone finished sending => every bump is in a
    //    ring; drain each ring fully in one pass.
    int bk;
    for (int src = 0; src < N; src++) {
        while (try_recv(grid, N, me, src, &bk)) lcounts[bk]++;
    }
    uint64_t t3 = csr_read(VX_CSR_MCYCLE);
    c_h = t3 - t2;
    (void)c_e;

    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];
    ph->send_data  = c_sd;
    ph->send_done  = c_sn;          // tree-barrier flag work only
    ph->handle_req = 0;
    ph->handle_rep = c_h;
    ph->empty_poll = 0;
    ph->tail_wait  = 0;
    ph->total      = t3 - t0;
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N;
    uint32_t block_dim = 1;
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
