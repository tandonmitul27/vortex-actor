// Bale histogram, two threads per actor: a MAIN thread that sends and a RECV thread
// that drains. We spawn 2N threads and pick the role from the physical warp id
// (first half of a core's warps are MAIN, second half RECV), so every warp is
// all-MAIN or all-RECV and there is no divergence. The two threads touch disjoint
// memory -- MAIN writes its outbound row, RECV reads its inbound column and the
// counts -- so they need no locking, and MAIN can use blocking sends since RECV
// keeps draining. MAIN and RECV run concurrently, so the wall-time metric is the
// RECV span (`total`), not the sum of the phase counters.

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

// blocking push of a data bump (spins while the ring is full)
static inline void send_data(ring_t* grid, int N, int dst, int sender, int bucket) {
    ring_t* out = &grid[sender * N + dst];
    while (out->tail - out->head >= CAP) { }
    out->slots[out->tail % CAP].sender    = sender;
    out->slots[out->tail % CAP].done_flag = 0;
    out->slots[out->tail % CAP].bucket    = bucket;
    vx_fence();
    out->tail = out->tail + 1;
}

static inline void broadcast_done(ring_t* grid, int N, int sender) {
    for (int dst = 0; dst < N; dst++) {
        ring_t* out = &grid[sender * N + dst];
        while (out->tail - out->head >= CAP) { }
        out->slots[out->tail % CAP].sender    = sender;
        out->slots[out->tail % CAP].done_flag = 1;
        vx_fence();
        out->tail = out->tail + 1;
    }
}

static inline int try_recv(ring_t* grid, int N, int me, int src, packet_t* out) {
    ring_t* in = &grid[src * N + me];
    if (in->head == in->tail) return 0;
    vx_fence();
    *out = in->slots[in->head % CAP];
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
    ring_t* grid   = (ring_t*)a->grid_addr;
    int*    counts = (int*)a->counts_addr;
    int     N      = a->N;
    // role + actor id from the physical warp (first half of a core's warps = main,
    // second half = recv); guarantees role-homogeneous warps for any layout.
    uint32_t TPW  = (uint32_t)vx_num_threads();
    uint32_t WPC  = (uint32_t)vx_num_warps();
    uint32_t w    = (uint32_t)vx_warp_id();
    uint32_t t    = (uint32_t)vx_thread_id();
    uint32_t core = (uint32_t)vx_core_id();
    int      is_main = (w < WPC / 2);
    uint32_t local_warp = is_main ? w : (w - WPC / 2);
    int      me   = (int)(core * (WPC / 2) * TPW + local_warp * TPW + t);
    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];

    if (is_main) {
        // MAIN: send M bumps, broadcast done, exit. (recv warps drain for us)
        uint32_t G_total = (uint32_t)(L * N);
        uint64_t s0 = csr_read(VX_CSR_MCYCLE);
        for (uint32_t i = 0; i < M; i++) {
            uint32_t g     = mix(me, i) % G_total;
            uint32_t owner = g % (uint32_t)N;
            uint32_t lidx  = g / (uint32_t)N;
            send_data(grid, N, (int)owner, me, (int)lidx);
        }
        uint64_t s1 = csr_read(VX_CSR_MCYCLE);
        broadcast_done(grid, N, me);
        uint64_t s2 = csr_read(VX_CSR_MCYCLE);
        ph->send_data = s1 - s0;
        ph->send_done = s2 - s1;
    } else {
        // RECV: drain my inbox until everyone is done, applying bumps.
        int* lcounts = &counts[me * L];
        int dones = 0;
        uint64_t c_h = 0, c_e = 0;
        uint64_t t0 = csr_read(VX_CSR_MCYCLE);
        uint64_t last_hit = t0;
        while (!vx_vote_all(dones == N)) {
            uint64_t i0 = csr_read(VX_CSR_MCYCLE);
            int hit = 0;
            for (int src = 0; src < N; src++) {
                packet_t pkt;
                if (try_recv(grid, N, me, src, &pkt)) {
                    hit = 1;
                    if (pkt.done_flag == 1) dones++;
                    else                    lcounts[pkt.bucket]++;
                }
            }
            uint64_t i1 = csr_read(VX_CSR_MCYCLE);
            if (hit) { c_h += i1 - i0; last_hit = i1; }
            else     { c_e += i1 - i0; }
        }
        uint64_t te = csr_read(VX_CSR_MCYCLE);
        uint64_t tail = te - last_hit;
        uint64_t ep = c_e > tail ? c_e - tail : 0;
        ph->handle_req = 0;
        ph->handle_rep = c_h;
        ph->empty_poll = ep;
        ph->tail_wait  = tail;
        ph->total      = te - t0;
    }
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = 2 * a->N;   // N main threads + N recv threads
    uint32_t block_dim = 1;
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
