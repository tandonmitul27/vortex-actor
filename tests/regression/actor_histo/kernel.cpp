// Bale-style histogram via message passing. Global bucket G lives on PE (G%N)
// at slot (G/N); each PE sends M random bumps to owners, who apply them.

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

// --- mailbox primitives (one SPSC ring per sender→receiver channel) ---

// push a data packet to dst (spins if the channel is full)
static inline void send_data(ring_t* grid, int N, int dst, int sender, int bucket) {
    ring_t* out = &grid[sender * N + dst];
    while (out->tail - out->head >= CAP) { }
    out->slots[out->tail % CAP].sender    = sender;
    out->slots[out->tail % CAP].done_flag = 0;
    out->slots[out->tail % CAP].bucket    = bucket;
    vx_fence();                              // publish slot before bumping tail
    out->tail = out->tail + 1;
}

// send a done-flagged packet to every PE
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

// non-blocking pop from src's channel; returns 0 if empty
static inline int try_recv(ring_t* grid, int N, int me, int src, packet_t* out) {
    ring_t* in = &grid[src * N + me];
    if (in->head == in->tail) return 0;
    vx_fence();                              // see payload before advancing head
    *out = in->slots[in->head % CAP];
    in->head = in->head + 1;
    return 1;
}

// deterministic hash (host replays it to verify)
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
    int     me     = blockIdx.x;

    int* lcounts = &counts[me * L];        // my buckets

    uint32_t G_total = (uint32_t)(L * N);
    int dones = 0;

    uint64_t t_start = csr_read(VX_CSR_MCYCLE);

    // 1. send M random bumps
    for (uint32_t i = 0; i < M; i++) {
        uint32_t g     = mix(me, i) % G_total;
        uint32_t owner = g % (uint32_t)N;
        uint32_t lidx  = g / (uint32_t)N;
        send_data(grid, N, (int)owner, me, (int)lidx);
    }
    uint64_t t_after_send = csr_read(VX_CSR_MCYCLE);

    // 2. broadcast done
    broadcast_done(grid, N, me);
    uint64_t t_after_done = csr_read(VX_CSR_MCYCLE);

    // 3. poll: apply incoming bumps, count dones, until everyone is done
    uint64_t handler_cyc    = 0;
    uint64_t empty_poll_cyc = 0;
    uint64_t last_hit_t     = t_after_done;
    while (!vx_vote_all(dones == N)) {
        uint64_t iter_s = csr_read(VX_CSR_MCYCLE);
        int hit = 0;
        for (int src = 0; src < N; src++) {
            packet_t pkt;
            if (try_recv(grid, N, me, src, &pkt)) {
                hit = 1;
                if (pkt.done_flag == 1) {
                    dones++;
                } else {
                    lcounts[pkt.bucket]++;
                }
            }
        }
        uint64_t iter_e = csr_read(VX_CSR_MCYCLE);
        if (hit) { handler_cyc += iter_e - iter_s; last_hit_t = iter_e; }
        else     { empty_poll_cyc += iter_e - iter_s; }
    }
    uint64_t t_exit = csr_read(VX_CSR_MCYCLE);

    uint64_t tail = t_exit - last_hit_t;            // idle after the last hit
    uint64_t empty_pretail = empty_poll_cyc > tail ? empty_poll_cyc - tail : 0;

    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];
    ph->send_data  = t_after_send - t_start;
    ph->send_done  = t_after_done - t_after_send;
    ph->handle_req = 0;                  // histo never replies
    ph->handle_rep = handler_cyc;
    ph->empty_poll = empty_pretail;
    ph->tail_wait  = tail;
    ph->total      = t_exit - t_start;
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N;
    uint32_t block_dim = 1;
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
