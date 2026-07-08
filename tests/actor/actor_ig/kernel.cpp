// 2-mailbox request/reply index-gather. Each PE sends M requests on mb0; the
// destination replies on mb1 with me*1000000 + queried_index, which the
// requester stores in tgt[]. Host recomputes expected tgt[] from the same RNG.

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

// --- mailbox primitives (SPSC ring; packets carry idx + value) ---

// push a packet to dst (spins if the channel is full)
static inline void send_data(ring_t* grid, int N, int dst, int sender,
                             int idx_field, int val_field) {
    ring_t* out = &grid[sender * N + dst];
    while (out->tail - out->head >= CAP) { }
    out->slots[out->tail % CAP].sender    = sender;
    out->slots[out->tail % CAP].done_flag = 0;
    out->slots[out->tail % CAP].idx       = idx_field;
    out->slots[out->tail % CAP].value     = val_field;
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
    ring_t* mb0 = (ring_t*)a->mb0_grid_addr;
    ring_t* mb1 = (ring_t*)a->mb1_grid_addr;
    int*    tgt = (int*)a->tgt_addr;
    int     N   = a->N;
    int     me  = blockIdx.x;

    int mb0_dones = 0;
    int mb1_dones = 0;
    int cascaded  = 0;

    uint64_t t_start = csr_read(VX_CSR_MCYCLE);

    // 1. send M requests on mb0
    for (uint32_t i = 0; i < M; i++) {
        int dst  = mix(me, 2u*i + 1u) % N;
        int lidx = mix(me, 2u*i + 2u) % 1000;
        send_data(mb0, N, dst, me, /*idx=*/(int)i, /*val=*/lidx);
    }
    uint64_t t_after_send = csr_read(VX_CSR_MCYCLE);

    // 2. broadcast done on mb0
    broadcast_done(mb0, N, me);
    uint64_t t_after_done0 = csr_read(VX_CSR_MCYCLE);

    // 3. poll: service requests (mb0) and land replies (mb1) until all done.
    //    mb0 and mb1 scans are timed separately → handle_req vs handle_rep.
    uint64_t handle_req_cyc   = 0;
    uint64_t handle_rep_cyc   = 0;
    uint64_t empty_poll_cyc   = 0;
    uint64_t cascade_done_cyc = 0;     // mb1 done broadcast → credited to send_done
    uint64_t last_work_t      = t_after_done0;

    while (!vx_vote_all(mb1_dones == N)) {
        uint64_t iter_s = csr_read(VX_CSR_MCYCLE);

        // service requests: drain mb0, reply on mb1
        int hit_req = 0;
        for (int src = 0; src < N; src++) {
            packet_t pkt;
            if (try_recv(mb0, N, me, src, &pkt)) {
                hit_req = 1;
                if (pkt.done_flag == 1) {
                    mb0_dones++;
                } else {
                    int reply_val = me * 1000000 + pkt.value;
                    send_data(mb1, N, pkt.sender, me, /*idx=*/pkt.idx, /*val=*/reply_val);
                }
            }
        }
        uint64_t mid = csr_read(VX_CSR_MCYCLE);   // req/rep boundary

        // once mb0 fully drained, broadcast done on mb1 (timed → send_done)
        uint64_t casc = 0;
        if (mb0_dones == N && !cascaded) {
            uint64_t c0 = csr_read(VX_CSR_MCYCLE);
            broadcast_done(mb1, N, me);
            uint64_t c1 = csr_read(VX_CSR_MCYCLE);
            casc = c1 - c0;
            cascade_done_cyc += casc;
            cascaded = 1;
        }

        // land replies: drain mb1, write tgt[]
        int hit_rep = 0;
        for (int src = 0; src < N; src++) {
            packet_t pkt;
            if (try_recv(mb1, N, me, src, &pkt)) {
                hit_rep = 1;
                if (pkt.done_flag == 1) {
                    mb1_dones++;
                } else {
                    tgt[me * M + pkt.idx] = pkt.value;
                }
            }
        }
        uint64_t iter_e = csr_read(VX_CSR_MCYCLE);

        // req tiles [iter_s, mid], rep tiles [mid, iter_e] - cascade → whole iter
        uint64_t req_span = mid - iter_s;
        uint64_t rep_span = (iter_e - mid) - casc;
        if (hit_req) handle_req_cyc += req_span; else empty_poll_cyc += req_span;
        if (hit_rep) handle_rep_cyc += rep_span; else empty_poll_cyc += rep_span;
        if (hit_req || hit_rep || casc) last_work_t = iter_e;
    }
    uint64_t t_exit = csr_read(VX_CSR_MCYCLE);

    uint64_t tail = t_exit - last_work_t;
    uint64_t empty_pretail = empty_poll_cyc > tail ? empty_poll_cyc - tail : 0;

    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];
    ph->send_data  = t_after_send - t_start;
    ph->send_done  = (t_after_done0 - t_after_send) + cascade_done_cyc;
    ph->handle_req = handle_req_cyc;
    ph->handle_rep = handle_rep_cyc;
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
