// Index-gather, two threads per actor: a MAIN thread that sends requests and a
// RECV thread that services requests and lands replies. We spawn 2N threads and
// pick the role from the physical warp id (first half of a core's warps are MAIN,
// second half RECV), so every warp is all-MAIN or all-RECV with no divergence.
// Requests flow on mb0, replies on mb1, and the roles write disjoint memory so no
// locking is needed: MAIN owns the mb0 outbound sends, RECV owns everything else.
// RECV services a request with peek-and-commit (it only pops a request once its
// reply ring has room), so it never blocks holding an unsendable reply. MAIN's
// sends overlap the RECV span, so the wall-time metric is `total` (the RECV span).

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

// blocking push of a request payload (spins while full)
static inline void send_req(ring_t* grid, int N, int dst, int sender,
                            int idx_field, int val_field) {
    ring_t* out = &grid[sender * N + dst];
    while (out->tail - out->head >= CAP) { }
    out->slots[out->tail % CAP].sender    = sender;
    out->slots[out->tail % CAP].done_flag = 0;
    out->slots[out->tail % CAP].idx       = idx_field;
    out->slots[out->tail % CAP].value     = val_field;
    vx_fence();
    out->tail = out->tail + 1;
}

static inline void broadcast_done_blk(ring_t* grid, int N, int sender) {
    for (int dst = 0; dst < N; dst++) {
        ring_t* out = &grid[sender * N + dst];
        while (out->tail - out->head >= CAP) { }
        out->slots[out->tail % CAP].sender    = sender;
        out->slots[out->tail % CAP].done_flag = 1;
        vx_fence();
        out->tail = out->tail + 1;
    }
}

// non-blocking done push to one dst; 0 if full
static inline int try_send_done(ring_t* grid, int N, int dst, int sender) {
    ring_t* out = &grid[sender * N + dst];
    if (out->tail - out->head >= CAP) return 0;
    out->slots[out->tail % CAP].sender    = sender;
    out->slots[out->tail % CAP].done_flag = 1;
    vx_fence();
    out->tail = out->tail + 1;
    return 1;
}

static inline int try_recv(ring_t* grid, int N, int me, int src, packet_t* out) {
    ring_t* in = &grid[src * N + me];
    if (in->head == in->tail) return 0;
    vx_fence();
    *out = in->slots[in->head % CAP];
    in->head = in->head + 1;
    return 1;
}

// service one mb0 channel: peek, commit only if the reply ring has room.
// returns 1 = request serviced, 2 = done consumed, 0 = nothing. *dones bumped on done.
static inline int service_one(ring_t* mb0, ring_t* mb1, int N, int me,
                              int src, int* dones) {
    ring_t* in = &mb0[src * N + me];
    if (in->head == in->tail) return 0;
    vx_fence();
    packet_t pkt = in->slots[in->head % CAP];      // peek
    if (pkt.done_flag == 1) {
        in->head = in->head + 1;
        (*dones)++;
        return 2;
    }
    ring_t* out = &mb1[me * N + (int)pkt.sender];
    if (out->tail - out->head >= CAP) return 0;    // reply ring full -> retry later
    in->head = in->head + 1;
    out->slots[out->tail % CAP].sender    = me;
    out->slots[out->tail % CAP].done_flag = 0;
    out->slots[out->tail % CAP].idx       = pkt.idx;
    out->slots[out->tail % CAP].value     = me * 1000000 + pkt.value;
    vx_fence();
    out->tail = out->tail + 1;
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
    ring_t* mb0 = (ring_t*)a->mb0_grid_addr;
    ring_t* mb1 = (ring_t*)a->mb1_grid_addr;
    int*    tgt = (int*)a->tgt_addr;
    int     N   = a->N;
    // role + actor id from the physical warp (first half of a core's warps = main,
    // second half = recv); guarantees role-homogeneous warps for any layout.
    uint32_t TPW  = (uint32_t)vx_num_threads();
    uint32_t WPC  = (uint32_t)vx_num_warps();
    uint32_t w    = (uint32_t)vx_warp_id();
    uint32_t t    = (uint32_t)vx_thread_id();
    uint32_t core = (uint32_t)vx_core_id();
    int      is_main = (w < WPC / 2);
    uint32_t local_warp = is_main ? w : (w - WPC / 2);
    int      me  = (int)(core * (WPC / 2) * TPW + local_warp * TPW + t);
    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];

    if (is_main) {
        // MAIN: fire M requests on mb0, broadcast mb0 done, exit.
        uint64_t s0 = csr_read(VX_CSR_MCYCLE);
        for (uint32_t i = 0; i < M; i++) {
            int dst  = (int)(mix(me, 2u*i + 1u) % (uint32_t)N);
            int lidx = (int)(mix(me, 2u*i + 2u) % 1000u);
            send_req(mb0, N, dst, me, /*idx=*/(int)i, /*val=*/lidx);
        }
        uint64_t s1 = csr_read(VX_CSR_MCYCLE);
        broadcast_done_blk(mb0, N, me);
        uint64_t s2 = csr_read(VX_CSR_MCYCLE);
        ph->send_data = s1 - s0;
        ph->send_done = s2 - s1;          // mb0 done (mb1 done added by recv below)
    } else {
        // RECV: service requests (mb0 -> reply on mb1) and land replies (mb1 -> tgt).
        int mb0_dones = 0, mb1_dones = 0, mb1_done_sent = 0;
        uint64_t c_hreq = 0, c_hrep = 0, c_empty = 0, c_sdone = 0;
        uint64_t t0 = csr_read(VX_CSR_MCYCLE);
        uint64_t last_work = t0;

        while (!vx_vote_all(mb1_done_sent == N && mb1_dones == N)) {
            // service requests on mb0
            uint64_t q0 = csr_read(VX_CSR_MCYCLE);
            int hit_req = 0;
            for (int src = 0; src < N; src++) {
                if (service_one(mb0, mb1, N, me, src, &mb0_dones)) hit_req = 1;
            }
            uint64_t q1 = csr_read(VX_CSR_MCYCLE);
            if (hit_req) { c_hreq += q1 - q0; last_work = q1; }
            else         { c_empty += q1 - q0; }

            // once all requests are in, all replies are sent -> announce mb1 done
            uint64_t d0 = csr_read(VX_CSR_MCYCLE);
            while (mb0_dones == N && mb1_done_sent < N) {
                if (!try_send_done(mb1, N, mb1_done_sent, me)) break;
                mb1_done_sent++; last_work = csr_read(VX_CSR_MCYCLE);
            }
            c_sdone += csr_read(VX_CSR_MCYCLE) - d0;

            // land replies on mb1
            uint64_t p0 = csr_read(VX_CSR_MCYCLE);
            int hit_rep = 0;
            for (int src = 0; src < N; src++) {
                packet_t pkt;
                if (try_recv(mb1, N, me, src, &pkt)) {
                    hit_rep = 1;
                    if (pkt.done_flag == 1) mb1_dones++;
                    else                    tgt[me * M + pkt.idx] = pkt.value;
                }
            }
            uint64_t p1 = csr_read(VX_CSR_MCYCLE);
            if (hit_rep) { c_hrep += p1 - p0; last_work = p1; }
            else         { c_empty += p1 - p0; }
        }
        uint64_t te = csr_read(VX_CSR_MCYCLE);
        uint64_t tail = te - last_work;
        uint64_t ep = c_empty > tail ? c_empty - tail : 0;
        // NOTE: ph->send_done is written by MAIN (mb0 done). To keep one writer
        // per field, recv folds its mb1-done broadcast into handle_req.
        ph->handle_req = c_hreq + c_sdone;
        ph->handle_rep = c_hrep;
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
