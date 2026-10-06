// Bale histogram over an MPSC shared inbox, one thread per actor. Each receiver has
// a single inbox that all senders push into (O(N) memory instead of the SPSC grid's
// O(N^2)), and receiving is O(1) per message rather than an O(N) channel scan.
// Producers claim a slot with an atomic fetch-add on `tail` (a CAS loop would
// livelock when many lanes hit the same tail). There is no blocking spin: one
// uniform loop runs every iteration doing bounded work -- claim one outgoing
// message, publish it once its slot is free, then drain the inbox -- and because
// every actor keeps draining until the global vote, no one can deadlock.

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

// single-consumer dequeue from my own inbox. returns 1 if a message was taken.
static inline int mpsc_recv(inbox_t* in, int* done_flag, int* bucket) {
    uint32_t pos = in->head;
    slot_t* s = &in->slots[pos % CAP];
    if (s->seq != pos + 1) return 0;               // empty / producer mid-write
    vx_fence();                                    // see payload before consuming
    *done_flag = (int)s->done_flag;
    *bucket    = s->bucket;
    s->seq = pos + CAP;                            // free slot for ticket pos+CAP
    in->head = pos + 1;
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
    inbox_t* inbox  = (inbox_t*)a->inbox_addr;
    int*     counts = (int*)a->counts_addr;
    int      N      = a->N;
    int      me     = blockIdx.x;                  // 1 actor per thread
    int*     lcounts = &counts[me * L];
    inbox_t* myin   = &inbox[me];
    uint32_t G_total = (uint32_t)(L * N);

    int send_idx    = 0;                           // messages issued: 0..M  bumps, M..M+N-1 dones
    int total_sends = (int)M + N;
    int dones       = 0;

    int      pending = 0;                          // a claimed-but-unpublished slot?
    uint32_t p_pos = 0; inbox_t* p_in = 0;
    int p_done = 0, p_bucket = 0;

    uint64_t c_ticket = 0, c_publish = 0;
    uint64_t c_sd = 0, c_sn = 0, c_h = 0, c_e = 0;
    uint64_t t0 = csr_read(VX_CSR_MCYCLE);
    uint64_t last_work = t0;

    while (!vx_vote_all(send_idx == total_sends && dones == N)) {
        // --- advance my sending (bounded: at most one claim + one publish) ---
        uint64_t s0 = csr_read(VX_CSR_MCYCLE);
        int doing_bump = 0, doing_send = 0;
        if (!pending && send_idx < total_sends) {  // claim the next message
            int dst;
            if (send_idx < (int)M) {               // a data bump
                uint32_t g = mix(me, (uint32_t)send_idx) % G_total;
                dst = (int)(g % (uint32_t)N);
                p_done = 0; p_bucket = (int)(g / (uint32_t)N); doing_bump = 1;
            } else {                                // a done flag to dst
                dst = send_idx - (int)M;
                p_done = 1; p_bucket = 0;
            }
            p_in = &inbox[dst];
            uint64_t k0 = csr_read(VX_CSR_MCYCLE);
            p_pos = (uint32_t)__atomic_fetch_add(&p_in->tail, 1, __ATOMIC_RELAXED);
            c_ticket += csr_read(VX_CSR_MCYCLE) - k0;   // BREAKDOWN: ticket alloc
            pending = 1; doing_send = 1;
        }
        if (pending) {                              // publish if my slot is free
            uint64_t q0 = csr_read(VX_CSR_MCYCLE);
            slot_t* s = &p_in->slots[p_pos % CAP];
            if (s->seq == p_pos) {
                s->sender    = me;
                s->done_flag = (uint32_t)p_done;
                s->bucket    = p_bucket;
                vx_fence();
                s->seq = p_pos + 1;
                pending = 0; send_idx++;
                last_work = csr_read(VX_CSR_MCYCLE);
            }
            c_publish += csr_read(VX_CSR_MCYCLE) - q0;  // BREAKDOWN: publish+spin
            doing_send = 1;
        }
        uint64_t s1 = csr_read(VX_CSR_MCYCLE);
        if (doing_send) { if (doing_bump || send_idx <= (int)M) c_sd += s1 - s0; else c_sn += s1 - s0; }

        // --- drain my inbox (always, every iteration) ---
        uint64_t r0 = csr_read(VX_CSR_MCYCLE);
        int hit = 0, df, bk;
        while (mpsc_recv(myin, &df, &bk)) {
            hit = 1;
            if (df == 1) dones++;
            else         lcounts[bk]++;
        }
        uint64_t r1 = csr_read(VX_CSR_MCYCLE);
        if (hit) { c_h += r1 - r0; last_work = r1; }
        else     { c_e += r1 - r0; }
    }
    uint64_t te = csr_read(VX_CSR_MCYCLE);

    uint64_t tail = te - last_work;
    uint64_t ep = c_e > tail ? c_e - tail : 0;
    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];
    ph->send_data  = c_sd;
    ph->bd_ticket  = c_ticket;
    ph->bd_publish = c_publish;
    ph->send_done  = c_sn;
    ph->handle_req = 0;
    ph->handle_rep = c_h;
    ph->empty_poll = ep;
    ph->tail_wait  = tail;
    ph->total      = te - t0;
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N;                     // 1 thread per actor
    uint32_t block_dim = 1;
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
