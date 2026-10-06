// Index-gather over MPSC shared inboxes, one thread per actor. Each actor is both a
// requester (it sends M lookup requests) and a responder (it answers requests for
// its own data). Requests flow on mb0, replies on mb1. The loop never blocks: it
// does bounded work each iteration and always drains. Each actor keeps two pending
// slots at once -- one for its outgoing request, one for a reply it owes -- so that
// answering an incoming request is never stuck behind its own outgoing request.
// Since mb1 is drained every iteration, replies always flow and no cycle can form.

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

enum { PT_REQ = 1, PT_MB0DONE, PT_MB1DONE };

static inline int mpsc_peek(inbox_t* in, int* df, int* sn, int* idx, int* val) {
    uint32_t pos = in->head;
    slot_t* s = &in->slots[pos % CAP];
    if (s->seq != pos + 1) return 0;
    vx_fence();
    *df = (int)s->done_flag; *sn = (int)s->sender; *idx = s->idx; *val = s->value;
    return 1;
}
static inline void mpsc_pop(inbox_t* in) {
    uint32_t pos = in->head;
    in->slots[pos % CAP].seq = pos + CAP;
    in->head = pos + 1;
}
// publish a claimed slot if it is free now; returns 1 if published.
static inline int mpsc_pub(inbox_t* in, uint32_t pos, int sender, int done, int idx, int val) {
    slot_t* s = &in->slots[pos % CAP];
    if (s->seq != pos) return 0;
    s->sender = (uint32_t)sender; s->done_flag = (uint32_t)done; s->idx = idx; s->value = val;
    vx_fence();
    s->seq = pos + 1;
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
    inbox_t* mb0 = (inbox_t*)a->mb0_addr;
    inbox_t* mb1 = (inbox_t*)a->mb1_addr;
    int*     tbl = (int*)a->table_addr;
    int*     tgt = (int*)a->tgt_addr;
    int      N   = a->N;
    int      me  = blockIdx.x;
    inbox_t* mb0me = &mb0[me];
    inbox_t* mb1me = &mb1[me];

    int req_sent = 0, mb0_done_sent = 0, mb1_done_sent = 0;
    int mb0_dones = 0, mb1_dones = 0;

    // requester-side pending (my outgoing request / done)
    int rq = 0, rq_type = 0, rq_done = 0, rq_idx = 0, rq_val = 0; uint32_t rq_pos = 0; inbox_t* rq_in = 0;
    // responder-side pending (a reply)
    int rp = 0, rp_idx = 0, rp_val = 0; uint32_t rp_pos = 0; inbox_t* rp_in = 0;

    uint64_t c_sd=0, c_sn=0, c_hreq=0, c_hrep=0, c_e=0;
    uint64_t t0 = csr_read(VX_CSR_MCYCLE);
    uint64_t last_work = t0;

    while (!vx_vote_all(req_sent == (int)M && mb0_done_sent == N && mb1_done_sent == N
                        && mb1_dones == N && !rq && !rp)) {
        int df, sn, ix, vl, worked = 0;

        // (1) responder: drain a done or service one request (if reply slot free)
        uint64_t q0 = csr_read(VX_CSR_MCYCLE);
        if (mpsc_peek(mb0me, &df, &sn, &ix, &vl)) {
            if (df == 1) { mpsc_pop(mb0me); mb0_dones++; worked = 1; }
            else if (!rp) {
                rp_pos = (uint32_t)__atomic_fetch_add(&mb1[sn].tail, 1, __ATOMIC_RELAXED);
                rp_in = &mb1[sn]; rp_idx = ix; rp_val = tbl[(size_t)me * TABLE_SIZE + (uint32_t)vl]; rp = 1;
                mpsc_pop(mb0me);                          // commit: consume request
                worked = 1;
            }
        }
        if (rp && mpsc_pub(rp_in, rp_pos, me, 0, rp_idx, rp_val)) { rp = 0; worked = 1; }
        uint64_t q1 = csr_read(VX_CSR_MCYCLE);
        c_hreq += q1 - q0;

        // (2) requester: claim my next outgoing if free, then publish it
        uint64_t s0 = csr_read(VX_CSR_MCYCLE);
        if (!rq) {
            if (req_sent < (int)M) {
                int i = req_sent;
                int dst  = (int)(mix(me, 2u*(uint32_t)i + 1u) % (uint32_t)N);
                int lidx = (int)(mix(me, 2u*(uint32_t)i + 2u) % (uint32_t)TABLE_SIZE);
                rq_pos = (uint32_t)__atomic_fetch_add(&mb0[dst].tail, 1, __ATOMIC_RELAXED);
                rq_in = &mb0[dst]; rq_type = PT_REQ; rq_done = 0; rq_idx = i; rq_val = lidx;
                rq = 1; req_sent = i + 1; worked = 1;
            } else if (mb0_done_sent < N) {
                rq_pos = (uint32_t)__atomic_fetch_add(&mb0[mb0_done_sent].tail, 1, __ATOMIC_RELAXED);
                rq_in = &mb0[mb0_done_sent]; rq_type = PT_MB0DONE; rq_done = 1; rq_idx = 0; rq_val = 0;
                rq = 1; mb0_done_sent++; worked = 1;
            } else if (mb0_dones == N && mb1_done_sent < N) {
                rq_pos = (uint32_t)__atomic_fetch_add(&mb1[mb1_done_sent].tail, 1, __ATOMIC_RELAXED);
                rq_in = &mb1[mb1_done_sent]; rq_type = PT_MB1DONE; rq_done = 1; rq_idx = 0; rq_val = 0;
                rq = 1; mb1_done_sent++; worked = 1;
            }
        }
        if (rq && mpsc_pub(rq_in, rq_pos, me, rq_done, rq_idx, rq_val)) { rq = 0; worked = 1; }
        uint64_t s1 = csr_read(VX_CSR_MCYCLE);
        if (rq_type == PT_REQ) c_sd += s1 - s0; else c_sn += s1 - s0;

        // (3) requester: land replies (drain mb1 unconditionally)
        uint64_t r0 = csr_read(VX_CSR_MCYCLE);
        while (mpsc_peek(mb1me, &df, &sn, &ix, &vl)) {
            if (df == 1) mb1_dones++;
            else         tgt[TGT_IDX(me, ix, N)] = vl;
            mpsc_pop(mb1me); worked = 1;
        }
        uint64_t r1 = csr_read(VX_CSR_MCYCLE);
        c_hrep += r1 - r0;

        if (worked) last_work = r1; else c_e += r1 - q0;
    }
    uint64_t te = csr_read(VX_CSR_MCYCLE);

    uint64_t tail = te - last_work;
    uint64_t ep = c_e > tail ? c_e - tail : 0;
    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];
    ph->send_data  = c_sd;
    ph->send_done  = c_sn;
    ph->handle_req = c_hreq;
    ph->handle_rep = c_hrep;
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
