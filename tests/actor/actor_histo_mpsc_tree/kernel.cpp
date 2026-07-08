// Bale histogram over an MPSC shared inbox with a tree barrier for termination,
// one thread per actor. Same non-spinning MPSC loop as actor_histo_mpsc (atomic
// ticket claim, one pending message, drain every iteration), but termination uses
// the combining tree barrier instead of a done broadcast. A per-actor phase runs
// through: 0 sending, 1 combine arrived[] up, 2 broadcast released[] down, 3 drain
// until empty, 4 done. Release only starts once the root has seen every actor
// arrive (so all bumps are published), so a phase-3 actor that finds its inbox
// empty knows nothing more can arrive and can safely finish.

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

// single-consumer dequeue from my inbox. returns 1 if a message was taken.
static inline int mpsc_recv(inbox_t* in, int* done_flag, int* bucket) {
    uint32_t pos = in->head;
    slot_t* s = &in->slots[pos % CAP];
    if (s->seq != pos + 1) return 0;               // empty / producer mid-write
    vx_fence();
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
    inbox_t* inbox    = (inbox_t*)a->inbox_addr;
    int*     counts   = (int*)a->counts_addr;
    int      N        = a->N;
    int      me       = blockIdx.x;
    volatile uint32_t* arrived  = (volatile uint32_t*)a->arrived_addr;
    volatile uint32_t* released = (volatile uint32_t*)a->released_addr;
    int*     lcounts  = &counts[me * L];
    inbox_t* myin     = &inbox[me];
    uint32_t G_total  = (uint32_t)(L * N);

    int Lc = 2 * me + 1, Rc = 2 * me + 2;
    int haveL = (Lc < N), haveR = (Rc < N);

    int send_idx = 0;                              // 0..M bumps issued
    int pending  = 0; uint32_t p_pos = 0; inbox_t* p_in = 0; int p_bucket = 0;

    uint64_t c_sd = 0, c_sn = 0, c_h = 0, c_e = 0;
    uint64_t t0 = csr_read(VX_CSR_MCYCLE);
    uint64_t last_work = t0;
    int phase = 0;

    while (!vx_vote_all(phase == 4)) {
        // --- advance my sending (bumps only; bounded: one claim + one publish) ---
        uint64_t s0 = csr_read(VX_CSR_MCYCLE);
        if (phase == 0) {
            if (!pending && send_idx < (int)M) {
                uint32_t g = mix(me, (uint32_t)send_idx) % G_total;
                int dst = (int)(g % (uint32_t)N);
                p_bucket = (int)(g / (uint32_t)N);
                p_in = &inbox[dst];
                p_pos = (uint32_t)__atomic_fetch_add(&p_in->tail, 1, __ATOMIC_RELAXED);
                pending = 1;
            }
            if (pending) {
                slot_t* s = &p_in->slots[p_pos % CAP];
                if (s->seq == p_pos) {
                    s->sender = me; s->done_flag = 0; s->bucket = p_bucket;
                    vx_fence();
                    s->seq = p_pos + 1;
                    pending = 0; send_idx++; last_work = csr_read(VX_CSR_MCYCLE);
                }
            }
            if (send_idx == (int)M && !pending) phase = 1;   // done sending
        }
        c_sd += csr_read(VX_CSR_MCYCLE) - s0;

        // --- barrier flag work (time ONLY the actual flag writes, O(1)/actor) ---
        if (phase == 1) {                                    // combine up
            if ((!haveL || arrived[Lc]) && (!haveR || arrived[Rc])) {
                uint64_t b0 = csr_read(VX_CSR_MCYCLE);
                arrived[me] = 1; vx_fence();
                c_sn += csr_read(VX_CSR_MCYCLE) - b0;
                phase = 2;
            }
        } else if (phase == 2) {                             // broadcast down
            if (me == 0 || released[me]) {
                uint64_t b0 = csr_read(VX_CSR_MCYCLE);
                if (haveL) released[Lc] = 1;
                if (haveR) released[Rc] = 1;
                vx_fence();
                c_sn += csr_read(VX_CSR_MCYCLE) - b0;
                phase = 3;
            }
        }

        // --- drain my shared inbox every iteration (O(1) per message) ---
        uint64_t r0 = csr_read(VX_CSR_MCYCLE);
        int hit = 0, df, bk;
        while (mpsc_recv(myin, &df, &bk)) { hit = 1; lcounts[bk]++; }
        uint64_t r1 = csr_read(VX_CSR_MCYCLE);
        if (hit) { c_h += r1 - r0; last_work = r1; }
        else     { c_e += r1 - r0; if (phase == 3) phase = 4; }  // empty after release
    }
    uint64_t te = csr_read(VX_CSR_MCYCLE);

    uint64_t tail = te - last_work;
    uint64_t ep = c_e > tail ? c_e - tail : 0;
    phase_cycles_t* ph = &((phase_cycles_t*)a->phase_addr)[me];
    ph->send_data  = c_sd;
    ph->send_done  = c_sn;
    ph->handle_req = 0;
    ph->handle_rep = c_h;
    ph->empty_poll = ep;
    ph->tail_wait  = tail;
    ph->total      = te - t0;
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N;
    uint32_t block_dim = 1;
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
