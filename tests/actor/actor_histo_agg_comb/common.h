// Bale histogram, aggregating actor + tree barrier, CORE-LEVEL COMBINING via LMEM.
// The P actors on a core share ONE per-core combined buffer core_out[owner][bucket]
// in LMEM: each actor amoadds its M bumps straight into it, so the core's P actors
// combine into a single contribution per owner. Each core then flushes its buffer to
// a small global array (C x N x L), and each owner sums its slice across the C cores.
// vs the private-mailbox variants this shrinks the global footprint from N*N*L to
// C*N*L and the drain from O(N) to O(C); the price is intra-core amoadd contention on
// the shared LMEM buffer (P writers, on-chip). The LMEM buffer is N*L ints (owner-
// indexed), so it fits the 16 KB scratchpad for any P as long as N*L*4 <= 16 KB.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

#ifndef L
#define L     10    // local buckets per PE (total = L * N)
#endif
#ifndef M
#define M     10    // updates per PE
#endif

typedef struct {
    uint64_t send_data;   // combine: amoadd M bumps into the shared LMEM core_out
    uint64_t send_done;   // tree-barrier flag ops
    uint64_t handle_req;  // flush: copy LMEM core_out -> global
    uint64_t handle_rep;  // drain: sum my owner-slice across the C cores
    uint64_t empty_poll;  // 0
    uint64_t tail_wait;   // 0
    uint64_t total;       // full kernel_body span
} phase_cycles_t;

typedef struct {
    uint64_t cout_addr;       // int[C*N*L]  per-core combined buffers (flush target)
    uint64_t counts_addr;     // int[N*L]
    uint64_t phase_addr;      // phase_cycles_t[N]
    uint64_t arrived_addr;    // volatile uint32_t[N]  tree-barrier arrival flags
    uint64_t released_addr;   // volatile uint32_t[N]  tree-barrier release flags
    uint32_t N;
} args_t;

#endif
