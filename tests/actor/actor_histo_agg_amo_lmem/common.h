// Bale histogram, aggregating actor + tree barrier, TWO-LEVEL messaging via LMEM,
// with INT (4B) message slots and amoadd accumulate (the agg_amo variant of
// actor_histo_agg_lmem). Same two-level routing -- same-core messages in LMEM,
// cross-core in global -- but each slot is an int[L] and the accumulate uses a
// single-instruction amoadd instead of a byte read-modify-write. The 4B slots make
// the per-core LMEM grid P*P*L*4 bytes, so it only fits the 16 KB scratchpad when
// P is small (P<=20); at P=32 it does NOT fit (40 KB) -- that is the byte-vs-int
// footprint tradeoff made concrete.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

#ifndef BD_FINE
#define BD_FINE 1   // 1 = per-update on/off-chip timing (ratio-only build); 0 = unperturbed
#endif
#ifndef L
#define L     10    // local buckets per PE (total = L * N)
#endif
#ifndef M
#define M     10    // updates per PE
#endif

// combined message: the owner's L counts as ints (amoadd targets). one per sender->owner.
typedef struct {
    int c[L];
} msg_t;

typedef struct {
    uint64_t send_data;   // accumulate (byte increments, LMEM + global) + publish fence
    uint64_t send_done;   // tree-barrier flag ops
    uint64_t handle_req;  // 0
    uint64_t handle_rep;  // drain: reduce LMEM (same-core) + global (cross-core) slots
    uint64_t empty_poll;  // 0
    uint64_t tail_wait;   // 0
    uint64_t total;       // full kernel_body span
    // fine-grained breakdown regions (subdivide send_data and handle_rep)
    uint64_t bd_acc_on;    // folds routed to the scratchpad
    uint64_t bd_acc_off;   // folds routed to global memory
    uint64_t bd_drain_on;  // drain of same-core scratchpad slots
    uint64_t bd_drain_off; // drain of cross-core global slots
} phase_cycles_t;

typedef struct {
    uint64_t msg_addr;        // msg_t[N*N]  global grid (only cross-core slots used)
    uint64_t counts_addr;     // int[N*L]
    uint64_t phase_addr;      // phase_cycles_t[N]
    uint64_t arrived_addr;    // volatile uint32_t[N]  tree-barrier arrival flags
    uint64_t released_addr;   // volatile uint32_t[N]  tree-barrier release flags
    uint32_t N;
} args_t;

#endif
