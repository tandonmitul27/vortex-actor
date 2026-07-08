// Bale histogram over an MPSC shared inbox with tree-barrier termination.
// The buffer is the MPSC shared inbox (O(N) memory, atomic-claim tail, seq
// handshake; see actor_histo_mpsc). Termination is the combining tree barrier --
// arrived[] up, released[] down -- so channels carry only bumps. Because the shared
// inbox can hold many messages, the consumer keeps draining every iteration, but
// each drain is O(1) so the barrier adds no scan cost.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

#ifndef CAP
#define CAP   16    // inbox capacity (shared across all senders to a receiver)
#endif
#define L     10    // local buckets per PE (total = L * N)
#ifndef M
#define M     10    // updates per PE
#endif

typedef struct {
    uint32_t sender;
    uint32_t done_flag;        // unused with the tree barrier (kept for parity)
    int      bucket;
    volatile uint32_t seq;     // MPSC slot handshake (ticket/ready marker)
} slot_t;

typedef struct {
    volatile uint32_t head;    // consumer index (single consumer)
    volatile uint32_t tail;    // producer ticket (atomic fetch-add)
    slot_t   slots[CAP];
} inbox_t;                      // one per receiver -> inbox_t[N], O(N) memory

typedef struct {
    uint64_t send_data;   // M bump messages
    uint64_t send_done;   // tree-barrier flag ops (was: O(N) done broadcast)
    uint64_t handle_req;  // 0 — histo never replies
    uint64_t handle_rep;  // recv bump + counts++ (O(1) per msg)
    uint64_t empty_poll;  // empty inbox checks before the last packet
    uint64_t tail_wait;   // idle after the last packet
    uint64_t total;       // full kernel_body span
} phase_cycles_t;

typedef struct {
    uint64_t inbox_addr;      // inbox_t[N]
    uint64_t counts_addr;     // int[N*L]
    uint64_t phase_addr;      // phase_cycles_t[N]
    uint64_t arrived_addr;    // volatile uint32_t[N]  tree-barrier arrival flags
    uint64_t released_addr;   // volatile uint32_t[N]  tree-barrier release flags
    uint32_t N;
} args_t;

#endif
