// Bale histogram over an MPSC shared inbox, one thread per actor.
// Each receiver has one shared inbox that all senders push into, so memory is O(N)
// instead of the SPSC grid's O(N^2), and receiving is O(1) per message. Producers
// claim a slot with an atomic fetch-add on `tail`; the single consumer drains in
// order. A per-slot `seq` field is the producer/consumer handshake (Vyukov-style
// bounded queue): a producer writes then sets seq = ticket+1, and the consumer
// reads then frees the slot with seq = head+CAP. Sending and draining are
// interleaved every tick so the inbox never overflows.
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
    uint32_t done_flag;        // 0 = data, 1 = done
    int      bucket;           // local bucket on the owner PE
    volatile uint32_t seq;     // MPSC slot handshake (ticket/ready marker)
} slot_t;

typedef struct {
    volatile uint32_t head;    // consumer index (single consumer)
    volatile uint32_t tail;    // producer ticket (CAS)
    slot_t   slots[CAP];
} inbox_t;                      // one per receiver -> inbox_t[N], O(N) memory

// per-PE cycle breakdown (same taxonomy as the other actor tests)
typedef struct {
    uint64_t send_data;   // M bump messages
    uint64_t send_done;   // done broadcast to all N
    uint64_t handle_req;  // 0 — histo never replies
    uint64_t handle_rep;  // recv bump + counts++ (O(1) per msg, no scan)
    uint64_t empty_poll;  // empty inbox checks before the last packet
    uint64_t tail_wait;   // idle after the last packet
    uint64_t total;
    uint64_t bd_ticket;   // atomic fetch-add for a slot ticket
    uint64_t bd_publish;  // write the slot + spin until the consumer frees it       // full kernel_body span
} phase_cycles_t;

typedef struct {
    uint64_t inbox_addr;      // inbox_t[N]   (O(N), not O(N^2))
    uint64_t counts_addr;     // int[N*L]
    uint64_t phase_addr;      // phase_cycles_t[N]
    uint32_t N;
} args_t;

#endif
