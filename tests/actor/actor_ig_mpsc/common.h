// Index-gather over MPSC shared inboxes, one thread per actor.
// Each receiver has two inboxes: mb0 for requests, mb1 for replies (O(N) memory
// each). Producers claim a slot with an atomic fetch-add on `tail`, a single
// consumer drains in order, and a per-slot `seq` field is the handshake (see
// actor_histo_mpsc/common.h). Receiving is O(1) per message, not an O(N) scan.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

#ifndef CAP
#define CAP   16   // inbox capacity (shared across all senders to a receiver)
#endif
#ifndef M
#define M     10   // queries per PE
#endif
#ifndef TABLE_SIZE
#define TABLE_SIZE  1000    // table entries owned by each actor (same as actor_ig)
#endif
#ifndef TGT_T
#define TGT_T 1
#endif
#define TGT_IDX(actor, q, n) (TGT_T ? ((size_t)(q) * (size_t)(n) + (size_t)(actor)) \
                                    : ((size_t)(actor) * (size_t)(M) + (size_t)(q)))

typedef struct {
    uint32_t sender;
    uint32_t done_flag;        // 0 = data, 1 = done
    int      idx;              // request: requester's tgt slot (echoed in reply)
    int      value;           // request: index to look up; reply: looked-up value
    volatile uint32_t seq;     // MPSC slot handshake (ticket/ready marker)
} slot_t;

typedef struct {
    volatile uint32_t head;    // consumer index (single consumer)
    volatile uint32_t tail;    // producer ticket (atomic fetch-add)
    slot_t   slots[CAP];
} inbox_t;                      // one per receiver, per mailbox -> inbox_t[N] x 2

// per-PE cycle breakdown (same taxonomy as the other actor tests)
typedef struct {
    uint64_t send_data;   // my M requests on mb0
    uint64_t send_done;   // mb0 + mb1 done broadcasts
    uint64_t handle_req;  // service a request (claim + publish reply on mb1)
    uint64_t handle_rep;  // land a reply (drain mb1 -> tgt)
    uint64_t empty_poll;  // idle iterations before the last packet
    uint64_t tail_wait;   // idle after the last packet
    uint64_t total;       // full kernel_body span
} phase_cycles_t;

typedef struct {
    uint64_t mb0_addr;        // inbox_t[N]  request channels
    uint64_t mb1_addr;        // inbox_t[N]  reply channels
    uint64_t table_addr;      // int[N*TABLE_SIZE] owned state, table[me*TABLE_SIZE + k]
    uint64_t tgt_addr;        // int[N*M] output, laid out by TGT_IDX
    uint64_t phase_addr;      // phase_cycles_t[N]
    uint32_t N;
} args_t;

#endif
