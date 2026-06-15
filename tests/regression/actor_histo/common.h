// actor_histo: Bale-style histogram via message passing.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

#define CAP   16    // ring capacity (power of 2)
#define L     10    // local buckets per PE (total = L * N)
#define M     10    // updates per PE

typedef struct {
    uint32_t sender;
    uint32_t done_flag;       // 0 = data, 1 = done
    int      bucket;          // local bucket on the owner PE
} packet_t;

typedef struct {
    volatile uint32_t head;
    volatile uint32_t tail;
    packet_t slots[CAP];
} ring_t;

// per-PE cycle breakdown; buckets sum to ~total
typedef struct {
    uint64_t send_data;   // M bump messages
    uint64_t send_done;   // done broadcast to all N
    uint64_t handle_req;  // 0 — histo never replies
    uint64_t handle_rep;  // recv bump + counts++
    uint64_t empty_poll;  // empty polls before the last packet
    uint64_t tail_wait;   // idle after the last packet
    uint64_t total;       // full kernel_body span
} phase_cycles_t;

typedef struct {
    uint64_t grid_addr;       // ring_t[N*N]
    uint64_t counts_addr;     // int[N*L] — counts[me*L + b]
    uint64_t phase_addr;      // phase_cycles_t[N]
    uint32_t N;
} args_t;

#endif
