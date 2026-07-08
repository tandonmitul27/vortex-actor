// Bale histogram, two threads per actor (MAIN + RECV), tree-barrier termination.
// Data flows over an N x N grid of SPSC rings (grid[sender*N + recv]). Termination
// uses two single-writer flag arrays: arrived[] combines up the tree and released[]
// broadcasts back down, so no atomics are needed.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

#ifndef CAP
#define CAP   16    // ring capacity (power of 2)
#endif
#define L     10    // local buckets per PE (total = L * N)
#ifndef M
#define M     10    // updates per PE
#endif

typedef struct {
    uint32_t sender;
    uint32_t done_flag;       // unused with the tree barrier (kept for layout parity)
    int      bucket;          // local bucket on the owner PE
} packet_t;

typedef struct {
    volatile uint32_t head;
    volatile uint32_t tail;
    packet_t slots[CAP];
} ring_t;

// per-PE cycle breakdown; send_done now measures the tree-barrier flag work only.
typedef struct {
    uint64_t send_data;   // M bump messages
    uint64_t send_done;   // tree-barrier flag ops
    uint64_t handle_req;  // 0 — histo never replies
    uint64_t handle_rep;  // recv bump + counts++
    uint64_t empty_poll;  // empty sweeps before the last packet
    uint64_t tail_wait;   // idle after the last packet
    uint64_t total;       // full kernel_body span
} phase_cycles_t;

typedef struct {
    uint64_t grid_addr;       // ring_t[N*N]
    uint64_t counts_addr;     // int[N*L]
    uint64_t phase_addr;      // phase_cycles_t[N]
    uint64_t arrived_addr;    // volatile uint32_t[N]  tree-barrier arrival flags
    uint64_t released_addr;   // volatile uint32_t[N]  tree-barrier release flags
    uint32_t N;
} args_t;

#endif
