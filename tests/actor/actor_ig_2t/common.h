// actor_ig: request/reply index-gather. mb0 carries requests, mb1 replies.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

#define CAP   16   // ring capacity (power of 2)
#define M     10   // queries per PE

typedef struct {
    uint32_t sender;
    uint32_t done_flag;       // 0 = data, 1 = done
    int      idx;             // request slot in tgt[]; echoed in the reply
    int      value;           // request: index to look up; reply: looked-up value
} packet_t;

typedef struct {
    volatile uint32_t head;
    volatile uint32_t tail;
    packet_t slots[CAP];
} ring_t;

// per-PE cycle breakdown; buckets sum to ~total
typedef struct {
    uint64_t send_data;   // M requests sent on mb0
    uint64_t send_done;   // mb0 + mb1 done broadcasts
    uint64_t handle_req;  // recv request (mb0) + send reply (mb1)
    uint64_t handle_rep;  // recv reply (mb1) + write tgt[]
    uint64_t empty_poll;  // empty polls before the last packet
    uint64_t tail_wait;   // idle after the last packet
    uint64_t total;       // full kernel_body span
} phase_cycles_t;

typedef struct {
    uint64_t mb0_grid_addr;   // request channel
    uint64_t mb1_grid_addr;   // reply channel
    uint64_t tgt_addr;        // int[N*M] output: tgt[me*M + i]
    uint64_t phase_addr;      // phase_cycles_t[N]
    uint32_t N;
} args_t;

#endif
