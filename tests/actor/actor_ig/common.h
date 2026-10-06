// actor_ig: request/reply index-gather. mb0 carries requests, mb1 replies.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

#ifndef CAP
#define CAP   16   // ring capacity (power of 2)
#endif
#ifndef M
#define M     10   // queries per PE
#endif

// THE OWNED TABLE. Every other index gather kernel here owns an int table of
// TABLE_SIZE entries per actor and answers a request by LOADING from it. This
// kernel used to answer arithmetically, reply = me*1000000 + lidx, which is the
// value the table would have held, with lidx taken modulo a hardcoded 1000.
// That is the same algorithm with the memory access deleted, so the actor
// baseline was doing strictly less work than the kernels measured against it,
// and -DTABLE_SIZE had no effect at all: three capacity points of the S3 sweep
// came back byte identical. The table is real now and TABLE_SIZE is honoured,
// so every index gather kernel touches the same state at the same capacity.
#ifndef TABLE_SIZE
#define TABLE_SIZE  1000    // table entries owned by each actor
#endif

// OUTPUT LAYOUT. tgt[actor*M + q] gives each actor one contiguous run of
// answers. That is the natural CPU layout and the worst possible GPU layout: the
// sixteen lanes of a warp are sixteen consecutive actors, so at question q they
// store to addresses M*4 bytes apart and not one pair shares a cache line.
// Measured on baseline_ig: 2,569,871 DRAM writes for 2,560,000 queries, one per
// query, nothing merged. TGT_T=1 stores tgt[q*N + actor] instead, so a warp's
// sixteen stores fall in one line.
//
// This changes the OUTPUT CONTRACT, not the transport, so it must be set the
// same way for every index gather kernel or the comparison is meaningless. It
// helps the non-actor baseline far more than it helps the actor kernels, since
// for us the landing store is a small share of total traffic. Measured at
// M=10000, TABLE_SIZE=16000: baseline_ig 26,482,914 -> 13,963,796 (1.896x),
// actor_ig 257,294,698 -> 256,799,222, actor_ig_msg 55,765,810 -> 54,875,789,
// actor_ig_dense 47,776,798 -> 47,561,593. TGT_T=1 is the honest layout and is
// the default, so every index gather ratio is against a baseline that coalesces
// its stores.
#ifndef TGT_T
#define TGT_T 1
#endif
#define TGT_IDX(actor, q, n) (TGT_T ? ((size_t)(q) * (size_t)(n) + (size_t)(actor)) \
                                    : ((size_t)(actor) * (size_t)(M) + (size_t)(q)))

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
    uint64_t table_addr;      // int[N*TABLE_SIZE] owned state, table[me*TABLE_SIZE + k]
    uint64_t tgt_addr;        // int[N*M] output: tgt[me*M + i]
    uint64_t phase_addr;      // phase_cycles_t[N]
    uint32_t N;
} args_t;

#endif
