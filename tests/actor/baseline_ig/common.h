// baseline_ig: same workload as actor_ig, but direct remote reads instead of request/reply.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

#ifndef M
#define M           10      // requests per PE
#endif
#ifndef TABLE_SIZE
#define TABLE_SIZE  1000    // per-PE table size
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
    uint64_t t_start;       // ABSOLUTE core cycles at body entry and end (T42): the
    uint64_t t_end;         // launch-free body window, as in the actor kernels
} phase_cycles_t;

typedef struct {
    uint64_t phase_addr;    // phase_cycles_t[N]
    uint64_t table_addr;    // int[N*TABLE_SIZE]; table[pe*TABLE_SIZE + lidx] = pe*1000000 + lidx
    uint64_t tgt_addr;      // int[N*M] output
    uint32_t N;
} args_t;

#endif
