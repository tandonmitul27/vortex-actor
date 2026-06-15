// baseline_ig: same workload as actor_ig, but direct remote reads instead of request/reply.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

#define M           10      // requests per PE
#define TABLE_SIZE  1000    // per-PE table size

typedef struct {
    uint64_t table_addr;    // int[N*TABLE_SIZE]; table[pe*TABLE_SIZE + lidx] = pe*1000000 + lidx
    uint64_t tgt_addr;      // int[N*M] output
    uint32_t N;
} args_t;

#endif
