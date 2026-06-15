// baseline_histo: same workload as actor_histo, but atomic updates instead of messages.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

#define L     10    // local buckets per PE (total = L * N)
#define M     10    // updates per PE

typedef struct {
    uint64_t counts_addr;     // int[N*L]
    uint32_t N;               // PE count
} args_t;

#endif
