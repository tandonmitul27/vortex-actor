// baseline_histo: same workload as actor_histo, but atomic updates instead of messages.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

#ifndef L
#define L     10    // local buckets per PE (total = L * N); smaller L => fewer buckets => more contention
#endif
#ifndef M
#define M     10    // updates per PE
#endif


// per-PE cycle breakdown. The atomics are posted writes (result unused), so the
// warp does not wait for them in-loop: `apply` sees only issue cost, and the
// contended memory backlog is captured by timing a fence after the loop (`drain`).
typedef struct {
    uint64_t apply;       // cycles inside the atomic-add issues
    uint64_t drain;       // fence after the loop: waiting out the contended backlog
    uint64_t total;       // full kernel_body span (apply + drain + compute + loop)
    uint64_t t_start;     // ABSOLUTE core cycles at body entry and end, for the
    uint64_t t_end;       // launch-free body window (T42), as in the actor kernels
} phase_cycles_t;

typedef struct {
    uint64_t counts_addr;     // int[N*L]
    uint64_t phase_addr;      // phase_cycles_t[N]
    uint32_t N;               // PE count
} args_t;

#endif
