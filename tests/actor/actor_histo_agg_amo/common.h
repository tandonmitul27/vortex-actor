// actor_histo, AGGREGATING (Conveyor-style combining) SPSC actor + tree barrier.
// Instead of one message per bump, each PE accumulates its M bumps LOCALLY into
// its outgoing ring slots (one slot per owner, single-writer -> no atomics), then
// sends ONE combined message per owner carrying that owner's L bucket counts.
// Messages drop from O(M) to <=N per PE -> the send/recv cost becomes M-independent
// and, at high M, the actor's cost collapses toward "M uncontended local adds",
// approaching the atomicAdd baseline (which pays M CONTENDED global atomics).
// Each sender->owner channel carries exactly one aggregated message, so CAP=1.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

#ifndef CAP
#define CAP   1     // each sender->owner channel carries exactly one aggregated message
#endif
#ifndef L
#define L     10    // local buckets per PE (total = L * N)
#endif
#ifndef M
#define M     10    // updates per PE
#endif

typedef struct {
    uint32_t sender;
    int      counts[L];       // aggregated bump counts for the owner's L buckets
} packet_t;

typedef struct {
    volatile uint32_t head;
    volatile uint32_t tail;
    packet_t slots[CAP];
} ring_t;

// per-PE cycle breakdown (same taxonomy as actor_histo). send_done now measures
// the tree-barrier flag work only -- it should collapse vs the O(N^2) broadcast.
typedef struct {
    uint64_t send_data;   // M bump messages
    uint64_t send_done;   // tree-barrier flag ops (was: O(N) done broadcast)
    uint64_t handle_req;  // 0 — histo never replies
    uint64_t handle_rep;  // recv bump + counts++
    uint64_t empty_poll;  // empty sweeps before the last packet
    uint64_t tail_wait;   // idle after the last packet
    uint64_t total;       // full kernel_body span
    // fine-grained breakdown regions (subdivide send_data)
    uint64_t bd_acc_off;  // O(M) folds into the global grid
    uint64_t bd_pub_off;  // O(N) combined-message publish
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
