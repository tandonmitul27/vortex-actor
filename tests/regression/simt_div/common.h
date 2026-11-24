#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// mode 0: each lane calls a different function through one call site.
// mode 1: each lane walks its own linked list in a while loop, so lanes
//         leave the loop at different times.
// mode 2: the same lists drained by a loop on a per-lane done flag.
// mode 3: the same lists drained by an endless loop with an early break.
// mode 4: the old actor drain loop: NUM_MB x NUM_SRC inboxes per lane (lengths
//         in head, packets in val), a handler table, a done count and a flag.
// mode 6: REPS indirect calls with one id for every lane.
// mode 8: a divergent switch compiled as a jump table.
// mode 9: mode 0 as a divergent tail call.
typedef struct {
  uint32_t num_points;
  uint32_t mode;
  uint64_t head_addr;
  uint64_t next_addr;
  uint64_t val_addr;
  uint64_t dst_addr;
} kernel_arg_t;

#define LIST_END 0xffffffffu

#define NUM_MB   2
#define NUM_SRC  4
#define MAX_PKTS 8
#define REPS     64

#endif
