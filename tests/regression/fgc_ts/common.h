#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include "../fgc_common/fgc_actor_args.h"

// Topological sort (HClib toposort_selector, the matching phase): the input
// is a random upper-triangular matrix with a full diagonal whose rows and
// columns were randomly permuted (Bale's generate_toposort_input). Row r and
// column c belong to actors r % N and c % N. A row with one nonzero left is
// matched to that column (its rowsum, the sum of its remaining column ids):
// the row's owner sends MATCHED to the column's owner, which queues the
// column and sends REMOVE to every row of the column; a REMOVE takes the
// column out of the row's count and sum and raises the row's level. Each
// actor's main loop drains its row and column queues and calls done once it
// has handled all its rows and columns.
//
// The permutations are built from the levels with global prefix sums (Bale's
// second phase), which is collective work, not messaging: the host builds
// them from the device's levels and matches and checks that the permuted
// matrix is upper triangular.
//
// config 0: non-actor reference, level-synchronous: ready rows update the
//           rows of their column with atomics; a global barrier ends a round.
// config 1: the actor runtime (fgc_actor.h).
enum { CFG_NON_ACTOR = 0, CFG_ACTOR = 1, CFG_COUNT };
enum { MB_TOPO = 0, NUM_MB = 1 };

#define TS_MATCHED 0x80000000u   // in ts_pkt_t.row: a MATCHED message

typedef struct {
  uint32_t row;    // REMOVE: the row's local index; MATCHED: TS_MATCHED
  uint32_t col;    // the column's global id
  uint32_t level;
} ts_pkt_t;

typedef struct {
  uint32_t config;
  uint32_t num_actors;    // N
  uint32_t rows;          // R rows (and columns) per actor
  uint32_t num_cores;     // global barrier participants (non-actor)
  uint64_t toff_addr;     // uint32_t[N * R + 1]: column offsets, by storage index owner * R + local
  uint64_t trows_addr;    // uint32_t[nnz]: each column's global row ids
  uint64_t rowcnt_addr;   // uint32_t[N * R]: nonzeros per row, by row storage index
  uint64_t rowsum_addr;   // uint32_t[N * R]: sum of each row's column ids (mod 2^32)
  uint64_t level_addr;    // uint32_t[N * R]: out
  uint64_t match_addr;    // uint32_t[N * R]: out, the matched column
  uint64_t work_addr;     // uint32_t[3][N * R]: row queue, column queue, column levels
  uint64_t flag_addr;     // uint32_t[2]: config 0's round flags
  uint64_t time_addr;     // uint64_t[cores][2]: each core's start and end cycle
  fgc_rt_args_t rt;
} kernel_arg_t;

#endif
