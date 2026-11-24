#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include "../fgc_common/fgc_actor_args.h"

// PageRank push (HClib pagerank_push_LU_selector): every actor owns R rows of
// an undirected Erdos-Renyi graph (global row r is actor r % N's local row
// r / N). Per iteration, for every nonzero (y, c) of its rows an actor sends
// rank(y) / degree(y) to c's owner, whose handler adds it into row c. After
// the iteration every actor sets rank = (1 - d) / n + d * sum on its own rows.
// A fixed number of iterations replaces HClib's convergence test.
//
// Ranks are fixed point, rank * n * 2^16 in 32 bits, so the non-actor version
// can push with the integer atomic add (the ISA has no float atomic) and both
// versions, whose sums are exact in any order, match the host bit for bit.
//
// config 0: non-actor reference, one atomic add per nonzero; a global barrier
//           and an L1 invalidation separate the phases.
// config 1: the actor runtime (fgc_actor.h), one epoch per iteration.
enum { CFG_NON_ACTOR = 0, CFG_ACTOR = 1, CFG_COUNT };
enum { MB_PUSH = 0, NUM_MB = 1 };

#define PS_ONE     65536.0f   // fixed-point rank of 1 / n
#define PS_DAMPING 0.85f

typedef struct {
  uint32_t dst;     // c's local row at its owner
  uint32_t contrib; // rank(y) / degree(y), fixed point
} ps_pkt_t;

typedef struct {
  uint32_t config;
  uint32_t num_actors;    // N
  uint32_t rows;          // R rows per actor
  uint32_t iters;
  uint32_t num_cores;     // global barrier participants (non-actor)
  uint32_t pad;
  uint64_t offsets_addr;  // uint32_t[N * R + 1], by storage index owner * R + local
  uint64_t cols_addr;     // uint32_t[nnz], global column ids
  uint64_t rank_addr;     // uint32_t[N * R], fixed point
  uint64_t acc_addr;      // uint32_t[N * R]: the iteration's sums
  uint64_t time_addr;     // uint64_t[cores][2]: each core's start and end cycle
  fgc_rt_args_t rt;
} kernel_arg_t;

// The new rank of a row from its sum; the host runs the same arithmetic, so
// no fused multiply-add.
static inline uint32_t ps_update(uint32_t acc) {
#ifdef __clang__
#pragma clang fp contract(off)
#endif
  return (uint32_t)((1.0f - PS_DAMPING) * PS_ONE + PS_DAMPING * (float)acc + 0.5f);
}

#endif
