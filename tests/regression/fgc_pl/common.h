#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include "../fgc_common/fgc_actor_args.h"

// PageRank pull (HClib pagerank_pull_LU_selector): every actor owns R rows of
// an undirected Erdos-Renyi graph (global row r is actor r % N's local row
// r / N). Per iteration, for every nonzero (y, c) of its rows an actor sends a
// REQUEST to c's owner; the owner's handler replies with c's rank and degree,
// and the asker's handler adds rank / degree into row y. After the iteration
// every actor sets rank = (1 - d) / n + d * sum on its own rows. A fixed
// number of iterations replaces HClib's convergence test (a global
// reduction, not messaging). Ranks are floats.
//
// config 0: non-actor reference, direct loads of the neighbours' ranks; a
//           global barrier and an L1 invalidation separate iterations.
// config 1: the actor runtime (fgc_actor.h), one epoch per iteration.
enum { CFG_NON_ACTOR = 0, CFG_ACTOR = 1, CFG_COUNT };
enum { MB_REQUEST = 0, MB_RESPONSE = 1, NUM_MB = 2 };

#define PL_DAMPING 0.85f

typedef struct {
  uint32_t src;   // REQUEST: c's local row at its owner
  uint32_t dst;   // the asker's local row y
  float    rank;  // RESPONSE: c's rank
  uint32_t deg;   // RESPONSE: c's degree
} pl_pkt_t;

typedef struct {
  uint32_t config;
  uint32_t num_actors;    // N
  uint32_t rows;          // R rows per actor
  uint32_t iters;
  uint32_t num_cores;     // global barrier participants (non-actor)
  uint32_t pad;
  uint64_t offsets_addr;  // uint32_t[N * R + 1], by storage index owner * R + local
  uint64_t cols_addr;     // uint32_t[nnz], global column ids
  uint64_t rank_addr;     // float[2][N * R]: config 0 alternates, config 1 uses [0]
  uint64_t acc_addr;      // float[N * R]: config 1's per-iteration sums
  uint64_t time_addr;     // uint64_t[cores][2]: each core's start and end cycle
  fgc_rt_args_t rt;
} kernel_arg_t;

#endif
