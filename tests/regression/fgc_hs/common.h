#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include "../fgc_common/fgc_actor_args.h"

// Histogram (HClib histo_selector): every actor sends M updates to random
// buckets of a table of N x L buckets; bucket g belongs to actor g >> lg and
// the message carries its local index. The handler adds one to the bucket.
// One actor is one thread, and every hardware thread is an actor.
//
// config 0: non-actor reference, one atomic add per update on the table.
// config 1: the actor runtime (fgc_actor.h).
enum { CFG_NON_ACTOR = 0, CFG_ACTOR = 1, CFG_COUNT };

typedef struct {
  uint32_t config;
  uint32_t num_actors;   // N
  uint32_t log_buckets;  // L = 1 << log_buckets buckets per actor
  uint32_t num_msgs;     // M updates sent per actor
  uint64_t counts_addr;  // uint32_t[N * L]
  uint64_t time_addr;    // uint64_t[cores][2]: each core's start and end cycle
  fgc_rt_args_t rt;
} kernel_arg_t;

// Update i of actor a goes to bucket hash_key(a, i, N * L).
static inline uint32_t hash_mix(uint32_t a, uint32_t i) {
  uint32_t x = a * 0x9E3779B9u + i * 0x85EBCA77u;
  x ^= x >> 13;
  x *= 0xC2B2AE3Du;
  x ^= x >> 16;
  return x;
}

// Maps the hash onto [0, range) with a multiply, not a divide.
static inline uint32_t hash_key(uint32_t a, uint32_t i, uint32_t range) {
  return (uint32_t)(((uint64_t)hash_mix(a, i) * range) >> 32);
}

#endif
