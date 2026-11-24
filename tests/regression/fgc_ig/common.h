#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include "../fgc_common/fgc_actor_args.h"

// Index gather (HClib ig_selector): every actor reads M random entries of a
// table of N x L entries; entry g belongs to actor g >> lg. A REQUEST carries
// the asker's output index and the owner's local index; the owner's handler
// sends a RESPONSE with the entry back to the asker, whose handler stores it.
// One actor is one thread, and every hardware thread is an actor.
//
// config 0: non-actor reference, one direct load per request.
// config 1: the actor runtime (fgc_actor.h).
enum { CFG_NON_ACTOR = 0, CFG_ACTOR = 1, CFG_COUNT };
enum { MB_REQUEST = 0, MB_RESPONSE = 1, NUM_MB = 2 };

typedef struct {
  uint32_t idx;   // the asker's output index
  uint32_t val;   // REQUEST: the owner's local index; RESPONSE: the entry
} ig_pkt_t;

typedef struct {
  uint32_t config;
  uint32_t num_actors;   // N
  uint32_t log_entries;  // L = 1 << log_entries entries per actor
  uint32_t num_msgs;     // M requests per actor
  uint64_t table_addr;   // uint32_t[N * L]
  uint64_t tgt_addr;     // uint32_t[N * M]: actor a's results at a * M
  uint64_t time_addr;    // uint64_t[cores][2]: each core's start and end cycle
  fgc_rt_args_t rt;
} kernel_arg_t;

// Request i of actor a reads entry hash_key(a, i, N * L).
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

// The table's contents, so a result shows which entry it came from.
static inline uint32_t table_value(uint32_t g) {
  return g * 0x2545F491u + 0x6A09E667u;
}

#endif
