#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include "../fgc_common/fgc_actor_args.h"

// Random permutation (HClib randperm_pure_selector, phase one: dart throwing,
// Gibbons, Matias and Ramachandran): every actor throws its R darts (the
// values i * N + me) at random slots of a target of 2 * N * R slots; slot g
// belongs to actor g % N as its local slot g / N. The owner's THROW handler
// keeps the dart if the slot is empty and REPLYs either way; the thrower's
// REPLY handler throws a rejected dart again at a new random slot and calls
// done(THROW) once all its darts have landed. HClib draws with lrand48; here
// every actor draws from its own counter-based stream.
//
// Phase two (pack the target and send the values out in order) needs global
// prefix sums, collective work: the host checks the target instead, every
// dart in exactly one slot.
//
// THROW and REPLY handlers send to each other. An actor has at most R darts
// in flight, so a REPLY ring of R slots always takes the reply, THROWs always
// drain, and the cycle cannot deadlock: the host sizes the REPLY ring so.
//
// config 0: non-actor reference, a compare-and-swap per throw on the target.
// config 1: the actor runtime (fgc_actor.h).
enum { CFG_NON_ACTOR = 0, CFG_ACTOR = 1, CFG_COUNT };
enum { MB_THROW = 0, MB_REPLY = 1, NUM_MB = 2 };

#define RP_EMPTY    0xFFFFFFFFu
#define RP_REJECTED 0x80000000u   // in rp_pkt_t.val of a REPLY

typedef struct {
  uint32_t idx;   // the slot's local index at its owner
  uint32_t val;   // the dart; a REPLY sets RP_REJECTED if the slot was taken
} rp_pkt_t;

typedef struct {
  uint32_t config;
  uint32_t num_actors;    // N
  uint32_t darts;         // R darts per actor
  uint32_t pad;
  uint64_t target_addr;   // uint32_t[N * 2R], by storage index owner * 2R + local
  uint64_t time_addr;     // uint64_t[cores][2]: each core's start and end cycle
  fgc_rt_args_t rt;
} kernel_arg_t;

// Throw k of actor a: a slot of the whole target.
static inline uint32_t rp_slot(uint32_t a, uint32_t k, uint32_t range) {
  uint32_t x = a * 0x9E3779B9u + k * 0x85EBCA77u;
  x ^= x >> 13;
  x *= 0xC2B2AE3Du;
  x ^= x >> 16;
  return (uint32_t)(((uint64_t)x * range) >> 32);
}

#endif
