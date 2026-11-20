#ifndef LMEM_BYTES
#define LMEM_BYTES 16384   // scratchpad bytes per core (sensitivity runs pass -DLMEM_BYTES; 2026-11-15)
#endif
// Random permutation (dart throwing), after HClib-Actor's bale_actor tests (srirajpaul/hclib, branch bale3_actor):
// test/randperm_selector.cpp (the actor version) and test/randperm_agi.cpp (the non-actor baseline).
// n = RP_LROWS * N items (-n), M = 2n target slots, distributed cyclically: slot r belongs to PE r % P at r / P.
// A dart's k-th throw goes to slot H(item, k) (hash, seed RP_SEED): both versions throw each item at the same
// sequence of slots, and no generator state is kept.
// ACTOR=1 (P = N actors): copied_rand_permp_selector. Phase 1, in rounds: every actor throws each of its darts not
//          yet landed (mailbox THROW, {sender | dart | slot}) to the slot's owner; the owner's handler claims the slot
//          if it is empty and replies (mailbox REPLY, {landed | dart}) to the sender, whose handler marks the dart
//          landed or leaves it to be thrown again; the rounds end when a round has no failure. Phase 2: every actor
//          packs its filled slots, offset = the actors' prefix of the counts (lgp_prior_add_l), and sends the i-th
//          value to PE (offset + i) % P, whose handler stores it at lperm[pos++] (arrival order).
//          On the append queue at our best: 32-bit words, one geometry (THROW queues t * N + owner keyed by the
//          owner's slot tile, REPLY queues rbase + sender); phase 1 never empties a queue (tails grow, consumers
//          keep heads), a failure REPLY re-throws at once, so a round is two barriers; phase 1 is one leaf. The claim: only the owner's warp touches the owner's slots, so CLAIM=0 needs no atomic
//          (load, store if empty, reload: the reload settles two lanes of one instruction); CLAIM=1 uses amocas.
// ACTOR=0 (P = 1): copied_rand_permp_agi on one GPU: a thread per item (grid stride) throws it until
//          lgp_cmp_and_swap (amocas) finds its slot empty; then the filled slots are packed in slot order into perm
//          (a parallel scan: one PE holds every slot).
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>
#include "../probe_apply.h"   // PROBE_APPLYT, ROUND1 (measurement only)
#ifndef NASSUME
#define NASSUME    256      // actors
#endif
#ifndef SPLIT
#define SPLIT      4        // warps per block (16 actors per block)
#endif
#ifndef ACTOR
#define ACTOR      1
#endif
#ifndef CLAIM
#define CLAIM      1        // 1: the claim is an amocas; 0: load, store if empty, reload (no atomic)
#endif
#ifndef HWBAR
#define HWBAR      0        // 1: the machine's global barrier; 0: the software barrier (atomic counter + poll)
#endif
#ifndef TILE0
#define TILE0      1        // 1: round 0 claims in the warp's scratchpad tile (the slots start EMPTY), written out once
#endif
#ifndef CNTW
#define CNTW       0        // 1: the THROW handler counts its wins into cnts[o] (no counting pass in phase 2)
#endif
#ifndef TILEK
#define TILEK      1        // rounds k < TILEK claim in the scratchpad tile (k >= 1 loads the tile first)
#endif
#ifndef SEND1
#define SEND1      1        // 1: phase 2 send in one coalesced pass (warp ballot gives each lane its rank)
#endif
#ifndef SKIPE
#define SKIPE      1        // rounds k >= TILEK: lane t fetches tile queue t's head/tail, a ballot skips the empty queues
#endif
#ifndef PIPE
#define PIPE       0        // rounds k >= TILEK, CLAIM=1: U claims (amocas) issued before their U replies
#endif
#ifndef RP_TB
#define RP_TB      10       // max bits of the in-tile slot: 1024 slots x 4 B x SPLIT warps = the 16 KB scratchpad
#endif
#define LANES      16
#define GLANES     16
#ifndef NBLK
#define NBLK       16                           // blocks = cores (one block of SPLIT warps per core); sensitivity runs set the core count
#endif
#define APB        ((NASSUME) / (NBLK))         // actors per block
#define OW         ((APB) / (SPLIT))            // actors per warp
static_assert((OW) >= 1 && (APB) % (SPLIT) == 0, "NASSUME: a multiple of NBLK * SPLIT");
#ifndef AQ_BASE
#define AQ_BASE    0xD0000000u
#endif
#ifndef AQ_SIZE
#define AQ_SIZE    0x100000u
#endif
#ifndef Q_CAP
#define Q_CAP      4096     // per-queue region (32-bit words); the rest overflows into pool chunks
#endif
#define QS32       (((((Q_CAP) * ((MSG64) ? 8 : 4) + 63) / 64) | 1) * 16)  // whole odd lines, in 32-bit words
#ifndef SWQ
#define SWQ        0        // 1: software append queue, no L2 append unit (../swq.h)
#endif
#ifndef OVF_CH
#define OVF_CH     256
#endif
#ifndef WC_U
#define WC_U       4
#endif
#define MAXR       64       // rounds
#define EMPTY      0xffffffffu
constexpr uint32_t rp_log2(uint32_t x) { return x <= 1 ? 0 : 1 + rp_log2(x / 2); }
#define AB         (rp_log2(NASSUME))   // bits of the sender
static_assert((1u << (AB)) == (NASSUME), "NASSUME: a power of 2");
constexpr uint32_t rp_clog2(uint32_t x) { return x <= 1 ? 0 : 1 + rp_clog2((x + 1) / 2); }
// the problem's geometry is compile time (as the index gather's constant N): every size an immediate
#ifndef RP_L
#define RP_L       1000     // items per PE (the tests' -n)
#endif
#ifndef RP_SEED
#define RP_SEED    101892u
#endif
#define LN         ((uint32_t)(RP_L))                 // darts per actor
#define LMS        (2u * (LN))                        // slots per actor
#define NN         ((LN) * (uint32_t)(NASSUME))       // items
#define MM         (2u * (NN))                        // slots
#define JB         (rp_clog2(LN) ? rp_clog2(LN) : 1u) // bits of the dart
#define LB         (rp_clog2(LMS))
#ifndef MSG64
#define MSG64      0        // 1: phase-1 messages (THROW, REPLY) are 64-bit (B off where the untiled slot field does not fit)
#endif
#if MSG64
typedef uint64_t msg_t;
#else
typedef uint32_t msg_t;
#endif
#define MSGB       ((MSG64) ? 64u : 32u)
#define TBS0       ((MSGB - (AB) - (JB)) < (LB) ? (MSGB - (AB) - (JB)) : (LB))
// Ablation switches (results/mapping/ABLATION.md, ../abl.h). Both default off (= the full design).
#ifndef NOTILE
#define NOTILE     0        // 1: piece B off: one THROW queue per owner (the slot field holds all its slots), claims in
#endif                      //    memory (needs TILE0=0)
#ifndef LANE_RECV
#define LANE_RECV  0        // 1: piece C off: a lane serves / drains a whole queue by itself; with tiles each lane has
#endif                      //    its own 2^(RP_TB - 4)-slot tile (THROW queues keyed by that smaller tile)
#define RP_TBE     ((NOTILE) ? 32u : (LANE_RECV) ? (uint32_t)(RP_TB) - 4u : (uint32_t)(RP_TB))
#define TBS        ((TBS0) < (RP_TBE) ? (TBS0) : (RP_TBE))   // bits of the in-tile slot
#define TILE_W     (1u << (TBS))                                                // slots per tile
#define TS         (((LMS) + (1u << (TBS)) - 1) >> (TBS))                       // slot tiles per owner
#define RBASE      ((TS) * (uint32_t)(NASSUME))       // first REPLY queue
#define NQ         ((RBASE) + (uint32_t)(NASSUME))
static_assert((AB) + (JB) < 32 && (NQ) <= (AQ_SIZE) / 64 - 1, "RP_L: the THROW word or the queue count does not fit");
static_assert((TILE0) || (TILEK) <= 1, "TILEK needs TILE0");
static_assert(!(NOTILE) || ((TS) == 1 && !(TILE0)), "NOTILE needs TILE0=0 and a slot field that holds the actor's slots");
#define SPW        ((LANE_RECV) ? (TILE_W) * (GLANES) : (TILE_W))   // scratchpad words per warp (TILE0)
static_assert(!(PIPE) || (CLAIM), "PIPE needs CLAIM=1");
static_assert(!(TILE0) || (TILE_W) * (SPLIT) * 4 <= (LMEM_BYTES), "TILE0: the tiles do not fit the 16 KB scratchpad");

// slot of an item's k-th throw: murmur3's finalizer of (item, k, seed), scaled to [0, M)
static inline uint32_t rp_slot(uint32_t item, uint32_t k, uint32_t seed, uint32_t M) {
    uint32_t h = item * 0x9e3779b1u + k * 0x85ebca77u + seed;
    h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
    return (uint32_t)(((uint64_t)h * M) >> 32);
}

#ifndef PROF
#define PROF       0        // 1: phase 1 step timers (round-0 throws, serve round 0, serve later, reply, barriers)
#endif
typedef struct { uint64_t p1, p2, total, msgs, throws, rounds, t0, t1, prof[5];
#if PROBE_APPLYT || ROUND1
                 uint64_t pa_slot[2], pa_h, pa_s, pa_r, pa_x, r1;   // handlers, their sends, receive calls, other sends
#endif
} phase_cycles_t;

typedef struct {
    uint64_t tgt_addr;        // ACTOR=1: uint32_t[N][lM] target slots, EMPTY; ACTOR=0: uint32_t[M]
    uint64_t land_addr;       // ACTOR=1: uint8_t[N][lN] landed flags
    uint64_t perm_addr;       // ACTOR=1: uint32_t[N][lN] lperm; ACTOR=0: uint32_t[n]
    uint64_t cnts_addr;       // ACTOR=1: uint32_t[N] filled slots per actor
    uint64_t fails_addr;      // ACTOR=1: uint32_t[MAXR] failed throws per round
    uint64_t heads_addr;      // ACTOR=1: uint32_t[rbase + N] consumer heads (phase 1 never empties a queue)
    uint64_t sums_addr;       // scan scratch: uint32_t[threads] (+ uint32_t[warps] after it, ACTOR=0)
    uint64_t qs_addr;         // ACTOR=1: queue regions, (rbase + N) x QS32 x 4 B
    uint64_t tails_addr;      // ACTOR=1: uint32_t[rbase + N] (model written)
    uint64_t pool_addr;       // ACTOR=1: overflow chunks
    uint64_t ctab_addr;       // ACTOR=1: uint64_t[(rbase + N) * ovf_maxc]
#if SWQ
    uint64_t swq_addr;        // SWQ: the pool counter (zeroed)
#endif
    uint64_t phase_addr;      // phase_cycles_t[warps]
    uint64_t bar_addr;        // volatile uint32_t[8]
    uint32_t N;               // actors
    uint32_t n;               // items
    uint32_t M;               // target slots, 2n
    uint32_t lN;              // ACTOR=1: items (darts) per actor, n / N
    uint32_t lM;              // ACTOR=1: slots per actor, M / N
    uint32_t seed;
    uint32_t ts;              // ACTOR=1: slot tiles per owner (THROW queues t * N + owner)
    uint32_t tbs;             // ACTOR=1: bits of the in-tile slot
    uint32_t jb;              // ACTOR=1: bits of the dart (AB + jb + tbs <= 32)
    uint32_t rbase;           // ACTOR=1: first REPLY queue, ts * N
    uint32_t ovf_maxc;        // ACTOR=1: chunks per queue
    uint32_t ovf_nch;         // ACTOR=1: pool size in chunks
} args_t;
#endif
