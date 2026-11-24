#ifndef _FGC_ACTOR_ARGS_H_
#define _FGC_ACTOR_ARGS_H_

#include <stdint.h>

// Runtime arguments shared by the host and the device side of fgc_actor.h.

// Mailboxes per actor; with two ring sets (epochs) each uses two entries.
#define FGC_MAX_MB 4

// Sender granularity: who claims ring slots. Each lane its own, the lanes of a
// warp sending to the same queue one claim, or the lanes of a CTA one claim.
#define FGC_SENDER_THREAD 0
#define FGC_SENDER_WARP   1
#define FGC_SENDER_CTA    2

// Topology: what one queue is. One per (producer, receiver) pair, where the
// producer is the sender unit (thread, warp or CTA); one per receiver actor;
// one per receiver warp; one per receiver CTA.
#define FGC_TOPO_SPSC       0
#define FGC_TOPO_MPSC_ACTOR 1
#define FGC_TOPO_MPSC_WARP  2
#define FGC_TOPO_MPSC_CTA   3

// Termination: a done marker to every actor (T0), global counters (T1), or
// counters aggregated per CTA before the global ones (T2).
#define FGC_TERM_T0 0
#define FGC_TERM_T1 1
#define FGC_TERM_T2 2

// The configuration a kernel and its host are built for (make FGC_TOPO=..
// FGC_SENDER=.. FGC_TERM=..): MPSC-actor, thread senders, T1 by default.
#ifndef FGC_TOPO
#define FGC_TOPO FGC_TOPO_MPSC_ACTOR
#endif
#ifndef FGC_SENDER
#define FGC_SENDER FGC_SENDER_THREAD
#endif
#ifndef FGC_TERM
#define FGC_TERM FGC_TERM_T1
#endif

// Messages a receiver group (MPSC-warp, MPSC-CTA) takes off its queue per
// batch at most: the staging in local memory does not grow with the CTA.
#ifndef FGC_STAGE
#define FGC_STAGE 256
#endif

// One mailbox: num_queues rings of 1 << lgcap slots.
typedef struct {
  uint64_t slots;      // slot[num_queues << lgcap], zeroed by the host
  uint64_t tail;       // uint32_t[num_queues], next ticket per queue
  uint64_t heads;      // SPSC: uint32_t[num_queues], the receiver's next ticket
  uint32_t lgcap;
  uint32_t pad;
} fgc_mb_args_t;

// Termination counters of one mailbox, alone on a 64-byte line.
typedef struct {
  uint32_t done;       // actors (T2: CTAs) that called done on this mailbox
  uint32_t sent;       // messages sent, added at done
  uint32_t recvd;      // messages processed, added in batches
  uint32_t pad[13];
} fgc_term_t;

typedef struct {
  uint32_t num_actors;
  uint32_t recv_batch;   // R: messages taken per receive pass
  uint32_t term_period;  // K: idle passes between termination checks
  uint32_t ring_sets;    // 2 if the app runs several epochs, else 1
  uint32_t sender;       // FGC_SENDER_*
  uint32_t topo;         // FGC_TOPO_*
  uint32_t term;         // FGC_TERM_*
  uint32_t lg_threads;   // log2 threads per warp
  uint32_t lg_block;     // log2 threads per CTA (one CTA per core)
  uint32_t lmem_size;    // local memory per CTA the runtime needs
  uint32_t lm_xchg;      // local memory offsets: the warps' word pairs,
  uint32_t lm_vote;      //   the CTA exit vote,
  uint32_t lm_claim;     //   the CTA claim table,
  uint32_t lm_term;      //   the T2 counters,
  uint32_t lm_stage[FGC_MAX_MB]; // the group receive staging per mailbox
  uint32_t lm_gbytes;    // staging bytes per receiver group
  uint32_t pad2;
  uint64_t term_addr;    // fgc_term_t[epochs][mailboxes]
  fgc_mb_args_t mb[2 * FGC_MAX_MB];
} fgc_rt_args_t;

#ifdef __cplusplus
namespace fgc {
// One ring slot: the sequence word (2 * lap free, 2 * lap + 1 published), the
// sender, the receiver (read when a queue serves a group) and the message.
// 8-byte aligned so the sequence word and the sender are one 64-bit load.
template <typename T>
struct alignas(8) Slot {
  uint32_t seq;
  uint32_t from;
  uint32_t to;
  T        msg;
};

// One staged message of a group receive: taken off the ring by any member of
// the group, then handed to its owner through a list in local memory.
template <typename T>
struct Staged {
  uint32_t next;       // the owner's next staged message
  uint32_t from;
  T        msg;
};

// The sender's done marker under T0 rides in the from word.
constexpr uint32_t kMarker = 0x80000000u;
} // namespace fgc
#endif

#endif
