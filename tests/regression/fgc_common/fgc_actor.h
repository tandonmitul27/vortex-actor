#ifndef _FGC_ACTOR_H_
#define _FGC_ACTOR_H_

// A GPU actor runtime with the semantics of HClib's bale_actor selectors.
//
// One actor is one thread. An actor has NMB mailboxes of messages of type T,
// each with a handler run by the owner, one message at a time. send() is
// non-blocking: a message the buffer cannot take yet waits in the sender's
// out-slot while the sender keeps processing its own mail. done(mb) says the
// actor sends nothing more on mailbox mb. A mailbox ends once every actor
// called done on it and every message sent on it was processed; then each
// actor calls done on the next mailbox. run() returns when every mailbox has
// ended. Sending and processing interleave; there is no barrier.
//
// An app that runs several finish scopes in a row (one per iteration, as
// HClib builds a new selector per iteration) calls next_epoch() between them:
// each epoch has its own termination counters, and epochs alternate between
// two ring sets, since an actor may start sending in epoch e + 1 while
// another still polls epoch e.
//
// Buffers. A mailbox is a set of queues, each a ring of 1 << lgcap slots. A
// sender takes a ticket on the queue's tail; the slot of ticket t is free when
// its sequence word holds 2 * lap(t) and published when it holds
// 2 * lap(t) + 1, so a zeroed ring is empty. The consumer frees a slot by
// writing 2 * lap + 2. The topology says what one queue is:
//   SPSC        one per (producer, receiver actor), the producer being the
//               sender unit (thread, warp or CTA), which alone owns the tail;
//               the receiver scans its queues round robin
//   MPSC-actor  one per receiver actor, which reads it alone
//   MPSC-warp   one per receiver warp: the warp takes up to a slot per lane,
//               then each lane processes the messages addressed to it
//   MPSC-CTA    one per receiver CTA, the same through the CTA's local memory
//
// send() only places the message in the sender's out-slot. The actor loop
// claims and publishes out-slots at one point where the warp (and, when a CTA
// works together, the CTA) is converged, so the sender granularity is a
// choice made there: each lane claims its own ticket (FGC_SENDER_THREAD); the
// lanes of a warp sending to the same queue share one claim, the leader
// handing out tickets through the warp's word pair in local memory
// (FGC_SENDER_WARP; shuffles do not span a warp on every machine); or the
// lanes of a CTA do, grouped by a hash table in local memory (FGC_SENDER_CTA).
//
// Termination: T0 sends a done marker to every actor after the actor's last
// message on the mailbox, and a mailbox has ended for an actor once it has a
// marker from everyone. T1 keeps per mailbox global counts of actors done,
// messages sent (added at done) and messages processed (added in batches),
// and has ended when every actor is done and processed == sent. T2 adds the
// done and sent counts per CTA in local memory first; the CTA's last actor
// publishes them.
//
// The L1 is not coherent across cores, so everything another core writes is
// read with LR, which the L2 answers like a plain read without touching the
// L1. Data only its owner touches (the app's state) uses plain accesses.

#include <vx_intrinsics.h>
#include <vx_spawn2.h>
#include "fgc_actor_args.h"

namespace fgc {

inline uint32_t l2_load32(const void* p) {
  uint32_t v;
  __asm__ volatile ("lr.w %0, (%1)" : "=r"(v) : "r"(p) : "memory");
  return v;
}

inline uint64_t l2_load64(const void* p) {
#if __riscv_xlen == 64
  uint64_t v;
  __asm__ volatile ("lr.d %0, (%1)" : "=r"(v) : "r"(p) : "memory");
  return v;
#else
  auto w = reinterpret_cast<const uint32_t*>(p);
  uint32_t lo = l2_load32(w);
  return ((uint64_t)l2_load32(w + 1) << 32) | lo;
#endif
}

template <typename T>
inline T l2_load(const T* p) {
  static_assert(sizeof(T) % 4 == 0, "messages are whole words");
  union { T t; uint32_t w[sizeof(T) / 4]; uint64_t d[(sizeof(T) + 7) / 8]; } u;
  if constexpr (sizeof(T) % 8 == 0 && alignof(T) >= 8) {
    for (uint32_t i = 0; i < sizeof(T) / 8; ++i)
      u.d[i] = l2_load64(reinterpret_cast<const uint64_t*>(p) + i);
  } else {
    for (uint32_t i = 0; i < sizeof(T) / 4; ++i)
      u.w[i] = l2_load32(reinterpret_cast<const uint32_t*>(p) + i);
  }
  return u.t;
}

// A zero computed from v: whatever uses it waits until v, typically an
// atomic's result, has come back.
inline uint32_t after(uint32_t v) {
  uint32_t z;
  __asm__ volatile ("sub %0, %1, %1" : "=r"(z) : "r"(v));
  return z;
}

template <typename T, uint32_t NMB>
class Actor {
  // the configuration is compiled in, so a kernel carries only its own path
  static constexpr uint32_t kTopo   = FGC_TOPO;
  static constexpr uint32_t kSender = FGC_SENDER;
  static constexpr uint32_t kTerm   = FGC_TERM;
  static constexpr bool kGroup = kTopo == FGC_TOPO_MPSC_WARP || kTopo == FGC_TOPO_MPSC_CTA;
  // the CTA works together each pass
  static constexpr bool kCta = kSender == FGC_SENDER_CTA || kTopo == FGC_TOPO_MPSC_CTA;

public:
  Actor(const fgc_rt_args_t* rt, uint32_t me) : rt_(rt), me_(me) {
    term_     = reinterpret_cast<fgc_term_t*>(rt->term_addr);
    set_      = 0;
    epoch_    = 0;
    pass_     = 0;
    lane_     = vx_thread_id();
    local_    = me & ((1u << rt->lg_block) - 1);
    lmem_     = reinterpret_cast<uint8_t*>(csr_read_nv(VX_CSR_CTA_LMEM_ADDR));
    xchg_     = reinterpret_cast<volatile uint32_t*>(lmem_ + rt->lm_xchg) + 2 * (local_ >> rt->lg_threads);
    lg_prod_  = kSender == FGC_SENDER_WARP ? rt->lg_threads
              : kSender == FGC_SENDER_CTA  ? rt->lg_block : 0;
    lg_group_ = kTopo == FGC_TOPO_MPSC_WARP ? rt->lg_threads
              : kTopo == FGC_TOPO_MPSC_CTA  ? rt->lg_block : 0;
    nprod_    = rt->num_actors >> lg_prod_;
    for (uint32_t mb = 0; mb < 2 * NMB; ++mb) {
      head_[mb] = 0;
    }
    for (uint32_t mb = 0; mb < NMB; ++mb) {
      rr_[mb]      = 0;
      pdst_[mb]    = 0;
      pend_[mb]    = false;
      claimed_[mb] = false;
    }
    this->reset();
    // the runtime's local memory starts zeroed
    auto w = reinterpret_cast<volatile uint32_t*>(lmem_);
    for (uint32_t i = local_; i < rt->lmem_size / 4; i += 1u << rt->lg_block) {
      w[i] = 0;
    }
    __syncthreads();
  }

  // Ends this finish scope's bookkeeping; every mailbox must have ended.
  void next_epoch() {
    term_ += NMB;
    ++epoch_;
    if (rt_->ring_sets == 2) {
      set_ ^= 1;
    }
    this->reset();
  }

  // Takes msg for actor dst unless mailbox mb's out-slot still holds an
  // earlier message.
  bool send(uint32_t mb, const T& msg, uint32_t dst) {
    if (pend_[mb]) {
      return false;
    }
    this->stage(mb, msg, dst, false);
    ++sent_[mb];
    return true;
  }

  void done(uint32_t mb) {
    if (done_[mb]) {
      return;
    }
    done_[mb] = true;
    // @step done
    if (kTerm == FGC_TERM_T0) {
      mark_next_[mb] = 0;
    } else if (kTerm == FGC_TERM_T1) {
      __atomic_fetch_add(&term_[mb].sent, sent_[mb], __ATOMIC_RELAXED);
      __atomic_thread_fence(__ATOMIC_RELEASE);
      __atomic_fetch_add(&term_[mb].done, 1, __ATOMIC_RELAXED);
    } else {
      // the CTA's counts for this epoch's parity: done, sent
      auto c = reinterpret_cast<uint32_t*>(lmem_ + rt_->lm_term) + 2 * (mb + NMB * (epoch_ & 1));
      uint32_t s = __atomic_fetch_add(&c[1], sent_[mb], __ATOMIC_RELAXED);
      uint32_t n = __atomic_fetch_add(&c[0], 1 + after(s), __ATOMIC_RELAXED);
      if (n + 1 == (1u << rt_->lg_block)) {
        // the CTA's last: every other actor's sent is in
        auto vc = reinterpret_cast<volatile uint32_t*>(c);
        uint32_t total = vc[1];
        vc[0] = 0;
        vc[1] = 0;
        __atomic_fetch_add(&term_[mb].sent, total, __ATOMIC_RELAXED);
        __atomic_thread_fence(__ATOMIC_RELEASE);
        __atomic_fetch_add(&term_[mb].done, 1, __ATOMIC_RELAXED);
      }
    }
    // @step end
  }

  // The actor loop. App provides main_step(Actor&), which sends until a send
  // fails or its work is over (and then calls done), process(Actor&, mb, msg,
  // from), and kSendsFrom[NMB], the mailboxes each handler may send on.
  template <typename App>
  void run(App& app) {
    uint32_t idle = 0;
    for (;;) {
      // @step loop
      bool all_ended = true;
      for (uint32_t mb = 0; mb < NMB; ++mb) {
        all_ended = all_ended && ended_[mb];
      }
      // the warp (or the CTA) leaves together
      if (this->leave(all_ended)) {
        break;
      }
      // the whole warp (or CTA) is converged here
      if (kTerm == FGC_TERM_T0) {
        for (uint32_t mb = 0; mb < NMB; ++mb) {
          if (!pend_[mb] && mark_next_[mb] < rt_->num_actors) {
            this->stage(mb, pmsg_[mb], mark_next_[mb]++, true);
          }
        }
      }
      bool progress = false;
      bool pending = false;
      for (uint32_t mb = 0; mb < NMB; ++mb) {
        progress = this->flush(mb) || progress;
        pending = pending || pend_[mb];
      }
      // @step loop
      if (!all_ended && !pending) {
        app.main_step(*this);
      }
      // @step loop
      for (uint32_t mb = 0; mb < NMB; ++mb) {
        progress = this->receive(app, mb, all_ended) || progress;
      }
      if (all_ended) {
        continue;
      }
      if (progress) {
        idle = 0;
      } else if (++idle >= rt_->term_period) {
        idle = 0;
        // @step term
        for (uint32_t mb = 0; mb < NMB; ++mb) {
          if (!ended_[mb] && done_[mb] && !pend_[mb] && this->ended(mb)) {
            ended_[mb] = true;
            if (mb + 1 < NMB) {
              this->done(mb + 1);
            }
          }
        }
        // @step end
      }
    }
  }

  uint32_t id() const { return me_; }

private:
  void stage(uint32_t mb, const T& msg, uint32_t dst, bool marker) {
    pdst_[mb]    = dst;
    pmsg_[mb]    = msg;
    pmark_[mb]   = marker;
    pend_[mb]    = true;
    claimed_[mb] = false;
  }

  // True when the warp's lanes (or the CTA's threads) have all ended. A CTA
  // vote counts the warps still running in one of three words, so the word a
  // pass resets is one no warp still reads or already adds to.
  bool leave(bool all_ended) {
    bool warp_ended = vx_vote_all(all_ended);
    if constexpr (!kCta) {
      return warp_ended;
    }
    auto v = reinterpret_cast<volatile uint32_t*>(lmem_ + rt_->lm_vote);
    uint32_t p = pass_;
    pass_ = (p == 2) ? 0 : p + 1;
    if (!warp_ended && lane_ == 0) {
      __atomic_fetch_add(&v[p], 1, __ATOMIC_RELAXED);
    }
    __syncthreads();
    bool leave = v[p] == 0;
    if (local_ == 0) {
      v[pass_ == 2 ? 0 : pass_ + 1] = 0;
    }
    return leave;
  }

  // The queue a message for actor dst goes to.
  uint32_t queue_of(uint32_t dst) const {
    switch (kTopo) {
      case FGC_TOPO_SPSC:      return dst * nprod_ + (me_ >> lg_prod_);
      case FGC_TOPO_MPSC_WARP: return dst >> rt_->lg_threads;
      case FGC_TOPO_MPSC_CTA:  return dst >> rt_->lg_block;
      default:                 return dst;
    }
  }

  // Takes n tickets of queue q: an atomic add, or a plain one where the
  // claimer alone owns the tail (SPSC).
  uint32_t take(uint32_t* tail, uint32_t q, uint32_t n) const {
    if (kTopo == FGC_TOPO_SPSC) {
      uint32_t t = tail[q];
      tail[q] = t + n;
      return t;
    }
    return __atomic_fetch_add(&tail[q], n, __ATOMIC_RELAXED);
  }

  // Claims a ticket for every lane whose out-slot of mailbox mb has none, then
  // publishes every claimed message whose slot is free. Warp-converged (and
  // CTA-converged with CTA senders).
  bool flush(uint32_t mb) {
    // @step claim
    bool want = pend_[mb] && !claimed_[mb];
    uint32_t q = this->queue_of(pdst_[mb]);
    auto tail = reinterpret_cast<uint32_t*>(this->args(mb).tail);
    if (kSender == FGC_SENDER_CTA) {
      this->cta_claim(mb, want, q, tail);
    } else if (kSender == FGC_SENDER_WARP) {
      size_t todo = vx_vote_ballot(want);
      while (todo != 0) {
        uint32_t leader = __builtin_ctzl(todo);
        if (lane_ == leader) {
          xchg_[0] = q;
        }
        uint32_t key = xchg_[0];
        bool mine = want && q == key;
        size_t group = vx_vote_ballot(mine);
        if (lane_ == leader) {
          xchg_[1] = this->take(tail, key, (uint32_t)__builtin_popcountl(group));
        }
        uint32_t base = xchg_[1];
        if (mine) {
          pticket_[mb] = base + (uint32_t)__builtin_popcountl(group & (((size_t)1 << lane_) - 1));
          claimed_[mb] = true;
          want = false;
        }
        todo &= ~group;
      }
    } else if (want) {
      pticket_[mb] = this->take(tail, q, 1);
      claimed_[mb] = true;
    }
    pq_[mb] = q;
    // @step end
    return pend_[mb] && claimed_[mb] && this->publish(mb);
  }

  // CTA claims: lanes insert their queue into a hash table in local memory
  // (key q + 1, 0 empty) and take a rank in its group; after a barrier each
  // thread claims the tickets of its table entries and clears them; after a
  // second barrier each lane reads its group's base.
  void cta_claim(uint32_t mb, bool want, uint32_t q, uint32_t* tail) {
    const uint32_t lg_h = rt_->lg_block + 1;
    const uint32_t H = 1u << lg_h;
    auto key  = reinterpret_cast<uint32_t*>(lmem_ + rt_->lm_claim);
    auto cnt  = key + H;
    auto base = cnt + H;
    uint32_t e = 0, rank = 0;
    if (want) {
      e = (q * 0x9E3779B1u) >> (32 - lg_h);
      for (;;) {
        uint32_t expect = 0;
        if (__atomic_compare_exchange_n(&key[e], &expect, q + 1, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)
         || expect == q + 1) {
          break;
        }
        e = (e + 1) & (H - 1);
      }
      rank = __atomic_fetch_add(&cnt[e], 1, __ATOMIC_RELAXED);
    }
    __syncthreads();
    for (uint32_t i = local_; i < H; i += 1u << rt_->lg_block) {
      uint32_t k = key[i];
      if (k != 0) {
        base[i] = this->take(tail, k - 1, cnt[i]);
        key[i]  = 0;
        cnt[i]  = 0;
      }
    }
    __syncthreads();
    if (want) {
      pticket_[mb] = base[e] + rank;
      claimed_[mb] = true;
    }
  }

  // Writes the claimed message of mailbox mb if its slot is free.
  bool publish(uint32_t mb) {
    // @step publish
    uint32_t lg  = this->args(mb).lgcap;
    auto ring    = reinterpret_cast<Slot<T>*>(this->args(mb).slots) + ((uint64_t)pq_[mb] << lg);
    auto slot    = ring + (pticket_[mb] & ((1u << lg) - 1));
    uint32_t lap = (pticket_[mb] >> lg) << 1;
    if (l2_load32(&slot->seq) != lap) {
      return false;
    }
    slot->from = pmark_[mb] ? (me_ | kMarker) : me_;
    if constexpr (kGroup) {
      slot->to = pdst_[mb];
    }
    slot->msg = pmsg_[mb];
    __atomic_thread_fence(__ATOMIC_RELEASE);
    *reinterpret_cast<volatile uint32_t*>(&slot->seq) = lap + 1;
    pend_[mb] = false;
    // @step end
    return true;
  }

  // A message is taken only while the out-slots of the mailboxes its handler
  // sends on (App::kSendsFrom[mb], a bit per mailbox) are free. A handler that
  // sends nothing is never held back, which keeps a request-reply chain free
  // of deadlock.
  template <typename App>
  bool blocked(uint32_t mb) const {
    uint32_t pending = 0;
    for (uint32_t m = 0; m < NMB; ++m) {
      pending |= (uint32_t)pend_[m] << m;
    }
    return (App::kSendsFrom[mb] & pending) != 0;
  }

  template <typename App>
  void deliver(App& app, uint32_t mb, const T& msg, uint32_t from) {
    if (from & kMarker) {
      ++marks_[mb];
      return;
    }
    ++unpub_[mb];
    // @step process
    app.process(*this, mb, msg, from);
    // @step recv
  }

  template <typename App>
  bool receive(App& app, uint32_t mb, bool all_ended) {
    if constexpr (kGroup) {
      return this->group_receive(app, mb);
    }
    if (all_ended) {
      return false;
    }
    if (kTopo == FGC_TOPO_SPSC) {
      return this->spsc_receive(app, mb);
    }
    return this->owner_receive(app, mb);
  }

  // Takes up to R messages off the ring at head, while not held back.
  template <typename App>
  uint32_t drain(App& app, uint32_t mb, Slot<T>* ring, uint32_t& head, uint32_t k) {
    uint32_t lg   = this->args(mb).lgcap;
    uint32_t mask = (1u << lg) - 1;
    while (k < rt_->recv_batch && !this->blocked<App>(mb)) {
      auto slot    = ring + (head & mask);
      uint32_t lap = (head >> lg) << 1;
      uint64_t hdr = l2_load64(slot);
      if ((uint32_t)hdr != lap + 1) {
        break;
      }
      T msg = l2_load(&slot->msg);
      *reinterpret_cast<volatile uint32_t*>(&slot->seq) = lap + 2;
      ++head;
      ++k;
      this->deliver(app, mb, msg, (uint32_t)(hdr >> 32));
    }
    return k;
  }

  // MPSC-actor: my own queue.
  template <typename App>
  bool owner_receive(App& app, uint32_t mb) {
    // @step recv
    auto ring = reinterpret_cast<Slot<T>*>(this->args(mb).slots) + ((uint64_t)me_ << this->args(mb).lgcap);
    uint32_t k = this->drain(app, mb, ring, head_[mb + NMB * set_], 0);
    // @step end
    return k != 0;
  }

  // SPSC: my queue from every producer, round robin from where I stopped.
  template <typename App>
  bool spsc_receive(App& app, uint32_t mb) {
    // @step recv
    uint32_t lg = this->args(mb).lgcap;
    uint64_t q0 = (uint64_t)me_ * nprod_;
    auto heads  = reinterpret_cast<uint32_t*>(this->args(mb).heads) + q0;
    auto slots  = reinterpret_cast<Slot<T>*>(this->args(mb).slots);
    uint32_t k  = 0;
    for (uint32_t i = 0; i < nprod_ && k < rt_->recv_batch && !this->blocked<App>(mb); ++i) {
      uint32_t p = rr_[mb];
      rr_[mb] = (p + 1 == nprod_) ? 0 : p + 1;
      uint32_t head = heads[p];
      k = this->drain(app, mb, slots + ((q0 + p) << lg), head, k);
      if (head != heads[p]) {
        heads[p] = head;
      }
    }
    // @step end
    return k != 0;
  }

  // A receiver group's staging in local memory: control words (1: messages
  // of the current batch not yet processed, 2: the batch size), each
  // member's list head (staged index + 1, 0 empty), the staged messages.
  volatile uint32_t* group_ctl(uint32_t mb) const {
    return reinterpret_cast<volatile uint32_t*>(lmem_ + rt_->lm_stage[mb]
                                                + (local_ >> lg_group_) * rt_->lm_gbytes);
  }

  void group_sync() const {
    if constexpr (kTopo == FGC_TOPO_MPSC_CTA) {
      __syncthreads();
    } else {
      __asm__ volatile ("" ::: "memory");
    }
  }

  // MPSC-warp and MPSC-CTA: once the group's batch is processed, the first
  // min(G, FGC_STAGE) members each take one slot of the group's queue; the
  // published prefix is the new batch, each message pushed on its owner's
  // list. Then every member processes up to R messages of its list.
  template <typename App>
  bool group_receive(App& app, uint32_t mb) {
    // @step recv
    const uint32_t G = 1u << lg_group_;
    const uint32_t S = G < FGC_STAGE ? G : FGC_STAGE;
    const uint32_t m = local_ & (G - 1);
    auto ctl   = this->group_ctl(mb);
    auto list  = ctl + 4;
    auto stage = reinterpret_cast<volatile Staged<T>*>(list + G);
    bool took  = false;
    // every member reads the count before any member's processing lowers it
    this->group_sync();
    uint32_t rem = ctl[1];
    this->group_sync();
    if (rem == 0) {
      uint32_t lg   = this->args(mb).lgcap;
      auto ring     = reinterpret_cast<Slot<T>*>(this->args(mb).slots) + ((uint64_t)(me_ >> lg_group_) << lg);
      uint32_t& head = head_[mb + NMB * set_];
      if (m == 0) {
        ctl[2] = S;
      }
      this->group_sync();
      // the first S members each read one slot
      auto slot    = ring + ((head + m) & ((1u << lg) - 1));
      uint32_t lap = ((head + m) >> lg) << 1;
      uint64_t hdr = 0;
      uint32_t r   = 0;
      if (m < S) {
        hdr = l2_load64(slot);
        if ((uint32_t)hdr != lap + 1) {
          r = __atomic_fetch_min(const_cast<uint32_t*>(&ctl[2]), m, __ATOMIC_RELAXED);
        }
      }
      this->group_sync();
      uint32_t c = ctl[2 + after(r)];
      if (m < c) {
        T msg = l2_load(&slot->msg);
        uint32_t to = l2_load32(&slot->to);
        *reinterpret_cast<volatile uint32_t*>(&slot->seq) = lap + 2;
        stage[m].from = (uint32_t)(hdr >> 32);
        const_cast<Staged<T>&>(stage[m]).msg = msg;
        stage[m].next = __atomic_exchange_n(const_cast<uint32_t*>(&list[to & (G - 1)]), m + 1, __ATOMIC_RELAXED);
      }
      head += c;
      if (m == 0) {
        ctl[1] = c;
      }
      took = c != 0;
      this->group_sync();
    }
    uint32_t k = 0;
    while (k < rt_->recv_batch && !this->blocked<App>(mb)) {
      uint32_t i = list[m];
      if (i == 0) {
        break;
      }
      list[m] = stage[i - 1].next;
      T msg = const_cast<const Staged<T>&>(stage[i - 1]).msg;
      uint32_t from = stage[i - 1].from;
      ++k;
      this->deliver(app, mb, msg, from);
    }
    if (k != 0) {
      __atomic_fetch_sub(const_cast<uint32_t*>(&ctl[1]), k, __ATOMIC_RELAXED);
    }
    // @step end
    return took || k != 0;
  }

  const fgc_mb_args_t& args(uint32_t mb) const {
    return rt_->mb[mb + NMB * set_];
  }

  void reset() {
    for (uint32_t mb = 0; mb < NMB; ++mb) {
      sent_[mb]      = 0;
      unpub_[mb]     = 0;
      marks_[mb]     = 0;
      mark_next_[mb] = rt_->num_actors;
      done_[mb]      = false;
      ended_[mb]     = false;
    }
  }

  // Mailbox end. T0: a marker from every actor, mine all sent, nothing of mine
  // staged. T1/T2: every actor (CTA) done and every sent message processed.
  bool ended(uint32_t mb) {
    if (kTerm == FGC_TERM_T0) {
      return marks_[mb] == rt_->num_actors && mark_next_[mb] == rt_->num_actors
          && (!kGroup || this->group_ctl(mb)[4 + (local_ & ((1u << lg_group_) - 1))] == 0);
    }
    if (unpub_[mb] != 0) {
      __atomic_fetch_add(&term_[mb].recvd, unpub_[mb], __ATOMIC_RELAXED);
      unpub_[mb] = 0;
    }
    uint32_t target = kTerm == FGC_TERM_T2 ? rt_->num_actors >> rt_->lg_block : rt_->num_actors;
    if (l2_load32(&term_[mb].done) != target) {
      return false;
    }
    return l2_load32(&term_[mb].recvd) == l2_load32(&term_[mb].sent);
  }

  const fgc_rt_args_t* rt_;
  fgc_term_t* term_;
  uint32_t me_;
  uint32_t local_;     // my thread in the CTA
  uint32_t lane_;      // my lane in the warp
  uint32_t set_;
  uint32_t epoch_;
  uint32_t pass_;      // the CTA vote's word
  uint32_t lg_prod_;   // log2 actors per producer (SPSC)
  uint32_t lg_group_;  // log2 actors per receiver group
  uint32_t nprod_;     // producers (SPSC)
  uint8_t* lmem_;
  volatile uint32_t* xchg_;  // the warp's word pair in local memory
  uint32_t head_[2 * NMB];
  uint32_t rr_[NMB];
  uint32_t sent_[NMB];
  uint32_t unpub_[NMB];
  uint32_t marks_[NMB];
  uint32_t mark_next_[NMB];
  uint32_t pticket_[NMB];
  uint32_t pdst_[NMB];
  uint32_t pq_[NMB];
  T        pmsg_[NMB];
  bool     pmark_[NMB];
  bool     pend_[NMB];
  bool     claimed_[NMB];
  bool     done_[NMB];
  bool     ended_[NMB];
};

} // namespace fgc

#endif
