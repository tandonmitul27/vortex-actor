#include <vx_spawn2.h>
#include "../fgc_common/fgc_actor.h"
#include "common.h"

using Rt = fgc::Actor<rp_pkt_t, NUM_MB>;

struct RandPerm {
  static constexpr uint32_t kSendsFrom[NUM_MB] = { 1u << MB_REPLY, 1u << MB_THROW };

  uint32_t  N;
  uint32_t  R;
  uint32_t  me;
  uint32_t  range;   // target slots in all
  uint32_t* target;  // my slots
  uint32_t  next;    // next dart to throw
  uint32_t  draws;   // my random stream's position
  uint32_t  hits;

  void main_step(Rt& rt) {
    // @step main
    while (next < R) {
      uint32_t r = rp_slot(me, draws, range);
      if (!rt.send(MB_THROW, rp_pkt_t{r / N, next * N + me}, r % N)) {
        return;
      }
      ++draws;
      ++next;
    }
    // @step end
  }

  void process(Rt& rt, uint32_t mb, rp_pkt_t pkt, uint32_t from) {
    // @step handler
    if (mb == MB_THROW) {
      if (target[pkt.idx] == RP_EMPTY) {
        target[pkt.idx] = pkt.val;
      } else {
        pkt.val |= RP_REJECTED;
      }
      rt.send(MB_REPLY, pkt, from);
    } else if (pkt.val & RP_REJECTED) {
      uint32_t r = rp_slot(me, draws++, range);
      rt.send(MB_THROW, rp_pkt_t{r / N, pkt.val & ~RP_REJECTED}, r % N);
    } else if (++hits == R) {
      rt.done(MB_THROW);
    }
    // @step end
  }
};

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto target = reinterpret_cast<uint32_t*>(arg->target_addr);
  uint32_t me    = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t N     = arg->num_actors;
  uint32_t R     = arg->darts;
  uint32_t range = 2 * N * R;
  auto time = reinterpret_cast<uint64_t*>(arg->time_addr) + 2 * blockIdx.x;

  // The core's span: from every warp present to every warp's memory done.
  __syncthreads();
  if (threadIdx.x == 0) {
    time[0] = csr_read(VX_CSR_MCYCLE);
  }

  if (arg->config == CFG_NON_ACTOR) {
    // @step throw
    uint32_t draws = 0;
    for (uint32_t i = 0; i < R; ++i) {
      for (;;) {
        uint32_t r = rp_slot(me, draws++, range);
        uint32_t expected = RP_EMPTY;
        if (__atomic_compare_exchange_n(&target[(uint64_t)(r % N) * 2 * R + r / N], &expected,
                                        i * N + me, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
          break;
        }
      }
    }
    // @step end
  } else {
    Rt rt(&arg->rt, me);
    RandPerm app{N, R, me, range, target + (uint64_t)me * 2 * R, 0, 0, 0};
    rt.run(app);
  }

  vx_fence();
  __syncthreads();
  if (threadIdx.x == 0) {
    time[1] = csr_read(VX_CSR_MCYCLE);
  }
}
