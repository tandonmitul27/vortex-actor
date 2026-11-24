#include <vx_spawn2.h>
#include "../fgc_common/fgc_actor.h"
#include "common.h"

using Rt = fgc::Actor<uint32_t, 1>;

struct Histogram {
  static constexpr uint32_t kSendsFrom[1] = { 0 };

  uint32_t  lg;
  uint32_t  range;
  uint32_t  num_msgs;
  uint32_t  next;
  uint32_t* mine;

  void main_step(Rt& rt) {
    // @step main
    while (next < num_msgs) {
      uint32_t g = hash_key(rt.id(), next, range);
      if (!rt.send(0, g & ((1u << lg) - 1), g >> lg)) {
        return;
      }
      ++next;
    }
    rt.done(0);
    // @step end
  }

  void process(Rt&, uint32_t, uint32_t idx, uint32_t) {
    // @step handler
    mine[idx] += 1;
    // @step end
  }
};

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto counts = reinterpret_cast<uint32_t*>(arg->counts_addr);
  uint32_t me    = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t lg    = arg->log_buckets;
  uint32_t range = arg->num_actors << lg;
  auto time = reinterpret_cast<uint64_t*>(arg->time_addr) + 2 * blockIdx.x;

  // The core's span: from every warp present to every warp's memory done.
  __syncthreads();
  if (threadIdx.x == 0) {
    time[0] = csr_read(VX_CSR_MCYCLE);
  }

  if (arg->config == CFG_NON_ACTOR) {
    // @step update
    for (uint32_t i = 0; i < arg->num_msgs; ++i) {
      __atomic_fetch_add(&counts[hash_key(me, i, range)], 1, __ATOMIC_RELAXED);
    }
    // @step end
  } else {
    Rt rt(&arg->rt, me);
    Histogram app{lg, range, arg->num_msgs, 0, counts + ((uint64_t)me << lg)};
    rt.run(app);
  }

  vx_fence();
  __syncthreads();
  if (threadIdx.x == 0) {
    time[1] = csr_read(VX_CSR_MCYCLE);
  }
}
