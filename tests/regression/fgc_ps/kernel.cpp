#include <vx_spawn2.h>
#include <vx_barrier.h>
#include "../fgc_common/fgc_actor.h"
#include "common.h"

using Rt = fgc::Actor<ps_pkt_t, NUM_MB>;

struct PageRankPush {
  static constexpr uint32_t kSendsFrom[NUM_MB] = { 0 };

  uint32_t        N;
  uint32_t        R;
  const uint32_t* offsets;  // my rows' offsets start here
  const uint32_t* cols;
  const uint32_t* rank;     // my rows
  uint32_t*       acc;      // my rows
  uint32_t        y;        // next local row to push from
  uint32_t        k;        // next nonzero of row y

  void main_step(Rt& rt) {
    // @step main
    while (y < R) {
      // an empty row sends nothing (and RISC-V division by zero does not trap)
      uint32_t contrib = rank[y] / (offsets[y + 1] - offsets[y]);
      while (k < offsets[y + 1]) {
        uint32_t c = cols[k];
        if (!rt.send(MB_PUSH, ps_pkt_t{c / N, contrib}, c % N)) {
          return;
        }
        ++k;
      }
      ++y;
    }
    rt.done(MB_PUSH);
    // @step end
  }

  void process(Rt&, uint32_t, ps_pkt_t pkt, uint32_t) {
    // @step handler
    acc[pkt.dst] += pkt.contrib;
    // @step end
  }
};

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto offsets = reinterpret_cast<const uint32_t*>(arg->offsets_addr);
  auto cols    = reinterpret_cast<const uint32_t*>(arg->cols_addr);
  auto rank    = reinterpret_cast<uint32_t*>(arg->rank_addr);
  auto acc     = reinterpret_cast<uint32_t*>(arg->acc_addr);
  uint32_t me  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t N   = arg->num_actors;
  uint32_t R   = arg->rows;
  auto time = reinterpret_cast<uint64_t*>(arg->time_addr) + 2 * blockIdx.x;

  // The core's span: from every warp present to every warp's memory done.
  __syncthreads();
  if (threadIdx.x == 0) {
    time[0] = csr_read(VX_CSR_MCYCLE);
  }

  const uint32_t* my_off = offsets + (uint64_t)me * R;
  uint32_t* my_rank = rank + (uint64_t)me * R;
  uint32_t* my_acc  = acc + (uint64_t)me * R;
  if (arg->config == CFG_NON_ACTOR) {
    vortex::gbarrier bar(1, arg->num_cores);
    for (uint32_t it = 0; it < arg->iters; ++it) {
      // @step push
      for (uint32_t y = 0; y < R; ++y) {
        uint32_t contrib = my_rank[y] / (my_off[y + 1] - my_off[y]);
        for (uint32_t k = my_off[y]; k < my_off[y + 1]; ++k) {
          uint32_t c = cols[k];
          __atomic_fetch_add(&acc[(c % N) * R + c / N], contrib, __ATOMIC_RELAXED);
        }
      }
      // @step sync
      vx_fence();
      bar.arrive_and_wait();
      vx_fence();
      // @step update
      for (uint32_t y = 0; y < R; ++y) {
        // other cores' atomics filled acc: read it at the L2
        my_rank[y] = ps_update(fgc::l2_load32(&my_acc[y]));
        my_acc[y]  = 0;
      }
      // @step sync
      vx_fence();
      bar.arrive_and_wait();
      // @step end
    }
  } else {
    Rt rt(&arg->rt, me);
    for (uint32_t it = 0; it < arg->iters; ++it) {
      if (it != 0) {
        rt.next_epoch();
      }
      PageRankPush app{N, R, my_off, cols, my_rank, my_acc, 0, my_off[0]};
      rt.run(app);
      // @step update
      for (uint32_t y = 0; y < R; ++y) {
        my_rank[y] = ps_update(my_acc[y]);
        my_acc[y]  = 0;
      }
      // @step end
    }
  }

  vx_fence();
  __syncthreads();
  if (threadIdx.x == 0) {
    time[1] = csr_read(VX_CSR_MCYCLE);
  }
}
