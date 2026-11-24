#include <vx_spawn2.h>
#include <vx_barrier.h>
#include "../fgc_common/fgc_actor.h"
#include "common.h"

using Rt = fgc::Actor<pl_pkt_t, NUM_MB>;

struct PageRankPull {
  static constexpr uint32_t kSendsFrom[NUM_MB] = { 1u << MB_RESPONSE, 0 };

  uint32_t        N;
  uint32_t        R;
  const uint32_t* offsets;  // my rows' offsets start here
  const uint32_t* cols;
  const float*    rank;     // my rows
  float*          acc;      // my rows
  uint32_t        y;        // next local row to request for
  uint32_t        k;        // next nonzero of row y

  void main_step(Rt& rt) {
    // @step main
    while (y < R) {
      while (k < offsets[y + 1]) {
        uint32_t c = cols[k];
        if (!rt.send(MB_REQUEST, pl_pkt_t{c / N, y, 0.0f, 0}, c % N)) {
          return;
        }
        ++k;
      }
      ++y;
    }
    rt.done(MB_REQUEST);
    // @step end
  }

  void process(Rt& rt, uint32_t mb, pl_pkt_t pkt, uint32_t from) {
    // @step handler
    if (mb == MB_REQUEST) {
      pkt.rank = rank[pkt.src];
      pkt.deg  = offsets[pkt.src + 1] - offsets[pkt.src];
      rt.send(MB_RESPONSE, pkt, from);
    } else {
      acc[pkt.dst] += pkt.rank / (float)pkt.deg;
    }
    // @step end
  }
};

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto offsets = reinterpret_cast<const uint32_t*>(arg->offsets_addr);
  auto cols    = reinterpret_cast<const uint32_t*>(arg->cols_addr);
  auto rank    = reinterpret_cast<float*>(arg->rank_addr);
  uint32_t me  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t N   = arg->num_actors;
  uint32_t R   = arg->rows;
  const float base = (1.0f - PL_DAMPING) / (float)(N * R);
  auto time = reinterpret_cast<uint64_t*>(arg->time_addr) + 2 * blockIdx.x;

  // The core's span: from every warp present to every warp's memory done.
  __syncthreads();
  if (threadIdx.x == 0) {
    time[0] = csr_read(VX_CSR_MCYCLE);
  }

  const uint32_t* my_off = offsets + (uint64_t)me * R;
  if (arg->config == CFG_NON_ACTOR) {
    vortex::gbarrier bar(1, arg->num_cores);
    const uint64_t n = (uint64_t)N * R;
    for (uint32_t it = 0; it < arg->iters; ++it) {
      const float* cur = rank + (it & 1) * n;
      float* next      = rank + ((it + 1) & 1) * n;
      // @step gather
      for (uint32_t y = 0; y < R; ++y) {
        float sum = 0.0f;
        for (uint32_t k = my_off[y]; k < my_off[y + 1]; ++k) {
          uint32_t c = cols[k];
          uint32_t s = (c % N) * R + c / N;
          // other cores wrote cur: read it at the L2
          sum += fgc::l2_load(&cur[s]) / (float)(offsets[s + 1] - offsets[s]);
        }
        next[(uint64_t)me * R + y] = base + PL_DAMPING * sum;
      }
      // @step sync
      vx_fence();
      bar.arrive_and_wait();
      vx_fence();
      // @step end
    }
  } else {
    float* my_rank = rank + (uint64_t)me * R;
    float* my_acc  = reinterpret_cast<float*>(arg->acc_addr) + (uint64_t)me * R;
    Rt rt(&arg->rt, me);
    for (uint32_t it = 0; it < arg->iters; ++it) {
      if (it != 0) {
        rt.next_epoch();
      }
      PageRankPull app{N, R, my_off, cols, my_rank, my_acc, 0, my_off[0]};
      rt.run(app);
      // @step update
      for (uint32_t y = 0; y < R; ++y) {
        my_rank[y] = base + PL_DAMPING * my_acc[y];
        my_acc[y]  = 0.0f;
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
