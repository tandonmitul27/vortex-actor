#include <vx_spawn2.h>
#include "../fgc_common/fgc_actor.h"
#include "common.h"

using Rt = fgc::Actor<ig_pkt_t, NUM_MB>;

struct IndexGather {
  static constexpr uint32_t kSendsFrom[NUM_MB] = { 1u << MB_RESPONSE, 0 };

  uint32_t        lg;
  uint32_t        range;
  uint32_t        num_msgs;
  uint32_t        next;
  const uint32_t* mine;   // my slice of the table
  uint32_t*       tgt;    // my results

  void main_step(Rt& rt) {
    // @step main
    while (next < num_msgs) {
      uint32_t g = hash_key(rt.id(), next, range);
      if (!rt.send(MB_REQUEST, ig_pkt_t{next, g & ((1u << lg) - 1)}, g >> lg)) {
        return;
      }
      ++next;
    }
    rt.done(MB_REQUEST);
    // @step end
  }

  void process(Rt& rt, uint32_t mb, ig_pkt_t pkt, uint32_t from) {
    // @step handler
    if (mb == MB_REQUEST) {
      pkt.val = mine[pkt.val];
      rt.send(MB_RESPONSE, pkt, from);
    } else {
      tgt[pkt.idx] = pkt.val;
    }
    // @step end
  }
};

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto table = reinterpret_cast<const uint32_t*>(arg->table_addr);
  auto tgt   = reinterpret_cast<uint32_t*>(arg->tgt_addr);
  uint32_t me    = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t lg    = arg->log_entries;
  uint32_t range = arg->num_actors << lg;
  uint32_t M     = arg->num_msgs;
  auto time = reinterpret_cast<uint64_t*>(arg->time_addr) + 2 * blockIdx.x;

  // The core's span: from every warp present to every warp's memory done.
  __syncthreads();
  if (threadIdx.x == 0) {
    time[0] = csr_read(VX_CSR_MCYCLE);
  }

  if (arg->config == CFG_NON_ACTOR) {
    // @step gather
    uint32_t* out = tgt + (uint64_t)me * M;
    for (uint32_t i = 0; i < M; ++i) {
      out[i] = table[hash_key(me, i, range)];
    }
    // @step end
  } else {
    Rt rt(&arg->rt, me);
    IndexGather app{lg, range, M, 0, table + ((uint64_t)me << lg), tgt + (uint64_t)me * M};
    rt.run(app);
  }

  vx_fence();
  __syncthreads();
  if (threadIdx.x == 0) {
    time[1] = csr_read(VX_CSR_MCYCLE);
  }
}
