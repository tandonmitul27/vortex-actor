#include <vx_spawn2.h>
#include "common.h"

// Known answers for tools/fgc_map.py: event "every4" runs iters/4 times per
// warp (a store, so the compiler keeps the branch).
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto dst_ptr = reinterpret_cast<uint32_t*>(arg->dst_addr);
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= arg->num_points) {
    return;
  }
  uint32_t acc = idx;
  for (uint32_t i = 0; i < arg->iters; ++i) {
    // @step compute
    acc = acc * 3 + i;
    // @step mix
    acc ^= i << 1;
    if ((i & 3) == 0) {
      dst_ptr[idx] = acc; // @event every4
    }
  }
  // @step end
  dst_ptr[idx] = acc;
}
