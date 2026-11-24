#include <vx_spawn2.h>
#include "common.h"

// Known answers for the fgc memory counters: one coalesced read of src, one
// coalesced write of dst and one atomic add on ctr[0] per thread.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto src = reinterpret_cast<uint32_t*>(arg->src_addr);
  auto dst = reinterpret_cast<uint32_t*>(arg->dst_addr);
  auto ctr = reinterpret_cast<uint32_t*>(arg->ctr_addr);
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= arg->num_points) {
    return;
  }
  dst[idx] = src[idx] + 1;
  __atomic_fetch_add(ctr, 1, __ATOMIC_RELAXED);
}
