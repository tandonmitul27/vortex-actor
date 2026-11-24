#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

typedef struct {
  uint32_t num_points;
  uint32_t iters;
  uint64_t dst_addr;
} kernel_arg_t;

#endif
