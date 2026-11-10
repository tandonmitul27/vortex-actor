// Scratchpad budget check (ablation hardening, 2026-11-03). vx_spawn gives every group resident on a core its own
// __local_mem(size) slice, so a core needs (groups per core) x (bytes per group) of local memory. D off puts
// N / 16 = 64 one-warp groups on 16 cores (4 per core); a size chosen for one group per core ran past the scratchpad
// into global memory without any error (toposort with B off: wrong output). This turns that into a build failure.
#ifndef LMEM_CHECK_H
#define LMEM_CHECK_H
#include <VX_config.h>
#define LMEM_CHECK(BYTES, GROUPS) static_assert((unsigned long long)(BYTES) * (((GROUPS) + (NUM_CORES) - 1) / (NUM_CORES)) \
    <= (1ull << (LMEM_LOG_SIZE)), "scratchpad: groups per core x bytes per group exceed the core's local memory")
#endif
