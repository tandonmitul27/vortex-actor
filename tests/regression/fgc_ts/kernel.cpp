#include <vx_spawn2.h>
#include <vx_barrier.h>
#include "../fgc_common/fgc_actor.h"
#include "common.h"

using Rt = fgc::Actor<ts_pkt_t, NUM_MB>;

struct TopoSort {
  static constexpr uint32_t kSendsFrom[NUM_MB] = { 0 };

  uint32_t        N;
  uint32_t        R;
  uint32_t        me;
  const uint32_t* toff;     // my columns' offsets start here
  const uint32_t* trows;
  uint32_t*       rowcnt;   // my rows
  uint32_t*       rowsum;
  uint32_t*       level;
  uint32_t*       match;
  uint32_t*       rowq;     // my ready rows
  uint32_t*       colq;     // my matched columns and their levels
  uint32_t*       colql;
  uint32_t        rownext, rowlast;
  uint32_t        colnext, collast;
  uint32_t        colstart, colend;
  uint32_t        curr_col, col_level;
  uint32_t        handled;  // my rows matched plus my columns walked

  void main_step(Rt& rt) {
    // @step main
    while (rownext < rowlast) {
      uint32_t row = rowq[rownext];
      uint32_t c   = rowsum[row];
      if (!rt.send(MB_TOPO, ts_pkt_t{TS_MATCHED, c, level[row]}, c % N)) {
        return;
      }
      match[row] = c;
      ++handled;
      ++rownext;
    }
    for (;;) {
      if (colstart == colend) {
        if (colnext == collast) {
          break;
        }
        curr_col  = colq[colnext];
        col_level = colql[colnext];
        ++colnext;
        colstart = toff[curr_col];
        colend   = toff[curr_col + 1];
      }
      uint32_t r = trows[colstart];
      if (!rt.send(MB_TOPO, ts_pkt_t{r / N, curr_col * N + me, col_level}, r % N)) {
        return;
      }
      if (++colstart == colend) {
        ++handled;
      }
    }
    if (handled == 2 * R) {
      rt.done(MB_TOPO);
    }
    // @step end
  }

  void process(Rt&, uint32_t, ts_pkt_t pkt, uint32_t) {
    // @step handler
    if (pkt.row & TS_MATCHED) {
      colq[collast]  = pkt.col / N;
      colql[collast] = pkt.level;
      ++collast;
    } else {
      uint32_t row = pkt.row;
      rowsum[row] -= pkt.col;
      rowcnt[row] -= 1;
      if (pkt.level >= level[row]) {
        level[row] = pkt.level + 1;
      }
      if (rowcnt[row] == 1) {
        rowq[rowlast++] = row;
      }
    }
    // @step end
  }
};

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto toff   = reinterpret_cast<const uint32_t*>(arg->toff_addr);
  auto trows  = reinterpret_cast<const uint32_t*>(arg->trows_addr);
  auto rowcnt = reinterpret_cast<uint32_t*>(arg->rowcnt_addr);
  auto rowsum = reinterpret_cast<uint32_t*>(arg->rowsum_addr);
  auto level  = reinterpret_cast<uint32_t*>(arg->level_addr);
  auto match  = reinterpret_cast<uint32_t*>(arg->match_addr);
  uint32_t me = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t N  = arg->num_actors;
  uint32_t R  = arg->rows;
  const uint64_t n = (uint64_t)N * R;
  const uint64_t mine = (uint64_t)me * R;
  uint32_t* rowq = reinterpret_cast<uint32_t*>(arg->work_addr) + mine;
  auto time = reinterpret_cast<uint64_t*>(arg->time_addr) + 2 * blockIdx.x;

  // The core's span: from every warp present to every warp's memory done.
  __syncthreads();
  if (threadIdx.x == 0) {
    time[0] = csr_read(VX_CSR_MCYCLE);
  }

  // @step init
  uint32_t rowlast = 0;
  for (uint32_t y = 0; y < R; ++y) {
    if (rowcnt[mine + y] == 1) {
      rowq[rowlast++] = y;
    }
  }
  // @step end

  if (arg->config == CFG_NON_ACTOR) {
    vortex::gbarrier bar(1, arg->num_cores);
    auto flag = reinterpret_cast<uint32_t*>(arg->flag_addr);
    for (uint32_t round = 0;; ++round) {
      // @step update
      for (uint32_t i = 0; i < rowlast; ++i) {
        uint32_t y   = rowq[i];
        // other cores' atomics change these: read them at the L2
        uint32_t c   = fgc::l2_load32(&rowsum[mine + y]);
        uint32_t lvl = fgc::l2_load32(&level[mine + y]);
        match[mine + y] = c;
        uint64_t cs = (uint64_t)(c % N) * R + c / N;
        for (uint32_t k = toff[cs]; k < toff[cs + 1]; ++k) {
          uint32_t r  = trows[k];
          uint64_t rs = (uint64_t)(r % N) * R + r / N;
          __atomic_fetch_sub(&rowsum[rs], c, __ATOMIC_RELAXED);
          __atomic_fetch_sub(&rowcnt[rs], 1, __ATOMIC_RELAXED);
          __atomic_fetch_max(&level[rs], lvl + 1, __ATOMIC_RELAXED);
        }
      }
      if (rowlast != 0) {
        __atomic_fetch_or(&flag[round], 1, __ATOMIC_RELAXED);
      }
      // @step sync
      vx_fence();
      bar.arrive_and_wait();
      vx_fence();
      // @step end
      if (fgc::l2_load32(&flag[round]) == 0) {
        break;
      }
      // @step scan
      rowlast = 0;
      for (uint32_t y = 0; y < R; ++y) {
        if (fgc::l2_load32(&rowcnt[mine + y]) == 1) {
          rowq[rowlast++] = y;
        }
      }
      // @step end
    }
  } else {
    Rt rt(&arg->rt, me);
    TopoSort app{N, R, me, toff + mine, trows, rowcnt + mine, rowsum + mine, level + mine, match + mine,
                 rowq, rowq + n, rowq + 2 * n, 0, rowlast, 0, 0, 0, 0, 0, 0, 0};
    rt.run(app);
  }

  vx_fence();
  __syncthreads();
  if (threadIdx.x == 0) {
    time[1] = csr_read(VX_CSR_MCYCLE);
  }
}
