#include <vx_spawn2.h>
#include "common.h"

// Divergence checks: an indirect call whose target differs across lanes, and
// a while loop whose exit differs across lanes.
static int op_add(int x) { return x + 1000; }
static int op_sub(int x) { return x - 1000; }
static int op_mul(int x) { return x * 7; }
static int op_neg(int x) { return -x; }

typedef int (*op_fn_t)(int);
static op_fn_t fns[4] = { op_add, op_sub, op_mul, op_neg };

static int32_t handle_add(int32_t acc, int32_t v) { return acc + v; }
static int32_t handle_xor(int32_t acc, int32_t v) { return acc ^ v; }

typedef int32_t (*handler_t)(int32_t, int32_t);
static handler_t handlers[2] = { handle_add, handle_xor };

// a call in tail position: compiled as a jump (jalr x0), not a linking call
__attribute__((noinline)) static int dispatch(uint32_t id, int x) {
  return fns[id](x);
}

// a dense switch: compiled as a jump table (jalr x0)
__attribute__((noinline)) static int pick(uint32_t c, int x) {
  switch (c) {
  case 0: return x + 1;
  case 1: return x * 3;
  case 2: return x - 7;
  case 3: return x ^ 5;
  case 4: return x << 2;
  case 5: return x >> 1;
  case 6: return -x;
  default: return x * x;
  }
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto dst = reinterpret_cast<int32_t*>(arg->dst_addr);
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= arg->num_points) {
    return;
  }
  if (arg->mode == 0) {
    op_fn_t f = fns[idx & 3];
    dst[idx] = f((int)idx);
  } else if (arg->mode == 8) {
    dst[idx] = pick(idx & 7, (int)idx);
  } else if (arg->mode == 9) {
    dst[idx] = dispatch(idx & 3, (int)idx);
  } else if (arg->mode == 6) {
    // cost: REPS calls with one id for the whole launch
    uint32_t id = arg->num_points & 3;
    int32_t acc = 0;
    for (uint32_t r = 0; r < REPS; ++r) {
      acc += fns[id]((int)(idx + r));
    }
    dst[idx] = acc;
  } else if (arg->mode == 1) {
    auto head = reinterpret_cast<const uint32_t*>(arg->head_addr);
    auto next = reinterpret_cast<const uint32_t*>(arg->next_addr);
    auto val  = reinterpret_cast<const int32_t*>(arg->val_addr);
    int32_t acc = 0;
    uint32_t p = head[idx];
    while (p != LIST_END) {
      acc += val[p];
      p = next[p];
    }
    dst[idx] = acc;
  } else if (arg->mode == 4) {
    // The old actor drain loop: a lane polls NUM_SRC inboxes of NUM_MB
    // mailboxes, runs the mailbox's handler through a table on each packet,
    // and leaves once it has seen every sender's done. Inbox lengths differ
    // per lane, so lanes leave at different times.
    auto len = reinterpret_cast<const uint32_t*>(arg->head_addr);
    auto val = reinterpret_cast<const int32_t*>(arg->val_addr);
    uint32_t pos[NUM_MB][NUM_SRC] = {};
    uint32_t dones = 0;
    int32_t acc = 0;
    int can_exit = 0;
    while (!can_exit) {
      for (uint32_t mb = 0; mb < NUM_MB; ++mb) {
        for (uint32_t src = 0; src < NUM_SRC; ++src) {
          uint32_t box = (idx * NUM_MB + mb) * NUM_SRC + src;
          uint32_t p = pos[mb][src];
          if (p <= len[box]) {
            if (p < len[box]) {
              acc = handlers[mb](acc, val[box * MAX_PKTS + p]);
            } else if (++dones == NUM_MB * NUM_SRC) {
              can_exit = 1;
            }
            pos[mb][src] = p + 1;
          }
        }
      }
    }
    dst[idx] = acc;
  } else {
    auto head = reinterpret_cast<const uint32_t*>(arg->head_addr);
    auto next = reinterpret_cast<const uint32_t*>(arg->next_addr);
    auto val  = reinterpret_cast<const int32_t*>(arg->val_addr);
    int32_t acc = 0;
    uint32_t p = head[idx];
    if (arg->mode == 2) {
      // a polling loop: each lane raises its own done flag
      bool done = false;
      while (!done) {
        if (p == LIST_END) {
          done = true;
        } else {
          acc += val[p];
          p = next[p];
        }
      }
    } else {
      // an early break out of an endless loop
      while (true) {
        if (p == LIST_END)
          break;
        acc += val[p];
        p = next[p];
      }
    }
    dst[idx] = acc;
  }
}
