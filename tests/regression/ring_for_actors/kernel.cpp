// SPSC ring smoke test: block 0 produces N packets, block 1 consumes and checks order.

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

void kernel_body(args_t* a) {
    if (threadIdx.x != 0) return;     // lane 0 only

    ring_t* r    = (ring_t*)a->ring_addr;
    int*    st   = (int*)a->status_addr;   // {errors, produced, consumed}
    int     N    = a->num_messages;

    if (blockIdx.x == 0) {
        // producer
        for (int i = 0; i < N; i++) {
            while (r->tail - r->head >= CAP) { }   // wait while full
            r->slots[r->tail % CAP].value = i;
            vx_fence();                            // publish slot before tail
            r->tail = r->tail + 1;
            st[1] = i + 1;
        }
    } else {
        // consumer
        int errors = 0;
        for (int i = 0; i < N; i++) {
            while (r->head == r->tail) { }         // wait while empty
            vx_fence();                            // see payload before reading
            int v = r->slots[r->head % CAP].value;
            r->head = r->head + 1;
            if (v != i) errors++;
            st[2] = i + 1;
        }
        st[0] = errors;
    }
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid  = 2;                  // block 0 = producer, block 1 = consumer
    uint32_t block = vx_num_threads();   // one warp per block
    return vx_spawn_threads(1, &grid, &block, (vx_kernel_func_cb)kernel_body, a);
}
