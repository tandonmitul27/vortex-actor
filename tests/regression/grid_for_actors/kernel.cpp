// Ring-topology check: each actor sends to its right neighbor and receives from
// its left, over the N×N channel grid.

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

void kernel_body(args_t* a) {
    ring_t* grid   = (ring_t*)a->grid_addr;
    int*    status = (int*)a->status_addr;
    int     N      = a->N;
    int     me     = blockIdx.x;          // my actor id
    int     right  = (me + 1) % N;        // I send here
    int     left   = (me + N - 1) % N;    // I should receive from here

    // send one packet to my right neighbor (channel me -> right)
    ring_t* out = &grid[me * N + right];
    while (out->tail - out->head >= CAP) { }   // wait for space
    out->slots[out->tail % CAP].sender = me;
    vx_fence();                                // publish slot before tail
    out->tail = out->tail + 1;

    // receive: scan my inbox column (channel src -> me) until a packet arrives
    while (1) {
        for (int src = 0; src < N; src++) {
            ring_t* in = &grid[src * N + me];
            if (in->head != in->tail) {
                vx_fence();                    // see payload before reading
                packet_t pkt = in->slots[in->head % CAP];
                in->head = in->head + 1;
                // pass only if it came from my left neighbor
                status[me] = (pkt.sender == (uint32_t)left && src == left) ? 1 : 0;
                return;
            }
        }
    }
}

int main(void) {
    args_t* a = (args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim  = a->N;     // one block per actor
    uint32_t block_dim = 1;        // one HW lane per actor
    return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, a);
}
