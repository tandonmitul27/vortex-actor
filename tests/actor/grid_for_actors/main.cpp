// grid_for_actors host: launch the ring-topology check; pass iff all N actors
// received from their left neighbor.

#include <iostream>
#include <vector>
#include <vortex.h>
#include "common.h"

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    vx_device_h dev;
    if (vx_dev_open(&dev) != 0) { std::cerr << "vx_dev_open failed\n"; return 1; }

    // one actor per hardware lane: N = cores * warps * threads
    uint64_t cores, warps, threads;
    vx_dev_caps(dev, VX_CAPS_NUM_CORES,   &cores);
    vx_dev_caps(dev, VX_CAPS_NUM_WARPS,   &warps);
    vx_dev_caps(dev, VX_CAPS_NUM_THREADS, &threads);
    uint32_t N = (uint32_t)(cores * warps * threads);
    std::cout << "N=" << N << " actors\n";

    size_t grid_bytes   = (size_t)N * N * sizeof(ring_t);  // N×N channels
    size_t status_bytes = (size_t)N * sizeof(int);         // one verdict per actor

    vx_buffer_h grid_buf, status_buf, kernel_buf, args_buf;
    vx_mem_alloc(dev, grid_bytes,   VX_MEM_READ_WRITE, &grid_buf);
    vx_mem_alloc(dev, status_bytes, VX_MEM_READ_WRITE, &status_buf);

    args_t args;
    vx_mem_address(grid_buf,   &args.grid_addr);
    vx_mem_address(status_buf, &args.status_addr);
    args.N = N;

    // Zero the buffers (alloc does not zero)
    std::vector<char> zg(grid_bytes,   0);  vx_copy_to_dev(grid_buf,   zg.data(), 0, grid_bytes);
    std::vector<char> zs(status_bytes, 0);  vx_copy_to_dev(status_buf, zs.data(), 0, status_bytes);

    // Upload kernel and args, launch
    vx_upload_kernel_file(dev, "kernel.vxbin", &kernel_buf);
    vx_upload_bytes(dev, &args, sizeof(args), &args_buf);
    vx_start(dev, kernel_buf, args_buf);
    vx_ready_wait(dev, VX_MAX_TIMEOUT);

    // Read back per-PE verdicts (1 = received from expected neighbor)
    std::vector<int> status(N);
    vx_copy_from_dev(status.data(), status_buf, 0, status_bytes);

    int ok = 0;
    for (uint32_t i = 0; i < N; i++) if (status[i] == 1) ok++;   // count passing actors

    std::cout << "result: " << ok << "/" << N << " PEs OK\n";
    std::cout << (ok == (int)N ? "PASSED" : "FAILED") << "\n";

    vx_mem_free(grid_buf);   vx_mem_free(status_buf);
    vx_mem_free(kernel_buf); vx_mem_free(args_buf);
    vx_dev_close(dev);
    return (ok == (int)N) ? 0 : 1;
}
