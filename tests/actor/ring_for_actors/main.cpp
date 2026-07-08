// ring_for_actors host: run the kernel, check N produced == N consumed, 0 errors.

#include <iostream>
#include <vector>
#include <cstdlib>
#include <vortex.h>
#include "common.h"

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 256;

    vx_device_h dev;
    if (vx_dev_open(&dev) != 0) { std::cerr << "vx_dev_open failed\n"; return 1; }

    vx_buffer_h ring_buf, status_buf, kernel_buf, args_buf;
    vx_mem_alloc(dev, sizeof(ring_t), VX_MEM_READ_WRITE, &ring_buf);
    vx_mem_alloc(dev, 3 * sizeof(int), VX_MEM_READ_WRITE, &status_buf);

    args_t args;
    vx_mem_address(ring_buf,   &args.ring_addr);
    vx_mem_address(status_buf, &args.status_addr);
    args.num_messages = N;

    // zero the buffers (alloc doesn't)
    std::vector<char> zr(sizeof(ring_t), 0);
    vx_copy_to_dev(ring_buf, zr.data(), 0, sizeof(ring_t));
    int zs[3] = {0, 0, 0};
    vx_copy_to_dev(status_buf, zs, 0, sizeof(zs));

    // upload, launch, wait
    vx_upload_kernel_file(dev, "kernel.vxbin", &kernel_buf);
    vx_upload_bytes(dev, &args, sizeof(args), &args_buf);
    vx_start(dev, kernel_buf, args_buf);
    vx_ready_wait(dev, VX_MAX_TIMEOUT);

    int status[3];   // {errors, produced, consumed}
    vx_copy_from_dev(status, status_buf, 0, sizeof(status));

    std::cout << "errors="   << status[0]
              << " produced=" << status[1]
              << " consumed=" << status[2] << "\n";

    bool ok = (status[0] == 0 && status[1] == N && status[2] == N);
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";

    vx_mem_free(ring_buf);
    vx_mem_free(status_buf);
    vx_mem_free(kernel_buf);
    vx_mem_free(args_buf);
    vx_dev_close(dev);
    return ok ? 0 : 1;
}
