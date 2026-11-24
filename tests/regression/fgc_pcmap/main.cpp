// fgc_pcmap: a kernel with step and event tags and known counts, for
// tools/fgc_map.py. Expected per warp: iters / 4 runs of event every4.
#include <vortex2.h>
#include "common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <vector>

#define CHECK(expr) do { \
    vx_result_t _r = (expr); \
    if (_r != VX_SUCCESS) { \
        std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n", \
                     __FILE__, __LINE__, #expr, vx_result_string(_r)); \
        std::exit(1); \
    } \
} while (0)

namespace {
const char* kernel_file = "kernel.vxbin";
uint32_t    size        = 256;
uint32_t    iters       = 100;

void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "n:i:k:h")) != -1) {
        switch (c) {
            case 'n': size        = std::atoi(optarg); break;
            case 'i': iters       = std::atoi(optarg); break;
            case 'k': kernel_file = optarg;            break;
            default:
                std::cout << "Usage: [-k kernel] [-n points] [-i iters] [-h]" << std::endl;
                std::exit(c == 'h' ? 0 : -1);
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    parse_args(argc, argv);
    const uint32_t num_points = size;
    const uint64_t buf_size   = num_points * sizeof(uint32_t);

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));
    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, &qi, &q));

    vx_buffer_h dst_buf = nullptr;
    CHECK(vx_buffer_create(dev, buf_size, VX_MEM_WRITE, &dst_buf));
    vx_module_h mod = nullptr;
    vx_kernel_h kern = nullptr;
    CHECK(vx_module_load_file(dev, kernel_file, &mod));
    CHECK(vx_module_get_kernel(mod, "main", &kern));

    kernel_arg_t kernel_arg{};
    kernel_arg.num_points = num_points;
    kernel_arg.iters      = iters;
    CHECK(vx_buffer_address(dst_buf, &kernel_arg.dst_addr));

    uint32_t grid[1], block[1];
    CHECK(vx_device_max_occupancy_grid(dev, 1, &num_points, grid, block));
    uint64_t num_threads = 0;
    CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &num_threads));
    uint32_t warps = (num_points + num_threads - 1) / num_threads;
    uint32_t runs  = (iters + 3) / 4;
    std::cout << "fgc_pcmap: n=" << num_points << " iters=" << iters
              << " grid=" << grid[0] << " block=" << block[0]
              << " expect every4: " << (uint64_t)warps * runs << " warp, "
              << (uint64_t)num_points * runs << " lanes" << std::endl;

    vx_launch_info_t li{};
    li.struct_size = sizeof(li);
    li.kernel      = kern;
    li.args_host   = &kernel_arg;
    li.args_size   = sizeof(kernel_arg);
    li.ndim        = 1;
    li.grid_dim[0] = grid[0];
    li.block_dim[0]= block[0];

    std::vector<uint32_t> h_dst(num_points);
    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
    CHECK(vx_enqueue_read(q, h_dst.data(), dst_buf, 0, buf_size, 1, &launch_ev, &read_ev));
    CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));

    int errors = 0;
    for (uint32_t idx = 0; idx < num_points; ++idx) {
        uint32_t acc = idx;
        for (uint32_t i = 0; i < iters; ++i) {
            acc = acc * 3 + i;
            acc ^= i << 1;
        }
        if (h_dst[idx] != acc) {
            if (errors < 16)
                std::printf("*** [%u] expected=%u actual=%u\n", idx, acc, h_dst[idx]);
            ++errors;
        }
    }

    vx_event_release(read_ev);
    vx_event_release(launch_ev);
    vx_buffer_release(dst_buf);
    vx_kernel_release(kern);
    vx_module_release(mod);
    vx_queue_release(q);
    vx_device_dump_perf(dev, stdout);
    vx_device_release(dev);

    if (errors) {
        std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
        return 1;
    }
    std::cout << "PASSED!" << std::endl;
    return 0;
}
