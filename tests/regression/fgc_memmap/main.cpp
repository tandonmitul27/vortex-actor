// fgc_memmap: known memory traffic for the fgc counters, by address tag.
#include <vortex2.h>
#include "common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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
uint32_t    size        = 1024;

void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "n:k:h")) != -1) {
        switch (c) {
            case 'n': size        = std::atoi(optarg); break;
            case 'k': kernel_file = optarg;            break;
            default:
                std::cout << "Usage: [-k kernel] [-n points] [-h]" << std::endl;
                std::exit(c == 'h' ? 0 : -1);
        }
    }
}

// Names a device range for the fgc counters (VORTEX_FGC_TAGS).
void fgc_tag(std::ofstream& os, const char* name, uint64_t base, uint64_t size) {
    if (os) {
        os << name << " " << std::hex << base << " " << size << std::dec << "\n";
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

    vx_buffer_h src_buf = nullptr, dst_buf = nullptr, ctr_buf = nullptr;
    CHECK(vx_buffer_create(dev, buf_size, VX_MEM_READ, &src_buf));
    CHECK(vx_buffer_create(dev, buf_size, VX_MEM_WRITE, &dst_buf));
    CHECK(vx_buffer_create(dev, 64, VX_MEM_READ_WRITE, &ctr_buf));
    vx_module_h mod = nullptr;
    vx_kernel_h kern = nullptr;
    CHECK(vx_module_load_file(dev, kernel_file, &mod));
    CHECK(vx_module_get_kernel(mod, "main", &kern));

    kernel_arg_t kernel_arg{};
    kernel_arg.num_points = num_points;
    CHECK(vx_buffer_address(src_buf, &kernel_arg.src_addr));
    CHECK(vx_buffer_address(dst_buf, &kernel_arg.dst_addr));
    CHECK(vx_buffer_address(ctr_buf, &kernel_arg.ctr_addr));

    if (const char* path = std::getenv("VORTEX_FGC_TAGS")) {
        std::ofstream os(path);
        fgc_tag(os, "src", kernel_arg.src_addr, buf_size);
        fgc_tag(os, "dst", kernel_arg.dst_addr, buf_size);
        fgc_tag(os, "ctr", kernel_arg.ctr_addr, 64);
    }

    std::vector<uint32_t> h_src(num_points), h_dst(num_points), h_ctr(16, 0);
    for (uint32_t i = 0; i < num_points; ++i) {
        h_src[i] = i * 7;
    }
    CHECK(vx_enqueue_write(q, src_buf, 0, h_src.data(), buf_size, 0, nullptr, nullptr));
    CHECK(vx_enqueue_write(q, ctr_buf, 0, h_ctr.data(), 64, 0, nullptr, nullptr));

    uint32_t grid[1], block[1];
    CHECK(vx_device_max_occupancy_grid(dev, 1, &num_points, grid, block));
    std::cout << "fgc_memmap: n=" << num_points << " grid=" << grid[0] << " block=" << block[0]
              << " expect: src " << buf_size << " B read, dst " << buf_size << " B written, ctr "
              << num_points << " atomics" << std::endl;

    vx_launch_info_t li{};
    li.struct_size = sizeof(li);
    li.kernel      = kern;
    li.args_host   = &kernel_arg;
    li.args_size   = sizeof(kernel_arg);
    li.ndim        = 1;
    li.grid_dim[0] = grid[0];
    li.block_dim[0]= block[0];

    vx_event_h launch_ev = nullptr, rd1 = nullptr, rd2 = nullptr;
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
    CHECK(vx_enqueue_read(q, h_dst.data(), dst_buf, 0, buf_size, 1, &launch_ev, &rd1));
    CHECK(vx_enqueue_read(q, h_ctr.data(), ctr_buf, 0, 64, 1, &launch_ev, &rd2));
    CHECK(vx_event_wait_value(rd2, 1, VX_TIMEOUT_INFINITE));
    CHECK(vx_event_wait_value(rd1, 1, VX_TIMEOUT_INFINITE));

    int errors = 0;
    for (uint32_t i = 0; i < num_points; ++i) {
        if (h_dst[i] != i * 7 + 1) {
            if (errors < 16)
                std::printf("*** dst[%u] expected=%u actual=%u\n", i, i * 7 + 1, h_dst[i]);
            ++errors;
        }
    }
    if (h_ctr[0] != num_points) {
        std::printf("*** ctr expected=%u actual=%u\n", num_points, h_ctr[0]);
        ++errors;
    }

    vx_event_release(rd2);
    vx_event_release(rd1);
    vx_event_release(launch_ev);
    vx_buffer_release(ctr_buf);
    vx_buffer_release(dst_buf);
    vx_buffer_release(src_buf);
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
