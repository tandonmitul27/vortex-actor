// simt_div: per-lane results through a divergent indirect call (mode 0) and
// divergent loop exits (modes 1-4: while condition, done flag, early break,
// the old actor drain loop), uniform indirect calls (mode 6), a divergent jump
// table (mode 8) and a divergent tail call (mode 9).
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
uint32_t    size        = 64;
uint32_t    mode        = 0;

void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "n:m:k:h")) != -1) {
        switch (c) {
            case 'n': size        = std::atoi(optarg); break;
            case 'm': mode        = std::atoi(optarg); break;
            case 'k': kernel_file = optarg;            break;
            default:
                std::cout << "Usage: [-k kernel] [-n points] [-m mode] [-h]" << std::endl;
                std::exit(c == 'h' ? 0 : -1);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    parse_args(argc, argv);
    const uint32_t n = size;

    // Lane i's list has i % 9 nodes, so lanes leave the loop at different times.
    std::vector<uint32_t> h_head(mode == 4 ? n * NUM_MB * NUM_SRC : n), h_next;
    std::vector<int32_t> h_val, expect(n);
    for (uint32_t i = 0; i < n; ++i) {
        if (mode == 8) {
            int x = (int)i;
            int r[8] = { x + 1, x * 3, x - 7, x ^ 5, x << 2, x >> 1, -x, x * x };
            expect[i] = r[i & 7];
            continue;
        }
        if (mode == 0 || mode == 9) {
            int x = (int)i;
            int r[4] = { x + 1000, x - 1000, x * 7, -x };
            expect[i] = r[i & 3];
            continue;
        }
        if (mode == 6) {
            int32_t acc = 0;
            for (uint32_t r = 0; r < REPS; ++r) {
                int x = (int)(i + r);
                int v[4] = { x + 1000, x - 1000, x * 7, -x };
                acc += v[n & 3];
            }
            expect[i] = acc;
            continue;
        }
        if (mode == 4) {
            // inbox (i, mb, src) holds (i + 3 mb + src) % 6 packets
            if (i == 0) h_val.assign((size_t)n * NUM_MB * NUM_SRC * MAX_PKTS, 0);
            for (uint32_t mb = 0; mb < NUM_MB; ++mb)
                for (uint32_t src = 0; src < NUM_SRC; ++src) {
                    uint32_t box = (i * NUM_MB + mb) * NUM_SRC + src;
                    h_head[box] = (i + 3 * mb + src) % 6;
                    for (uint32_t k = 0; k < h_head[box]; ++k) {
                        int32_t v = (int32_t)(i * 31 + src * 7 + k * 5 + 1);
                        h_val[box * MAX_PKTS + k] = v;
                    }
                }
            // replay the per-lane packet order of the kernel
            int32_t a = 0;
            uint32_t pos[NUM_MB][NUM_SRC] = {}, dones = 0;
            while (dones < NUM_MB * NUM_SRC)
                for (uint32_t mb = 0; mb < NUM_MB; ++mb)
                    for (uint32_t src = 0; src < NUM_SRC; ++src) {
                        uint32_t box = (i * NUM_MB + mb) * NUM_SRC + src;
                        uint32_t p = pos[mb][src];
                        if (p <= h_head[box]) {
                            if (p < h_head[box]) {
                                int32_t v = h_val[box * MAX_PKTS + p];
                                a = mb ? (a ^ v) : (a + v);
                            } else {
                                ++dones;
                            }
                            pos[mb][src] = p + 1;
                        }
                    }
            expect[i] = a;
            continue;
        }
        uint32_t len = i % 9;
        h_head[i] = len ? (uint32_t)h_val.size() : LIST_END;
        int32_t sum = 0;
        for (uint32_t k = 0; k < len; ++k) {
            int32_t v = (int32_t)(i * 31 + k * 7 + 1);
            sum += v;
            h_val.push_back(v);
            h_next.push_back(k + 1 < len ? (uint32_t)h_val.size() : LIST_END);
        }
        expect[i] = sum;
    }
    h_val.push_back(0);
    h_next.push_back(LIST_END);

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));
    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, &qi, &q));

    if (h_next.size() < h_val.size())
        h_next.resize(h_val.size(), LIST_END);
    const uint64_t lane_bytes = n * sizeof(uint32_t);
    const uint64_t head_bytes = h_head.size() * sizeof(uint32_t);
    const uint64_t node_bytes = h_val.size() * sizeof(uint32_t);
    vx_buffer_h head_buf = nullptr, next_buf = nullptr, val_buf = nullptr, dst_buf = nullptr;
    CHECK(vx_buffer_create(dev, head_bytes, VX_MEM_READ, &head_buf));
    CHECK(vx_buffer_create(dev, node_bytes, VX_MEM_READ, &next_buf));
    CHECK(vx_buffer_create(dev, node_bytes, VX_MEM_READ, &val_buf));
    CHECK(vx_buffer_create(dev, lane_bytes, VX_MEM_WRITE, &dst_buf));
    vx_module_h mod = nullptr;
    vx_kernel_h kern = nullptr;
    CHECK(vx_module_load_file(dev, kernel_file, &mod));
    CHECK(vx_module_get_kernel(mod, "main", &kern));

    kernel_arg_t kernel_arg{};
    kernel_arg.num_points = n;
    kernel_arg.mode = mode;
    CHECK(vx_buffer_address(head_buf, &kernel_arg.head_addr));
    CHECK(vx_buffer_address(next_buf, &kernel_arg.next_addr));
    CHECK(vx_buffer_address(val_buf, &kernel_arg.val_addr));
    CHECK(vx_buffer_address(dst_buf, &kernel_arg.dst_addr));

    CHECK(vx_enqueue_write(q, head_buf, 0, h_head.data(), head_bytes, 0, nullptr, nullptr));
    CHECK(vx_enqueue_write(q, next_buf, 0, h_next.data(), node_bytes, 0, nullptr, nullptr));
    CHECK(vx_enqueue_write(q, val_buf, 0, h_val.data(), node_bytes, 0, nullptr, nullptr));

    uint32_t grid[1], block[1];
    CHECK(vx_device_max_occupancy_grid(dev, 1, &n, grid, block));
    std::cout << "simt_div: mode=" << mode << " n=" << n << " grid=" << grid[0]
              << " block=" << block[0] << std::endl;

    vx_launch_info_t li{};
    li.struct_size = sizeof(li);
    li.kernel      = kern;
    li.args_host   = &kernel_arg;
    li.args_size   = sizeof(kernel_arg);
    li.ndim        = 1;
    li.grid_dim[0] = grid[0];
    li.block_dim[0]= block[0];

    std::vector<int32_t> h_dst(n);
    vx_event_h launch_ev = nullptr, rd = nullptr;
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
    CHECK(vx_enqueue_read(q, h_dst.data(), dst_buf, 0, lane_bytes, 1, &launch_ev, &rd));
    CHECK(vx_event_wait_value(rd, 1, VX_TIMEOUT_INFINITE));

    int errors = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (h_dst[i] != expect[i]) {
            if (errors < 16)
                std::printf("*** dst[%u] expected=%d actual=%d\n", i, expect[i], h_dst[i]);
            ++errors;
        }
    }

    vx_event_release(rd);
    vx_event_release(launch_ev);
    vx_buffer_release(dst_buf);
    vx_buffer_release(val_buf);
    vx_buffer_release(next_buf);
    vx_buffer_release(head_buf);
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
