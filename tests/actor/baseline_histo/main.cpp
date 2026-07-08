// baseline_histo host: launch the kernel, verify by replaying the RNG on the host.

#include <iostream>
#include <vector>
#include <vortex.h>
#include "common.h"

// must match the device-side mix() exactly so the host reference agrees
static inline uint32_t mix(uint32_t a, uint32_t b) {
    uint32_t x = a * 0x9E3779B9u + b * 0x85EBCA77u;
    x ^= (x >> 13);
    x *= 0xC2B2AE3Du;
    x ^= (x >> 16);
    return x;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    vx_device_h dev;
    if (vx_dev_open(&dev) != 0) { std::cerr << "vx_dev_open failed\n"; return 1; }

    // one PE per hardware lane
    uint64_t cores, warps, threads;
    vx_dev_caps(dev, VX_CAPS_NUM_CORES,   &cores);
    vx_dev_caps(dev, VX_CAPS_NUM_WARPS,   &warps);
    vx_dev_caps(dev, VX_CAPS_NUM_THREADS, &threads);
    uint32_t N = (uint32_t)(cores * warps * threads);

    std::cout << "N=" << N << " threads, L=" << L << " buckets/PE, M=" << M
              << " updates/PE (baseline histogram via atomic_fetch_add)\n";

    size_t counts_bytes = (size_t)N * L * sizeof(int);

    vx_buffer_h counts_buf, kernel_buf, args_buf;
    vx_mem_alloc(dev, counts_bytes, VX_MEM_READ_WRITE, &counts_buf);

    args_t args;
    vx_mem_address(counts_buf, &args.counts_addr);
    args.N = N;

    std::vector<char> zc(counts_bytes, 0);
    vx_copy_to_dev(counts_buf, zc.data(), 0, counts_bytes);

    vx_upload_kernel_file(dev, "kernel.vxbin", &kernel_buf);
    vx_upload_bytes(dev, &args, sizeof(args), &args_buf);
    vx_start(dev, kernel_buf, args_buf);
    vx_ready_wait(dev, VX_MAX_TIMEOUT);

    std::vector<int> counts(N * L);
    vx_copy_from_dev(counts.data(), counts_buf, 0, counts_bytes);

    // Expected: replay the kernel's RNG on the host to build the golden histogram
    std::vector<int> expected(N * L, 0);
    uint32_t G_total = L * N;
    for (uint32_t me = 0; me < N; me++) {
        for (uint32_t i = 0; i < M; i++) {
            uint32_t g = mix(me, i) % G_total;
            expected[g]++;
        }
    }

    int mismatches = 0;
    uint64_t total = 0;
    for (uint32_t i = 0; i < N * L; i++) {
        total += counts[i];
        if (counts[i] != expected[i]) {
            if (mismatches < 8) {
                std::cout << "  mismatch bucket " << i
                          << ": got " << counts[i] << ", expected " << expected[i] << "\n";
            }
            mismatches++;
        }
    }

    std::cout << "result: total_updates=" << (uint64_t)N * M
              << " sum_of_counts=" << total
              << " mismatches=" << mismatches << "\n";
    std::cout << (mismatches == 0 ? "PASSED" : "FAILED") << "\n";

    vx_mem_free(counts_buf);
    vx_mem_free(kernel_buf);
    vx_mem_free(args_buf);
    vx_dev_close(dev);
    return (mismatches == 0) ? 0 : 1;
}
