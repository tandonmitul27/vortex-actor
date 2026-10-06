// baseline_histo host: launch the kernel, verify by replaying the RNG on the host.

#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <vortex.h>
#include "common.h"
#ifndef SPLIT
#define SPLIT 1
#endif

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
    // Optional -n N: override the actor count (defaults to total HW lanes).
    // Without this the N axis of any sweep is silently void: the host prints and
    // uses the derived lane count whatever -n says.
    int N_arg = 0;
    for (int i = 1; i + 1 < argc; i++)
        if (std::string(argv[i]) == "-n") N_arg = atoi(argv[i + 1]);
    uint32_t hw_lanes = (uint32_t)(cores * warps * threads);
    uint32_t N = (N_arg > 0) ? (uint32_t)N_arg : hw_lanes / (uint32_t)(SPLIT);   // SPLIT, log T41

    std::cout << "N=" << N << " threads, L=" << L << " buckets/PE, M=" << M
              << " updates/PE (baseline histogram via atomic_fetch_add)\n";

    size_t counts_bytes = (size_t)N * L * sizeof(int);
    size_t phase_bytes  = (size_t)N * sizeof(phase_cycles_t);

    vx_buffer_h counts_buf, phase_buf, kernel_buf, args_buf;
    vx_mem_alloc(dev, counts_bytes, VX_MEM_READ_WRITE, &counts_buf);
    vx_mem_alloc(dev, phase_bytes,  VX_MEM_READ_WRITE, &phase_buf);

    args_t args;
    vx_mem_address(counts_buf, &args.counts_addr);
    vx_mem_address(phase_buf,  &args.phase_addr);
    args.N = N;

    std::vector<char> zc(counts_bytes, 0);
    vx_copy_to_dev(counts_buf, zc.data(), 0, counts_bytes);
    std::vector<char> zp(phase_bytes, 0);
    vx_copy_to_dev(phase_buf, zp.data(), 0, phase_bytes);

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

    {   // per-PE cycle breakdown: atomic applies vs index compute + loop
        std::vector<phase_cycles_t> ph(N);
        vx_copy_from_dev(ph.data(), phase_buf, 0, phase_bytes);
        uint64_t s_apply = 0, s_drain = 0, s_total = 0, mx_drain = 0, mx_total = 0;
        for (uint32_t i = 0; i < N; i++) {
            s_apply += ph[i].apply;
            s_drain += ph[i].drain;  mx_drain = std::max(mx_drain, ph[i].drain);
            s_total += ph[i].total;  mx_total = std::max(mx_total, ph[i].total);
        }
        std::cout << "PHASES (cycles per PE — mean, max across " << N << " PEs):\n";
        std::cout << "  apply:   mean=" << s_apply / N << "  (atomic-add issues)\n";
        std::cout << "  drain:   mean=" << s_drain / N << "  max=" << mx_drain
                  << "  (fence: contended backlog)\n";
        std::cout << "  compute: mean=" << (s_total - s_apply - s_drain) / N
                  << "  (index hash + loop overhead)\n";
        { uint64_t s0=~0ull, e1=0; for (uint32_t i = 0; i < N; i++) { s0 = std::min(s0, ph[i].t_start); e1 = std::max(e1, ph[i].t_end); }
          std::cout << "span (absolute core cycles): first start=" << s0 << " last end=" << e1 << "\n"; }
        std::cout << "BREAKDOWN apply=" << s_apply / N
                  << " drain=" << s_drain / N
                  << " compute=" << (s_total - s_apply - s_drain) / N
                  << " span=" << s_total / N << " maxspan=" << mx_total << "\n";
    }

    vx_mem_free(phase_buf);
    vx_mem_free(counts_buf);
    vx_mem_free(kernel_buf);
    vx_mem_free(args_buf);
    vx_dev_close(dev);
    return (mismatches == 0) ? 0 : 1;
}
