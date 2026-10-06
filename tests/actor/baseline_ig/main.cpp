// baseline_ig host: fill a known table, launch the gather, verify via host RNG replay.
// Usage: ./baseline_ig [-n N]   (default N = device lane count)

#include <iostream>
#include <vector>
#include <algorithm>
#include <vortex.h>
#include "common.h"
#ifndef SPLIT
#define SPLIT 1
#endif

// must match the device-side mix() exactly
static inline uint32_t mix(uint32_t a, uint32_t b) {
    uint32_t x = a * 0x9E3779B9u + b * 0x85EBCA77u;
    x ^= (x >> 13);
    x *= 0xC2B2AE3Du;
    x ^= (x >> 16);
    return x;
}

int main(int argc, char** argv) {
    // optional "-n N" to override the actor count
    int N_arg = 0;
    for (int i = 1; i + 1 < argc; i++) {
        if (std::string(argv[i]) == "-n") N_arg = atoi(argv[i + 1]);
    }

    vx_device_h dev;
    if (vx_dev_open(&dev) != 0) { std::cerr << "vx_dev_open failed\n"; return 1; }

    uint64_t cores, warps, threads;
    vx_dev_caps(dev, VX_CAPS_NUM_CORES,   &cores);
    vx_dev_caps(dev, VX_CAPS_NUM_WARPS,   &warps);
    vx_dev_caps(dev, VX_CAPS_NUM_THREADS, &threads);
    uint32_t hw_lanes = (uint32_t)(cores * warps * threads);
    uint32_t N = (N_arg > 0) ? (uint32_t)N_arg : hw_lanes / (uint32_t)(SPLIT);   // SPLIT, log T41

    std::cout << "N=" << N << " threads, M=" << M
              << " requests/PE, table=" << TABLE_SIZE
              << " entries/PE (baseline IG via indirect a[b[i]])\n";

    size_t table_bytes = (size_t)N * TABLE_SIZE * sizeof(int);
    size_t tgt_bytes   = (size_t)N * M * sizeof(int);

    vx_buffer_h table_buf, tgt_buf, kernel_buf, args_buf, phase_buf;
    vx_mem_alloc(dev, table_bytes, VX_MEM_READ_WRITE, &table_buf);
    vx_mem_alloc(dev, tgt_bytes,   VX_MEM_READ_WRITE, &tgt_buf);
    size_t phase_bytes = (size_t)N * sizeof(phase_cycles_t);
    vx_mem_alloc(dev, phase_bytes, VX_MEM_READ_WRITE, &phase_buf);

    args_t args;
    vx_mem_address(table_buf, &args.table_addr);
    vx_mem_address(tgt_buf,   &args.tgt_addr);
    vx_mem_address(phase_buf, &args.phase_addr);
    args.N = N;

    // table[pe*TABLE_SIZE + lidx] = pe*1e6 + lidx  (so the host knows every value)
    std::vector<int> table(N * TABLE_SIZE);
    for (uint32_t pe = 0; pe < N; pe++) {
        for (uint32_t lidx = 0; lidx < TABLE_SIZE; lidx++) {
            table[pe * TABLE_SIZE + lidx] = (int)(pe * 1000000 + lidx);
        }
    }
    vx_copy_to_dev(table_buf, table.data(), 0, table_bytes);

    std::vector<char> zt(tgt_bytes, 0);
    vx_copy_to_dev(tgt_buf, zt.data(), 0, tgt_bytes);

    vx_upload_kernel_file(dev, "kernel.vxbin", &kernel_buf);
    vx_upload_bytes(dev, &args, sizeof(args), &args_buf);
    vx_start(dev, kernel_buf, args_buf);
    vx_ready_wait(dev, VX_MAX_TIMEOUT);

    std::vector<int> tgt(N * M);
    vx_copy_from_dev(tgt.data(), tgt_buf, 0, tgt_bytes);
    {   // the launch-free body window (T42), as the actor kernels report it
        std::vector<phase_cycles_t> ph(N);
        vx_copy_from_dev(ph.data(), phase_buf, 0, phase_bytes);
        uint64_t s0 = ~0ull, e1 = 0;
        for (uint32_t i = 0; i < N; i++) { s0 = std::min(s0, ph[i].t_start); e1 = std::max(e1, ph[i].t_end); }
        std::cout << "span (absolute core cycles): first start=" << s0 << " last end=" << e1 << "\n";
    }

    // expected output: replay the same RNG on the host
    std::vector<int> expected(N * M, 0);
    for (uint32_t me = 0; me < N; me++) {
        for (uint32_t i = 0; i < M; i++) {
            uint32_t dst  = mix(me, 2*i + 1) % N;
            uint32_t lidx = mix(me, 2*i + 2) % TABLE_SIZE;
            expected[me * M + i] = (int)(dst * 1000000 + lidx);
        }
    }

    int mismatches = 0;
    for (uint32_t me = 0; me < N; me++) {
        for (uint32_t i = 0; i < M; i++) {
            int actual = tgt[TGT_IDX(me, i, N)];
            int exp    = expected[me * M + i];
            if (actual != exp) {
                if (mismatches < 8) {
                    std::cout << "  mismatch PE " << me << " query " << i
                              << ": got " << actual << ", expected " << exp << "\n";
                }
                mismatches++;
            }
        }
    }

    std::cout << "result: total_requests=" << (uint64_t)N * M
              << " mismatches=" << mismatches << "\n";
    std::cout << (mismatches == 0 ? "PASSED" : "FAILED") << "\n";

    vx_mem_free(table_buf);
    vx_mem_free(tgt_buf);
    vx_mem_free(phase_buf);
    vx_mem_free(kernel_buf);
    vx_mem_free(args_buf);
    vx_dev_close(dev);
    return (mismatches == 0) ? 0 : 1;
}
