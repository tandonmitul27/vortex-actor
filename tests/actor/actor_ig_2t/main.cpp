// actor_ig host: launch the request/reply kernel, verify via host RNG replay,
// print the per-PE phase cycle breakdown.
// Usage: ./actor_ig [-n N]   (default N = device lane count)

#include <iostream>
#include <vector>
#include <vortex.h>
#include "common.h"

// must match the kernel's mix()
static inline uint32_t mix(uint32_t a, uint32_t b) {
    uint32_t x = a * 0x9E3779B9u + b * 0x85EBCA77u;
    x ^= (x >> 13);
    x *= 0xC2B2AE3Du;
    x ^= (x >> 16);
    return x;
}

int main(int argc, char** argv) {
    // Optional -n N: override the actor count (defaults to total HW lanes).
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
    // 2 hardware threads per actor (main + recv), so N actors use 2N lanes
    uint32_t N = (N_arg > 0) ? (uint32_t)N_arg : hw_lanes / 2;
    std::cout << "N=" << N << " actors (2 threads each), M=" << M
              << " requests/PE (HW lanes available=" << hw_lanes << ")\n";

    size_t grid_bytes = (size_t)N * N * sizeof(ring_t);   // one N×N grid per mailbox
    size_t tgt_bytes  = (size_t)N * M * sizeof(int);      // gathered results
    size_t phase_bytes = (size_t)N * sizeof(phase_cycles_t);

    // mb0 = request channels, mb1 = reply channels (both N×N grids)
    vx_buffer_h mb0_buf, mb1_buf, tgt_buf, phase_buf, kernel_buf, args_buf;
    vx_mem_alloc(dev, grid_bytes, VX_MEM_READ_WRITE, &mb0_buf);
    vx_mem_alloc(dev, grid_bytes, VX_MEM_READ_WRITE, &mb1_buf);
    vx_mem_alloc(dev, tgt_bytes,  VX_MEM_READ_WRITE, &tgt_buf);
    vx_mem_alloc(dev, phase_bytes, VX_MEM_READ_WRITE, &phase_buf);

    args_t args;
    vx_mem_address(mb0_buf, &args.mb0_grid_addr);
    vx_mem_address(mb1_buf, &args.mb1_grid_addr);
    vx_mem_address(tgt_buf, &args.tgt_addr);
    vx_mem_address(phase_buf, &args.phase_addr);
    args.N = N;

    std::vector<char> zg(grid_bytes, 0);
    vx_copy_to_dev(mb0_buf, zg.data(), 0, grid_bytes);
    vx_copy_to_dev(mb1_buf, zg.data(), 0, grid_bytes);
    std::vector<char> zt(tgt_bytes, 0);
    vx_copy_to_dev(tgt_buf, zt.data(), 0, tgt_bytes);
    std::vector<char> zp(phase_bytes, 0);
    vx_copy_to_dev(phase_buf, zp.data(), 0, phase_bytes);

    vx_upload_kernel_file(dev, "kernel.vxbin", &kernel_buf);
    vx_upload_bytes(dev, &args, sizeof(args), &args_buf);
    vx_start(dev, kernel_buf, args_buf);
    vx_ready_wait(dev, VX_MAX_TIMEOUT);

    std::vector<int> tgt(N * M);
    vx_copy_from_dev(tgt.data(), tgt_buf, 0, tgt_bytes);

    // ---- Recompute the expected tgt[] on the host (mirrors the responder formula) ----
    std::vector<int> expected(N * M, 0);
    for (uint32_t me = 0; me < N; me++) {
        for (uint32_t i = 0; i < M; i++) {
            uint32_t dst  = mix(me, 2*i + 1) % N;     // who PE me queried
            uint32_t lidx = mix(me, 2*i + 2) % 1000;  // the index it asked for
            expected[me * M + i] = (int)(dst * 1000000 + lidx);  // dst's reply value
        }
    }

    int mismatches = 0;
    for (uint32_t me = 0; me < N; me++) {
        for (uint32_t i = 0; i < M; i++) {
            int actual = tgt[me * M + i];
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

    // ---- Phase cycle breakdown ----
    std::vector<phase_cycles_t> ph(N);
    vx_copy_from_dev(ph.data(), phase_buf, 0, phase_bytes);

    uint64_t sum_send_data=0, sum_send_done=0, sum_hreq=0, sum_hrep=0, sum_empty=0, sum_tail=0, sum_span=0;
    uint64_t max_send_data=0, max_send_done=0, max_hreq=0, max_hrep=0, max_empty=0, max_tail=0;
    for (uint32_t i = 0; i < N; i++) {
        sum_send_data += ph[i].send_data;  max_send_data = std::max(max_send_data, ph[i].send_data);
        sum_send_done += ph[i].send_done;  max_send_done = std::max(max_send_done, ph[i].send_done);
        sum_hreq      += ph[i].handle_req; max_hreq      = std::max(max_hreq,      ph[i].handle_req);
        sum_hrep      += ph[i].handle_rep; max_hrep      = std::max(max_hrep,      ph[i].handle_rep);
        sum_empty     += ph[i].empty_poll; max_empty     = std::max(max_empty,     ph[i].empty_poll);
        sum_tail      += ph[i].tail_wait;  max_tail      = std::max(max_tail,      ph[i].tail_wait);
        sum_span      += ph[i].total;
    }
    auto p = [N](uint64_t s){ return (double)s / N; };
    uint64_t mean_total = (sum_send_data + sum_send_done + sum_hreq + sum_hrep + sum_empty + sum_tail) / N;
    uint64_t mean_span  = sum_span / N;   // mean per-PE kernel_body span
    auto pct = [&](uint64_t s){ return mean_total ? 100.0 * (p(s) / mean_total) : 0.0; };

    std::cout << "PHASES (cycles per PE — mean, max across " << N << " PEs):\n";
    std::cout << "  send_data:  mean=" << (uint64_t)p(sum_send_data)
              << "  max=" << max_send_data << "  (" << pct(sum_send_data) << "% of mean total)\n";
    std::cout << "  send_done:  mean=" << (uint64_t)p(sum_send_done)
              << "  max=" << max_send_done << "  (" << pct(sum_send_done) << "%)\n";
    std::cout << "  handle_req: mean=" << (uint64_t)p(sum_hreq)
              << "  max=" << max_hreq     << "  (" << pct(sum_hreq) << "%)\n";
    std::cout << "  handle_rep: mean=" << (uint64_t)p(sum_hrep)
              << "  max=" << max_hrep     << "  (" << pct(sum_hrep) << "%)\n";
    std::cout << "  empty_poll: mean=" << (uint64_t)p(sum_empty)
              << "  max=" << max_empty     << "  (" << pct(sum_empty) << "%)\n";
    std::cout << "  tail_wait:  mean=" << (uint64_t)p(sum_tail)
              << "  max=" << max_tail      << "  (" << pct(sum_tail) << "%)\n";
    std::cout << "  mean_total: " << mean_total << "\n";           // sum of 6 phases
    std::cout << "  mean_span:  " << mean_span << "\n";            // measured kernel_body span
    std::cout << "  other:      " << (mean_span > mean_total ? mean_span - mean_total : 0)
              << "  (unmeasured vote_all/loop overhead)\n";

    vx_mem_free(mb0_buf);    vx_mem_free(mb1_buf);   vx_mem_free(tgt_buf);
    vx_mem_free(phase_buf);
    vx_mem_free(kernel_buf); vx_mem_free(args_buf);
    vx_dev_close(dev);
    return (mismatches == 0) ? 0 : 1;
}
