// actor_histo_mpsc host: 1 actor per hardware lane, MPSC shared inboxes (O(N)
// memory). Launches the kernel, verifies by replaying the RNG, prints the
// per-PE phase breakdown and the inbox memory vs the SPSC N^2 grid.

#include <iostream>
#include <vector>
#include <vortex.h>
#include "common.h"

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

    uint64_t cores, warps, threads;
    vx_dev_caps(dev, VX_CAPS_NUM_CORES,   &cores);
    vx_dev_caps(dev, VX_CAPS_NUM_WARPS,   &warps);
    vx_dev_caps(dev, VX_CAPS_NUM_THREADS, &threads);
    uint32_t N = (uint32_t)(cores * warps * threads);   // 1 actor per lane

    size_t inbox_bytes  = (size_t)N * sizeof(inbox_t);          // O(N)
    size_t spsc_bytes   = (size_t)N * N * sizeof(slot_t);       // ~ N^2 grid (ref)
    size_t counts_bytes = (size_t)N * L * sizeof(int);
    size_t phase_bytes  = (size_t)N * sizeof(phase_cycles_t);

    std::cout << "N=" << N << " actors (1 thread each, MPSC shared inbox), L=" << L
              << " buckets/PE, M=" << M << " updates/PE\n";
    std::cout << "inbox memory = " << inbox_bytes << " bytes (O(N)); an SPSC N*N grid"
              << " would be ~" << spsc_bytes << " bytes\n";

    vx_buffer_h inbox_buf, counts_buf, phase_buf, kernel_buf, args_buf;
    vx_mem_alloc(dev, inbox_bytes,  VX_MEM_READ_WRITE, &inbox_buf);
    vx_mem_alloc(dev, counts_bytes, VX_MEM_READ_WRITE, &counts_buf);
    vx_mem_alloc(dev, phase_bytes,  VX_MEM_READ_WRITE, &phase_buf);

    args_t args;
    vx_mem_address(inbox_buf,  &args.inbox_addr);
    vx_mem_address(counts_buf, &args.counts_addr);
    vx_mem_address(phase_buf,  &args.phase_addr);
    args.N = N;

    // initialize inboxes: head=tail=0, slots[j].seq = j
    std::vector<inbox_t> ib(N);
    for (uint32_t n = 0; n < N; n++) {
        ib[n].head = 0; ib[n].tail = 0;
        for (int j = 0; j < CAP; j++) {
            ib[n].slots[j].sender = 0;
            ib[n].slots[j].done_flag = 0;
            ib[n].slots[j].bucket = 0;
            ib[n].slots[j].seq = (uint32_t)j;
        }
    }
    vx_copy_to_dev(inbox_buf, ib.data(), 0, inbox_bytes);

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

    // expected: replay the kernel's RNG on the host
    std::vector<int> expected(N * L, 0);
    uint32_t G_total = L * N;
    for (uint32_t me = 0; me < N; me++) {
        for (uint32_t i = 0; i < M; i++) {
            uint32_t g     = mix(me, i) % G_total;
            uint32_t owner = g % N;
            uint32_t lidx  = g / N;
            expected[owner * L + lidx]++;
        }
    }

    int mismatches = 0;
    uint64_t total_counts = 0;
    for (uint32_t i = 0; i < N * L; i++) {
        total_counts += counts[i];
        if (counts[i] != expected[i]) {
            if (mismatches < 8)
                std::cout << "  mismatch bucket " << i << ": got " << counts[i]
                          << ", expected " << expected[i] << "\n";
            mismatches++;
        }
    }

    std::cout << "result: total_updates=" << (uint64_t)N * M
              << " sum_of_counts=" << total_counts
              << " mismatches=" << mismatches << "\n";
    std::cout << (mismatches == 0 ? "PASSED" : "FAILED") << "\n";

    // ---- phase breakdown ----
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
    uint64_t mean_span  = sum_span / N;
    auto pct = [&](uint64_t s){ return mean_total ? 100.0 * (p(s) / mean_total) : 0.0; };

    std::cout << "PHASES (cycles per PE — mean, max across " << N << " PEs):\n";
    std::cout << "  send_data:  mean=" << (uint64_t)p(sum_send_data) << "  max=" << max_send_data << "  (" << pct(sum_send_data) << "% of mean total)\n";
    std::cout << "  send_done:  mean=" << (uint64_t)p(sum_send_done) << "  max=" << max_send_done << "  (" << pct(sum_send_done) << "%)\n";
    std::cout << "  handle_req: mean=" << (uint64_t)p(sum_hreq)      << "  max=" << max_hreq      << "  (" << pct(sum_hreq) << "%)\n";
    std::cout << "  handle_rep: mean=" << (uint64_t)p(sum_hrep)      << "  max=" << max_hrep      << "  (" << pct(sum_hrep) << "%)\n";
    std::cout << "  empty_poll: mean=" << (uint64_t)p(sum_empty)     << "  max=" << max_empty     << "  (" << pct(sum_empty) << "%)\n";
    std::cout << "  tail_wait:  mean=" << (uint64_t)p(sum_tail)      << "  max=" << max_tail      << "  (" << pct(sum_tail) << "%)\n";
    std::cout << "  mean_total: " << mean_total << "\n";
    std::cout << "  mean_span:  " << mean_span << "\n";
    std::cout << "  other:      " << (mean_span > mean_total ? mean_span - mean_total : 0) << "\n";

    vx_mem_free(inbox_buf);  vx_mem_free(counts_buf);  vx_mem_free(phase_buf);
    vx_mem_free(kernel_buf); vx_mem_free(args_buf);
    vx_dev_close(dev);
    return (mismatches == 0) ? 0 : 1;
}
