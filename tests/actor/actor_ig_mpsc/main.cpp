// actor_ig_mpsc host: 1 actor per hardware lane, MPSC shared inboxes (mb0/mb1,
// O(N) each). Launches the kernel, verifies via host RNG replay, prints the
// per-PE phase breakdown and the inbox memory vs the SPSC 2*N^2 grid.

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

    size_t inbox_bytes = (size_t)N * sizeof(inbox_t);          // per mailbox, O(N)
    size_t spsc_bytes  = (size_t)2 * N * N * sizeof(slot_t);   // ~ 2*N^2 grid (ref)
    size_t tgt_bytes   = (size_t)N * M * sizeof(int);
    size_t phase_bytes = (size_t)N * sizeof(phase_cycles_t);

    std::cout << "N=" << N << " actors (1 thread each, MPSC shared inboxes), M=" << M
              << " requests/PE\n";
    std::cout << "inbox memory = " << (2 * inbox_bytes) << " bytes (mb0+mb1, O(N)); "
              << "an SPSC 2*N*N grid would be ~" << spsc_bytes << " bytes\n";

    vx_buffer_h mb0_buf, mb1_buf, tgt_buf, phase_buf, kernel_buf, args_buf;
    vx_mem_alloc(dev, inbox_bytes, VX_MEM_READ_WRITE, &mb0_buf);
    vx_mem_alloc(dev, inbox_bytes, VX_MEM_READ_WRITE, &mb1_buf);
    vx_mem_alloc(dev, tgt_bytes,   VX_MEM_READ_WRITE, &tgt_buf);
    vx_mem_alloc(dev, phase_bytes, VX_MEM_READ_WRITE, &phase_buf);

    args_t args;
    vx_mem_address(mb0_buf,   &args.mb0_addr);
    vx_mem_address(mb1_buf,   &args.mb1_addr);
    vx_mem_address(tgt_buf,   &args.tgt_addr);
    vx_mem_address(phase_buf, &args.phase_addr);
    args.N = N;

    // initialize both inbox arrays: head=tail=0, slots[j].seq = j
    std::vector<inbox_t> ib(N);
    for (uint32_t n = 0; n < N; n++) {
        ib[n].head = 0; ib[n].tail = 0;
        for (int j = 0; j < CAP; j++) {
            ib[n].slots[j].sender = 0; ib[n].slots[j].done_flag = 0;
            ib[n].slots[j].idx = 0; ib[n].slots[j].value = 0;
            ib[n].slots[j].seq = (uint32_t)j;
        }
    }
    vx_copy_to_dev(mb0_buf, ib.data(), 0, inbox_bytes);
    vx_copy_to_dev(mb1_buf, ib.data(), 0, inbox_bytes);

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

    // expected: replay the same RNG on the host
    std::vector<int> expected(N * M, 0);
    for (uint32_t me = 0; me < N; me++) {
        for (uint32_t i = 0; i < M; i++) {
            uint32_t dst  = mix(me, 2*i + 1) % N;
            uint32_t lidx = mix(me, 2*i + 2) % 1000;
            expected[me * M + i] = (int)(dst * 1000000 + lidx);
        }
    }

    int mismatches = 0;
    for (uint32_t me = 0; me < N; me++) {
        for (uint32_t i = 0; i < M; i++) {
            if (tgt[me * M + i] != expected[me * M + i]) {
                if (mismatches < 8)
                    std::cout << "  mismatch PE " << me << " query " << i << ": got "
                              << tgt[me * M + i] << ", expected " << expected[me * M + i] << "\n";
                mismatches++;
            }
        }
    }

    std::cout << "result: total_requests=" << (uint64_t)N * M
              << " mismatches=" << mismatches << "\n";
    std::cout << (mismatches == 0 ? "PASSED" : "FAILED") << "\n";

    // ---- phase breakdown ----
    std::vector<phase_cycles_t> ph(N);
    vx_copy_from_dev(ph.data(), phase_buf, 0, phase_bytes);

    uint64_t sum_send_data=0, sum_send_done=0, sum_hreq=0, sum_hrep=0, sum_empty=0, sum_tail=0, sum_span=0;
    uint64_t max_hreq=0, max_hrep=0, max_empty=0;
    for (uint32_t i = 0; i < N; i++) {
        sum_send_data += ph[i].send_data;
        sum_send_done += ph[i].send_done;
        sum_hreq      += ph[i].handle_req; max_hreq  = std::max(max_hreq,  ph[i].handle_req);
        sum_hrep      += ph[i].handle_rep; max_hrep  = std::max(max_hrep,  ph[i].handle_rep);
        sum_empty     += ph[i].empty_poll; max_empty = std::max(max_empty, ph[i].empty_poll);
        sum_tail      += ph[i].tail_wait;
        sum_span      += ph[i].total;
    }
    auto p = [N](uint64_t s){ return (uint64_t)((double)s / N); };
    uint64_t mean_span = sum_span / N;

    std::cout << "PHASES (mean cycles per PE across " << N << " PEs):\n";
    std::cout << "  send_data:  " << p(sum_send_data) << "\n";
    std::cout << "  send_done:  " << p(sum_send_done) << "\n";
    std::cout << "  handle_req: " << p(sum_hreq) << "  max=" << max_hreq << "\n";
    std::cout << "  handle_rep: " << p(sum_hrep) << "  max=" << max_hrep << "\n";
    std::cout << "  empty_poll: " << p(sum_empty) << "  max=" << max_empty << "\n";
    std::cout << "  tail_wait:  " << p(sum_tail) << "\n";
    std::cout << "  mean_span:  " << mean_span << "\n";

    vx_mem_free(mb0_buf);    vx_mem_free(mb1_buf);   vx_mem_free(tgt_buf);
    vx_mem_free(phase_buf);  vx_mem_free(kernel_buf); vx_mem_free(args_buf);
    vx_dev_close(dev);
    return (mismatches == 0) ? 0 : 1;
}
