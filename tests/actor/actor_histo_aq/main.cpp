// Host for actor_histo_aq. Allocates the queue storage the model appends into,
// audits it, and checks every bucket against a replayed reference.
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vortex.h>
#include "common.h"
#ifndef PROBE_NOAPPLY
#define PROBE_NOAPPLY 0   // 1: measurement only, the receive skips the apply (results are not checked, counts are)
#endif
#ifndef ACTOR
#define ACTOR 1
#endif
#include "../isa2.h"
#include "../skew.h"

static inline uint32_t mix(uint32_t a, uint32_t b) {
    uint32_t x = a * 0x9E3779B9u + b * 0x85EBCA77u;
    x ^= (x >> 13);
    x *= 0xC2B2AE3Du;
    x ^= (x >> 16);
    return x;
}

int main(int argc, char** argv) {
    int N_arg = 0;
    for (int i = 1; i + 1 < argc; i++)
        if (std::string(argv[i]) == "-n") N_arg = atoi(argv[i + 1]);
#if !defined(BASELINE) || !BASELINE   // BASELINE: no L2-append window, so no window environment
    const char* eb = getenv("VORTEX_AQ_BASE");
    const char* es = getenv("VORTEX_AQ_SIZE");
    if (!eb || !es || strtoull(eb, nullptr, 0) != (unsigned long long)AQ_BASE
                   || strtoull(es, nullptr, 0) != (unsigned long long)AQ_SIZE) {
        std::cerr << "set VORTEX_AQ_BASE=0x" << std::hex << AQ_BASE
                  << " VORTEX_AQ_SIZE=0x" << AQ_SIZE << std::dec << "\n";
        return 1;
    }
#endif
    vx_device_h dev;
    if (vx_dev_open(&dev) != 0) { std::cerr << "vx_dev_open failed\n"; return 1; }
    uint64_t cores, warps, threads;
    vx_dev_caps(dev, VX_CAPS_NUM_CORES,   &cores);
    vx_dev_caps(dev, VX_CAPS_NUM_WARPS,   &warps);
    vx_dev_caps(dev, VX_CAPS_NUM_THREADS, &threads);
    // SPLIT lanes share one actor, so the machine's lanes no longer dictate the
    // actor count: N = lanes / SPLIT.
    uint32_t N = N_arg > 0 ? (uint32_t)N_arg
                           : (uint32_t)(cores * warps * threads) * (uint32_t)(VPATH ? (VACT) : 1) / (uint32_t)(SPLIT);
    if ((uint32_t)threads != LANES) { std::cerr << "LANES mismatch\n"; return 1; }
    if (N % GLANES != 0) { std::cerr << "N must be a multiple of GLANES\n"; return 1; }
    if (N != (uint32_t)NASSUME) { std::cerr << "rebuild with -DNASSUME=" << N << "\n"; return 1; }
    uint32_t NB = VPATH ? (uint32_t)(NBA) : N / GLANES;   // sender keys (VPATH: N / (GLANES*VACT*QSG))
#if NCONST
    if (N != (uint32_t)NASSUME) {
        std::cout << "NCONST built for NASSUME=" << NASSUME << " but N=" << N << "\nFAILED\n";
        return 1;
    }
#endif
    size_t NQ = (size_t)((QMAJOR) ? BQ_ROW(N) : N) * NB;   // QROWPAD spare queues per row (QMAJOR)
    std::cout << "N=" << N << " actors, L=" << L << ", M=" << M
              << ", queues=" << NQ << " x " << AQ_QCAP << " (mean " << ((double)M/NB) << ")"
              << ", qs=" << (NQ*AQ_QCAP*4.0/(1024*1024)) << " MB"
              << ", combining lines=" << (NQ*64.0/1024) << " KB"
              << ", lmem=" << (BUF_WORDS*4) << " B\n";
    size_t qs_bytes = NQ * QSTRIDE_W * sizeof(int);   // padded stride, see QPAD
    size_t tl_bytes = NQ * sizeof(uint32_t);
    size_t cnt_bytes = (size_t)N * L * sizeof(int);
    size_t phase_bytes = (size_t)N * sizeof(phase_cycles_t) + (LANE_TS ? (size_t)N * (SPLIT) * 4 * 8 : 0);
    vx_buffer_h qsb, tlb, cb, ph_b, bar_b, kb, ab_, bsb, btb;
    vx_mem_alloc(dev, qs_bytes,  VX_MEM_READ_WRITE, &qsb);
    vx_mem_alloc(dev, tl_bytes,  VX_MEM_READ_WRITE, &tlb);
    vx_mem_alloc(dev, cnt_bytes, VX_MEM_READ_WRITE, &cb);
    vx_mem_alloc(dev, phase_bytes, VX_MEM_READ_WRITE, &ph_b);
    vx_mem_alloc(dev, sizeof(uint32_t)*8, VX_MEM_READ_WRITE, &bar_b);
#if BUCKET_TILE
    size_t bs_bytes = (size_t)aq_region_bytes((uint64_t)BQ_ROW(N) * BUCKET_TILE, (uint64_t)BQ_STRIDE_W * sizeof(int));   // ISA2: the program's queue layout
    size_t bt_bytes = (size_t)BQ_ROW(N) * BUCKET_TILE * sizeof(uint32_t);
    vx_mem_alloc(dev, bs_bytes, VX_MEM_READ_WRITE, &bsb);
    vx_mem_alloc(dev, bt_bytes, VX_MEM_READ_WRITE, &btb);
#endif
    args_t args;
    vx_mem_address(qsb,&args.qs_addr); vx_mem_address(tlb,&args.tails_addr);
    vx_mem_address(cb,&args.counts_addr); vx_mem_address(ph_b,&args.phase_addr);
    vx_mem_address(bar_b,&args.bar_addr); args.N = N;
    args.bs_addr = 0; args.btails_addr = 0;
#if BUCKET_TILE
    vx_mem_address(bsb,&args.bs_addr); vx_mem_address(btb,&args.btails_addr);
#if OVF
    vx_buffer_h poolb, ctabb;
    { size_t pool_bytes = (size_t)(OVF_NCH) * (OVF_CH) * sizeof(int);
      size_t ctab_bytes = (size_t)BQ_ROW(N) * BUCKET_TILE * (OVF_MAXC) * sizeof(uint64_t);
      vx_mem_alloc(dev, pool_bytes, VX_MEM_READ_WRITE, &poolb);
      vx_mem_alloc(dev, ctab_bytes, VX_MEM_READ_WRITE, &ctabb);
      vx_mem_address(poolb, &args.pool_addr); vx_mem_address(ctabb, &args.ctab_addr);
      std::cout << "  overflow pool: " << (OVF_NCH) << " chunks x " << (OVF_CH) << " words, chunk table "
                << (ctab_bytes >> 10) << " KB\n"; }
#endif
    { std::vector<char> z(bs_bytes,0); vx_copy_to_dev(bsb, z.data(), 0, bs_bytes); }
    { std::vector<char> z(bt_bytes,0); vx_copy_to_dev(btb, z.data(), 0, bt_bytes); }
    std::cout << "  bucket tiles: " << BUCKET_TILE << " x " << BTILE_SZ
              << " (" << (BTILE_SZ*4/1024.0) << " KB/lane), queues=" << ((size_t)N*BUCKET_TILE) << "\n";
#endif
    { std::vector<char> z(qs_bytes,0); vx_copy_to_dev(qsb, z.data(), 0, qs_bytes); }
    { std::vector<char> z(tl_bytes,0); vx_copy_to_dev(tlb, z.data(), 0, tl_bytes); }
    { std::vector<char> z(cnt_bytes,0); vx_copy_to_dev(cb, z.data(), 0, cnt_bytes); }
    { std::vector<char> z(phase_bytes,0); vx_copy_to_dev(ph_b, z.data(), 0, phase_bytes); }
    { uint32_t z[8]={0,0,0,0,0,0,0,0}; vx_copy_to_dev(bar_b, z, 0, sizeof(z)); }
    vx_upload_kernel_file(dev, "kernel.vxbin", &kb);
    vx_upload_bytes(dev, &args, sizeof(args), &ab_);
    vx_start(dev, kb, ab_);
    vx_ready_wait(dev, VX_MAX_TIMEOUT);

    std::vector<int> counts((size_t)N*L);
    std::vector<uint32_t> tails(NQ);
    std::vector<phase_cycles_t> ph(N + (LANE_TS ? (N * (SPLIT) * 4 * 8 + sizeof(phase_cycles_t) - 1) / sizeof(phase_cycles_t) : 0));
    vx_copy_from_dev(counts.data(), cb, 0, cnt_bytes);
    vx_copy_from_dev(tails.data(), tlb, 0, tl_bytes);
    vx_copy_from_dev(ph.data(), ph_b, 0, phase_bytes);

    int tbad = 0;
#if BUCKET_TILE && !AQ_ON   // ISA2: the counts live in the L2 (the count instruction), not in a tails array; simx reports DROPPED
    { std::vector<uint32_t> bt((size_t)BQ_ROW(N) * BUCKET_TILE);
      vx_copy_from_dev(bt.data(), btb, 0, bt_bytes);
      unsigned long long sum = 0; uint32_t mx = 0;
      for (size_t q = 0; q < bt.size(); q++) { sum += bt[q]; if (bt[q] > mx) mx = bt[q]; }
      // With chunked rounds only the LAST chunk's appends are still in the
      // tails: every earlier chunk was drained and the queues emptied.
      const unsigned long long want_appends = (unsigned long long)N*M / (CHUNKS);
      std::cout << "tile queue audit: " << sum << " updates (want " << want_appends
                << "), max tail " << mx << " (cap " << BQ_CAP << ")\n";
      if (sum != want_appends) { std::cout << "  TQUEUE: count wrong\n"; tbad++; }
      if (!(OVF) && mx > (uint32_t)BQ_CAP) { std::cout << "  TQUEUE: exceeds capacity\n"; tbad++; }
    }
#endif
#if !DIRECT_BT
    { unsigned long long sum = 0; uint32_t mx = 0, mn = ~0u; size_t used = 0;
      for (size_t q = 0; q < NQ; q++) { sum += tails[q]; mx = std::max(mx, tails[q]);
                                        mn = std::min(mn, tails[q]); used += tails[q] != 0; }
      const unsigned long long want_appends = (unsigned long long)N*M / (CHUNKS);
      std::cout << "queue audit: " << sum << " appends (want " << want_appends
                << "), queues used " << used << "/" << NQ << ", tail min/mean/max = "
                << mn << "/" << (double)sum/NQ << "/" << mx << " (cap " << AQ_QCAP << ")\n";
      if (sum != want_appends) { std::cout << "  QUEUE: append count wrong\n"; tbad++; }
      if (mx > (uint32_t)AQ_QCAP) { std::cout << "  QUEUE: exceeds capacity\n"; tbad++; }
    }
#endif
    std::vector<int> want((size_t)N*L, 0);
    uint32_t G = (uint32_t)(L * N);
    for (uint32_t s = 0; s < N; s++)
        for (uint32_t i = 0; i < (uint32_t)M; i++) {
            uint32_t g = skew_key(mix(s, i), G);
            want[(size_t)(g % N) * L + (g / N)]++;
        }
    int bad = 0;
    for (uint32_t o = 0; o < N; o++)
        for (uint32_t b = 0; b < (uint32_t)L; b++) {
            int got = counts[CNT_IDX(o, b)];
            int w   = want[(size_t)o * L + b];
            if (got != w) {
                if (bad < 8) std::cout << "  bucket " << o << ":" << b
                                       << " got " << got << " want " << w << "\n";
                bad++;
            }
        }
    double se=0,ba=0,re=0,to=0;
    for (uint32_t i=0;i<N;i++){ se+=ph[i].send; ba+=ph[i].barrier; re+=ph[i].recv; to+=ph[i].total; }
#if LANE_TS
    { const uint64_t* lt = (const uint64_t*)&ph[N]; const uint32_t NL = N * (SPLIT);
      const char* nm[3] = {"send","wait","recv"};
      uint64_t s0=~0ull, e1=0, sk=0; for (uint32_t l=0;l<NL;l++){ s0=std::min(s0,lt[l*4]); e1=std::max(e1,lt[l*4+3]); }
      for (uint32_t l=0;l<NL;l++) sk=std::max(sk,lt[l*4]-s0);
      std::cout << "LANE_TS lanes=" << NL << " wall=" << (e1-s0) << " start_skew=" << sk << "\n";
      for (int k=0;k<3;k++){ double sm=0; uint64_t mn=~0ull,mx=0, se=~0ull, ee=0;
        for (uint32_t l=0;l<NL;l++){ uint64_t d=lt[l*4+k+1]-lt[l*4+k]; sm+=d; mn=std::min(mn,d); mx=std::max(mx,d);
                                     se=std::min(se,lt[l*4+k]); ee=std::max(ee,lt[l*4+k+1]); }
        std::cout << "LANE_TS " << nm[k] << " mean=" << (uint64_t)(sm/NL) << " min=" << mn << " max=" << mx
                  << " window=" << (ee-se) << "\n"; } }
#endif
    std::cout << "phases (mean cycles/actor): send=" << (uint64_t)(se/N)
              << " barrier=" << (uint64_t)(ba/N) << " recv=" << (uint64_t)(re/N)
              << " total=" << (uint64_t)(to/N) << "\n";
    { uint64_t s0=~0ull, s1=0, e0=~0ull, e1=0, tmax=0;
      for (uint32_t i=0;i<N;i++){ s0=std::min(s0,ph[i].t_start); s1=std::max(s1,ph[i].t_start);
                                  e0=std::min(e0,ph[i].t_end);   e1=std::max(e1,ph[i].t_end);
                                  tmax=std::max(tmax,ph[i].total); }
      std::cout << "span (absolute core cycles): first start=" << s0 << " last start=" << s1
                << " first end=" << e0 << " last end=" << e1 << " max total=" << tmax << "\n"; }
    std::cout << "result: total_updates=" << (uint64_t)N*M << " errors=" << bad
              << " table_problems=" << tbad << "\n"
              << (((bad==0 || PROBE_NOAPPLY) && tbad==0) ? "PASSED" : "FAILED") << "\n";
#if BUCKET_TILE
    vx_mem_free(bsb); vx_mem_free(btb);
#endif
    vx_mem_free(qsb); vx_mem_free(tlb); vx_mem_free(cb); vx_mem_free(ph_b);
    vx_mem_free(bar_b); vx_mem_free(kb); vx_mem_free(ab_);
    vx_dev_close(dev);
    return (bad==0 && tbad==0) ? 0 : 1;
}
