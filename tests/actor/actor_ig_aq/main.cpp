// Host for actor_ig_aq. Allocates the queue storage and tails the simulator's
// append model writes into, audits them, and checks tgt exactly as the dense
// host does. Run with VORTEX_AQ_BASE=0xD0000000 VORTEX_AQ_SIZE=0x100000 (the
// values of AQ_BASE / AQ_SIZE in common.h); without them every append is an
// ordinary store to an unmapped address and the run fails.
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <vortex.h>
#include "common.h"
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
                  << " VORTEX_AQ_SIZE=0x" << AQ_SIZE << std::dec << " in the environment\n";
        return 1;
    }
#endif
    vx_device_h dev;
    if (vx_dev_open(&dev) != 0) { std::cerr << "vx_dev_open failed\n"; return 1; }
    uint64_t cores, warps, threads;
    vx_dev_caps(dev, VX_CAPS_NUM_CORES,   &cores);
    vx_dev_caps(dev, VX_CAPS_NUM_WARPS,   &warps);
    vx_dev_caps(dev, VX_CAPS_NUM_THREADS, &threads);
    // SPLIT lanes share one actor (log T41), so N = lanes / SPLIT.
    uint32_t N = N_arg > 0 ? (uint32_t)N_arg
                           : (uint32_t)(cores * warps * threads) / (uint32_t)(SPLIT);
    if ((uint32_t)threads != LANES) {
        std::cerr << "LANES=" << LANES << " but NUM_THREADS=" << threads << "\n"; return 1; }
    if (N % GLANES != 0) { std::cerr << "N must be a multiple of GLANES\n"; return 1; }
    if (N != (uint32_t)NASSUME) {
        std::cerr << "N=" << N << " but sizing assumes NASSUME=" << NASSUME
                  << "; rebuild with -DNASSUME=" << N << "\n"; return 1; }
    uint32_t blocks_per_core = (uint32_t)warps / (PW * (SPLIT));   // a block is SPLIT warps wide
    if (blocks_per_core == 0) blocks_per_core = 1;
    if (blocks_per_core * (size_t)BUF_WORDS * sizeof(int) > 16384) {
        std::cerr << "scratchpad overflow\n"; return 1; }
    uint32_t NB = N / GLANES;
    size_t NQ       = (size_t)N * NB;                          // queues
    size_t QS_WORDS = NQ * AQ_QSTRIDE_W;   // req_t words, line aligned stride
#if REPLY_TILE
    size_t NRQ_R    = (size_t)N * REPLY_TILE;   // reply queues, keyed by question tile
    size_t RP_SLOTS = (size_t)aq_region_bytes(NRQ_R, (uint64_t)RT_SSTRIDE * 8) / 8;   // RQ_ALIGN pads each queue to odd lines; ISA2: the program's queue layout
    size_t PT_ENT   = 1;
#elif REPLY_AQ
    size_t NRQ_R    = NQ;
    size_t RP_SLOTS = NQ * RQ_CAP;        // reply QUEUE storage, 8 bytes a slot
    size_t PT_ENT   = 1;                  // no run table
#else
    size_t NRQ_R    = NQ;
    size_t RP_SLOTS = (size_t)NB * NB * NCHUNK * CAP2;
    size_t PT_ENT   = (size_t)NB * NB * NCHUNK * TSTRIDE;
#endif
    std::cout << "N=" << N << " actors, TABLE_SIZE=" << TABLE_SIZE << ", M=" << M
              << ", queues=" << NQ << " x " << AQ_QCAP << " words (mean "
              << ((double)M * GLANES / N) << "), piece=" << PIECE << ", chunks=" << NCHUNK
              << ", CAP2=" << CAP2 << ", blocks=" << NB
              << ", qs=" << (QS_WORDS*4.0/(1024*1024)) << " MB"
              << ", rep=" << (2*RP_SLOTS*4.0/(1024*1024)) << " MB"
              << ", combining lines=" << (NQ*64.0/1024) << " KB"
              << (REPLY_AQ ? ", REPLY ON QUEUE" : "")
#if REPLY_TILE
              << ", reply keyed by question tile: " << REPLY_TILE << " x " << RTILE_SZ
              << " (" << (RTILE_SZ*GLANES*4/1024.0) << " KB scratchpad, "
              << ((size_t)N*REPLY_TILE) << " queues)"
#endif
              << ", lmem=" << blocks_per_core << "x" << (BUF_WORDS*4) << " B\n";
    size_t table_bytes = (size_t)N * TABLE_SIZE * sizeof(int);
    size_t qs_bytes    = QS_WORDS * sizeof(req_t);
    size_t tl_bytes    = NQ * sizeof(uint32_t);
#if REPLY_AQ
    size_t rp_bytes    = RP_SLOTS * sizeof(uint64_t);   // reply queue storage
    size_t rt_bytes    = NRQ_R * sizeof(uint32_t);      // reply queue lengths
    size_t pt_bytes    = PT_ENT * sizeof(int);
#else
    size_t rp_bytes    = RP_SLOTS * sizeof(int);
    size_t rt_bytes    = rp_bytes;
    size_t pt_bytes    = PT_ENT * sizeof(int);
#endif
    size_t tgt_bytes   = (size_t)N * M * sizeof(int);
    size_t phase_bytes = (size_t)N * sizeof(phase_cycles_t) + (LANE_TS ? (size_t)N * (SPLIT) * 6 * 8 : 0);
    vx_buffer_h tb, qsb, tlb, rtb, rvb, ptb, tg, ph_b, bar_b, kb, ab_;
    vx_buffer_h tsb, ttb;
    vx_mem_alloc(dev, table_bytes, VX_MEM_READ_WRITE, &tb);
    vx_mem_alloc(dev, qs_bytes,    VX_MEM_READ_WRITE, &qsb);
    vx_mem_alloc(dev, tl_bytes,    VX_MEM_READ_WRITE, &tlb);
    vx_mem_alloc(dev, rp_bytes,    VX_MEM_READ_WRITE, &rtb);
    vx_mem_alloc(dev, rt_bytes,    VX_MEM_READ_WRITE, &rvb);
    vx_mem_alloc(dev, pt_bytes,    VX_MEM_READ_WRITE, &ptb);
    vx_mem_alloc(dev, tgt_bytes,   VX_MEM_READ_WRITE, &tg);
    vx_mem_alloc(dev, phase_bytes, VX_MEM_READ_WRITE, &ph_b);
    vx_mem_alloc(dev, sizeof(uint32_t)*8, VX_MEM_READ_WRITE, &bar_b);
#if TILE_AQ
#if REQ32
    size_t TQ_SLOTS = (size_t)aq_region_bytes((uint64_t)N * TILE_AQ, (uint64_t)TQ_STRIDE32 * 4) / 8;   // ISA2: the program's queue layout
    if (TQ_SLOTS < (size_t)N * TILE_AQ * TQ_STRIDE_W) TQ_SLOTS = (size_t)N * TILE_AQ * TQ_STRIDE_W;
#else
    size_t TQ_SLOTS = (size_t)aq_region_bytes((uint64_t)N * TILE_AQ, (uint64_t)TQ_STRIDE_W * 8) / 8;
#endif
    size_t ts_bytes = TQ_SLOTS * sizeof(uint64_t);
    size_t tt_bytes = (size_t)N * TILE_AQ * sizeof(uint32_t);
    vx_mem_alloc(dev, ts_bytes, VX_MEM_READ_WRITE, &tsb);
    vx_mem_alloc(dev, tt_bytes, VX_MEM_READ_WRITE, &ttb);
#endif
    args_t args;
    vx_mem_address(tb,&args.table_addr);   vx_mem_address(qsb,&args.qs_addr);
    vx_mem_address(tlb,&args.tails_addr);
    vx_mem_address(rtb,&args.rp_tag_addr); vx_mem_address(rvb,&args.rp_val_addr);
    vx_mem_address(ptb,&args.ptbl_addr);
    vx_mem_address(tg,&args.tgt_addr);     vx_mem_address(ph_b,&args.phase_addr);
    vx_mem_address(bar_b,&args.bar_addr);  args.N = N;
    args.ts_addr = 0; args.ttails_addr = 0;
#if TILE_AQ
    vx_mem_address(tsb,&args.ts_addr); vx_mem_address(ttb,&args.ttails_addr);
#if OVF
    vx_buffer_h poolb, ctabb;
    { size_t pool_bytes = (size_t)(OVF_NCH) * (OVF_CH) * sizeof(uint64_t);
      size_t ctab_bytes = (size_t)(NASSUME) * TILE_AQ * (OVF_MAXC) * sizeof(uint64_t);
      vx_mem_alloc(dev, pool_bytes, VX_MEM_READ_WRITE, &poolb);
      vx_mem_alloc(dev, ctab_bytes, VX_MEM_READ_WRITE, &ctabb);
      vx_mem_address(poolb, &args.pool_addr); vx_mem_address(ctabb, &args.ctab_addr);
      std::cout << "  overflow pool: " << (OVF_NCH) << " chunks x " << (OVF_CH) << " words\n"; }
#endif
    { std::vector<char> z(ts_bytes,0); vx_copy_to_dev(tsb, z.data(), 0, ts_bytes); }
    { std::vector<char> z(tt_bytes,0); vx_copy_to_dev(ttb, z.data(), 0, tt_bytes); }
    std::cout << "  tile reorder: " << TILE_AQ << " tiles of " << TILE_SZ
              << " words (" << (TILE_SZ*4/1024.0) << " KB/lane, "
              << (TILE_SZ*4*GLANES/1024.0) << " KB/core), tile queues="
              << (size_t)N*TILE_AQ << " x " << TQ_CAP << ", ts=" << (ts_bytes/(1024.0*1024)) << " MB\n";
#endif
    if (args.qs_addr + qs_bytes > AQ_BASE && args.qs_addr < AQ_BASE + AQ_SIZE) {
        std::cerr << "queue storage overlaps the window\n"; return 1; }
    { std::vector<int> t((size_t)N*TABLE_SIZE);
      for (uint32_t pe=0; pe<N; pe++) for (uint32_t k=0;k<TABLE_SIZE;k++)
          t[(size_t)pe*TABLE_SIZE+k] = (int)(pe*1000000u + k);
      vx_copy_to_dev(tb, t.data(), 0, table_bytes); }
    { std::vector<char> z(qs_bytes,0);   vx_copy_to_dev(qsb, z.data(), 0, qs_bytes); }
    { std::vector<char> z(tl_bytes,0);   vx_copy_to_dev(tlb, z.data(), 0, tl_bytes); }
    { std::vector<char> z(rp_bytes,0);   vx_copy_to_dev(rtb, z.data(), 0, rp_bytes); }
    { std::vector<char> z(rt_bytes,0);   vx_copy_to_dev(rvb, z.data(), 0, rt_bytes); }
    { std::vector<int>  z(PT_ENT, -1);   vx_copy_to_dev(ptb, z.data(), 0, pt_bytes); }
    { std::vector<char> z(tgt_bytes,0);   vx_copy_to_dev(tg,   z.data(), 0, tgt_bytes); }
    { std::vector<char> z(phase_bytes,0); vx_copy_to_dev(ph_b, z.data(), 0, phase_bytes); }
    { uint32_t z[8]={0,0,0,0,0,0,0,0};    vx_copy_to_dev(bar_b, z, 0, sizeof(z)); }
    vx_upload_kernel_file(dev, "kernel.vxbin", &kb);
    vx_upload_bytes(dev, &args, sizeof(args), &ab_);
    vx_start(dev, kb, ab_);
    vx_ready_wait(dev, VX_MAX_TIMEOUT);

    std::vector<int> tgt((size_t)N*M);
    std::vector<phase_cycles_t> ph(N + (LANE_TS ? (N * (SPLIT) * 6 * 8 + sizeof(phase_cycles_t) - 1) / sizeof(phase_cycles_t) : 0)); uint32_t bar[8];
    std::vector<int> ptbl(PT_ENT);
    std::vector<uint32_t> tails(NQ);
    vx_copy_from_dev(tgt.data(), tg, 0, tgt_bytes);
    vx_copy_from_dev(ph.data(), ph_b, 0, phase_bytes);
    vx_copy_from_dev(bar, bar_b, 0, sizeof(bar));
#if !REPLY_AQ
    vx_copy_from_dev(ptbl.data(), ptb, 0, pt_bytes);
#endif
    vx_copy_from_dev(tails.data(), tlb, 0, tl_bytes);

    int tbad = 0;
#if !DIRECT_TILE
    // queue audit: every request appended somewhere, and no queue past its capacity
    // (under DIRECT_TILE the request queues are never used; the tile audit covers it)
    { unsigned long long sum = 0; uint32_t mx = 0, mn = ~0u; size_t used = 0;
      for (size_t q = 0; q < NQ; q++) { sum += tails[q]; mx = std::max(mx, tails[q]);
                                        mn = std::min(mn, tails[q]); used += tails[q] != 0; }
      std::cout << "queue audit: " << sum << " appends (want " << (unsigned long long)N*M
                << "), queues used " << used << "/" << NQ << ", tail min/mean/max = "
                << mn << "/" << (double)sum/NQ << "/" << mx << " (cap " << AQ_QCAP << ")\n";
      if (sum != (unsigned long long)N * M) { std::cout << "  QUEUE: append count wrong\n"; tbad++; }
      if (mx > (uint32_t)AQ_QCAP) { std::cout << "  QUEUE: a queue exceeds AQ_QCAP; raise NCHUNK\n"; tbad++; }
    }
#endif
#if REPLY_AQ && AQ_ON
    // ISA2: the reply counts live in the L2 (the count instruction), not in a tails array; the reply tiles are exact
    // and every tgt entry is checked below, and simx reports DROPPED appends
    long long pcov = (long long)N * M;
    std::cout << "reply queue audit: skipped (ISA2)\n";
#elif REPLY_AQ
    long long pcov = 0;
    { std::vector<uint32_t> rt(NRQ_R);
      vx_copy_from_dev(rt.data(), rvb, 0, rt_bytes);
      unsigned long long sum = 0; uint32_t mx = 0; size_t used = 0;
#if REPLY_TILE
      size_t cap = (size_t)RT_CAP;
#else
      size_t cap = (size_t)RQ_CAP;
#endif
      for (size_t q = 0; q < NRQ_R; q++) { sum += rt[q]; if (rt[q] > mx) mx = rt[q]; used += rt[q] != 0; }
      pcov = (long long)sum;
      std::cout << "reply queue audit: " << sum << " replies (want " << (unsigned long long)N*M
                << "), queues used " << used << "/" << NRQ_R << ", max tail " << mx
                << " (cap " << cap << ")\n";
      if (mx > (uint32_t)cap) { std::cout << "  RQUEUE: a reply queue exceeds its capacity\n"; tbad++; }
    }
#else
    long long pcov = 0;
    for (uint32_t ob = 0; ob < NB && tbad < 8; ob++)
        for (uint32_t ab = 0; ab < NB && tbad < 8; ab++)
            for (uint32_t ch = 0; ch < (uint32_t)NCHUNK; ch++) {
                const int* o = &ptbl[((size_t)(ob*NB + ab) * NCHUNK + ch) * TSTRIDE];
                int prev = 0; bool ok = true;
                for (int k = 0; k <= GLANES; k++) {
                    if (o[k] < prev || o[k] > CAP2) {
                        if (tbad < 8) std::cout << "  PTABLE: owner blk " << ob << " asker blk "
                            << ab << " chunk " << ch << " entry " << k << " = " << o[k]
                            << " after " << prev << "\n";
                        tbad++; ok = false; break;
                    }
                    prev = o[k];
                }
                if (ok && o[0] != 0) { if (tbad < 8) std::cout << "  PTABLE: pool does not start at 0\n"; tbad++; }
                if (ok) pcov += o[GLANES] - o[0];
            }
#endif
    if (pcov != (long long)N * M) {
        std::cout << "  PTABLE: runs cover " << pcov << " replies, want " << (long long)N*M << "\n"; tbad++; }
    std::cout << "table audit: replies " << pcov << ", " << tbad << " problems\n";

    int bad = 0;
#if ORDER_CHECK && !REPLY_AQ
    {
        std::vector<int> rtag(RP_SLOTS), rval(RP_SLOTS);
        vx_copy_from_dev(rtag.data(), rtb, 0, rp_bytes);
        vx_copy_from_dev(rval.data(), rvb, 0, rp_bytes);
        uint64_t checked = 0;
        std::vector<long long> last(N);
        for (uint32_t blk2 = 0; blk2 < NB; blk2++)
          for (uint32_t tid2 = 0; tid2 < (uint32_t)GLANES; tid2++) {
            uint32_t me2 = blk2 * GLANES + tid2;
            std::fill(last.begin(), last.end(), -1);
            for (uint32_t ob = 0; ob < NB; ob++)
              for (uint32_t ch = 0; ch < (uint32_t)NCHUNK; ch++) {
                const int* o = &ptbl[((size_t)(ob*NB + blk2) * NCHUNK + ch) * TSTRIDE];
                size_t rb = ((size_t)(ob*NB + blk2) * NCHUNK + ch) * CAP2;
                for (int i = o[tid2]; i < o[tid2 + 1]; i++) {
                    uint32_t q = (uint32_t)rtag[rb + i];
                    uint32_t owner = (uint32_t)rval[rb + i] / 1000000u;
                    if (owner >= N) { if (bad<8) std::cout << "  bad owner in reply\n"; bad++; continue; }
                    if ((long long)q <= last[owner]) {
                        if (bad<8) std::cout << "  ORDER VIOLATION asker " << me2 << " from owner "
                            << owner << ": q" << q << " arrived after q" << last[owner] << "\n";
                        bad++;
                    }
                    last[owner] = q; checked++;
                }
              }
          }
        std::cout << "order check: " << checked << " replies, " << bad << " out-of-order\n";
    }
#endif
    for (uint32_t s = 0; s < N; s++)
        for (uint32_t i = 0; i < (uint32_t)M; i++) {
            uint32_t dst  = mix(s, 2u*i+1u) % N;
            uint32_t lidx = mix(s, 2u*i+2u) % (uint32_t)TABLE_SIZE;
#if SKEW > 1
            { const uint32_t k_ = skew_key(mix(s, 2u*i+1u), (uint32_t)(N) * (uint32_t)TABLE_SIZE); dst = k_ % (uint32_t)(N); lidx = (k_ / (uint32_t)(N)) % (uint32_t)TABLE_SIZE; }
#endif
            int expect = (int)(dst*1000000u + lidx);
            if (tgt[TGT_IDX(s, i, N)] != expect) {
                if (bad<8) std::cout << "  tgt mismatch actor " << s << " q" << i
                                     << ": got " << tgt[TGT_IDX(s, i, N)] << " want " << expect << "\n";
                bad++;
            }
        }
    double rq_c=0,ba=0,se=0,dr=0,to=0;
    for (uint32_t i=0;i<N;i++){ rq_c+=ph[i].request; ba+=ph[i].barrier; se+=ph[i].serve;
                                dr+=ph[i].drain; to+=ph[i].total; }
    std::cout << "phases (mean cycles/actor): request=" << (uint64_t)(rq_c/N)
              << " barrier=" << (uint64_t)(ba/N) << " serve=" << (uint64_t)(se/N)
              << " drain=" << (uint64_t)(dr/N) << " total=" << (uint64_t)(to/N) << "\n";
    { uint64_t s0=~0ull, e1=0; for (uint32_t i=0;i<N;i++){ s0=std::min(s0,ph[i].t_start); e1=std::max(e1,ph[i].t_end); }
      std::cout << "span (absolute core cycles): first start=" << s0 << " last end=" << e1 << "\n"; }
#if LANE_TS
    { // per-lane segments: ask, wait1, serve, wait2, drain
      const uint64_t* lt = (const uint64_t*)&ph[N]; const uint32_t NL = N * (SPLIT);
      const char* nm[5] = {"ask","wait1","serve","wait2","drain"};
      uint64_t s0=~0ull, e1=0; for (uint32_t l=0;l<NL;l++){ s0=std::min(s0,lt[l*6]); e1=std::max(e1,lt[l*6+5]); }
      std::cout << "LANE_TS lanes=" << NL << " wall=" << (e1-s0) << " start_skew=";
      { uint64_t m=0; for (uint32_t l=0;l<NL;l++) m=std::max(m,lt[l*6]-s0); std::cout << m << "\n"; }
      for (int k=0;k<5;k++){ double sm=0; uint64_t mn=~0ull,mx=0, se=~0ull, ee=0;
        for (uint32_t l=0;l<NL;l++){ uint64_t d=lt[l*6+k+1]-lt[l*6+k]; sm+=d; mn=std::min(mn,d); mx=std::max(mx,d);
                                     se=std::min(se,lt[l*6+k]); ee=std::max(ee,lt[l*6+k+1]); }
        std::cout << "LANE_TS " << nm[k] << " mean=" << (uint64_t)(sm/NL) << " min=" << mn << " max=" << mx
                  << " window=" << (ee-se) << "\n"; } }
#endif
#if PROFILE_SUB
    { double vc=0,vs=0,vp=0,vu=0;
      for (uint32_t i=0;i<N;i++){ vc+=ph[i].v_count; vs+=ph[i].v_scan; vp+=ph[i].v_place; vu+=ph[i].v_pub; }
      std::cout << "  serve sub-stages: count=" << (uint64_t)(vc/N) << " scan=" << (uint64_t)(vs/N)
                << " place=" << (uint64_t)(vp/N) << " publish=" << (uint64_t)(vu/N) << "\n"; }
#endif
    if (bar[7]) { std::cout << "STUCK BARRIERS bar[7]=0x" << std::hex << bar[7] << std::dec << " (nibble i = slot i timed out, count in that nibble)\n";
                  for (int s2 = 0; s2 < 7; s2++) std::cout << "    bar[" << s2 << "] = " << bar[s2] << "\n"; }
    if (bar[2]) std::cout << "POOL OVERFLOW: " << bar[2] << " replies dropped -- raise CAP2\n";
#if TILE_AQ
    { std::vector<uint32_t> tt((size_t)N*TILE_AQ);
      vx_copy_from_dev(tt.data(), ttb, 0, tt_bytes);
      unsigned long long sum=0; uint32_t mx=0;
      for (size_t q=0;q<(size_t)N*TILE_AQ;q++){ sum+=tt[q]; if(tt[q]>mx) mx=tt[q]; }
      std::cout << "tile queue audit: " << sum << " requests (want " << (unsigned long long)N*M
                << "), max tail " << mx << " (cap " << TQ_CAP << ")\n";
      if (sum != (unsigned long long)N*M) { std::cout << "  TQUEUE: count wrong\n"; tbad++; }
      if (!(OVF) && mx > (uint32_t)TQ_CAP) { std::cout << "  TQUEUE: exceeds TQ_CAP\n"; tbad++; }
    }
#endif
    if (bar[3]) std::cout << "QUEUE OVERFLOW: " << bar[3] << " queues longer than AQ_QCAP -- raise NCHUNK\n";
    bool ok = (bad==0 && tbad==0 && bar[2]==0 && bar[3]==0);
    std::cout << "result: total_queries=" << (uint64_t)N*M << " errors=" << bad
              << " table_problems=" << tbad << "\n" << (ok ? "PASSED" : "FAILED") << "\n";
    vx_mem_free(tb); vx_mem_free(qsb); vx_mem_free(tlb); vx_mem_free(rtb); vx_mem_free(rvb);
#if TILE_AQ
    vx_mem_free(tsb); vx_mem_free(ttb);
#endif
    vx_mem_free(ptb); vx_mem_free(tg); vx_mem_free(ph_b); vx_mem_free(bar_b); vx_mem_free(kb); vx_mem_free(ab_);
    vx_dev_close(dev);
    return ok ? 0 : 1;
}
