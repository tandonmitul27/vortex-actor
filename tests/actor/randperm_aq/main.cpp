// Host for randperm_aq: n = RP_LROWS * N items (the tests' -n per PE), M = 2n slots; validates that the device's
// perm is a permutation of [0, n), and (ACTOR=1) that every message was received: 2 per throw, plus n in phase 2.
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vortex.h>
#include "common.h"
#include "../isa2.h"

int main() {
    const uint64_t l_numrows = RP_L;   // -n, compile time
    const uint32_t seed = RP_SEED;
#if ACTOR == 1
#if !defined(BASELINE) || !BASELINE   // BASELINE: no L2-append window, so no window environment
    const char* env_b = getenv("VORTEX_AQ_BASE"); const char* env_s = getenv("VORTEX_AQ_SIZE");
    if (!env_b || !env_s || strtoull(env_b, nullptr, 0) != (unsigned long long)AQ_BASE || strtoull(env_s, nullptr, 0) != (unsigned long long)AQ_SIZE) {
        std::cerr << "set VORTEX_AQ_BASE=0x" << std::hex << AQ_BASE << " VORTEX_AQ_SIZE=0x" << AQ_SIZE << std::dec << "\n"; return 1; }
#endif
#endif
    vx_device_h dev;
    if (vx_dev_open(&dev) != 0) { std::cerr << "vx_dev_open failed\n"; return 1; }
    uint64_t cores, warps, threads;
    vx_dev_caps(dev, VX_CAPS_NUM_CORES, &cores); vx_dev_caps(dev, VX_CAPS_NUM_WARPS, &warps); vx_dev_caps(dev, VX_CAPS_NUM_THREADS, &threads);
    const uint32_t N = NASSUME;
    if (cores != NBLK || warps != SPLIT || threads != LANES) { std::cerr << "machine is not NBLK x SPLIT x 16, rebuild\nFAILED\n"; return 1; }

    const uint32_t n = NN, M = MM, lN = LN, lM = LMS;
    // THROW word {sender AB | dart JB | in-tile slot TBS}, queue (slot tile) * N + owner; REPLY queues RBASE + sender
    const uint32_t jb = JB, tbs = TBS, ts = TS, rbase = RBASE, nq = NQ;
    if (getenv("RP_LROWS") && strtoull(getenv("RP_LROWS"), nullptr, 0) != l_numrows) { std::cerr << "RP_LROWS is compile time: build with -DRP_L\nFAILED\n"; return 1; }
    // overflow: phase 1 never empties a queue, so a queue holds every message of every round. The host plays the
    // rounds (items in id order) for the per-queue totals; the device's winners differ, so a margin is added.
    // Phase 2 (after the queues are emptied) brings lN per actor.
    std::vector<uint64_t> qn(nq, 0);
    uint32_t hrounds = 0;
    { std::vector<uint8_t> occ(M, 0); std::vector<uint32_t> pend(n), next;
      for (uint32_t it = 0; it < n; it++) pend[it] = it;
      for (uint32_t k = 0; !pend.empty(); k++, hrounds++) {
          next.clear();
          for (auto it : pend) { const uint32_t r = rp_slot(it, k, seed, M);
              qn[((r >> (AB)) >> tbs) * N + (r & (N - 1))]++; qn[rbase + it % N]++;
              if (occ[r]) next.push_back(it); else occ[r] = 1; }
          pend.swap(next);
      } }
    uint64_t need = 0, maxc = 0, need2 = 0;
    for (auto m : qn) if (m > (uint64_t)(Q_CAP)) { uint64_t c = (m - (Q_CAP) + (OVF_CH) - 1) / (OVF_CH); need += c; maxc = std::max(maxc, c); }
    if (lN > (uint32_t)(Q_CAP)) { uint64_t c = (lN - (Q_CAP) + (OVF_CH) - 1) / (OVF_CH); need2 = c * N; maxc = std::max(maxc, c); }
    need = std::max(need, need2);
    std::cout << std::unitbuf << "RP n=" << n << " (l_numrows " << l_numrows << ") M=" << M << " ACTOR=" << ACTOR << " CLAIM=" << CLAIM << " TILE0=" << TILE0 << " CNTW=" << CNTW << " HWBAR=" << HWBAR
              << "\n  THROW bits sender " << AB << " dart " << jb << " in-tile slot " << tbs << ", slot tiles/owner " << ts
              << "; queues " << nq << "; overflow chunks " << need << " (max " << maxc << " per queue); host rounds " << hrounds << "\n";

    // ---- device buffers
    auto alloc = [&](size_t bytes, vx_buffer_h* b, uint64_t* addr) {
        vx_mem_alloc(dev, std::max<size_t>(bytes, 4), VX_MEM_READ_WRITE, b); vx_mem_address(*b, addr); };
    auto fill = [&](size_t words, uint32_t v, uint64_t* addr) { vx_buffer_h b; std::vector<uint32_t> z(words, v);
        alloc(words * 4, &b, addr); vx_copy_to_dev(b, z.data(), 0, words * 4); };
    args_t args; memset(&args, 0, sizeof(args));
    const size_t nwarps = (size_t)(NBLK) * (SPLIT), nthr = nwarps * GLANES;
    args.N = N; args.n = n; args.M = M; args.lN = lN; args.lM = lM; args.seed = seed;
    args.ts = ts; args.tbs = tbs; args.jb = jb; args.rbase = rbase;
    // the device picks different winners (unfair to late senders: measured 4995 vs host ~4150), so a wide margin
    args.ovf_maxc = (uint32_t)(2 * maxc + 4); args.ovf_nch = (uint32_t)(2 * need + 256);
    vx_buffer_h pmb, phb, barb, kb, ab;
    fill(M, EMPTY, &args.tgt_addr);
    alloc((size_t)n * 4, &pmb, &args.perm_addr);
#if ACTOR == 1
    fill((lN * (size_t)N + 3) / 4, 0, &args.land_addr);
    fill(N, 0, &args.cnts_addr);
    fill(MAXR, 0, &args.fails_addr);
    fill(nq, 0, &args.heads_addr);
    fill(nthr, 0, &args.sums_addr);
    { vx_buffer_h b; alloc((size_t)aq_region_bytes((uint64_t)nq, (uint64_t)QS32 * 4), &b, &args.qs_addr)   /* ISA2: the program's queue layout */; }
    fill(nq, 0, &args.tails_addr);
    { vx_buffer_h b; alloc((size_t)args.ovf_nch * (OVF_CH) * 8, &b, &args.pool_addr); }
#if SWQ
    fill((size_t)nq * args.ovf_maxc * 2, 0, &args.ctab_addr); fill(16, 0, &args.swq_addr);
#else
    { vx_buffer_h b; alloc((size_t)nq * args.ovf_maxc * 8, &b, &args.ctab_addr); }
#endif
#else
    fill(nthr + nwarps, 0, &args.sums_addr);
#endif
    { std::vector<phase_cycles_t> z(nwarps); memset(z.data(), 0, z.size() * sizeof(phase_cycles_t));
      alloc(nwarps * sizeof(phase_cycles_t), &phb, &args.phase_addr); vx_copy_to_dev(phb, z.data(), 0, nwarps * sizeof(phase_cycles_t)); }
    { uint32_t z[8] = {0}; alloc(sizeof(z), &barb, &args.bar_addr); vx_copy_to_dev(barb, z, 0, sizeof(z)); }
    vx_upload_kernel_file(dev, "kernel.vxbin", &kb);
    vx_upload_bytes(dev, &args, sizeof(args), &ab);
    vx_start(dev, kb, ab);
    vx_ready_wait(dev, VX_MAX_TIMEOUT);

    // ---- validate
    std::vector<uint32_t> perm(n); vx_copy_from_dev(perm.data(), pmb, 0, (size_t)n * 4);
    std::vector<phase_cycles_t> ph(nwarps); vx_copy_from_dev(ph.data(), phb, 0, nwarps * sizeof(phase_cycles_t));
    double s1 = 0, s2 = 0, st = 0; uint64_t tmax = 0, recvd = 0, throws = 0, rounds = ph[0].rounds;
    for (auto& x : ph) { s1 += x.p1; s2 += x.p2; st += x.total; tmax = std::max(tmax, x.total); recvd += x.msgs; throws += x.throws; }
    if (PROF) { double pf[5] = {0, 0, 0, 0, 0}; for (auto& x : ph) for (int k = 0; k < 5; k++) pf[k] += x.prof[k];
        std::cout << "p1 steps (mean/warp): throw0=" << (uint64_t)(pf[0] / nwarps) << " serve0=" << (uint64_t)(pf[1] / nwarps)
                  << " serveK=" << (uint64_t)(pf[2] / nwarps) << " reply=" << (uint64_t)(pf[3] / nwarps) << " barrier=" << (uint64_t)(pf[4] / nwarps) << "\n"; }
    std::cout << "phases (mean cycles/actor): p1=" << (uint64_t)(s1 / nwarps) << " p2=" << (uint64_t)(s2 / nwarps)
              << " total=" << (uint64_t)(st / nwarps) << " max_total=" << tmax << "\n";
    std::vector<uint8_t> seen(n, 0); uint64_t bad = 0;
    for (auto v : perm) { if (v >= n || seen[v]) bad++; else seen[v] = 1; }
    const uint64_t want = 2 * throws + n;
    const bool ok = bad == 0 && (!ACTOR || recvd == want);
    std::cout << "validation: perm entries out of range or repeated " << bad << " of " << n << "; throws " << throws
              << " (" << (double)throws / n << " per item)";
    if (ACTOR) std::cout << " in " << rounds << " rounds, messages received " << recvd << " of " << want;
    std::cout << "\n";
    vx_dev_close(dev);
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
    return ok ? 0 : 1;
}
