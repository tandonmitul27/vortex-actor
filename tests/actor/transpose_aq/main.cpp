// Host for transpose_aq: builds A as the bale_actor transpose tests do (erdos_renyi_random_graph(n, p, DIRECTED,
// NOLOOPS), seed + 2), distributes its rows cyclically over P PEs (P = N actors, or 1 for the agi baseline),
// transposes it on the host and validates the device's At. Rows of At are compared as sets: the device fills
// a row in arrival order.
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include <vortex.h>
#include "common.h"
#include "../isa2.h"

int main() {
    const uint64_t l_numrows = getenv("TR_LROWS") ? strtoull(getenv("TR_LROWS"), nullptr, 0) : 1000;   // -n
    const int64_t nz_per_row = getenv("TR_NZ") ? strtoll(getenv("TR_NZ"), nullptr, 0) : 10;            // -Z
    const uint64_t seed = getenv("TR_SEED") ? strtoull(getenv("TR_SEED"), nullptr, 0) : 101892;        // -s
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
    const uint32_t N = (uint32_t)(cores * warps * threads) / (SPLIT);
    if (N != (uint32_t)NASSUME || threads != LANES) { std::cerr << "machine gives N=" << N << ", rebuild\nFAILED\n"; return 1; }

    // ---- A: the tests' numrows = l_numrows * THREADS and erdos_renyi_prob = 2 (nz_per_row - 1) / numrows;
    // erdos_renyi_random_graph's geometric gaps (Batagelj-Brandes ALG1) over every column of every row
    const uint32_t n = (uint32_t)(l_numrows * N);
    double p = (2.0 * (nz_per_row - 1)) / n;
    if (p > 1.0) p = 1.0;
    std::vector<int> rp(n + 1, 0), cols;
    { std::mt19937_64 rng(seed + 2); std::uniform_real_distribution<double> U(0.0, 1.0);
      const double D = std::log(1 - p);
      int64_t col = 1 + (int64_t)std::floor(std::log(1 - U(rng)) / D);
      for (uint32_t row = 0; row < n; row++) {
          while (col < (int64_t)n) { cols.push_back((int)col); col += 1 + (int64_t)std::floor(std::log(1 - U(rng)) / D); }
          rp[row + 1] = (int)cols.size();
          col -= n;
      } }
    const uint32_t nnz = (uint32_t)cols.size();

    // ---- rows cyclic over P PEs: PE p's CSR (absolute offsets), PE by PE
    const int P = ACTOR ? (int)N : 1;
    const uint32_t lrows = (n + P - 1) / P;
    std::vector<int> aoff((size_t)P * (lrows + 1)), anz(nnz);
    { uint32_t k = 0;
      for (int pe = 0; pe < P; pe++)
          for (uint32_t l = 0; l <= lrows; l++) {
              aoff[(size_t)pe * (lrows + 1) + l] = (int)k;
              const uint32_t r = l * P + pe;
              if (l < lrows && r < n) for (int j = rp[r]; j < rp[r + 1]; j++) anz[k++] = cols[j];
          } }

    // ---- reference At; messages per owner (each phase sends one per nonzero to owner col % N)
    std::vector<int> tref(n + 1, 0), tnzref(nnz);
    for (auto c : cols) tref[c + 1]++;
    for (uint32_t c = 0; c < n; c++) tref[c + 1] += tref[c];
    { std::vector<int> cur(tref.begin(), tref.end() - 1);
      for (uint32_t r = 0; r < n; r++) for (int j = rp[r]; j < rp[r + 1]; j++) tnzref[cur[cols[j]]++] = (int)r; }
    // queue t * N + owner, t = the owner's column tile
    const uint32_t lnc = (n + N - 1) / N;
    const uint32_t T = ACTOR == 1 ? (lnc + (KW) - 1) / (KW) : 1;
    if (ACTOR == 1 && NOTILE && T != 1) { std::cerr << "NOTILE needs TILE_W >= lnc = " << lnc << "\nFAILED\n"; return 1; }
    if (ACTOR == 1 && !(MSG64) && (uint64_t)n > (1ull << (32 - (TB)))) { std::cerr << "a row id and an in-tile column exceed 32 bits\nFAILED\n"; return 1; }
    if ((uint64_t)N * T > (AQ_SIZE) / 64 - 1) { std::cerr << "too many queues: " << N * T << "\nFAILED\n"; return 1; }
    std::vector<uint64_t> per_owner(N, 0), per_q((size_t)N * T, 0);
    for (uint32_t c = 0; c < n; c++) { const uint64_t m = (uint64_t)(tref[c + 1] - tref[c]);
        per_owner[c % N] += m; per_q[(size_t)(T > 1 ? (c / N) / (KW) : 0) * N + c % N] += m; }
    uint64_t need = 0, maxc = 0, omax = 0;
    for (auto m : per_owner) omax = std::max(omax, m);
    for (auto m : per_q)
        if (m > (uint64_t)(Q_CAP)) { uint64_t c = (m - (Q_CAP) + (OVF_CH) - 1) / (OVF_CH); need += c; maxc = std::max(maxc, c); }
    std::cout << std::unitbuf << "TR n=" << n << " (l_numrows " << l_numrows << ") nz_per_row=" << nz_per_row << " p=" << p
              << " nnz=" << nnz << " (" << (double)nnz / n << "/row)\n  per owner: mean " << nnz / N << " max " << omax
              << "  Q_CAP " << Q_CAP << ": overflow chunks " << need << " (max " << maxc << " per queue) ACTOR=" << ACTOR << " P=" << P << " tiles/actor " << T << "\n";

    // ---- device buffers
    auto alloc = [&](size_t bytes, vx_buffer_h* b, uint64_t* addr) {
        vx_mem_alloc(dev, bytes, VX_MEM_READ_WRITE, b); vx_mem_address(*b, addr); };
    auto zeros = [&](size_t words, uint64_t* addr) { vx_buffer_h b; std::vector<uint32_t> z(words, 0);
        alloc(words * 4, &b, addr); vx_copy_to_dev(b, z.data(), 0, words * 4); };
    args_t args; memset(&args, 0, sizeof(args));
    const size_t nwarps = (size_t)(N / GLANES) * (SPLIT), nthr = nwarps * GLANES;
    args.ntile = T;
    args.N = N; args.numrows = n; args.lrows = lrows; args.lnc = lnc; args.tcap = (uint32_t)std::max<uint64_t>(omax, 1);
    args.ovf_maxc = (uint32_t)maxc + 1; args.ovf_nch = (uint32_t)need + 16;
    vx_buffer_h aob, anb, tob, tnb, phb, barb, kb, ab;
    alloc(aoff.size() * 4, &aob, &args.aoff_addr); vx_copy_to_dev(aob, aoff.data(), 0, aoff.size() * 4);
    alloc(std::max<size_t>(nnz, 1) * 4, &anb, &args.anz_addr); vx_copy_to_dev(anb, anz.data(), 0, (size_t)nnz * 4);
#if ACTOR == 1
    const size_t ntoff = (size_t)N * (lnc + 1), ntnz = (size_t)N * args.tcap;
    zeros((size_t)N * lnc, &args.cnt_addr); zeros((size_t)N * lnc, &args.wrk_addr);
    zeros(nthr, &args.sums_addr);
    { vx_buffer_h b; alloc((size_t)aq_region_bytes((uint64_t)N * T, (uint64_t)QS32 * 4), &b, &args.qs_addr); }   // ISA2: the program's queue layout
    zeros((size_t)N * T, &args.tails_addr);
    { vx_buffer_h b; alloc((size_t)args.ovf_nch * (OVF_CH) * 8, &b, &args.pool_addr); }
#if SWQ
    zeros((size_t)N * T * args.ovf_maxc * 2, &args.ctab_addr); zeros(16, &args.swq_addr);
#else
    { vx_buffer_h b; alloc((size_t)N * T * args.ovf_maxc * 8, &b, &args.ctab_addr); }
#endif
#else
    const size_t ntoff = (size_t)n + 1, ntnz = std::max<size_t>(nnz, 1);
    zeros(n, &args.cnt_addr); zeros(n, &args.wrk_addr);
    zeros(nthr + nwarps, &args.sums_addr);
#endif
    alloc(ntoff * 4, &tob, &args.toff_addr);
    alloc(ntnz * 4, &tnb, &args.tnz_addr);
    { std::vector<uint8_t> z(nwarps * sizeof(phase_cycles_t), 0);   // zeroed: LANE_RECV lanes add their msgs
      alloc(z.size(), &phb, &args.phase_addr); vx_copy_to_dev(phb, z.data(), 0, z.size()); }
    { uint32_t z[8] = {0,0,0,0,0,0,0,0}; alloc(sizeof(z), &barb, &args.bar_addr); vx_copy_to_dev(barb, z, 0, sizeof(z)); }
    vx_upload_kernel_file(dev, "kernel.vxbin", &kb);
    vx_upload_bytes(dev, &args, sizeof(args), &ab);
    vx_start(dev, kb, ab);
    vx_ready_wait(dev, VX_MAX_TIMEOUT);

    // ---- validate
    std::vector<uint32_t> toff(ntoff); vx_copy_from_dev(toff.data(), tob, 0, ntoff * 4);
    std::vector<int> tnz(ntnz); vx_copy_from_dev(tnz.data(), tnb, 0, ntnz * 4);
    std::vector<phase_cycles_t> ph(nwarps); vx_copy_from_dev(ph.data(), phb, 0, nwarps * sizeof(phase_cycles_t));
    double s1 = 0, ss = 0, s2 = 0, st = 0; uint64_t tmax = 0, recvd = 0;
    for (auto& x : ph) { s1 += x.p1; ss += x.scan; s2 += x.p2; st += x.total; tmax = std::max(tmax, x.total); recvd += x.msgs; }
    std::cout << "phases (mean cycles/actor): p1=" << (uint64_t)(s1 / nwarps) << " scan=" << (uint64_t)(ss / nwarps)
              << " p2=" << (uint64_t)(s2 / nwarps) << " total=" << (uint64_t)(st / nwarps) << " max_total=" << tmax << "\n";
#if PROBE_TIMERS
    { double ps = 0, pr = 0; for (auto& x : ph) { ps += x.snd; pr += x.rcv; }
      std::cout << "probe timers (mean per warp): send=" << (uint64_t)(ps / nwarps) << " recv=" << (uint64_t)(pr / nwarps) << "\n"; }
#endif
#if PROBE_APPLYT
    { double h = 0, s = 0; for (auto& x : ph) { h += x.pa_h; s += x.pa_s; }
      std::cout << "probe apply (mean per warp): handler=" << (uint64_t)(h / nwarps) << " hsend=" << (uint64_t)(s / nwarps)
                << " total=" << (uint64_t)(st / nwarps) << "\n"; }
#endif
    uint64_t badoff = 0, badrow = 0;
    std::vector<int> got;
    for (uint32_t c = 0; c < n; c++) {
#if ACTOR == 1
        const size_t o = c % N, l = c / N;
        const uint32_t s = toff[o * (lnc + 1) + l], e = toff[o * (lnc + 1) + l + 1];
        if (l == 0 && s != 0) badoff++;
        const int* seg = &tnz[o * args.tcap];
#else
        const uint32_t s = toff[c], e = toff[c + 1];
        if (s != (uint32_t)tref[c]) badoff++;
        const int* seg = tnz.data();
#endif
        if (e < s || e - s != (uint32_t)(tref[c + 1] - tref[c])) { badoff++; continue; }
        got.assign(seg + s, seg + e); std::sort(got.begin(), got.end());
        if (!std::equal(got.begin(), got.end(), tnzref.begin() + tref[c])) badrow++;
    }
    const bool ok = ((badoff == 0 && badrow == 0) || PROBE_NOAPPLY) && (!ACTOR || recvd == 2ull * nnz);
    std::cout << "validation: rows of At with wrong offsets " << badoff << ", wrong contents " << badrow << " of " << n;
    if (ACTOR) std::cout << ", messages received " << recvd << " of " << 2ull * nnz;
    std::cout << "\n";
    vx_dev_close(dev);
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
    return ok ? 0 : 1;
}
