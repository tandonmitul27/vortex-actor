// Host for permute_aq: builds the bale_actor permute tests' inputs (rand_permp(n, seed), rand_permp(n, seed + 1),
// erdos_renyi_random_graph(n, p, DIRECTED, NOLOOPS, seed + 2)), distributes them cyclically over P PEs (P = N
// actors, or 1 for the agi baseline), permutes on the host and validates the device's Ap. Rows of Ap are
// compared as sets: the actor version fills a row in arrival order.
#include <iostream>
#include <vector>
#include <numeric>
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
    const uint64_t l_numrows = getenv("PM_LROWS") ? strtoull(getenv("PM_LROWS"), nullptr, 0) : 1000;   // -n
    const int64_t nz_per_row = getenv("PM_NZ") ? strtoll(getenv("PM_NZ"), nullptr, 0) : 10;            // -Z
    const uint64_t seed = getenv("PM_SEED") ? strtoull(getenv("PM_SEED"), nullptr, 0) : 101892;        // -s
#if ACTOR == 1 && (!defined(BASELINE) || !BASELINE)   // BASELINE: no L2-append window, so no window environment
    const char* env_b = getenv("VORTEX_AQ_BASE"); const char* env_s = getenv("VORTEX_AQ_SIZE");
    if (!env_b || !env_s || strtoull(env_b, nullptr, 0) != (unsigned long long)AQ_BASE || strtoull(env_s, nullptr, 0) != (unsigned long long)AQ_SIZE) {
        std::cerr << "set VORTEX_AQ_BASE=0x" << std::hex << AQ_BASE << " VORTEX_AQ_SIZE=0x" << AQ_SIZE << std::dec << "\n"; return 1; }
#endif
    vx_device_h dev;
    if (vx_dev_open(&dev) != 0) { std::cerr << "vx_dev_open failed\n"; return 1; }
    uint64_t cores, warps, threads;
    vx_dev_caps(dev, VX_CAPS_NUM_CORES, &cores); vx_dev_caps(dev, VX_CAPS_NUM_WARPS, &warps); vx_dev_caps(dev, VX_CAPS_NUM_THREADS, &threads);
    const uint32_t N = (uint32_t)(cores * warps * threads) / (SPLIT);
    if (N != (uint32_t)NASSUME || threads != LANES) { std::cerr << "machine gives N=" << N << ", rebuild\nFAILED\n"; return 1; }

    // ---- the tests' inputs: numrows = l_numrows * THREADS, erdos_renyi_prob = 2 (nz_per_row - 1) / numrows
    const uint32_t n = (uint32_t)(l_numrows * N);
    std::vector<int> rperminv(n), cperminv(n);
    std::iota(rperminv.begin(), rperminv.end(), 0); std::iota(cperminv.begin(), cperminv.end(), 0);
    { std::mt19937_64 g0(seed); std::shuffle(rperminv.begin(), rperminv.end(), g0);
      std::mt19937_64 g1(seed + 1); std::shuffle(cperminv.begin(), cperminv.end(), g1); }
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

    // ---- cyclic over P PEs: A's CSR (absolute offsets), and the permutations' local parts
    const int P = ACTOR ? (int)N : 1;
    const uint32_t lrows = n / P;
    std::vector<int> aoff((size_t)P * (lrows + 1)), anz(nnz), rpi((size_t)P * lrows), cpi((size_t)P * lrows);
    { uint32_t k = 0;
      for (int pe = 0; pe < P; pe++)
          for (uint32_t l = 0; l <= lrows; l++) {
              aoff[(size_t)pe * (lrows + 1) + l] = (int)k;
              if (l == lrows) break;
              const uint32_t r = l * P + pe;
              rpi[(size_t)pe * lrows + l] = rperminv[r]; cpi[(size_t)pe * lrows + l] = cperminv[r];
              for (int j = rp[r]; j < rp[r + 1]; j++) anz[k++] = cols[j];
          } }

    // ---- reference Ap (rows sorted); messages per owner in each request round
    std::vector<int> pref(n + 1, 0), pnzref(nnz);
    for (uint32_t r = 0; r < n; r++) pref[rperminv[r] + 1] = rp[r + 1] - rp[r];
    for (uint32_t i = 0; i < n; i++) pref[i + 1] += pref[i];
    for (uint32_t r = 0; r < n; r++) {
        int* d = &pnzref[pref[rperminv[r]]];
        for (int j = rp[r]; j < rp[r + 1]; j++) *d++ = cperminv[cols[j]];
        std::sort(&pnzref[pref[rperminv[r]]], d);
    }
    // queue keys t * N + owner: round 1 t = 0; round 2 requests t = the column owner's table tile (t < T3), forwards
    // t = T3 + the new row owner's row tile
    std::vector<uint64_t> c2(N, 0);
    int maxlen = 0;
    for (uint32_t r = 0; r < n; r++) { c2[rperminv[r] % N] += (uint64_t)(rp[r + 1] - rp[r]); maxlen = std::max(maxlen, rp[r + 1] - rp[r]); }
    uint64_t tcap = 0;
    for (auto m : c2) tcap = std::max(tcap, m);
    const uint32_t T2 = ACTOR == 1 ? (lrows + (RK) - 1) / (RK) : 1, T3 = ACTOR == 1 ? (lrows + (KW) - 1) / (KW) : 1, TQ = T2 + T3;
    if (ACTOR == 1 && NOTILE && (T2 != 1 || T3 != 1)) { std::cerr << "NOTILE needs TILE_W and RTILE >= lrows = " << lrows << "\nFAILED\n"; return 1; }
    if (ACTOR == 1 && (uint64_t)n > (1ull << (32 - (TB)))) { std::cerr << "a new row id and an in-tile column exceed 32 bits\nFAILED\n"; return 1; }
    if (ACTOR == 1 && (lrows >= 65536 || maxlen >= 65536 || (uint64_t)n > (1ull << (CB)) || (uint64_t)N * TQ > (AQ_SIZE) / 64 - 1)) {
        std::cerr << "a message field does not fit, or too many queues: lrows " << lrows << " maxlen " << maxlen << " n " << n
                  << " (CB " << (CB) << ") queues " << (uint64_t)N * TQ << " of " << (AQ_SIZE) / 64 - 1 << "\nFAILED\n"; return 1; }
    // overflow: round 1 alone; requests and forwards are live together in round 2
    std::vector<uint64_t> c1(N, 0), cq((size_t)N * TQ, 0);
    if (ACTOR == 1) {
    for (uint32_t r = 0; r < n; r++) { const uint32_t nr = rperminv[r]; const uint64_t m = (uint64_t)(rp[r + 1] - rp[r]);
        c1[nr % N]++; cq[(size_t)(T3 + (nr / N) / (RK)) * N + nr % N] += m; }
    for (auto c : cols) cq[(size_t)(((uint32_t)c / N) / (KW)) * N + c % N]++; }
    uint64_t need = 0, maxc = 0;
    for (auto* cv : {&c1, &cq}) { uint64_t nd = 0;
        for (auto m : *cv) if (m > (uint64_t)(Q_CAP)) { uint64_t c = (m - (Q_CAP) + (OVF_CH) - 1) / (OVF_CH); nd += c; maxc = std::max(maxc, c); }
        need = std::max(need, nd); }
    std::cout << std::unitbuf << "PM n=" << n << " (l_numrows " << l_numrows << ") nz_per_row=" << nz_per_row << " p=" << p
              << " nnz=" << nnz << " (" << (double)nnz / n << "/row)\n  per owner max: rows " << *std::max_element(c1.begin(), c1.end())
              << " nonzeros " << tcap << "  Q_CAP " << Q_CAP << ": overflow chunks " << need << " (max " << maxc
              << " per queue) ACTOR=" << ACTOR << " P=" << P << " SWQ=" << SWQ << " NOTILE=" << NOTILE << " LANE_RECV=" << LANE_RECV
              << "\n  tiles/actor: table " << T3 << ", row " << T2
              << " (" << (KW) << " columns, " << (RK) << " rows); queues " << (uint64_t)N * TQ << "\n";

    // ---- device buffers
    auto alloc = [&](size_t bytes, vx_buffer_h* b, uint64_t* addr) {
        vx_mem_alloc(dev, bytes, VX_MEM_READ_WRITE, b); vx_mem_address(*b, addr); };
    auto upload = [&](const void* src, size_t bytes, uint64_t* addr) { vx_buffer_h b;
        alloc(std::max<size_t>(bytes, 4), &b, addr); vx_copy_to_dev(b, src, 0, bytes); };
    auto zeros = [&](size_t words, uint64_t* addr) { std::vector<uint32_t> z(words, 0); upload(z.data(), words * 4, addr); };
    args_t args; memset(&args, 0, sizeof(args));
    const size_t nwarps = (size_t)(N / GLANES) * (SPLIT), nthr = nwarps * GLANES;
    args.ntile = T2; args.ntile3 = T3;
    args.N = N; args.numrows = n; args.lrows = lrows; args.tcap = (uint32_t)std::max<uint64_t>(tcap, 1);
    args.ovf_maxc = (uint32_t)maxc + 1; args.ovf_nch = (uint32_t)need + 16;
    vx_buffer_h pob, pnb, phb, barb, kb, ab;
    upload(aoff.data(), aoff.size() * 4, &args.aoff_addr);
    upload(anz.data(), (size_t)nnz * 4, &args.anz_addr);
    upload(rpi.data(), rpi.size() * 4, &args.rpi_addr);
    upload(cpi.data(), cpi.size() * 4, &args.cpi_addr);
#if ACTOR == 1
    const size_t npoff = (size_t)N * (lrows + 1), npnz = (size_t)N * args.tcap;
    zeros((size_t)N * lrows, &args.cnt_addr);
    zeros(nthr, &args.sums_addr);
    { vx_buffer_h b; alloc((size_t)aq_region_bytes((uint64_t)N * TQ, (uint64_t)QS32 * 4), &b, &args.qs_addr); }   // ISA2: the program's queue layout
    zeros((size_t)N * TQ, &args.tails_addr);
    { vx_buffer_h b; alloc((size_t)args.ovf_nch * (OVF_CH) * 8, &b, &args.pool_addr); }
#if SWQ
    zeros((size_t)N * TQ * args.ovf_maxc * 2, &args.ctab_addr); zeros(16, &args.swq_addr);
#else
    { vx_buffer_h b; alloc((size_t)N * TQ * args.ovf_maxc * 8, &b, &args.ctab_addr); }
#endif
#if NOTILE
    zeros((size_t)N * lrows, &args.wrk_addr);
#endif
#else
    const size_t npoff = (size_t)n + 1, npnz = std::max<size_t>(nnz, 1);
    zeros(n, &args.cnt_addr);
    zeros(nthr + nwarps, &args.sums_addr);
#endif
    alloc(npoff * 4, &pob, &args.poff_addr);
    alloc(npnz * 4, &pnb, &args.pnz_addr);
{ std::vector<uint8_t> z(nwarps * sizeof(phase_cycles_t), 0);   // zeroed: LANE_RECV lanes add their msgs
      alloc(z.size(), &phb, &args.phase_addr); vx_copy_to_dev(phb, z.data(), 0, z.size()); }
    { uint32_t z[NBAR] = {0}; alloc(sizeof(z), &barb, &args.bar_addr); vx_copy_to_dev(barb, z, 0, sizeof(z)); }
    vx_upload_kernel_file(dev, "kernel.vxbin", &kb);
    vx_upload_bytes(dev, &args, sizeof(args), &ab);
    vx_start(dev, kb, ab);
    vx_ready_wait(dev, VX_MAX_TIMEOUT);

    // ---- validate
    std::vector<uint32_t> poff(npoff); vx_copy_from_dev(poff.data(), pob, 0, npoff * 4);
    std::vector<int> pnz(npnz); vx_copy_from_dev(pnz.data(), pnb, 0, npnz * 4);
    std::vector<phase_cycles_t> ph(nwarps); vx_copy_from_dev(ph.data(), phb, 0, nwarps * sizeof(phase_cycles_t));
    double s1 = 0, ss = 0, s2 = 0, s3 = 0, st = 0; uint64_t tmax = 0, recvd = 0;
    for (auto& x : ph) { s1 += x.p1; ss += x.scan; s2 += x.p2; s3 += x.p3; st += x.total; tmax = std::max(tmax, x.total); recvd += x.msgs; }
    std::cout << "phases (mean cycles/actor): p1=" << (uint64_t)(s1 / nwarps) << " scan=" << (uint64_t)(ss / nwarps)
              << " p2=" << (uint64_t)(s2 / nwarps) << " p3=" << (uint64_t)(s3 / nwarps) << " total=" << (uint64_t)(st / nwarps)
              << " max_total=" << tmax << "\n";
#if PROBE_TIMERS
    { double ps = 0, pr = 0; for (auto& x : ph) { ps += x.snd; pr += x.rcv; }
      std::cout << "probe timers (mean per warp): send=" << (uint64_t)(ps / nwarps) << " recv=" << (uint64_t)(pr / nwarps) << "\n"; }
#endif
    uint64_t badoff = 0, badrow = 0;
    std::vector<int> got;
    for (uint32_t r = 0; r < n; r++) {
#if ACTOR == 1
        const size_t o = r % N, l = r / N;
        const uint32_t s = poff[o * (lrows + 1) + l], e = poff[o * (lrows + 1) + l + 1];
        if (l == 0 && s != 0) badoff++;
        const int* seg = &pnz[o * args.tcap];
#else
        const uint32_t s = poff[r], e = poff[r + 1];
        if (s != (uint32_t)pref[r]) badoff++;
        const int* seg = pnz.data();
#endif
        if (e < s || e - s != (uint32_t)(pref[r + 1] - pref[r])) { badoff++; continue; }
        got.assign(seg + s, seg + e); std::sort(got.begin(), got.end());
        if (!std::equal(got.begin(), got.end(), pnzref.begin() + pref[r])) badrow++;
    }
    const uint64_t want = (uint64_t)n + 2ull * nnz;   // row lengths, requests, forwards
    const bool ok = ((badoff == 0 && badrow == 0) || PROBE_NOAPPLY) && (!ACTOR || recvd == want);
    std::cout << "validation: rows of Ap with wrong offsets " << badoff << ", wrong contents " << badrow << " of " << n;
    if (ACTOR) std::cout << ", messages received " << recvd << " of " << want;
    std::cout << "\n";
    vx_dev_close(dev);
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
    return ok ? 0 : 1;
}
