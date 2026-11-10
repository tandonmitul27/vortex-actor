// Host for pr_aq: builds A2 as the bale_actor pagerank_push_LU tests do (erdos_renyi_random_graph(n, p, UNDIRECTED,
// NOLOOPS), seed 12345, then A2 = L + U), distributes its rows cyclically over P PEs (P = N actors, or 1 for the agi
// baseline), runs the tests' PageRank on the host in double and validates the device's ranks and iteration count.
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
    const uint64_t l_numrows = getenv("PR_LROWS") ? strtoull(getenv("PR_LROWS"), nullptr, 0) : 1000;   // -n
    const int64_t nz_per_row = getenv("PR_NZ") ? strtoll(getenv("PR_NZ"), nullptr, 0) : 35;            // fixed in the tests
    const uint64_t seed = getenv("PR_SEED") ? strtoull(getenv("PR_SEED"), nullptr, 0) : 12345;
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

    // ---- L: the tests' numrows = l_numrows * THREADS and erdos_renyi_prob = 2 (nz_per_row - 1) / numrows;
    // erdos_renyi_random_graph's geometric gaps (Batagelj-Brandes ALG1), UNDIRECTED: columns below the diagonal
    const uint32_t n = (uint32_t)(l_numrows * N);
    double p = (2.0 * (nz_per_row - 1)) / n;
    if (p > 1.0) p = 1.0;
    std::vector<uint32_t> rpl(n + 1, 0), deg(n, 0);
    std::vector<int> lcols;
    { std::mt19937_64 rng(seed); std::uniform_real_distribution<double> U(0.0, 1.0);
      const double D = std::log(1 - p);
      int64_t col = 1 + (int64_t)std::floor(std::log(1 - U(rng)) / D);
      for (uint32_t row = 0; row < n; row++) {
          while (col < (int64_t)row) { lcols.push_back((int)col); deg[row]++; deg[col]++; col += 1 + (int64_t)std::floor(std::log(1 - U(rng)) / D); }
          rpl[row + 1] = (uint32_t)lcols.size();
          col -= row;
      } }
    // ---- A2 = L + U: row i is L's row i (columns < i) then U's (columns > i), both ascending
    std::vector<uint32_t> rp(n + 1, 0);
    for (uint32_t i = 0; i < n; i++) rp[i + 1] = rp[i] + deg[i];
    const uint64_t nnz = rp[n];
    std::vector<int> cols(nnz);
    { std::vector<uint32_t> cur(rp.begin(), rp.end() - 1);
      for (uint32_t r = 0; r < n; r++) for (uint32_t j = rpl[r]; j < rpl[r + 1]; j++) cols[cur[r]++] = lcols[j];   // L part of row r
      for (uint32_t r = 0; r < n; r++) for (uint32_t j = rpl[r]; j < rpl[r + 1]; j++) cols[cur[lcols[j]]++] = (int)r; }   // U part: rows ascending
    std::vector<int>().swap(lcols);

    // ---- rows cyclic over P PEs: PE p's CSR (absolute offsets), PE by PE
    const int P = ACTOR ? (int)N : 1;
    const uint32_t lrows = (n + P - 1) / P;
    std::vector<int> aoff((size_t)P * (lrows + 1)), anz(nnz);
    { uint64_t k = 0;
      for (int pe = 0; pe < P; pe++)
          for (uint32_t l = 0; l <= lrows; l++) {
              aoff[(size_t)pe * (lrows + 1) + l] = (int)k;
              const uint64_t r = (uint64_t)l * P + pe;
              if (l < lrows && r < n) for (uint32_t j = rp[r]; j < rp[r + 1]; j++) anz[k++] = cols[j];
          } }

    // ---- reference: the tests' loop, in double
    std::vector<double> rpr(n, 1.0 / n), rcur(n);
    uint32_t rit = 1;
    double rerr = 0;
    for (;;) {
        std::fill(rcur.begin(), rcur.end(), 0.0);
        for (uint32_t y = 0; y < n; y++) { const double prf = rpr[y] / (double)(rp[y + 1] - rp[y]);
            for (uint32_t k = rp[y]; k < rp[y + 1]; k++) rcur[cols[k]] += prf; }
        rerr = 0;
        for (uint32_t i = 0; i < n; i++) { rcur[i] = ((1 - (DAMPING)) / n) + (DAMPING) * rcur[i]; rerr += std::fabs(rcur[i] - rpr[i]); }
        if (rerr < (ERR_DELTA) || rit >= (uint32_t)(MAXIT)) break;
        rpr = rcur;
        rit++;
    }

    // ---- queue t * N + owner, t = the owner's row tile; messages per queue per iteration (one per nonzero)
    const uint32_t T = ACTOR == 1 ? (lrows + (KW) - 1) / (KW) : 1;
    if (ACTOR == 1 && NOTILE && T != 1) { std::cerr << "NOTILE needs TILE_W >= lrows = " << lrows << "\nFAILED\n"; return 1; }
    if ((uint64_t)N * T > (AQ_SIZE) / 64 - 1) { std::cerr << "too many queues: " << N * T << " of " << (AQ_SIZE) / 64 - 1 << "\nFAILED\n"; return 1; }
    std::vector<uint64_t> per_owner(N, 0), per_q((size_t)N * T, 0);
    for (uint32_t c = 0; c < n; c++) { per_owner[c % N] += deg[c]; per_q[(size_t)(T > 1 ? (c / N) / (KW) : 0) * N + c % N] += deg[c]; }
    uint64_t need = 0, maxc = 0, omax = 0, dmin = UINT64_MAX;
    for (auto m : per_owner) omax = std::max(omax, m);
    for (auto d : deg) dmin = std::min<uint64_t>(dmin, d);
    for (auto m : per_q)
        if (m > (uint64_t)(Q_CAP)) { uint64_t c = (m - (Q_CAP) + (OVF_CH) - 1) / (OVF_CH); need += c; maxc = std::max(maxc, c); }
    std::cout << std::unitbuf << "PR n=" << n << " (l_numrows " << l_numrows << ") nz_per_row=" << nz_per_row << " p=" << p
              << " nnz=" << nnz << " (" << (double)nnz / n << "/row, min degree " << dmin << ")\n  reference: iterations " << rit
              << " error " << rerr << "\n  per owner per iteration: mean " << nnz / N << " max " << omax
              << "  Q_CAP " << Q_CAP << ": overflow chunks " << need << " (max " << maxc << " per queue) ACTOR=" << ACTOR << " P=" << P << " tiles/actor " << T << "\n";

    // ---- device buffers
    auto alloc = [&](size_t bytes, vx_buffer_h* b, uint64_t* addr) {
        vx_mem_alloc(dev, bytes, VX_MEM_READ_WRITE, b); vx_mem_address(*b, addr); };
    args_t args; memset(&args, 0, sizeof(args));
    const size_t nwarps = (size_t)(N / GLANES) * (SPLIT);
    args.ntile = T;
    args.N = N; args.numrows = n; args.lrows = lrows;
    args.ovf_maxc = (uint32_t)maxc + 1; args.ovf_nch = (uint32_t)need + 16;
    vx_buffer_h aob, anb, prb, cub, erb, phb, barb, kb, ab;
    alloc(aoff.size() * 4, &aob, &args.aoff_addr); vx_copy_to_dev(aob, aoff.data(), 0, aoff.size() * 4);
    alloc(std::max<size_t>(nnz, 1) * 4, &anb, &args.anz_addr); vx_copy_to_dev(anb, anz.data(), 0, (size_t)nnz * 4);
    const size_t npr = (size_t)P * lrows;
    { std::vector<double> pr0(npr, 0.0);
      for (int pe = 0; pe < P; pe++) for (uint32_t l = 0; l < lrows; l++) if ((uint64_t)l * P + pe < n) pr0[(size_t)pe * lrows + l] = 1.0 / n;
      alloc(npr * 8, &prb, &args.pr_addr); vx_copy_to_dev(prb, pr0.data(), 0, npr * 8);
      std::fill(pr0.begin(), pr0.end(), 0.0);
      alloc(npr * 8, &cub, &args.cur_addr);   // NOTILE: curr starts at 0 (the handlers add into it in memory)
      if (ACTOR == 1 && NOTILE) { std::vector<double> z(npr, 0.0); vx_copy_to_dev(cub, z.data(), 0, npr * 8); }
      else vx_copy_to_dev(cub, pr0.data(), 0, npr * 8); }
    alloc((nwarps * GLANES + nwarps) * 8, &erb, &args.err_addr);
#if ACTOR == 1
    { vx_buffer_h b; alloc((size_t)aq_region_bytes((uint64_t)N * T, (uint64_t)QS64 * 8), &b, &args.qs_addr)   /* ISA2: the program's queue layout */; }
    { vx_buffer_h b; std::vector<uint32_t> z((size_t)N * T, 0); alloc(z.size() * 4, &b, &args.tails_addr); vx_copy_to_dev(b, z.data(), 0, z.size() * 4); }
    { vx_buffer_h b; alloc((size_t)args.ovf_nch * (OVF_CH) * 8, &b, &args.pool_addr); }
#if SWQ
    { vx_buffer_h b; std::vector<uint64_t> z((size_t)N * T * args.ovf_maxc, 0); alloc(z.size() * 8, &b, &args.ctab_addr); vx_copy_to_dev(b, z.data(), 0, z.size() * 8); }
    { vx_buffer_h b; std::vector<uint32_t> z(16, 0); alloc(z.size() * 4, &b, &args.swq_addr); vx_copy_to_dev(b, z.data(), 0, z.size() * 4); }
#else
    { vx_buffer_h b; alloc((size_t)N * T * args.ovf_maxc * 8, &b, &args.ctab_addr); }
#endif
#endif
    { std::vector<uint8_t> z(nwarps * sizeof(phase_cycles_t), 0);   // zeroed: LANE_RECV lanes add their msgs
      alloc(z.size(), &phb, &args.phase_addr); vx_copy_to_dev(phb, z.data(), 0, z.size()); }
    { uint32_t z[8] = {0,0,0,0,0,0,0,0}; alloc(sizeof(z), &barb, &args.bar_addr); vx_copy_to_dev(barb, z, 0, sizeof(z)); }
    vx_upload_kernel_file(dev, "kernel.vxbin", &kb);
    vx_upload_bytes(dev, &args, sizeof(args), &ab);
    vx_start(dev, kb, ab);
    vx_ready_wait(dev, VX_MAX_TIMEOUT);

    // ---- validate
    std::vector<double> pr(npr); vx_copy_from_dev(pr.data(), prb, 0, npr * 8);
    std::vector<phase_cycles_t> ph(nwarps); vx_copy_from_dev(ph.data(), phb, 0, nwarps * sizeof(phase_cycles_t));
    double s1 = 0, s2 = 0, s3 = 0, st = 0; uint64_t tmax = 0, recvd = 0, itmin = UINT64_MAX, itmax = 0;
    for (auto& x : ph) { s1 += x.send; s2 += x.recv; s3 += x.upd; st += x.total; tmax = std::max(tmax, x.total); recvd += x.msgs;
                         itmin = std::min(itmin, x.iters); itmax = std::max(itmax, x.iters); }
    std::cout << "phases (mean cycles/actor): send=" << (uint64_t)(s1 / nwarps) << " recv=" << (uint64_t)(s2 / nwarps)
              << " upd=" << (uint64_t)(s3 / nwarps) << " total=" << (uint64_t)(st / nwarps) << " max_total=" << tmax
              << " iters=" << itmax << "\n";
    double maxrel = 0, sum = 0;
    for (uint32_t r = 0; r < n; r++) {
        const double g = pr[(size_t)(r % P) * lrows + r / P];
        sum += g;
        maxrel = std::max(maxrel, std::fabs(g - rpr[r]) / rpr[r]);
    }
    const bool ok = itmin == itmax && itmax == rit && maxrel < 1e-9 && std::fabs(sum - 1.0) < 0.001
                    && (!ACTOR || recvd == (uint64_t)rit * nnz);
    std::cout << "validation: iterations " << itmax << " (warps agree " << (itmin == itmax) << ") of " << rit
              << ", max relative rank error " << maxrel << ", rank sum " << sum;
    if (ACTOR) std::cout << ", messages received " << recvd << " of " << (uint64_t)rit * nnz;
    std::cout << "\n";
    vx_dev_close(dev);
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
    return ok ? 0 : 1;
}
