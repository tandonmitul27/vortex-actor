// Host for topo_aq: builds the bale_actor toposort tests' input (generate_toposort_input: the transpose of
// erdos_renyi_random_graph(n, p, UNDIRECTED, LOOPS, seed 5), rows and columns randomly permuted), distributes it over
// P PEs (P = N actors, or 1 for the agi baseline), runs the level-synchronous toposort on the host for the levels and
// validates the device's permutations (permutations, upper triangular with a full diagonal) and level counts.
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
    const uint64_t l_numrows = getenv("TP_LROWS") ? strtoull(getenv("TP_LROWS"), nullptr, 0) : 1000;   // -n
    const double nz_per_row = getenv("TP_NZ") ? strtod(getenv("TP_NZ"), nullptr) : 10.0;               // -Z
    const uint64_t seed = getenv("TP_SEED") ? strtoull(getenv("TP_SEED"), nullptr, 0) : 5;             // PE 0's rand_seed
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

    // ---- L = erdos_renyi_random_graph(n, p, UNDIRECTED, LOOPS): geometric gaps below the diagonal, then the diagonal;
    // T = L^T (upper triangular): T's row j holds j and every row of L with column j
    const uint32_t n = (uint32_t)(l_numrows * N);
    if (!(MSG64) && (uint64_t)n > (1ull << (32 - TB))) { std::cerr << "n=" << n << " does not fit the 32-bit messages (TILE_W " << TILE_W << ")\nFAILED\n"; return 1; }
    double p = (2.0 * nz_per_row) / (n - 1);
    if (p > 1.0) p = 1.0;
    std::vector<uint32_t> trp(n + 1, 0);                 // T's row pointers
    std::vector<uint32_t> lr, lc;                        // L's nonzeros (row, col), col <= row
    { std::mt19937_64 rng(seed); std::uniform_real_distribution<double> U(0.0, 1.0);
      const double D = std::log(1 - p);
      int64_t col = 1 + (int64_t)std::floor(std::log(1 - U(rng)) / D);
      for (uint32_t row = 0; row < n; row++) {
          while (col < (int64_t)row) { lr.push_back(row); lc.push_back((uint32_t)col); col += 1 + (int64_t)std::floor(std::log(1 - U(rng)) / D); }
          lr.push_back(row); lc.push_back(row);          // LOOPS: the diagonal
          col -= row;
      } }
    const uint64_t nnz = lr.size();
    for (uint64_t k = 0; k < nnz; k++) trp[lc[k] + 1]++;
    for (uint32_t j = 0; j < n; j++) trp[j + 1] += trp[j];
    std::vector<uint32_t> tcol(nnz);
    { std::vector<uint32_t> cur(trp.begin(), trp.end() - 1);
      for (uint64_t k = 0; k < nnz; k++) tcol[cur[lc[k]]++] = lr[k]; }
    std::vector<uint32_t>().swap(lr); std::vector<uint32_t>().swap(lc);
    // ---- M = permute_matrix(T, rperminv, cperminv): T's row i -> row rpi[i], column j -> column cpi[j]
    std::vector<uint32_t> rpi(n), cpi(n);
    for (uint32_t i = 0; i < n; i++) rpi[i] = cpi[i] = i;
    { std::mt19937_64 g1(1230), g2(45); std::shuffle(rpi.begin(), rpi.end(), g1); std::shuffle(cpi.begin(), cpi.end(), g2); }
    std::vector<uint32_t> mrp(n + 1, 0), mcol(nnz);       // M by rows
    { std::vector<uint32_t> inv(n);
      for (uint32_t i = 0; i < n; i++) inv[rpi[i]] = i;
      for (uint32_t r = 0; r < n; r++) mrp[r + 1] = mrp[r] + (trp[inv[r] + 1] - trp[inv[r]]);
      for (uint32_t r = 0; r < n; r++) { uint64_t k = mrp[r]; for (uint32_t e = trp[inv[r]]; e < trp[inv[r] + 1]; e++) mcol[k++] = cpi[tcol[e]]; } }
    std::vector<uint32_t>().swap(trp); std::vector<uint32_t>().swap(tcol);
    std::vector<uint32_t> ccp(n + 1, 0), crow(nnz);       // tmat = M^T: M by columns
    for (uint64_t k = 0; k < nnz; k++) ccp[mcol[k] + 1]++;
    for (uint32_t c = 0; c < n; c++) ccp[c + 1] += ccp[c];
    { std::vector<uint32_t> cur(ccp.begin(), ccp.end() - 1);
      for (uint32_t r = 0; r < n; r++) for (uint32_t k = mrp[r]; k < mrp[r + 1]; k++) crow[cur[mcol[k]]++] = r; }

    // ---- reference: level-synchronous toposort (levels of every row)
    std::vector<uint32_t> rcnt(n), rsum(n, 0), rlev(n, 0);
    for (uint32_t r = 0; r < n; r++) { rcnt[r] = mrp[r + 1] - mrp[r]; for (uint32_t k = mrp[r]; k < mrp[r + 1]; k++) rsum[r] += mcol[k]; }
    const std::vector<uint32_t> cnt0 = rcnt, sum0 = rsum;
    uint32_t nlev = 0;
    { std::vector<uint32_t> curq, nxt;
      for (uint32_t r = 0; r < n; r++) if (rcnt[r] == 1) curq.push_back(r);
      uint64_t done = 0;
      while (!curq.empty()) {
          nxt.clear();
          for (uint32_t r : curq) { rlev[r] = nlev; done++; }
          for (uint32_t r : curq) { const uint32_t c = rsum[r];
              for (uint32_t k = ccp[c]; k < ccp[c + 1]; k++) { const uint32_t rr = crow[k]; rsum[rr] -= c; if (--rcnt[rr] == 1) nxt.push_back(rr); } }
          curq.swap(nxt); nlev++;
      }
      if (done != n) { std::cerr << "reference toposort matched " << done << " of " << n << " rows\nFAILED\n"; return 1; } }
    if (nlev > (uint32_t)(MAXL) || 2 * nlev + 2 > (uint32_t)(MAXR)) { std::cerr << "levels " << nlev << " exceed MAXL/MAXR\nFAILED\n"; return 1; }

    // ---- P PEs: row r and column c of M belong to PE r % P / c % P at local index r / P / c / P
    const int P = ACTOR ? (int)N : 1;
    const uint32_t lrows = (n + P - 1) / P;
    std::vector<uint32_t> toff((size_t)P * (lrows + 1)), tnz(nnz);
    { uint64_t k = 0;
      for (int pe = 0; pe < P; pe++)
          for (uint32_t l = 0; l <= lrows; l++) {
              toff[(size_t)pe * (lrows + 1) + l] = (uint32_t)k;
              const uint64_t c = (uint64_t)l * P + pe;
              if (l < lrows && c < n) for (uint32_t e = ccp[c]; e < ccp[c + 1]; e++) tnz[k++] = crow[e];
          } }
    const size_t npe = (size_t)P * lrows;
    std::vector<uint32_t> dcnt(npe, 0), dsum(npe, 0), iq, iqoff(P + 1, 0);
    for (int pe = 0; pe < P; pe++) {
        for (uint32_t l = 0; l < lrows; l++) { const uint64_t r = (uint64_t)l * P + pe; if (r >= n) continue;
            dcnt[(size_t)pe * lrows + l] = cnt0[r]; dsum[(size_t)pe * lrows + l] = sum0[r];
            if (cnt0[r] == 1) iq.push_back(ACTOR ? l : (uint32_t)r); }
        iqoff[pe + 1] = (uint32_t)iq.size();
    }
    const uint32_t iq0 = (uint32_t)iq.size();

    // ---- queues: MATCH to owner o = q o (each column matched once); DEC to (owner, row tile) = q N + t N + o (one per
    // nonzero of the row); CPERM reuses the DEC queues after the reset (one per column, never past Q_CAP)
    if (ACTOR == 1 && lrows != (uint32_t)(LR)) { std::cerr << "rebuild with -DTP_L=" << lrows << "\nFAILED\n"; return 1; }
    const uint32_t T = ACTOR == 1 ? (lrows + (KW) - 1) / (KW) : 1;
    const uint64_t NQ = (uint64_t)N * (1 + T);
    if (ACTOR == 1 && NQ > (AQ_SIZE) / 64 - 1) { std::cerr << "too many queues: " << NQ << " of " << (AQ_SIZE) / 64 - 1 << "\nFAILED\n"; return 1; }
    std::vector<uint64_t> per_q(NQ, 0);
    for (uint32_t c = 0; c < n; c++) per_q[c % N]++;
    if (ACTOR == 1) for (uint32_t r = 0; r < n; r++) per_q[N + (size_t)((r / N) / (KW)) * N + r % N] += cnt0[r];
    uint64_t need = 0, maxc = 0, qmax = 0;
    for (auto m : per_q) {
        qmax = std::max(qmax, m);
        if (m > (uint64_t)(Q_CAP)) { uint64_t c = (m - (Q_CAP) + (OVF_CH) - 1) / (OVF_CH); need += c; maxc = std::max(maxc, c); }
    }
    uint32_t dmax = 0; for (uint32_t r = 0; r < n; r++) dmax = std::max(dmax, cnt0[r]);
    std::cout << std::unitbuf << "TOPO n=" << n << " (l_numrows " << l_numrows << ") nz_per_row=" << nz_per_row << " p=" << p
              << " nnz=" << nnz << " (" << (double)nnz / n << "/row, max " << dmax << ")\n  reference: levels " << nlev
              << ", level-0 rows " << iq0 << "\n  queues " << NQ << ", deepest (whole run) " << qmax << "  Q_CAP " << Q_CAP
              << ": overflow chunks " << need << " (max " << maxc << " per queue) ACTOR=" << ACTOR << " P=" << P
              << " tiles/actor " << T << "\n";

    // ---- device buffers
    auto alloc = [&](size_t bytes, vx_buffer_h* b, uint64_t* addr) {
        vx_mem_alloc(dev, std::max<size_t>(bytes, 64), VX_MEM_READ_WRITE, b); vx_mem_address(*b, addr); };
    auto up = [&](const void* src, size_t bytes, uint64_t* addr) {
        vx_buffer_h b; alloc(bytes, &b, addr); if (bytes) vx_copy_to_dev(b, src, 0, bytes); return b; };
    args_t args; memset(&args, 0, sizeof(args));
    const size_t nwarps = (size_t)(N / GLANES) * (SPLIT);
    args.N = N; args.numrows = n; args.lrows = lrows; args.ntile = T; args.iq0 = iq0;
    args.ovf_maxc = (uint32_t)maxc + 1; args.ovf_nch = (uint32_t)need + 16;
    up(toff.data(), toff.size() * 4, &args.toff_addr);
    up(tnz.data(), (size_t)nnz * 4, &args.tnz_addr);
    up(dcnt.data(), npe * 4, &args.cnt_addr);
    up(dsum.data(), npe * 4, &args.sum_addr);
    const std::vector<uint32_t> zn(npe, 0);
    vx_buffer_h rpb = up(zn.data(), npe * 4, &args.rperm_addr);
    vx_buffer_h cpb = up(zn.data(), npe * 4, &args.cperm_addr);
    vx_buffer_h lvb = nullptr;
    { std::vector<uint32_t> z(nwarps * GLANES, 0); up(z.data(), z.size() * 4, &args.sums_addr); }
#if ACTOR == 1
    lvb = up(zn.data(), npe * 4, &args.lev_addr);
    up(zn.data(), npe * 4, &args.mat_addr);
    up(iq.data(), iq.size() * 4, &args.iq_addr);
    up(iqoff.data(), iqoff.size() * 4, &args.iqoff_addr);
    { std::vector<uint32_t> z((size_t)(MAXL) * N, 0); up(z.data(), z.size() * 4, &args.hist_addr); }
    { std::vector<uint32_t> z(MAXL, 0); up(z.data(), z.size() * 4, &args.ltot_addr); }
    { std::vector<uint32_t> z(MAXR, 0); up(z.data(), z.size() * 4, &args.flag_addr); }
    { vx_buffer_h b; alloc((size_t)aq_region_bytes((uint64_t)NQ, (uint64_t)QS32 * 4), &b, &args.qs_addr)   /* ISA2: the program's queue layout */; }
    { std::vector<uint32_t> z(NQ, 0); up(z.data(), z.size() * 4, &args.tails_addr); up(z.data(), z.size() * 4, &args.heads_addr); }
    { vx_buffer_h b; alloc((size_t)args.ovf_nch * (OVF_CH) * ((MSG64) ? 8 : 4), &b, &args.pool_addr); }
#if SWQ
    { std::vector<uint32_t> z((size_t)NQ * args.ovf_maxc * 2, 0); up(z.data(), z.size() * 4, &args.ctab_addr); }
    { std::vector<uint32_t> z(16, 0); up(z.data(), z.size() * 4, &args.swq_addr); }
#else
    { vx_buffer_h b; alloc((size_t)NQ * args.ovf_maxc * 8, &b, &args.ctab_addr); }
#endif
#else
    { std::vector<uint32_t> q(n, 0); std::copy(iq.begin(), iq.end(), q.begin()); up(q.data(), (size_t)n * 4, &args.iq_addr); }
    { uint32_t c[2] = {0, iq0}; up(c, sizeof(c), &args.ctr_addr); }
#endif
    vx_buffer_h phb, barb, kb, ab;
    { std::vector<uint8_t> z(nwarps * sizeof(phase_cycles_t), 0);   // zeroed: LANE_RECV lanes add their msgs
      alloc(z.size(), &phb, &args.phase_addr); vx_copy_to_dev(phb, z.data(), 0, z.size()); }
    { uint32_t z[8] = {0,0,0,0,0,0,0,0}; alloc(sizeof(z), &barb, &args.bar_addr); vx_copy_to_dev(barb, z, 0, sizeof(z)); }
    vx_upload_kernel_file(dev, "kernel.vxbin", &kb);
    vx_upload_bytes(dev, &args, sizeof(args), &ab);
    vx_start(dev, kb, ab);
    vx_ready_wait(dev, VX_MAX_TIMEOUT);

    // ---- validate
    std::vector<uint32_t> drp(npe), dcp(npe);
    vx_copy_from_dev(drp.data(), rpb, 0, npe * 4); vx_copy_from_dev(dcp.data(), cpb, 0, npe * 4);
    std::vector<phase_cycles_t> ph(nwarps); vx_copy_from_dev(ph.data(), phb, 0, nwarps * sizeof(phase_cycles_t));
    double s1 = 0, s2 = 0, st = 0, sp[8] = {0, 0, 0, 0, 0, 0, 0, 0}; uint64_t tmax = 0, recvd = 0, lvmin = UINT64_MAX, lvmax = 0, rdmax = 0;
    for (auto& x : ph) { s1 += x.p1; s2 += x.p2; st += x.total; tmax = std::max(tmax, x.total); recvd += x.msgs;
                         lvmin = std::min(lvmin, x.levels); lvmax = std::max(lvmax, x.levels); rdmax = std::max(rdmax, x.rounds);
                         for (int k = 0; k < 8; k++) sp[k] += x.prof[k]; }
    std::cout << "phases (mean cycles/actor): levels=" << (uint64_t)(s1 / nwarps) << " perm=" << (uint64_t)(s2 / nwarps)
              << " total=" << (uint64_t)(st / nwarps) << " max_total=" << tmax << " nlev=" << lvmax << " rounds=" << rdmax;
    if (PROF) std::cout << " match=" << (uint64_t)(sp[0] / nwarps) << " dec=" << (uint64_t)(sp[1] / nwarps) << " bar=" << (uint64_t)(sp[2] / nwarps)
                         << " m_state=" << (uint64_t)(sp[3] / nwarps) << " m_drain=" << (uint64_t)(sp[4] / nwarps)
                         << " d_state=" << (uint64_t)(sp[5] / nwarps) << " d_drain=" << (uint64_t)(sp[6] / nwarps);
    std::cout << "\n";
#if PROBE_APPLYT
    { double h = 0, s = 0, r = 0, x = 0; for (auto& y : ph) { h += y.pa_h; s += y.pa_s; r += y.pa_r; x += y.pa_x; }
      std::cout << "probe apply (mean per warp): handler=" << (uint64_t)(h / nwarps) << " hsend=" << (uint64_t)(s / nwarps)
                << " recv=" << (uint64_t)(r / nwarps) << " send=" << (uint64_t)(x / nwarps) << " total=" << (uint64_t)(st / nwarps) << "\n"; }
#endif
#if ROUND1
    { double r1 = 0; for (auto& y : ph) r1 += y.r1; std::cout << "probe round1 send=" << (uint64_t)(r1 / nwarps) << "\n"; }
#endif
    std::vector<uint32_t> rperm(n), cperm(n);
    for (uint32_t r = 0; r < n; r++) { rperm[r] = drp[(size_t)(r % P) * lrows + r / P]; cperm[r] = dcp[(size_t)(r % P) * lrows + r / P]; }
    bool perm_ok = true;
    { std::vector<uint8_t> sr(n, 0), sc(n, 0);
      for (uint32_t i = 0; i < n; i++) {
          if (rperm[i] >= n || sr[rperm[i]]++) perm_ok = false;
          if (cperm[i] >= n || sc[cperm[i]]++) perm_ok = false; } }
    uint64_t below = 0, diag = 0;
    if (perm_ok)
        for (uint32_t r = 0; r < n; r++) for (uint32_t k = mrp[r]; k < mrp[r + 1]; k++) {
            const uint32_t pr = rperm[r], pc = cperm[mcol[k]];
            below += pr > pc; diag += pr == pc; }
    uint64_t levbad = 0;
#if ACTOR == 1
    { std::vector<uint32_t> dl(npe); vx_copy_from_dev(dl.data(), lvb, 0, npe * 4);
      for (uint32_t r = 0; r < n; r++) levbad += dl[(size_t)(r % P) * lrows + r / P] != rlev[r]; }
#else
    (void)lvb;
#endif
    const uint64_t want = ACTOR ? (uint64_t)n + nnz + n : 0;
    const bool ok = perm_ok && below == 0 && diag == n && lvmin == lvmax && lvmax == nlev && levbad == 0 && recvd == want;
    std::cout << "validation: permutations " << perm_ok << ", nonzeros below the diagonal " << below << ", diagonal " << diag
              << " of " << n << ", levels " << lvmax << " (warps agree " << (lvmin == lvmax) << ") of " << nlev;
    if (ACTOR) std::cout << ", row levels wrong " << levbad << ", messages received " << recvd << " of " << want;
    std::cout << "\n";
    vx_dev_close(dev);
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
    return ok ? 0 : 1;
}
