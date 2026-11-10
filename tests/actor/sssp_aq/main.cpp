// Host for sssp_aq: bale's sssp input (directed Erdos-Renyi, no loops, weights uniform in [0, 1)), distributed by
// vertex v % P; runs bale's Bellman-Ford on the host twice: exactly (the baseline's answer) and with the actor
// version's message rounding (its answer, message counts per queue and rounds, bit for bit), and validates.
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

static inline double as_f64(uint64_t b) { double d; memcpy(&d, &b, 8); return d; }
static inline uint64_t as_u64(double d) { uint64_t b; memcpy(&b, &d, 8); return b; }

int main() {
    const uint64_t l_numrows = getenv("SS_LROWS") ? strtoull(getenv("SS_LROWS"), nullptr, 0) : 1000;
    const double nz = getenv("SS_NZ") ? strtod(getenv("SS_NZ"), nullptr) : 10.0;
    const uint64_t seed = getenv("SS_SEED") ? strtoull(getenv("SS_SEED"), nullptr, 0) : 1;
#if ACTOR == 1
#if !defined(BASELINE) || !BASELINE   // BASELINE: no L2-append window, so no window environment
    const char* env_b = getenv("VORTEX_AQ_BASE"); const char* env_s = getenv("VORTEX_AQ_SIZE");
    if (!env_b || !env_s || strtoull(env_b, nullptr, 0) != (unsigned long long)AQ_BASE || strtoull(env_s, nullptr, 0) != (unsigned long long)AQ_SIZE) {
        std::cerr << "set VORTEX_AQ_BASE=0x" << std::hex << AQ_BASE << " VORTEX_AQ_SIZE=0x" << AQ_SIZE << std::dec << "\n"; return 1; }
#endif
    if (l_numrows != (uint64_t)(SS_L)) { std::cerr << "SS_LROWS " << l_numrows << " != SS_L " << SS_L << ": rebuild with -DSS_L=\nFAILED\n"; return 1; }
#endif
    vx_device_h dev;
    if (vx_dev_open(&dev) != 0) { std::cerr << "vx_dev_open failed\n"; return 1; }
    uint64_t cores, warps, threads;
    vx_dev_caps(dev, VX_CAPS_NUM_CORES, &cores); vx_dev_caps(dev, VX_CAPS_NUM_WARPS, &warps); vx_dev_caps(dev, VX_CAPS_NUM_THREADS, &threads);
    const uint32_t N = (uint32_t)(cores * warps * threads) / (SPLIT);
    if (N != (uint32_t)NASSUME || threads != LANES) { std::cerr << "machine gives N=" << N << ", rebuild\nFAILED\n"; return 1; }

    // ---- graph: every off-diagonal (v, w) with probability p (geometric gaps), weights uniform in [0, 1)
    const uint32_t n = (uint32_t)(l_numrows * N);
    const double p = std::min(1.0, nz / n);
    std::vector<uint32_t> rp(n + 1, 0), cols; std::vector<double> wts;
    { std::mt19937_64 rng(seed); std::uniform_real_distribution<double> U(0.0, 1.0);
      const double D = std::log(1 - p);
      for (uint32_t v = 0; v < n; v++) {
          int64_t c = (int64_t)std::floor(std::log(1 - U(rng)) / D);
          while (c < (int64_t)n) { if (c != (int64_t)v) { cols.push_back((uint32_t)c); wts.push_back(U(rng)); } c += 1 + (int64_t)std::floor(std::log(1 - U(rng)) / D); }
          rp[v + 1] = (uint32_t)cols.size();
      } }
    const uint64_t m = cols.size();
    const uint32_t src = 0;

    // ---- host Bellman-Ford (bale's rounds: the vertices that changed relax their out-edges); RND: the actor
    // version's message rounding. Returns the weights; counts messages per queue ((parity * NT + tile) * N + owner)
    const int P = ACTOR ? (int)N : 1;
    const uint32_t lrows = (n + P - 1) / P;
    auto bf = [&](bool rnd, std::vector<uint64_t>& tent, std::vector<uint64_t>* per_q, uint64_t& msgs, uint32_t& rounds) {
        tent.assign(n, SS_INF); tent[src] = 0;
        std::vector<uint64_t> nm(n, SS_INF); std::vector<uint32_t> f{src}, touched, nf;
        msgs = 0; rounds = 0;
        for (uint32_t r = 0; !f.empty(); r++) {
            bool sent = false;
            for (uint32_t v : f) { const double d = as_f64(tent[v]);
                for (uint32_t e = rp[v]; e < rp[v + 1]; e++) {
                    const uint32_t w = cols[e];
                    uint64_t x = as_u64(d + wts[e]);
                    if (rnd) x = (x + (1ull << ((TB) - 1))) & ~(uint64_t)((KW) - 1);
                    if (per_q) (*per_q)[((size_t)(((r & 1) ^ 1) * (NT)) + (w / N) / (KW)) * N + w % N]++;
                    msgs++; sent = true;
                    if (x < nm[w]) { if (nm[w] == SS_INF) touched.push_back(w); nm[w] = x; } } }
            rounds += sent;
            nf.clear();
            for (uint32_t w : touched) { if (nm[w] < tent[w]) { tent[w] = nm[w]; nf.push_back(w); } nm[w] = SS_INF; }
            touched.clear(); f.swap(nf);
        }
    };
    std::vector<uint64_t> ex, rd, per_q(ACTOR ? (size_t)2 * (NT) * N : 0, 0);
    uint64_t exm, rdm; uint32_t exr, rdr;
    bf(false, ex, nullptr, exm, exr);
    if (ACTOR) bf(true, rd, &per_q, rdm, rdr);
    uint64_t reached = 0; double dmax = 0;
    for (auto t : ex) if (t != SS_INF) { reached++; dmax = std::max(dmax, as_f64(t)); }
    uint64_t need = 0, maxc = 0, qmax = 0;
    for (auto c : per_q) { qmax = std::max(qmax, c); if (c > (uint64_t)(Q_CAP)) { uint64_t k = (c - (Q_CAP) + (OVF_CH) - 1) / (OVF_CH); need += k; maxc = std::max(maxc, k); } }
    if (ACTOR == 1 && (uint64_t)2 * (NT) * N > (AQ_SIZE) / 64 - 1) { std::cerr << "too many queues\nFAILED\n"; return 1; }
    std::cout << std::unitbuf << "SSSP n=" << n << " (l_numrows " << l_numrows << ") nz=" << nz << " m=" << m << " src=" << src
              << " reached=" << reached << " max weight " << dmax << "\n  host Bellman-Ford exact: rounds " << exr << " relaxations " << exm;
    if (ACTOR) std::cout << "; rounded: rounds " << rdr << " relaxations " << rdm << " (max per queue " << qmax << ", Q_CAP " << Q_CAP
                         << ": overflow chunks " << need << ", max " << maxc << ")";
    std::cout << " ACTOR=" << ACTOR << " P=" << P << "\n";

    // ---- device buffers
    auto alloc = [&](size_t bytes, vx_buffer_h* b, uint64_t* addr) {
        vx_mem_alloc(dev, bytes, VX_MEM_READ_WRITE, b); vx_mem_address(*b, addr); };
    args_t args; memset(&args, 0, sizeof(args));
    args.N = N; args.n = n; args.lrows = lrows; args.src = src;
    args.ovf_maxc = (uint32_t)maxc + 1; args.ovf_nch = (uint32_t)need + 16;
    const size_t nwarps = (size_t)(N / GLANES) * (SPLIT);
    vx_buffer_h ob, nbb, wb, tb, frb, phb, barb, kb, ab;
    { std::vector<uint32_t> off((size_t)P * (lrows + 1)), nb(m); std::vector<double> wv(m); size_t k = 0;
      for (int pe = 0; pe < P; pe++)
          for (uint32_t l = 0; l <= lrows; l++) {
              off[(size_t)pe * (lrows + 1) + l] = (uint32_t)k;
              const uint64_t v = (uint64_t)l * P + pe;
              if (l < lrows && v < n) for (uint32_t e = rp[v]; e < rp[v + 1]; e++) { nb[k] = cols[e]; wv[k] = wts[e]; k++; }
          }
      alloc(off.size() * 4, &ob, &args.off_addr); vx_copy_to_dev(ob, off.data(), 0, off.size() * 4);
      alloc(std::max<size_t>(m, 1) * 4, &nbb, &args.nbr_addr); vx_copy_to_dev(nbb, nb.data(), 0, m * 4);
      alloc(std::max<size_t>(m, 1) * 8, &wb, &args.wt_addr); vx_copy_to_dev(wb, wv.data(), 0, m * 8); }
    const size_t nt = (size_t)P * lrows;
    std::vector<uint64_t> tent(nt, SS_INF); tent[(size_t)(src % P) * lrows + src / P] = 0;
    alloc(nt * 8, &tb, &args.tent_addr); vx_copy_to_dev(tb, tent.data(), 0, nt * 8);
#if ACTOR == 1
    { std::vector<uint32_t> fr(nt, 0); fr[(size_t)(src % P) * lrows] = src / P;
      alloc(fr.size() * 4, &frb, &args.fr_addr); vx_copy_to_dev(frb, fr.data(), 0, fr.size() * 4); }
    { vx_buffer_h b; std::vector<uint32_t> z(MAXR, 0); alloc(z.size() * 4, &b, &args.flag_addr); vx_copy_to_dev(b, z.data(), 0, z.size() * 4); }
    if (NOTILE || LANE_RECV) {   // the ablation variants' per-actor sender counts and (NOTILE) round stamps
      vx_buffer_h b; std::vector<uint32_t> z(nt, 0); alloc(z.size() * 4, &b, &args.stamp_addr); vx_copy_to_dev(b, z.data(), 0, z.size() * 4);
      vx_buffer_h c; alloc((size_t)N * 4, &c, &args.cnt_addr); vx_copy_to_dev(c, z.data(), 0, (size_t)N * 4); }
    const size_t nq = (size_t)2 * (NT) * N;
    { vx_buffer_h b; alloc((size_t)aq_region_bytes((uint64_t)nq, (uint64_t)QS64 * 8), &b, &args.qs_addr)   /* ISA2: the program's queue layout */; }
    { vx_buffer_h b; std::vector<uint32_t> z(nq, 0); alloc(z.size() * 4, &b, &args.tails_addr); vx_copy_to_dev(b, z.data(), 0, z.size() * 4);
      vx_buffer_h c; alloc(z.size() * 4, &c, &args.heads_addr); vx_copy_to_dev(c, z.data(), 0, z.size() * 4); }
    { vx_buffer_h b; alloc((size_t)args.ovf_nch * (OVF_CH) * 8, &b, &args.pool_addr); }
#if SWQ
    { vx_buffer_h b; std::vector<uint64_t> z(nq * args.ovf_maxc, 0); alloc(z.size() * 8, &b, &args.ctab_addr); vx_copy_to_dev(b, z.data(), 0, z.size() * 8); }
    { vx_buffer_h b; std::vector<uint32_t> z(16, 0); alloc(z.size() * 4, &b, &args.swq_addr); vx_copy_to_dev(b, z.data(), 0, z.size() * 4); }
#else
    { vx_buffer_h b; alloc(nq * args.ovf_maxc * 8, &b, &args.ctab_addr); }
#endif
#else
    { std::vector<uint32_t> fq((size_t)2 * n, 0); fq[0] = src;
      alloc(fq.size() * 4, &frb, &args.fr_addr); vx_copy_to_dev(frb, fq.data(), 0, fq.size() * 4); }
    { vx_buffer_h b; std::vector<uint32_t> z(n, 0); alloc(z.size() * 4, &b, &args.stamp_addr); vx_copy_to_dev(b, z.data(), 0, z.size() * 4); }
    { vx_buffer_h b; uint32_t c3[3] = {1, 0, 0}; alloc(sizeof(c3), &b, &args.cnt_addr); vx_copy_to_dev(b, c3, 0, sizeof(c3)); }
#endif
    { std::vector<uint8_t> z(nwarps * sizeof(phase_cycles_t), 0);   // zeroed: LANE_RECV lanes add their msgs
      alloc(z.size(), &phb, &args.phase_addr); vx_copy_to_dev(phb, z.data(), 0, z.size()); }
    { uint32_t z[8] = {0,0,0,0,0,0,0,0}; alloc(sizeof(z), &barb, &args.bar_addr); vx_copy_to_dev(barb, z, 0, sizeof(z)); }
    vx_upload_kernel_file(dev, "kernel.vxbin", &kb);
    vx_upload_bytes(dev, &args, sizeof(args), &ab);
    vx_start(dev, kb, ab);
    vx_ready_wait(dev, VX_MAX_TIMEOUT);

    // ---- validate: ACTOR=1 bit for bit against the rounded host run (and within 1e-9 of exact); ACTOR=0 exact
    vx_copy_from_dev(tent.data(), tb, 0, nt * 8);
    std::vector<phase_cycles_t> ph(nwarps); vx_copy_from_dev(ph.data(), phb, 0, nwarps * sizeof(phase_cycles_t));
    uint64_t bad = 0; double maxrel = 0;
    for (uint32_t v = 0; v < n; v++) {
        const uint64_t g = tent[(size_t)(v % P) * lrows + v / P], want = ACTOR ? rd[v] : ex[v];
        if (g != want) { if (bad < 8) std::cout << "  vertex " << v << ": " << as_f64(g) << " want " << as_f64(want) << "\n"; bad++; }
        if (ex[v] != SS_INF && ex[v] != 0) maxrel = std::max(maxrel, std::fabs(as_f64(g) - as_f64(ex[v])) / as_f64(ex[v]));
    }
    double dr = 0, se = 0, to = 0; uint64_t tmax = 0, rmin = UINT64_MAX, rmax = 0, recvd = 0, mdr = 0, mse = 0;
    for (auto& x : ph) { dr += x.drain; se += x.send; to += x.total; tmax = std::max(tmax, x.total); recvd += x.msgs;
                         rmin = std::min(rmin, x.rounds); rmax = std::max(rmax, x.rounds); mdr = std::max(mdr, x.drain); mse = std::max(mse, x.send); }
    std::cout << "phases (mean cycles/actor): drain=" << (uint64_t)(dr / nwarps) << " send=" << (uint64_t)(se / nwarps)
              << " barrier=" << (uint64_t)((to - dr - se) / nwarps) << " total=" << (uint64_t)(to / nwarps) << " max_total=" << tmax
              << " max_drain=" << mdr << " max_send=" << mse << " rounds=" << rmax << "\n";
    const bool ok = bad == 0 && maxrel < 1e-9 && rmin == rmax && (!ACTOR || (rmax == rdr && recvd == rdm));
    std::cout << "validation: " << bad << " wrong weights of " << n << ", max relative error vs exact " << maxrel
              << ", rounds " << rmax << " (warps agree " << (rmin == rmax) << ")";
    if (ACTOR) std::cout << " of " << rdr << ", messages drained " << recvd << " of " << rdm;
    std::cout << "\n";
    vx_dev_close(dev);
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
    return ok ? 0 : 1;
}
