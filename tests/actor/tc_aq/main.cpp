// Host for tc_aq: reads L from a Matrix Market file (env TC_MTX) the way bale-gpu's
// read_mtx_coo_host + coo_to_csr do, for P PEs (P = N actors, or 1 for the agp baseline),
// counts the triangles on the host (agp algorithm) and validates the device count.
#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vortex.h>
#include "common.h"
#include "../isa2.h"

struct Coo { int row, col; };

// bale-gpu read_mtx_coo_host for PE mype of npes; the value column, if any, is skipped
static void read_mtx(const char* fn, int npes, int mype, int* numrows, int* numcols, int* nnz, int* lnumrows,
                     std::vector<Coo>& coo) {
    FILE* fp = fopen(fn, "r");
    if (!fp) { std::cerr << "Could not open " << fn << "\n"; exit(1); }
    char line[512];
    do { if (!fgets(line, sizeof(line), fp)) { std::cerr << "Invalid MTX header\n"; exit(1); } } while (line[0] == '%');
    sscanf(line, "%d %d %d", numrows, numcols, nnz);
    coo.clear(); *lnumrows = 0;
    for (int i = 0; i < *nnz; i++) {
        int r, c;
        if (!fgets(line, sizeof(line), fp) || sscanf(line, "%d %d", &r, &c) != 2) { std::cerr << "short MTX\n"; exit(1); }
        if ((r - 1) % npes == mype) {
            int lrow = (r - 1) / npes;
            coo.push_back({lrow, c - 1});
            if (lrow > *lnumrows) *lnumrows = lrow;
        }
    }
    (*lnumrows) += 1;
    fclose(fp);
}

// bale-gpu coo_to_csr: offsets are written at base + local row, absolute into lnonzero from nzbase
static void coo_to_csr(int lrows, const std::vector<Coo>& coo, int* loffset, std::vector<int>& lnonzero, int nzbase) {
    std::vector<int> off(lrows + 1, 0);
    for (auto& e : coo) off[e.row + 1]++;
    for (int i = 0; i < lrows; i++) off[i + 1] += off[i];
    std::vector<int> cursor(off.begin(), off.end() - 1);
    for (auto& e : coo) lnonzero[nzbase + cursor[e.row]++] = e.col;
    for (int i = 0; i < lrows; i++) std::sort(lnonzero.begin() + nzbase + off[i], lnonzero.begin() + nzbase + off[i + 1]);
    for (int i = 0; i <= lrows; i++) loffset[i] = nzbase + off[i];
}

int main() {
    const char* fn = getenv("TC_MTX");
    if (!fn) { std::cerr << "set TC_MTX=<lower triangular .mtx>\nFAILED\n"; return 1; }
#if ACTOR == 1 || ACTOR == 4
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

    // ---- the graph, for P PEs
    const bool MSG = ACTOR == 1 || ACTOR == 2 || ACTOR == 4;    // the actor versions (P = N); agp and Polak run on one PE
    const int P = MSG ? (int)N : 1;
    int numrows = 0, numcols = 0, nnz = 0;
    std::vector<Coo> all;
    { int lr; read_mtx(fn, 1, 0, &numrows, &numcols, &nnz, &lr, all);
      size_t bad = 0; for (auto& e : all) bad += (e.col >= e.row);
      if (numrows != numcols || bad) { std::cerr << "input must be square and strictly lower triangular (" << bad << " entries are not)\nFAILED\n"; return 1; } }
    // TC_DEGORD=1: Polak's degree order. Relabel by descending (degree, id), so row u of L holds exactly the
    // neighbours Polak's list of u holds (higher degree, or equal degree and larger id)
    const bool degord = getenv("TC_DEGORD") && atoi(getenv("TC_DEGORD"));
    if (degord) {
        std::vector<int> deg(numrows, 0), ord(numrows), nid(numrows);
        for (auto& e : all) { deg[e.row]++; deg[e.col]++; }
        for (int i = 0; i < numrows; i++) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](int x, int y) { return deg[x] != deg[y] ? deg[x] > deg[y] : x > y; });
        for (int i = 0; i < numrows; i++) nid[ord[i]] = i;
        for (auto& e : all) { int r = nid[e.row], c = nid[e.col]; e.row = std::max(r, c); e.col = std::min(r, c); }
    }
    const int lrows = (numrows + P - 1) / P;
    std::vector<int> loffset((size_t)P * (lrows + 1)), lnonzero(nnz);
    { int nzbase = 0; std::vector<Coo> c;
      for (int p = 0; p < P; p++) {      // read_mtx's row filter for PE p, on the one read
          c.clear(); for (auto& e : all) if (e.row % P == p) c.push_back({e.row / P, e.col});
          coo_to_csr(lrows, c, &loffset[(size_t)p * (lrows + 1)], lnonzero, nzbase);
          nzbase += (int)c.size(); } }
    auto row = [&](int u, int* s, int* e) { int p = u % P, l = u / P;
        *s = loffset[(size_t)p * (lrows + 1) + l]; *e = loffset[(size_t)p * (lrows + 1) + l + 1]; };

    // ---- reference: agp on the host; messages per owner and per sender (push)
    uint64_t ref = 0, wedges = 0, maxdeg = 0;
    // MSG32: w takes wbits, v_local lb bits; past 32, the top tbits of v_local pick one of the owner's 1 << tbits queues
    auto bitw = [](uint64_t x) { uint32_t b = 0; while (x) { b++; x >>= 1; } return b; };
    const uint32_t wbits = std::max<uint32_t>(1, bitw((uint64_t)numrows - 1)), lb = bitw((uint64_t)lrows - 1);
    const uint32_t tbits = MSG32 && lb + wbits > 32 ? lb + wbits - 32 : 0, vbits = lb - tbits;
#if RTILE
    // RTILE: each owner's rows cut into tiles of <= TS_W words (offsets + columns) and <= 1 << rbits rows; a row
    // longer than a tile is a tile of its own in global memory. rmap[v] = v's queue << 10 | v's in-tile row.
    const uint32_t rbits = std::min<uint32_t>(LMB, 31 - wbits);   // key {row, w} < 2^31: never HS_EMPTY
    std::vector<uint32_t> qofs(N + 1, 0), tdesc, rmap(std::max(numrows, 1), 0);
    uint32_t nlong = 0;
    for (uint32_t o = 0; o < N; o++) {
        const int* lo = &loffset[(size_t)o * (lrows + 1)];
        for (uint32_t r0 = 0, r1; r0 < (uint32_t)lrows; r0 = r1) {
            const uint32_t q = (uint32_t)tdesc.size() / 2;
            // RTILE=1: offsets + columns <= TS_W words; RTILE 2/3: <= HMAX edges
            auto fits = [&](uint32_t a_, uint32_t b_) { const uint32_t e = (uint32_t)(lo[b_] - lo[a_]);
                return RTILE >= 2 ? e <= (uint32_t)(HMAX) : (b_ - a_ + 1) + e <= (uint32_t)(TS_W); };
            if (!fits(r0, r0 + 1)) { r1 = r0 + 1; tdesc.push_back(r0); tdesc.push_back(0x80000001u); nlong++; }
            else { r1 = r0 + 1;
                   while (r1 < (uint32_t)lrows && r1 - r0 < (1u << rbits) && fits(r0, r1 + 1)) r1++;
                   tdesc.push_back(r0); tdesc.push_back(r1 - r0); }
            for (uint32_t r = r0; r < r1; r++) { const uint64_t v = (uint64_t)r * N + o; if (v < (uint64_t)numrows) rmap[v] = q << (LMB) | (r - r0); }
        }
        qofs[o + 1] = (uint32_t)tdesc.size() / 2;
    }
    const uint32_t NQ = qofs[N];
    if (NQ > (uint32_t)(AQ_SIZE) / 64 - 1 || NQ >= (1u << 22)) { std::cerr << "RTILE: " << NQ << " queues exceed AQ_SIZE\nFAILED\n"; return 1; }
    std::vector<uint32_t> lmap(std::max(nnz, 1), 0);
    for (int i = 0; i < nnz; i++) lmap[i] = rmap[lnonzero[i]];
#if SEDGE
    // SEDGE: i's position in its row in bits 24..31 (255: the kernel finds the row); the queue keeps bits 10..23
    if (NQ >= (1u << (24 - (LMB)))) { std::cerr << "SEDGE: " << NQ << " queues exceed " << 24 - (LMB) << " bits\nFAILED\n"; return 1; }
    for (uint32_t o = 0; o < N; o++) { const int* lo = &loffset[(size_t)o * (lrows + 1)];
        for (int r = 0; r < lrows; r++) for (int i = lo[r]; i < lo[r + 1]; i++) lmap[i] |= (uint32_t)std::min(i - lo[r], (int)(SEDGE_PMAX)) << 24; }
#endif
#if RTILE == 2
    // the tiles' hash images: edge {in-tile row, w} at HS_HASH, linear probing
    std::vector<uint32_t> timg((size_t)NQ * (HS), HS_EMPTY);
    for (uint32_t o = 0; o < N; o++) {
        const int* lo = &loffset[(size_t)o * (lrows + 1)];
        for (uint32_t q = qofs[o]; q < qofs[o + 1]; q++) {
            const uint32_t r0 = tdesc[2 * q], nr = tdesc[2 * q + 1];
            if (nr >> 31) continue;
            uint32_t* T = &timg[(size_t)q * (HS)];
            for (uint32_t r = r0; r < r0 + nr; r++)
                for (int j = lo[r]; j < lo[r + 1]; j++) {
                    const uint32_t k = (r - r0) << wbits | (uint32_t)lnonzero[j];
                    uint32_t h = HS_HASH(k); while (T[h] != HS_EMPTY) h = (h + 1) & ((HS) - 1);
                    T[h] = k; } } }
#endif
#else
    const uint32_t NQ = (uint32_t)N << tbits;
#if SEDGE
    // SEDGE without tiles (piece B off): lmap[i] = i's position in its row in bits 24..31 only
    std::vector<uint32_t> lmap(std::max(nnz, 1), 0);
    for (uint32_t o = 0; o < N; o++) { const int* lo = &loffset[(size_t)o * (lrows + 1)];
        for (int r = 0; r < lrows; r++) for (int i = lo[r]; i < lo[r + 1]; i++) lmap[i] = (uint32_t)std::min(i - lo[r], (int)(SEDGE_PMAX)) << 24; }
#endif
#endif
    std::vector<uint64_t> per_owner(N, 0), per_sender(N, 0), per_q(NQ, 0);
    for (int u = 0; u < numrows; u++) {
        int us, ue; row(u, &us, &ue);
        maxdeg = std::max<uint64_t>(maxdeg, (uint64_t)(ue - us));
        per_sender[u % N] += (uint64_t)(ue - us) * (ue - us - 1) / 2;
        for (int j = us; j < ue; j++) {
            int v = lnonzero[j], vs, ve; row(v, &vs, &ve);
            per_owner[v % N] += (uint64_t)(j - us); wedges += (uint64_t)(j - us);
#if RTILE
            per_q[rmap[v] >> 10] += (uint64_t)(j - us);
#else
            per_q[((uint32_t)(v % N) << tbits) + ((uint32_t)(v / N) >> vbits)] += (uint64_t)(j - us);
#endif
            int i1 = us, i2 = vs;
            while (i1 < ue && i2 < ve) { int d = lnonzero[i1] - lnonzero[i2]; if (d == 0) { ref++; i1++; i2++; } else if (d < 0) i1++; else i2++; }
        }
    }
    // ACTOR=4 (bulk-pull): a request per edge (u, v) whose v another actor owns; owner(u) receives L[v]
    std::vector<uint64_t> dsz(N, 0), hsz(N, 0); uint64_t nreq = 0, dwords = 0;
    if (ACTOR == 4) {
        std::fill(per_q.begin(), per_q.end(), 0);
        for (int u = 0; u < numrows; u++) { int us, ue; row(u, &us, &ue);
            for (int j = us; j < ue; j++) { const int v = lnonzero[j]; if (v % (int)N == u % (int)N) continue;
                int vs, ve; row(v, &vs, &ve);
                per_q[v % N]++; dsz[u % N] += (uint64_t)(ve - vs); hsz[u % N]++; nreq++; dwords += (uint64_t)(ve - vs); } }
    }
    const uint64_t nmsg = ACTOR == 4 ? nreq : wedges;
    auto stats = [&](const std::vector<uint64_t>& x, double* mean, double* sd, uint64_t* mx) {
        double s = 0, s2 = 0; *mx = 0; for (auto y : x) { s += y; s2 += (double)y * y; *mx = std::max(*mx, y); }
        *mean = s / x.size(); *sd = std::sqrt(std::max(0.0, s2 / x.size() - *mean * *mean)); };
    double om, osd, sm, ssd; uint64_t omx, smx; stats(per_owner, &om, &osd, &omx); stats(per_sender, &sm, &ssd, &smx);
    uint64_t need = 0, maxc = 0;
    for (auto m : per_q) if (m > (uint64_t)(Q_CAP)) { uint64_t c = (m - (Q_CAP) + (OVF_CH) - 1) / (OVF_CH); need += c; maxc = std::max(maxc, c); }
    std::cout << std::unitbuf << "TC " << fn << (degord ? " DEGORD" : "") << " rows=" << numrows << " nnz=" << nnz << " maxdeg(L)=" << maxdeg << " triangles(ref)=" << ref
              << " messages=" << wedges << "\n  per owner: mean " << (uint64_t)om << " sd " << (uint64_t)osd << " max " << omx
              << " (max/mean " << (om > 0 ? omx / om : 0) << ")  per sender: mean " << (uint64_t)sm << " sd " << (uint64_t)ssd << " max " << smx
              << "\n  Q_CAP " << Q_CAP << ": overflow chunks " << need << " (max " << maxc << " per queue) ACTOR=" << ACTOR << " P=" << P
              << " MSG32=" << MSG32 << " wbits=" << wbits << " vbits=" << vbits << " queues/owner=" << (1u << tbits) << "\n";
#if RTILE
    std::cout << "  RTILE: " << NQ << " tiles (" << nlong << " long rows in global memory), in-tile row bits " << rbits << "\n";
#endif
    if (ACTOR == 4) std::cout << "  bulk-pull: requests " << nreq << ", row words sent " << dwords << "\n";
    const size_t nwarps = (size_t)(N / GLANES) * (SPLIT);

#if BAL
    // ---- balanced work: weight of pair source jj of row u is jj + 1 (ACTOR=1: jj messages) or jj + d(v) + 1
    // (agp: intersect's bound); a row heavier than W/(N*16) is cut into chunks of ~W/(4N), placed largest first
    // on the least loaded sender: actor u % N (ACTOR=1), or the warp of thread u % (N*SPLIT) (agp)
    const uint32_t nsend = ACTOR ? N : (uint32_t)nwarps;
    auto sender = [&](int u) { return ACTOR ? (uint32_t)u % N : ((uint32_t)u % (N * (SPLIT))) / GLANES; };
    auto wt = [&](int us, int jj) { uint64_t w = (uint64_t)jj + 1;
        if (!ACTOR) { int vs, ve; row(lnonzero[us + jj], &vs, &ve); w += (uint64_t)(ve - vs); } return w; };
    std::vector<uint64_t> wrow(numrows, 0); uint64_t W = 0;
    for (int u = 0; u < numrows; u++) { int us, ue; row(u, &us, &ue); for (int jj = 0; jj < ue - us; jj++) wrow[u] += wt(us, jj); W += wrow[u]; }
    const uint64_t th_row = W / ((uint64_t)N * GLANES), th_ch = std::max<uint64_t>(1, W / (4ull * N));
    struct Chunk { uint64_t w; int u; uint32_t ja, jb, s; };
    std::vector<Chunk> chs; std::vector<uint64_t> load(nsend, 0), load0(nsend, 0);
    std::vector<int> rend((size_t)P * lrows);
    uint32_t nsplit = 0;
    for (int u = 0; u < numrows; u++) {
        int us, ue; row(u, &us, &ue);
        load0[sender(u)] += wrow[u];
        const bool split = wrow[u] > th_row;
        rend[(size_t)(u % P) * lrows + u / P] = split ? us : ue;
        if (!split) { load[sender(u)] += wrow[u]; continue; }
        nsplit++; uint64_t acc = 0; uint32_t ja = 0;
        for (int jj = 0; jj < ue - us; jj++) {
            acc += wt(us, jj);
            if (acc >= th_ch || jj == ue - us - 1) { chs.push_back({acc, u, ja, (uint32_t)jj + 1, 0}); acc = 0; ja = (uint32_t)jj + 1; }
        }
    }
    for (int u = numrows; u < P * lrows; u++) rend[(size_t)(u % P) * lrows + u / P] = 0;   // padding rows: empty
    std::stable_sort(chs.begin(), chs.end(), [](const Chunk& x, const Chunk& y) { return x.w > y.w; });
    for (auto& c : chs) { c.s = (uint32_t)(std::min_element(load.begin(), load.end()) - load.begin()); load[c.s] += c.w; }
    std::stable_sort(chs.begin(), chs.end(), [](const Chunk& x, const Chunk& y) { return x.s < y.s; });
    std::vector<uint32_t> chv(std::max<size_t>(chs.size(), 1) * 3, 0), chofs(nsend + 1, 0); std::vector<int> repv;
    for (size_t i = 0; i < chs.size(); i++) {
        int us, ue; row(chs[i].u, &us, &ue);
        chofs[chs[i].s + 1]++;
        if (ACTOR) { chv[3 * i] = (uint32_t)repv.size(); repv.insert(repv.end(), lnonzero.begin() + us, lnonzero.begin() + us + chs[i].jb); }
        else chv[3 * i] = (uint32_t)chs[i].u;
        chv[3 * i + 1] = chs[i].ja; chv[3 * i + 2] = chs[i].jb;
    }
    for (uint32_t s = 0; s < nsend; s++) chofs[s + 1] += chofs[s];
    { const double mean = (double)W / nsend;
      std::cout << "BAL: W=" << W << " split rows " << nsplit << " (weight > " << th_row << "), chunks " << chs.size() << " (~" << th_ch
                << "), replica words " << repv.size() << "; sender load max/mean " << *std::max_element(load0.begin(), load0.end()) / mean
                << " -> " << *std::max_element(load.begin(), load.end()) / mean << " over " << nsend << " senders\n"; }
#endif

#if ACTOR == 3
    // ---- Polak's preprocessing (CPU path): the undirected edges (MakeUndirected) keep the direction
    // RemoveBackwardEdgesCPU keeps, (s, t) with deg[s] > deg[t] or deg equal and s > t; SortEdges sorts the
    // pairs as uint64 (t the high word, s the low); UnzipEdges; CalculateNodePointers<false> over the t's
    std::vector<int> deg(numrows, 0);
    for (auto& e : all) { deg[e.row]++; deg[e.col]++; }
    std::vector<uint64_t> fe; fe.reserve(nnz);
    for (auto& e : all) { int s = e.row, t = e.col;
        if (!(deg[s] > deg[t] || (deg[s] == deg[t] && s > t))) std::swap(s, t);
        fe.push_back(((uint64_t)(uint32_t)t << 32) | (uint32_t)s); }
    std::sort(fe.begin(), fe.end());
    if (std::adjacent_find(fe.begin(), fe.end()) != fe.end()) { std::cerr << "duplicate edges in L\nFAILED\n"; return 1; }
    const int m = (int)fe.size();
    std::vector<int> pedges(2 * (size_t)m), pnodes(numrows + 1);
    for (int i = 0; i < m; i++) { pedges[i] = (int)(uint32_t)fe[i]; pedges[m + i] = (int)(fe[i] >> 32); }
    for (int j = 0, i = 0; j <= numrows; j++) { while (i < m && pedges[m + i] < j) i++; pnodes[j] = i; }
    { uint64_t work = 0; int mx = 0;
      for (int i = 0; i < m; i++) { int u = pedges[i], v = pedges[m + i];
          work += (uint64_t)(pnodes[u + 1] - pnodes[u]) + (pnodes[v + 1] - pnodes[v]); }
      for (int j = 0; j < numrows; j++) mx = std::max(mx, pnodes[j + 1] - pnodes[j]);
      std::cout << "Polak: m=" << m << " max out-degree " << mx << " merge bound sum(d(u)+d(v)) " << work << "\n"; }
#endif

#if ACTOR == 7
    // ---- TRUST's preprocessing (TC-Compare CSR2TrustCSR.cpp): sort by degree ascending (cmp1; stable here, ties by id)
    // and orient every edge to the later vertex (orientation); sort by out-degree descending (cmp2) and renumber
    // (computeCSR); reassignID: groups [out > 100 | 2..100 | < 2], each renumbered neighbours-first; CSR in the new ids
    std::vector<int> tbeg(numrows + 1), tadj;
    int wfv;
    {
        std::vector<int> deg(numrows, 0);
        for (auto& e : all) { deg[e.row]++; deg[e.col]++; }
        std::vector<int> o1(numrows); for (int i = 0; i < numrows; i++) o1[i] = i;
        std::stable_sort(o1.begin(), o1.end(), [&](int x, int y) { return deg[x] < deg[y]; });
        std::vector<int> p1(numrows); for (int i = 0; i < numrows; i++) p1[o1[i]] = i;
        std::vector<std::vector<int>> out(numrows);                  // by cmp1 position
        for (auto& e : all) { int x = p1[e.row], y = p1[e.col]; if (x < y) out[x].push_back(y); else out[y].push_back(x); }
        std::vector<int> o2(numrows); for (int i = 0; i < numrows; i++) o2[i] = i;
        std::stable_sort(o2.begin(), o2.end(), [&](int x, int y) { return out[x].size() > out[y].size(); });
        std::vector<int> p2(numrows); for (int i = 0; i < numrows; i++) p2[o2[i]] = i;
        std::vector<std::vector<int>> V(numrows);
        for (int i = 0; i < numrows; i++) for (int v : out[o2[i]]) V[i].push_back(p2[v]);
        // reassignID
        int k1 = 0, k2 = -1, k3 = -1;
        for (int i = 0; i < numrows; i++) {
            if (k2 == -1 && V[i].size() <= 100) k2 = i;
            if (k3 == -1 && V[i].size() < 2) k3 = i;
        }
        const int s2 = k2, s3 = k3;
        std::vector<int> nid(numrows, -1);
        auto give = [&](int v) { if (nid[v] == -1) nid[v] = v >= s3 ? k3++ : v >= s2 ? k2++ : k1++; };
        for (int i = 0; i < numrows; i++) { if (V[i].size() <= 2) break; for (int v : V[i]) give(v); }
        for (int i = 0; i < numrows; i++) give(i);
        std::vector<std::vector<int>> W(numrows);
        for (int i = 0; i < numrows; i++) for (int v : V[i]) W[nid[i]].push_back(nid[v]);
        for (int i = 0; i < numrows; i++) { tbeg[i] = (int)tadj.size(); tadj.insert(tadj.end(), W[i].begin(), W[i].end()); }
        tbeg[numrows] = (int)tadj.size();
        tadj.resize(tadj.size() + 64, 0);                            // the kernel's walk reads up to 2 superwarps past a list
        // my_binary_search(vertex_count, USE_CTA, beg_pos) + 1
        int l = 0, r = numrows;
        while (l < r - 1) { int mid = (l + r) / 2; if (tbeg[mid + 1] - tbeg[mid] > 100) l = mid; else r = mid; }
        wfv = (tbeg[l + 1] - tbeg[l] <= 100) ? 0 : l + 1;
        int mx = 0; for (int i = 0; i < numrows; i++) mx = std::max(mx, tbeg[i + 1] - tbeg[i]);
        std::cout << "TRUST: m=" << tbeg[numrows] << " CTA vertices " << wfv << " warp vertices " << s3 - s2
                  << " max out-degree " << mx << "\n";
    }
#endif

    // ---- device buffers
    auto alloc = [&](size_t bytes, vx_buffer_h* b, uint64_t* addr) {
        vx_mem_alloc(dev, bytes, VX_MEM_READ_WRITE, b); vx_mem_address(*b, addr); };
    args_t args; memset(&args, 0, sizeof(args));
    args.N = N; args.numrows = (uint32_t)numrows; args.lrows = (uint32_t)lrows;
#if ACTOR == 3
    args.m = (uint32_t)m;
#elif ACTOR == 5 || ACTOR == 6
    args.m = (uint32_t)lnonzero.size();
#endif
    args.wbits = wbits; args.vbits = vbits; args.tbits = tbits;
    args.ovf_maxc = (uint32_t)maxc + 1; args.ovf_nch = (uint32_t)need + 16;
    vx_buffer_h lob, lnb, cntb, qsb, tlb, poolb, ctabb, phb, barb, gridb, kb, ab;
    const size_t nph = ACTOR == 2 ? (size_t)N : nwarps;              // phase records: per actor (plain) or per warp
    const size_t ncnt = std::max(nwarps * GLANES, (size_t)N);
#if ACTOR == 7
    alloc(tbeg.size() * 4, &lob, &args.loffset_addr); vx_copy_to_dev(lob, tbeg.data(), 0, tbeg.size() * 4);
    alloc(tadj.size() * 4, &lnb, &args.lnonzero_addr); vx_copy_to_dev(lnb, tadj.data(), 0, tadj.size() * 4);
    { args.wfv = (uint32_t)wfv;
      const int nblk = (int)(N / GLANES), nwarp = nblk * (SPLIT);
      vx_buffer_h b; alloc((size_t)nblk * (TR_BB) * 100 * 4, &b, &args.part_addr);
      int nowindex[3] = { nwarp, nblk, wfv + nwarp };                // chunk_size 1: n_blocks*n_threads/T_Group, n_blocks, ...
      alloc(sizeof(nowindex), &b, &args.gidx_addr); vx_copy_to_dev(b, nowindex, 0, sizeof(nowindex)); }
#elif ACTOR == 3
    alloc(pnodes.size() * 4, &lob, &args.loffset_addr); vx_copy_to_dev(lob, pnodes.data(), 0, pnodes.size() * 4);
    alloc(std::max<size_t>(pedges.size(), 1) * 4, &lnb, &args.lnonzero_addr); vx_copy_to_dev(lnb, pedges.data(), 0, pedges.size() * 4);
#else
    alloc(loffset.size() * 4, &lob, &args.loffset_addr); vx_copy_to_dev(lob, loffset.data(), 0, loffset.size() * 4);
    alloc(std::max<size_t>(lnonzero.size(), 1) * 4, &lnb, &args.lnonzero_addr); vx_copy_to_dev(lnb, lnonzero.data(), 0, lnonzero.size() * 4);
#endif
#if ACTOR == 5 || ACTOR == 6
    { std::vector<int> erow(lnonzero.size());
      for (int u = 0; u < numrows; u++) for (int j = loffset[u]; j < loffset[u + 1]; j++) erow[j] = u;
      vx_buffer_h b; alloc(std::max<size_t>(erow.size(), 1) * 4, &b, &args.erow_addr); vx_copy_to_dev(b, erow.data(), 0, erow.size() * 4); }
#endif
    { std::vector<uint32_t> z(ncnt, 0); alloc(ncnt * 4, &cntb, &args.cnt_addr); vx_copy_to_dev(cntb, z.data(), 0, ncnt * 4); }
#if ACTOR == 2
    { size_t gb = (size_t)N * N * sizeof(ring_t); std::vector<char> z(gb, 0);
      alloc(gb, &gridb, &args.grid_addr); vx_copy_to_dev(gridb, z.data(), 0, gb); }
#endif
#if ACTOR == 4
    { std::vector<uint32_t> dofs(N + 1, 0), hofs(N + 1, 0), rt(2 * N);
      for (uint32_t p = 0; p < N; p++) { dofs[p + 1] = dofs[p] + (uint32_t)dsz[p]; hofs[p + 1] = hofs[p] + (uint32_t)hsz[p]; }
      for (uint32_t p = 0; p < N; p++) { rt[2 * p] = dofs[p]; rt[2 * p + 1] = hofs[p]; }   // tails start at the region starts
      vx_buffer_h b;
      alloc(std::max<size_t>(dofs[N], 1) * 4, &b, &args.data_addr);
      alloc(std::max<size_t>(hofs[N], 1) * 12, &b, &args.hdr_addr);
      alloc((N + 1) * 4, &b, &args.hofs_addr); vx_copy_to_dev(b, hofs.data(), 0, (N + 1) * 4);
      alloc(2 * N * 4, &b, &args.rtail_addr); vx_copy_to_dev(b, rt.data(), 0, 2 * N * 4); }
#endif
#if ACTOR == 1 || ACTOR == 4
    { size_t qb = (size_t)aq_region_bytes((uint64_t)NQ, (uint64_t)QSTRIDE_E * sizeof(msg_t)); alloc(qb, &qsb, &args.qs_addr);
      std::vector<uint32_t> z(NQ, 0); alloc(NQ * 4, &tlb, &args.tails_addr); vx_copy_to_dev(tlb, z.data(), 0, NQ * 4);
      alloc((size_t)args.ovf_nch * (OVF_CH) * sizeof(msg_t), &poolb, &args.pool_addr);
#if SWQ
      { std::vector<uint64_t> zc((size_t)NQ * args.ovf_maxc, 0); alloc(zc.size() * 8, &ctabb, &args.ctab_addr); vx_copy_to_dev(ctabb, zc.data(), 0, zc.size() * 8); }
      { vx_buffer_h sb; std::vector<uint32_t> zs(16, 0); alloc(zs.size() * 4, &sb, &args.swq_addr); vx_copy_to_dev(sb, zs.data(), 0, zs.size() * 4); }
#else
      alloc((size_t)NQ * args.ovf_maxc * 8, &ctabb, &args.ctab_addr);
#endif
    }
#endif
#if BAL
    { vx_buffer_h b;
      alloc(rend.size() * 4, &b, &args.rend_addr); vx_copy_to_dev(b, rend.data(), 0, rend.size() * 4);
      alloc(chv.size() * 4, &b, &args.ch_addr); vx_copy_to_dev(b, chv.data(), 0, chv.size() * 4);
      alloc(chofs.size() * 4, &b, &args.chofs_addr); vx_copy_to_dev(b, chofs.data(), 0, chofs.size() * 4);
      if (!repv.empty()) { alloc(repv.size() * 4, &b, &args.rep_addr); vx_copy_to_dev(b, repv.data(), 0, repv.size() * 4); } }
#endif
#if SEDGE && !RTILE
    { vx_buffer_h b; alloc(lmap.size() * 4, &b, &args.lmap_addr); vx_copy_to_dev(b, lmap.data(), 0, lmap.size() * 4); }
#endif
#if RTILE
    { vx_buffer_h b;
      alloc(lmap.size() * 4, &b, &args.lmap_addr); vx_copy_to_dev(b, lmap.data(), 0, lmap.size() * 4);
#if RTILE == 2
      alloc(timg.size() * 4, &b, &args.timg_addr); vx_copy_to_dev(b, timg.data(), 0, timg.size() * 4);
#endif
      alloc(tdesc.size() * 4, &b, &args.tdesc_addr); vx_copy_to_dev(b, tdesc.data(), 0, tdesc.size() * 4);
      alloc(qofs.size() * 4, &b, &args.qofs_addr); vx_copy_to_dev(b, qofs.data(), 0, qofs.size() * 4); }
#endif
    { std::vector<uint8_t> z(nph * sizeof(phase_cycles_t), 0);   // zeroed: LANE_RECV lanes add their msgs
      alloc(z.size(), &phb, &args.phase_addr); vx_copy_to_dev(phb, z.data(), 0, z.size()); }
    { uint32_t z[8] = {0,0,0,0,0,0,0,0}; alloc(sizeof(z), &barb, &args.bar_addr); vx_copy_to_dev(barb, z, 0, sizeof(z)); }
    vx_upload_kernel_file(dev, "kernel.vxbin", &kb);
    vx_upload_bytes(dev, &args, sizeof(args), &ab);
    vx_start(dev, kb, ab);
    vx_ready_wait(dev, VX_MAX_TIMEOUT);

    // ---- validate (the host sums the per-lane counters: ALLREDUCE)
    std::vector<uint32_t> cnt(ncnt); vx_copy_from_dev(cnt.data(), cntb, 0, cnt.size() * 4);
    std::vector<phase_cycles_t> ph(nph); vx_copy_from_dev(ph.data(), phb, 0, nph * sizeof(phase_cycles_t));
    uint64_t got = 0, recvd = 0; for (auto c : cnt) got += c;
    double se = 0, ba = 0, re = 0, to = 0, sv = 0; uint64_t smax = 0, rmax = 0;
    for (auto& x : ph) { se += x.send; ba += x.barrier; re += x.recv; to += x.total; recvd += x.msgs; sv += x.pad;
                         smax = std::max(smax, x.send); rmax = std::max(rmax, x.recv); }
    std::cout << "phases (mean cycles/actor): send=" << (uint64_t)(se / nph) << " barrier=" << (uint64_t)(ba / nph)
              << " recv=" << (uint64_t)(re / nph) << " total=" << (uint64_t)(to / nph)
              << " max_send=" << smax << " max_recv=" << rmax;
    if (ACTOR == 4) std::cout << " serve=" << (uint64_t)(sv / nph);
    std::cout << "\n";
    const bool ok = got == ref && (!MSG || recvd == nmsg);
    std::cout << "validation: triangles " << got << " of " << ref;
    if (MSG) std::cout << ", messages received " << recvd << " of " << nmsg;
    std::cout << "\n";
    vx_dev_close(dev);
    std::cout << (ok ? "PASSED" : "FAILED") << "\n";
    return ok ? 0 : 1;
}
