#ifndef _FGC_GRAPH_H_
#define _FGC_GRAPH_H_

// Host-side graphs for the fgc apps, distributed the way Bale distributes
// rows: global row r belongs to actor r % N as its local row r / N, and each
// actor's rows are stored contiguously (storage index owner * R + local).

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace fgc {

struct Csr {
    uint32_t num_actors;          // N
    uint32_t rows_per_actor;      // R
    std::vector<uint32_t> offsets; // [N * R + 1], by storage index
    std::vector<uint32_t> cols;    // global column ids

    uint32_t num_rows() const { return num_actors * rows_per_actor; }
    uint32_t storage(uint32_t r) const { return (r % num_actors) * rows_per_actor + r / num_actors; }
};

// Undirected Erdos-Renyi graph without self loops, as Bale's
// erdos_renyi_random_graph: each pair i > j is an edge with probability p,
// drawn by geometric skips (O(edges)), then mirrored.
inline Csr erdos_renyi(uint32_t num_actors, uint32_t rows_per_actor, double nz_per_row, uint64_t seed) {
    Csr g{num_actors, rows_per_actor, {}, {}};
    const uint32_t n = g.num_rows();
    double p = n > 1 ? (2.0 * (nz_per_row - 1)) / n : 0.0;  // Bale's choice from nz_per_row
    if (p > 1.0) p = 1.0;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<std::vector<uint32_t>> adj(n);
    if (p > 0.0) {
        const double lq = std::log(1.0 - p);
        for (uint32_t i = 1; i < n; ++i) {
            int64_t j = -1;
            for (;;) {
                double x = u(rng);
                j += 1 + (p >= 1.0 ? 0 : (int64_t)std::floor(std::log(1.0 - x) / lq));
                if (j >= (int64_t)i) break;
                adj[i].push_back((uint32_t)j);
                adj[(uint32_t)j].push_back(i);
            }
        }
    }
    g.offsets.assign((size_t)n + 1, 0);
    for (uint32_t r = 0; r < n; ++r) {
        g.offsets[g.storage(r) + 1] = (uint32_t)adj[r].size();
    }
    for (uint32_t s = 0; s < n; ++s) {
        g.offsets[s + 1] += g.offsets[s];
    }
    g.cols.resize(g.offsets[n]);
    for (uint32_t r = 0; r < n; ++r) {
        std::copy(adj[r].begin(), adj[r].end(), g.cols.begin() + g.offsets[g.storage(r)]);
    }
    return g;
}

} // namespace fgc

#endif
