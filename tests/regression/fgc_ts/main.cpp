// fgc_ts: the matching phase of topological sort through the actor runtime or
// the non-actor reference. The device's levels and matches are compared with
// a host run of the same algorithm, then turned into row and column
// permutations as Bale's second phase does, and the permuted matrix must be
// upper triangular.
#include "../fgc_common/fgc_host.h"
#include "common.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <unistd.h>

namespace {
const char* kernel_file = "kernel.vxbin";
const char* config_names[CFG_COUNT] = { "non_actor", "actor" };
uint32_t    config      = CFG_ACTOR;
uint32_t    rows        = 4;
double      nz_per_row  = 4;
uint32_t    ring_slots  = 32;
uint32_t    recv_batch  = 8;
uint32_t    term_period = 16;
const fgc::RtConfig rtcfg;

bool pow2(uint32_t x) { return x != 0 && (x & (x - 1)) == 0; }

uint32_t log2u(uint32_t x) {
    uint32_t l = 0;
    while ((1u << l) < x) ++l;
    return l;
}

void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "c:l:z:q:r:t:k:h")) != -1) {
        switch (c) {
            case 'c': config      = std::atoi(optarg); break;
            case 'l': rows        = std::atoi(optarg); break;
            case 'z': nz_per_row  = std::atof(optarg); break;
            case 'q': ring_slots  = std::atoi(optarg); break;
            case 'r': recv_batch  = std::atoi(optarg); break;
            case 't': term_period = std::atoi(optarg); break;
            case 'k': kernel_file = optarg;            break;
            default:
                std::cout << "Usage: [-c config] [-l rows per actor] [-z nonzeros per row]"
                             " [-q ring slots] [-r recv batch] [-t idle passes per termination check]"
                             " [-k kernel] [-h]" << std::endl;
                std::exit(c == 'h' ? 0 : -1);
        }
    }
    if (config >= CFG_COUNT || rows == 0 || !pow2(ring_slots) || recv_batch == 0 || term_period == 0) {
        std::cout << "invalid arguments" << std::endl;
        std::exit(-1);
    }
}

// Bale's generate_toposort_input: upper triangular with a full diagonal, each
// entry above it present with probability 2 * nz / (n - 1) (geometric skips),
// then rows and columns randomly permuted. Returns each row's column ids.
std::vector<std::vector<uint32_t>> toposort_input(uint32_t n, double nz, uint64_t seed) {
    double p = n > 1 ? 2.0 * nz / (n - 1) : 0.0;
    if (p > 1.0) p = 1.0;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<uint32_t> rp(n), cp(n);
    std::iota(rp.begin(), rp.end(), 0u);
    std::iota(cp.begin(), cp.end(), 0u);
    std::shuffle(rp.begin(), rp.end(), rng);
    std::shuffle(cp.begin(), cp.end(), rng);
    std::vector<std::vector<uint32_t>> m(n);
    const double lq = p > 0.0 && p < 1.0 ? std::log(1.0 - p) : 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        m[rp[i]].push_back(cp[i]);
        if (p <= 0.0) continue;
        int64_t j = i;
        for (;;) {
            j += 1 + (p >= 1.0 ? 0 : (int64_t)std::floor(std::log(1.0 - u(rng)) / lq));
            if (j >= (int64_t)n) break;
            m[rp[i]].push_back(cp[(uint32_t)j]);
        }
    }
    return m;
}
} // namespace

int main(int argc, char** argv) {
    parse_args(argc, argv);

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));
    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, &qi, &q));

    const fgc::Machine mach = fgc::query_machine(dev);
    const uint32_t N = mach.actors();
    const uint32_t n = N * rows;
    auto storage = [&](uint32_t g) { return (g % N) * rows + g / N; };

    const auto m = toposort_input(n, nz_per_row, 12345);
    std::vector<std::vector<uint32_t>> colrows(n);
    std::vector<uint32_t> rowcnt(n), rowsum(n, 0);
    for (uint32_t r = 0; r < n; ++r) {
        rowcnt[storage(r)] = (uint32_t)m[r].size();
        for (uint32_t c : m[r]) {
            rowsum[storage(r)] += c;
            colrows[c].push_back(r);
        }
    }
    std::vector<uint32_t> toff((size_t)n + 1, 0), trows;
    for (uint32_t c = 0; c < n; ++c) toff[storage(c) + 1] = (uint32_t)colrows[c].size();
    for (uint32_t s = 0; s < n; ++s) toff[s + 1] += toff[s];
    trows.resize(toff[n]);
    for (uint32_t c = 0; c < n; ++c) {
        std::copy(colrows[c].begin(), colrows[c].end(), trows.begin() + toff[storage(c)]);
    }

    // The same algorithm on the host, sequentially: matches and levels do not
    // depend on the order in which ready rows are taken.
    std::vector<uint32_t> ref_level(n, 0), ref_match(n, UINT32_MAX);
    {
        std::vector<uint32_t> cnt = rowcnt, sum = rowsum, ready;
        for (uint32_t s = 0; s < n; ++s) if (cnt[s] == 1) ready.push_back(s);
        for (size_t i = 0; i < ready.size(); ++i) {
            uint32_t s = ready[i], c = sum[s], lvl = ref_level[s];
            ref_match[s] = c;
            for (uint32_t r : colrows[c]) {
                uint32_t rs = storage(r);
                sum[rs] -= c;
                ref_level[rs] = std::max(ref_level[rs], lvl + 1);
                if (--cnt[rs] == 1) ready.push_back(rs);
            }
        }
    }

    int errors = 0;
    {
        fgc::Buffers bufs(dev, q);
        kernel_arg_t arg{};
        arg.config     = config;
        arg.num_actors = N;
        arg.rows       = rows;
        arg.num_cores  = mach.cores;
        const uint64_t vec = (uint64_t)n * sizeof(uint32_t);
        const uint64_t time_size = (uint64_t)mach.cores * 2 * sizeof(uint64_t);
        bufs.filled("toff", toff.data(), toff.size() * sizeof(uint32_t), &arg.toff_addr);
        bufs.filled("trows", trows.data(), trows.size() * sizeof(uint32_t), &arg.trows_addr);
        bufs.filled("rowcnt", rowcnt.data(), vec, &arg.rowcnt_addr);
        bufs.filled("rowsum", rowsum.data(), vec, &arg.rowsum_addr);
        vx_buffer_h level_buf = bufs.zeroed("level", vec, &arg.level_addr);
        vx_buffer_h match_buf = bufs.zeroed("match", vec, &arg.match_addr);
        bufs.zeroed("work", 3 * vec, &arg.work_addr);
        bufs.zeroed("flag", vec + 8, &arg.flag_addr);
        vx_buffer_h time_buf = bufs.zeroed("time", time_size, &arg.time_addr);
        if (config == CFG_ACTOR) {
            fgc::alloc_runtime<ts_pkt_t>(bufs, &arg.rt, mach, rtcfg, NUM_MB, log2u(ring_slots),
                               recv_batch, term_period);
        }
        bufs.write_tags();

        vx_module_h mod = nullptr;
        vx_kernel_h kern = nullptr;
        CHECK(vx_module_load_file(dev, kernel_file, &mod));
        CHECK(vx_module_get_kernel(mod, "main", &kern));

        uint32_t max_level = *std::max_element(ref_level.begin(), ref_level.end());
        std::cout << "fgc_ts: config=" << config_names[config] << " actors=" << N
                  << " rows/actor=" << rows << " nnz=" << trows.size() << " levels=" << max_level
                  << " ring=" << ring_slots << " R=" << recv_batch << " " << fgc::rt_describe(rtcfg) << " K=" << term_period
                  << " grid=" << mach.cores << " block=" << mach.block() << std::endl;

        vx_launch_info_t li{};
        li.struct_size  = sizeof(li);
        li.kernel       = kern;
        li.args_host    = &arg;
        li.args_size    = sizeof(arg);
        li.ndim         = 1;
        li.grid_dim[0]  = mach.cores;
        li.block_dim[0] = mach.block();
        li.lmem_size    = fgc::lmem_bytes(arg.rt);

        std::vector<uint32_t> level(n), match(n);
        std::vector<uint64_t> h_time((uint64_t)mach.cores * 2);
        vx_event_h launch_ev = nullptr, rd1 = nullptr, rd2 = nullptr, rd3 = nullptr;
        CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
        CHECK(vx_enqueue_read(q, level.data(), level_buf, 0, vec, 1, &launch_ev, &rd1));
        CHECK(vx_enqueue_read(q, match.data(), match_buf, 0, vec, 1, &launch_ev, &rd2));
        CHECK(vx_enqueue_read(q, h_time.data(), time_buf, 0, time_size, 1, &launch_ev, &rd3));
        CHECK(vx_event_wait_value(rd3, 1, VX_TIMEOUT_INFINITE));
        CHECK(vx_event_wait_value(rd2, 1, VX_TIMEOUT_INFINITE));
        CHECK(vx_event_wait_value(rd1, 1, VX_TIMEOUT_INFINITE));

        const uint64_t cycles = fgc::kernel_cycles(h_time);
        std::cout << "KERNEL: cycles=" << cycles << " per_nonzero="
                  << (double)cycles / (double)std::max<size_t>(trows.size(), 1) << std::endl;

        for (uint32_t s = 0; s < n; ++s) {
            if (level[s] != ref_level[s] || match[s] != ref_match[s]) {
                if (errors < 16)
                    std::printf("*** row %u: level %u (expected %u), match %u (expected %u)\n",
                                s, level[s], ref_level[s], match[s], ref_match[s]);
                ++errors;
            }
        }

        // Bale's second phase: rows ordered by level, then actor, then local
        // row; the matched column takes its row's position.
        if (errors == 0) {
            std::vector<uint32_t> order(n);
            std::iota(order.begin(), order.end(), 0u);
            std::stable_sort(order.begin(), order.end(),
                             [&](uint32_t a, uint32_t b) { return level[a] < level[b]; });
            std::vector<uint32_t> rperm(n), cperm(n, UINT32_MAX);
            for (uint32_t i = 0; i < n; ++i) rperm[order[i]] = (n - 1) - i;
            for (uint32_t s = 0; s < n; ++s) {
                if (match[s] >= n || cperm[match[s]] != UINT32_MAX) {
                    if (errors < 16) std::printf("*** column %u matched twice or out of range\n", match[s]);
                    ++errors;
                } else {
                    cperm[match[s]] = rperm[s];
                }
            }
            for (uint32_t r = 0; r < n && errors == 0; ++r) {
                for (uint32_t c : m[r]) {
                    if (cperm[c] < rperm[storage(r)]) {
                        if (errors < 16) std::printf("*** nonzero (%u, %u) below the diagonal\n", r, c);
                        ++errors;
                    }
                }
            }
        }

        vx_event_release(rd3);
        vx_event_release(rd2);
        vx_event_release(rd1);
        vx_event_release(launch_ev);
        vx_kernel_release(kern);
        vx_module_release(mod);
    }
    vx_queue_release(q);
    vx_device_dump_perf(dev, stdout);
    vx_device_release(dev);

    if (errors) {
        std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
        return 1;
    }
    std::cout << "PASSED!" << std::endl;
    return 0;
}
