// fgc_pl: PageRank pull through the actor runtime or the non-actor reference,
// checked against a host computation in double precision.
#include "../fgc_common/fgc_host.h"
#include "../fgc_common/fgc_graph.h"
#include "common.h"
#include <cmath>
#include <unistd.h>

namespace {
const char* kernel_file = "kernel.vxbin";
const char* config_names[CFG_COUNT] = { "non_actor", "actor" };
uint32_t    config      = CFG_ACTOR;
uint32_t    rows        = 4;
double      nz_per_row  = 8;
uint32_t    iters       = 2;
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
    while ((c = getopt(argc, argv, "c:l:z:i:q:r:t:k:h")) != -1) {
        switch (c) {
            case 'c': config      = std::atoi(optarg); break;
            case 'l': rows        = std::atoi(optarg); break;
            case 'z': nz_per_row  = std::atof(optarg); break;
            case 'i': iters       = std::atoi(optarg); break;
            case 'q': ring_slots  = std::atoi(optarg); break;
            case 'r': recv_batch  = std::atoi(optarg); break;
            case 't': term_period = std::atoi(optarg); break;
            case 'k': kernel_file = optarg;            break;
            default:
                std::cout << "Usage: [-c config] [-l rows per actor] [-z nonzeros per row] [-i iterations]"
                             " [-q ring slots] [-r recv batch] [-t idle passes per termination check]"
                             " [-k kernel] [-h]" << std::endl;
                std::exit(c == 'h' ? 0 : -1);
        }
    }
    if (config >= CFG_COUNT || rows == 0 || iters == 0 || !pow2(ring_slots) || recv_batch == 0 || term_period == 0) {
        std::cout << "invalid arguments" << std::endl;
        std::exit(-1);
    }
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
    const fgc::Csr g = fgc::erdos_renyi(N, rows, nz_per_row, 12345);
    const uint32_t n = g.num_rows();

    // Only a neighbour's degree divides, and a neighbour has degree >= 1.
    std::vector<double> ref(n, 1.0 / n), sum(n);
    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t s = 0; s < n; ++s) {
            double acc = 0.0;
            for (uint32_t k = g.offsets[s]; k < g.offsets[s + 1]; ++k) {
                uint32_t c = g.storage(g.cols[k]);
                acc += ref[c] / (double)(g.offsets[c + 1] - g.offsets[c]);
            }
            sum[s] = (1.0 - PL_DAMPING) / n + PL_DAMPING * acc;
        }
        ref.swap(sum);
    }

    int errors = 0;
    {
        fgc::Buffers bufs(dev, q);
        kernel_arg_t arg{};
        arg.config     = config;
        arg.num_actors = N;
        arg.rows       = rows;
        arg.iters      = iters;
        arg.num_cores  = mach.cores;
        std::vector<float> rank0(2 * (size_t)n, 1.0f / n);
        const uint64_t rank_size = rank0.size() * sizeof(float);
        const uint64_t time_size = (uint64_t)mach.cores * 2 * sizeof(uint64_t);
        bufs.filled("offsets", g.offsets.data(), g.offsets.size() * sizeof(uint32_t), &arg.offsets_addr);
        bufs.filled("cols", g.cols.data(), std::max<size_t>(g.cols.size(), 1) * sizeof(uint32_t), &arg.cols_addr);
        vx_buffer_h rank_buf = bufs.filled("rank", rank0.data(), rank_size, &arg.rank_addr);
        bufs.zeroed("acc", (uint64_t)n * sizeof(float), &arg.acc_addr);
        vx_buffer_h time_buf = bufs.zeroed("time", time_size, &arg.time_addr);
        if (config == CFG_ACTOR) {
            fgc::alloc_runtime<pl_pkt_t>(bufs, &arg.rt, mach, rtcfg, NUM_MB, log2u(ring_slots),
                               recv_batch, term_period, iters);
        }
        bufs.write_tags();

        vx_module_h mod = nullptr;
        vx_kernel_h kern = nullptr;
        CHECK(vx_module_load_file(dev, kernel_file, &mod));
        CHECK(vx_module_get_kernel(mod, "main", &kern));

        std::cout << "fgc_pl: config=" << config_names[config] << " actors=" << N
                  << " rows/actor=" << rows << " nnz=" << g.cols.size() << " iters=" << iters
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

        std::vector<float> h_rank(2 * (size_t)n);
        std::vector<uint64_t> h_time((uint64_t)mach.cores * 2);
        vx_event_h launch_ev = nullptr, rd1 = nullptr, rd2 = nullptr;
        CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
        CHECK(vx_enqueue_read(q, h_rank.data(), rank_buf, 0, rank_size, 1, &launch_ev, &rd1));
        CHECK(vx_enqueue_read(q, h_time.data(), time_buf, 0, time_size, 1, &launch_ev, &rd2));
        CHECK(vx_event_wait_value(rd2, 1, VX_TIMEOUT_INFINITE));
        CHECK(vx_event_wait_value(rd1, 1, VX_TIMEOUT_INFINITE));

        const uint64_t cycles = fgc::kernel_cycles(h_time);
        std::cout << "KERNEL: cycles=" << cycles << " per_nonzero_iter="
                  << (double)cycles / ((double)std::max<size_t>(g.cols.size(), 1) * iters) << std::endl;

        // config 0 leaves the result in the set of the last iteration's output
        const float* got = h_rank.data() + (config == CFG_NON_ACTOR ? (iters & 1) * (size_t)n : 0);
        for (uint32_t s = 0; s < n; ++s) {
            if (std::fabs(got[s] - ref[s]) > 1e-3 * ref[s] + 1e-9) {
                if (errors < 16)
                    std::printf("*** rank[%u] expected=%g actual=%g\n", s, ref[s], got[s]);
                ++errors;
            }
        }

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
