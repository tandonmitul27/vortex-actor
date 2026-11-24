// fgc_hs: histogram through the actor runtime or the non-actor reference,
// checked against a host replay of every update.
#include "../fgc_common/fgc_host.h"
#include "common.h"
#include <unistd.h>

namespace {
const char* kernel_file = "kernel.vxbin";
const char* config_names[CFG_COUNT] = { "non_actor", "actor" };
uint32_t    config      = CFG_ACTOR;
uint32_t    num_msgs    = 16;
uint32_t    num_buckets = 16;
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
    while ((c = getopt(argc, argv, "c:m:l:q:r:t:k:h")) != -1) {
        switch (c) {
            case 'c': config      = std::atoi(optarg); break;
            case 'm': num_msgs    = std::atoi(optarg); break;
            case 'l': num_buckets = std::atoi(optarg); break;
            case 'q': ring_slots  = std::atoi(optarg); break;
            case 'r': recv_batch  = std::atoi(optarg); break;
            case 't': term_period = std::atoi(optarg); break;
            case 'k': kernel_file = optarg;            break;
            default:
                std::cout << "Usage: [-c config] [-m msgs per actor] [-l buckets per actor]"
                             " [-q ring slots] [-r recv batch] [-t idle passes per termination check]"
                             " [-k kernel] [-h]" << std::endl;
                std::exit(c == 'h' ? 0 : -1);
        }
    }
    if (config >= CFG_COUNT || !pow2(num_buckets) || !pow2(ring_slots) || recv_batch == 0 || term_period == 0) {
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
    const uint32_t N  = mach.actors();
    const uint32_t lg = log2u(num_buckets);

    int errors = 0;
    {
        fgc::Buffers bufs(dev, q);
        kernel_arg_t arg{};
        arg.config      = config;
        arg.num_actors  = N;
        arg.log_buckets = lg;
        arg.num_msgs    = num_msgs;
        const uint64_t counts_size = (uint64_t)N * num_buckets * sizeof(uint32_t);
        const uint64_t time_size   = (uint64_t)mach.cores * 2 * sizeof(uint64_t);
        vx_buffer_h counts_buf = bufs.zeroed("counts", counts_size, &arg.counts_addr);
        vx_buffer_h time_buf   = bufs.zeroed("time", time_size, &arg.time_addr);
        if (config == CFG_ACTOR) {
            fgc::alloc_runtime<uint32_t>(bufs, &arg.rt, mach, rtcfg, 1, log2u(ring_slots),
                               recv_batch, term_period);
        }
        bufs.write_tags();

        vx_module_h mod = nullptr;
        vx_kernel_h kern = nullptr;
        CHECK(vx_module_load_file(dev, kernel_file, &mod));
        CHECK(vx_module_get_kernel(mod, "main", &kern));

        std::cout << "fgc_hs: config=" << config_names[config] << " actors=" << N
                  << " msgs/actor=" << num_msgs << " buckets/actor=" << num_buckets
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

        std::vector<uint32_t> h_counts((uint64_t)N * num_buckets);
        std::vector<uint64_t> h_time((uint64_t)mach.cores * 2);
        vx_event_h launch_ev = nullptr, rd1 = nullptr, rd2 = nullptr;
        CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
        CHECK(vx_enqueue_read(q, h_counts.data(), counts_buf, 0, counts_size, 1, &launch_ev, &rd1));
        CHECK(vx_enqueue_read(q, h_time.data(), time_buf, 0, time_size, 1, &launch_ev, &rd2));
        CHECK(vx_event_wait_value(rd2, 1, VX_TIMEOUT_INFINITE));
        CHECK(vx_event_wait_value(rd1, 1, VX_TIMEOUT_INFINITE));

        const uint64_t cycles = fgc::kernel_cycles(h_time);
        std::cout << "KERNEL: cycles=" << cycles << " per_update="
                  << (double)cycles / ((double)N * num_msgs) << std::endl;

        std::vector<uint32_t> ref((uint64_t)N * num_buckets, 0);
        const uint32_t range = N << lg;
        for (uint32_t a = 0; a < N; ++a) {
            for (uint32_t i = 0; i < num_msgs; ++i) {
                ++ref[hash_key(a, i, range)];
            }
        }
        for (uint64_t g = 0; g < ref.size(); ++g) {
            if (h_counts[g] != ref[g]) {
                if (errors < 16)
                    std::printf("*** counts[%llu] expected=%u actual=%u\n",
                                (unsigned long long)g, ref[g], h_counts[g]);
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
