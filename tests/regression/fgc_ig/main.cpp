// fgc_ig: index gather through the actor runtime or the non-actor reference,
// every result checked against the table entry it asked for.
#include "../fgc_common/fgc_host.h"
#include "common.h"
#include <unistd.h>

namespace {
const char* kernel_file = "kernel.vxbin";
const char* config_names[CFG_COUNT] = { "non_actor", "actor" };
uint32_t    config      = CFG_ACTOR;
uint32_t    num_msgs    = 16;
uint32_t    num_entries = 16;
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
            case 'l': num_entries = std::atoi(optarg); break;
            case 'q': ring_slots  = std::atoi(optarg); break;
            case 'r': recv_batch  = std::atoi(optarg); break;
            case 't': term_period = std::atoi(optarg); break;
            case 'k': kernel_file = optarg;            break;
            default:
                std::cout << "Usage: [-c config] [-m requests per actor] [-l table entries per actor]"
                             " [-q ring slots] [-r recv batch] [-t idle passes per termination check]"
                             " [-k kernel] [-h]" << std::endl;
                std::exit(c == 'h' ? 0 : -1);
        }
    }
    if (config >= CFG_COUNT || !pow2(num_entries) || !pow2(ring_slots) || recv_batch == 0 || term_period == 0) {
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
    const uint32_t lg = log2u(num_entries);

    int errors = 0;
    {
        fgc::Buffers bufs(dev, q);
        kernel_arg_t arg{};
        arg.config      = config;
        arg.num_actors  = N;
        arg.log_entries = lg;
        arg.num_msgs    = num_msgs;
        std::vector<uint32_t> table((uint64_t)N * num_entries);
        for (uint64_t g = 0; g < table.size(); ++g) {
            table[g] = table_value((uint32_t)g);
        }
        const uint64_t table_size = table.size() * sizeof(uint32_t);
        const uint64_t tgt_size   = (uint64_t)N * num_msgs * sizeof(uint32_t);
        const uint64_t time_size  = (uint64_t)mach.cores * 2 * sizeof(uint64_t);
        bufs.filled("table", table.data(), table_size, &arg.table_addr);
        vx_buffer_h tgt_buf  = bufs.zeroed("tgt", tgt_size, &arg.tgt_addr);
        vx_buffer_h time_buf = bufs.zeroed("time", time_size, &arg.time_addr);
        if (config == CFG_ACTOR) {
            fgc::alloc_runtime<ig_pkt_t>(bufs, &arg.rt, mach, rtcfg, NUM_MB, log2u(ring_slots),
                               recv_batch, term_period);
        }
        bufs.write_tags();

        vx_module_h mod = nullptr;
        vx_kernel_h kern = nullptr;
        CHECK(vx_module_load_file(dev, kernel_file, &mod));
        CHECK(vx_module_get_kernel(mod, "main", &kern));

        std::cout << "fgc_ig: config=" << config_names[config] << " actors=" << N
                  << " requests/actor=" << num_msgs << " entries/actor=" << num_entries
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

        std::vector<uint32_t> h_tgt((uint64_t)N * num_msgs);
        std::vector<uint64_t> h_time((uint64_t)mach.cores * 2);
        vx_event_h launch_ev = nullptr, rd1 = nullptr, rd2 = nullptr;
        CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
        CHECK(vx_enqueue_read(q, h_tgt.data(), tgt_buf, 0, tgt_size, 1, &launch_ev, &rd1));
        CHECK(vx_enqueue_read(q, h_time.data(), time_buf, 0, time_size, 1, &launch_ev, &rd2));
        CHECK(vx_event_wait_value(rd2, 1, VX_TIMEOUT_INFINITE));
        CHECK(vx_event_wait_value(rd1, 1, VX_TIMEOUT_INFINITE));

        const uint64_t cycles = fgc::kernel_cycles(h_time);
        std::cout << "KERNEL: cycles=" << cycles << " per_request="
                  << (double)cycles / ((double)N * num_msgs) << std::endl;

        const uint32_t range = N << lg;
        for (uint32_t a = 0; a < N; ++a) {
            for (uint32_t i = 0; i < num_msgs; ++i) {
                uint32_t want = table_value(hash_key(a, i, range));
                uint32_t got  = h_tgt[(uint64_t)a * num_msgs + i];
                if (got != want) {
                    if (errors < 16)
                        std::printf("*** tgt[%u][%u] expected=%u actual=%u\n", a, i, want, got);
                    ++errors;
                }
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
