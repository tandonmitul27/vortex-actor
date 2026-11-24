#ifndef _FGC_HOST_H_
#define _FGC_HOST_H_

// Host helpers shared by the fgc tests: error checks, zeroed buffers, address
// tags for the fgc counters, the actor runtime's buffers and the kernel span.

#include <vortex2.h>
#include "fgc_actor_args.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Exits on a fatal error without running static destructors: the device is
// still open and its worker thread may be using them.
[[noreturn]] inline void fgc_fail_exit() {
    std::fflush(nullptr);
    std::_Exit(1);
}

#define CHECK(expr) do { \
    vx_result_t _r = (expr); \
    if (_r != VX_SUCCESS) { \
        std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n", \
                     __FILE__, __LINE__, #expr, vx_result_string(_r)); \
        fgc_fail_exit(); \
    } \
} while (0)

namespace fgc {

// Every hardware thread is an actor: one CTA of warps x threads per core.
struct Machine {
    uint32_t cores, warps, threads;
    uint32_t block() const { return warps * threads; }
    uint32_t actors() const { return cores * block(); }
};

inline Machine query_machine(vx_device_h dev) {
    uint64_t c, w, t;
    CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES, &c));
    CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &w));
    CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &t));
    return { (uint32_t)c, (uint32_t)w, (uint32_t)t };
}

// The buffer configuration this host and its kernel are built for
// (make FGC_TOPO=.. FGC_SENDER=.. FGC_TERM=..; see fgc_actor_args.h).
struct RtConfig {
    uint32_t topo   = FGC_TOPO;
    uint32_t sender = FGC_SENDER;
    uint32_t term   = FGC_TERM;
};

inline std::string rt_describe(const RtConfig& cfg) {
    static const char* topo[]   = { "spsc", "mpsc_actor", "mpsc_warp", "mpsc_cta" };
    static const char* sender[] = { "thread", "warp", "cta" };
    return std::string("topo=") + topo[cfg.topo] + " sender=" + sender[cfg.sender]
         + " term=T" + std::to_string(cfg.term);
}

// Device buffers, zeroed before the launch, and their address tags.
class Buffers {
public:
    Buffers(vx_device_h dev, vx_queue_h q) : dev_(dev), q_(q) {}

    ~Buffers() {
        for (auto b : bufs_) vx_buffer_release(b);
    }

    vx_buffer_h zeroed(const char* name, uint64_t size, uint64_t* addr) {
        vx_buffer_h buf = nullptr;
        CHECK(vx_buffer_create(dev_, size, VX_MEM_READ_WRITE, &buf));
        CHECK(vx_buffer_address(buf, addr));
        // writes are asynchronous: the source must outlive the queue's work
        if (zeros_.size() < size) {
            CHECK(vx_queue_finish(q_, VX_TIMEOUT_INFINITE));
            zeros_.assign(size, 0);
        }
        CHECK(vx_enqueue_write(q_, buf, 0, zeros_.data(), size, 0, nullptr, nullptr));
        bufs_.push_back(buf);
        tags_.push_back({ name, *addr, size });
        return buf;
    }

    // A buffer filled from host data; the copy lives until the buffers go.
    vx_buffer_h filled(const char* name, const void* data, uint64_t size, uint64_t* addr) {
        vx_buffer_h buf = nullptr;
        CHECK(vx_buffer_create(dev_, size, VX_MEM_READ_WRITE, &buf));
        CHECK(vx_buffer_address(buf, addr));
        auto src = static_cast<const uint8_t*>(data);
        copies_.emplace_back(src, src + size);
        CHECK(vx_enqueue_write(q_, buf, 0, copies_.back().data(), size, 0, nullptr, nullptr));
        bufs_.push_back(buf);
        tags_.push_back({ name, *addr, size });
        return buf;
    }

    // Names every buffer for the fgc counters (VORTEX_FGC_TAGS).
    void write_tags() const {
        const char* path = std::getenv("VORTEX_FGC_TAGS");
        if (!path) return;
        std::ofstream os(path);
        for (auto& t : tags_) {
            os << t.name << " " << std::hex << t.base << " " << t.size << std::dec << "\n";
        }
    }

private:
    struct Tag { std::string name; uint64_t base, size; };
    vx_device_h dev_;
    vx_queue_h q_;
    std::vector<vx_buffer_h> bufs_;
    std::vector<Tag> tags_;
    std::vector<uint8_t> zeros_;
    std::deque<std::vector<uint8_t>> copies_;
};

inline uint32_t exact_log2(uint32_t x, const char* what) {
    if (x == 0 || (x & (x - 1)) != 0) {
        std::fprintf(stderr, "the actor runtime needs a power-of-two %s (got %u)\n", what, x);
        fgc_fail_exit();
    }
    return (uint32_t)__builtin_ctz(x);
}

// The runtime's buffers for configuration cfg on machine m. Per mailbox and
// ring set: the queues (rings of 1 << lgcap slots per consumer actor, so a
// queue a group reads holds the group's share; mb_lgcap[mb] if given), their
// tails and, for SPSC, the receivers' heads; termination counters for every
// epoch; and the local-memory layout per CTA.
template <typename T>
inline void alloc_runtime(Buffers& b, fgc_rt_args_t* rt, const Machine& m, const RtConfig& cfg,
                          uint32_t num_mb, uint32_t lgcap, uint32_t recv_batch, uint32_t term_period,
                          uint32_t epochs = 1, const uint32_t* mb_lgcap = nullptr) {
    const uint32_t N = m.actors();
    rt->num_actors  = N;
    rt->recv_batch  = recv_batch;
    rt->term_period = term_period;
    rt->ring_sets   = epochs > 1 ? 2 : 1;
    rt->topo        = cfg.topo;
    rt->sender      = cfg.sender;
    rt->term        = cfg.term;
    rt->lg_threads  = exact_log2(m.threads, "warp size");
    rt->lg_block    = exact_log2(m.block(), "CTA size");
    const uint32_t lg_prod  = cfg.sender == FGC_SENDER_WARP ? rt->lg_threads
                            : cfg.sender == FGC_SENDER_CTA  ? rt->lg_block : 0;
    const uint32_t lg_group = cfg.topo == FGC_TOPO_MPSC_WARP ? rt->lg_threads
                            : cfg.topo == FGC_TOPO_MPSC_CTA  ? rt->lg_block : 0;
    const uint64_t queues   = cfg.topo == FGC_TOPO_SPSC ? (uint64_t)N * (N >> lg_prod) : (uint64_t)(N >> lg_group);
    b.zeroed("term", (uint64_t)epochs * num_mb * sizeof(fgc_term_t), &rt->term_addr);
    for (uint32_t set = 0; set < rt->ring_sets; ++set) {
        for (uint32_t mb = 0; mb < num_mb; ++mb) {
            auto& q = rt->mb[mb + num_mb * set];
            std::string id = std::to_string(mb + num_mb * set);
            q.lgcap = (mb_lgcap ? mb_lgcap[mb] : lgcap) + lg_group;
            b.zeroed(("ring" + id).c_str(), (queues << q.lgcap) * sizeof(Slot<T>), &q.slots);
            b.zeroed(("tail" + id).c_str(), queues * sizeof(uint32_t), &q.tail);
            q.heads = 0;
            if (cfg.topo == FGC_TOPO_SPSC) {
                b.zeroed(("head" + id).c_str(), queues * sizeof(uint32_t), &q.heads);
            }
        }
    }
    // local memory: the warps' word pairs, the CTA vote, the CTA claim table
    // (two entries of key, count, base per thread), the T2 counters (done and
    // sent per mailbox and epoch parity), the group staging per mailbox (a
    // list head per member, at most FGC_STAGE staged messages per group)
    auto align8 = [](uint32_t x) { return (x + 7) & ~7u; };
    uint32_t off = 0;
    rt->lm_xchg = off;   off += m.warps * 2 * sizeof(uint32_t);
    rt->lm_vote = off;   off += 4 * sizeof(uint32_t);
    rt->lm_claim = off;
    if (cfg.sender == FGC_SENDER_CTA) off += 3 * 2 * m.block() * sizeof(uint32_t);
    rt->lm_term = off;
    if (cfg.term == FGC_TERM_T2) off += 2 * num_mb * 2 * sizeof(uint32_t);
    rt->lm_gbytes = 0;
    if (lg_group != 0) {
        const uint32_t G = 1u << lg_group;
        const uint32_t S = std::min<uint32_t>(G, FGC_STAGE);
        rt->lm_gbytes = align8(4 * sizeof(uint32_t) + G * sizeof(uint32_t) + S * sizeof(Staged<T>));
    }
    for (uint32_t mb = 0; mb < num_mb; ++mb) {
        off = align8(off);
        rt->lm_stage[mb] = off;
        if (lg_group != 0) off += (m.block() >> lg_group) * rt->lm_gbytes;
    }
    rt->lmem_size = align8(off);
}

// Local memory per CTA to launch with (the runtime's, or a word).
inline uint32_t lmem_bytes(const fgc_rt_args_t& rt) {
    return std::max<uint32_t>(rt.lmem_size, 4);
}

// Kernel cycles: the first core's start to the last core's end.
inline uint64_t kernel_cycles(const std::vector<uint64_t>& time) {
    uint64_t start = UINT64_MAX, end = 0;
    for (size_t c = 0; c + 1 < time.size(); c += 2) {
        start = std::min(start, time[c]);
        end   = std::max(end, time[c + 1]);
    }
    return end - start;
}

} // namespace fgc

#endif
