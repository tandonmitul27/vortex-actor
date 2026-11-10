// ISA2 (2026-11-10): the final L2 append mechanism as the program sees it (sim/simx/types.h "ISA2").
//  - append: a new one-way atomic instruction, operands = queue number + message (4 or 8 B), no result;
//  - count:  a second new instruction, operand = queue number, returns the queue's message count this round;
//  - L1INV:  each core invalidates its L1 at the barrier that ends a send phase (all versions);
//  - layout: the program places queue q's line k at qbase + AQ_LSTR * ((q / AQ_BS) * LQ + k) + 64 * hash(q),
//            where the L2's ordinary address mapping puts every line of q in q's slice and set.
// ISA2=0 (with VORTEX_LEGACY=1 in the simulator) rebuilds the old kernels exactly.
// Include after the kernel's common.h (needs ACTOR, SWQ, GLANES).
#pragma once
#include <stdint.h>
#include <VX_config.h>
#ifndef ISA2
#define ISA2 1
#endif
#ifndef SWQ
#define SWQ 0
#endif
// the L2 append is in use: our actor versions with the L2 append on (SWQ = the append off / the baseline actor)
#if defined(ACTOR) && ACTOR == 1 && !SWQ && ISA2
#define AQ_ON 1
#else
#define AQ_ON 0
#endif

// L2 geometry: banks x sets (the hash range) and the distance between two lines of one queue
#define AQ_BS   ((uint32_t)(L2_CACHE_SIZE / (64 * L2_NUM_WAYS)))
#define AQ_LSTR ((uint64_t)64 * AQ_BS)
static inline uint32_t aq_qhash(uint32_t q) {            // the slice/set hash of the append (bijective per AQ_BS block)
    const uint64_t bs = AQ_BS, hi = q / bs;
    uint64_t lo = q % bs;
    lo = (lo ^ (hi * 0x9E5u)) & (bs - 1);
    lo = (lo * 0x9E3779B1u) & (bs - 1);
    return (uint32_t)(lo ^ (lo >> 7));
}
// line 0 of queue q in a region at qbase whose queues are lq lines long (qstride = 64 * lq)
static inline uint64_t aq_qline0(uint64_t qbase, uint32_t lq, uint32_t q) {
    return qbase + AQ_LSTR * ((uint64_t)(q / AQ_BS) * lq) + 64ull * aq_qhash(q);
}
// bytes a region of nq queues of qstride bytes needs (host allocation); qstride must be whole lines
static inline uint64_t aq_region_bytes(uint64_t nq, uint64_t qstride) {
#if AQ_ON
    return ((nq + AQ_BS - 1) / AQ_BS) * (qstride / 64) * AQ_LSTR;
#else
    return nq * qstride;
#endif
}

#if defined(__riscv)
// append: AMO opcode, funct5 0x1e; .w = 4-byte message, .d = 8-byte message; rd = x0 (one-way)
static inline void vx_aq_append32(uint32_t q, uint32_t w) {
    asm volatile(".insn r 0x2f, 0x2, 0x78, x0, %0, %1" :: "r"(q), "r"(w) : "memory");
}
static inline void vx_aq_append64(uint32_t q, uint64_t w) {
    asm volatile(".insn r 0x2f, 0x3, 0x78, x0, %0, %1" :: "r"(q), "r"(w) : "memory");
}
// count: AMO opcode, funct5 0x1f; answered by the queue's L2 slice
static inline uint32_t vx_aq_count(uint32_t q) {
    uint32_t c;
    asm volatile(".insn r 0x2f, 0x2, 0x7c, %0, %1, x0" : "=r"(c) : "r"(q) : "memory");
    return c;
}
// invalidate this core's L1 data cache (MISC-MEM funct3 7)
static inline void vx_l1inv(void) {
    asm volatile(".insn i 0x0f, 0x7, x0, x0, 0" ::: "memory");
}
// ISA2_NOINV=1: TIMING PROBE ONLY, not a machine. simx keeps data coherent outside its caches, so a run without the
// invalidate still passes; it prices the invalidate (every version pays it at each barrier).
#if ISA2 && !defined(ISA2_NOINV)
#define ISA2_L1INV() vx_l1inv()
#else
#define ISA2_L1INV() do { } while (0)
#endif
#endif

// Receive-side addressing. QCOUNT(QI): queue QI's count. QPTR(QI): its first element. QIX(k): element k of a queue
// relative to QPTR. QSTEP: elements between lane ln's consecutive warp-cooperative entries (k*GLANES + ln).
// Need `qs` (typed element pointer = the region base), `tails`, QS_W (queue stride in elements).
// The _ forms name the region explicitly (B = typed element pointer to the region base, W = queue stride in elements).
#if AQ_ON
#define QWPL_(B)        (64u / (uint32_t)sizeof(*(B)))                   // elements per line
#define QLSE_(B)        ((uint64_t)(AQ_LSTR / sizeof(*(B))))             // elements between two lines of a queue
#define QCOUNT(QI)      vx_aq_count((uint32_t)(QI))
#define QPTR_(B, W, QI) ((decltype(B))(uintptr_t)aq_qline0((uint64_t)(uintptr_t)(B), (uint32_t)((uint64_t)(W) * sizeof(*(B)) / 64), (uint32_t)(QI)))
#define QIX_(B, k)      ((size_t)((k) / QWPL_(B)) * QLSE_(B) + (k) % QWPL_(B))
#define QSTEP_(B)       ((size_t)((uint32_t)(GLANES) / QWPL_(B)) * QLSE_(B))
#else
#define QCOUNT(QI)      tails[QI]
#define QPTR_(B, W, QI) ((B) + (size_t)(QI) * (W))
#define QIX_(B, k)      (k)
#define QSTEP_(B)       GLANES
#endif
#define QPTR(QI)        QPTR_(qs, QS_W, QI)
#define QIX(k)          QIX_(qs, k)
#define QSTEP           QSTEP_(qs)

#if defined(__riscv)
// the append instruction for a 4- or 8-byte message type
#define aq_isa2_put(q, word) do { if (sizeof(word) == 8) vx_aq_append64((q), (uint64_t)(word)); \
                                  else vx_aq_append32((q), (uint32_t)(word)); } while (0)
#endif
