// Skewed keys for the skewed-traffic study (shared by every actor kernel, every
// baseline and their host references, so all of them draw the SAME keys).
// SKEW=1 (default): the uniform key h % K, bit for bit the old code.
// SKEW=p > 1: u^p sampling makes low keys hot (a power law), then a multiplicative
// scatter spreads the hot keys over owners and tiles, as hashing keys to actors does.
#pragma once
#include <stdint.h>
#ifndef SKEW
#define SKEW 1
#endif
static inline uint32_t skew_key(uint32_t h, uint32_t K) {
#if SKEW <= 1
    return h % K;
#else
    uint64_t v = h;
    for (int p = 1; p < (SKEW); p++) v = (v * (uint64_t)h) >> 32;   // u^SKEW in 32-bit fixed point
    const uint32_t x = (uint32_t)((v * (uint64_t)K) >> 32);           // hot near 0
    return (uint32_t)(((uint64_t)x * 2654435761u + 0x9E3779B9u) % K);
#endif
}
