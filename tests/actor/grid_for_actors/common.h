// grid_for_actors: N×N channel grid; grid[s*N + r] = sender s -> receiver r.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

#define CAP 16   // ring capacity (power of 2)

typedef struct {
    uint32_t sender;            // who sent it
} packet_t;

typedef struct {
    volatile uint32_t head;     // consumer writes
    volatile uint32_t tail;     // producer writes
    packet_t slots[CAP];
} ring_t;

typedef struct {
    uint64_t grid_addr;         // ring_t[N*N]
    uint64_t status_addr;       // int[N] per-PE verdict (1=ok, 0=bad)
    uint32_t N;
} args_t;

#endif
