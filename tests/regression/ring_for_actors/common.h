// ring_for_actors: single-producer/single-consumer ring smoke test.
#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>

#define CAP 16   // ring capacity (power of 2)

typedef struct {
    int value;
} packet_t;

// SPSC ring; buffered count = tail - head (counters free-run, never wrap)
typedef struct {
    volatile uint32_t head;   // consumer writes
    volatile uint32_t tail;   // producer writes
    packet_t slots[CAP];
} ring_t;

typedef struct {
    uint64_t ring_addr;
    uint64_t status_addr;     // status[3] = {errors, produced, consumed}
    uint32_t num_messages;
} args_t;

#endif
