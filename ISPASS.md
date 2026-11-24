# ISPASS branch: a base Vortex for fine-grained communication studies

This branch is upstream Vortex (`ccf811e11`) with four additions. The first
three are in SimX, and the fourth is software that runs on it:

1. **Four base machines** (`profiles/base_*.json`), from 1 to 16 cores.
2. **Per-PC counters.** Each warp-cycle is charged to one instruction and
   one state. Each memory request is counted at every level, by the
   instruction that issued it and by the buffer it touches.
3. **Two SimX bug fixes** that hang or abort large machines on upstream.
4. **An actor runtime and six actor applications**: histogram, index gather,
   PageRank pull, PageRank push, topological sort and random permutation.
   The runtime has 4 queue shapes, 3 sender granularities and 3 termination
   methods, chosen at compile time.

Everything is off by default except in the `base_*` profiles, so upstream
behaviour is unchanged unless you opt in. The RTL is untouched and has not
been checked against these changes.

Details of the actor runtime and the applications:
[`tests/regression/fgc_common/README.md`](tests/regression/fgc_common/README.md).

## Quick start

```bash
# 1. build a 4-core machine (SimX only; a full `make` also needs Verilator)
mkdir build_b4c && cd build_b4c
../configure --xlen=32 --tooldir=$HOME/tools_v3 --profile=base_4c
make -s -j8 third_party software
make -s -j8 -C sim/simx
make -s -C sw/runtime/simx DESTDIR=$PWD/sw/runtime

# 2. build and run the actor histogram (default runtime configuration)
mkdir -p tests/regression/hs && cd tests/regression/hs
cp ../../../../tests/regression/fgc_hs/Makefile .
make -s fgc_hs kernel.vxbin
LD_LIBRARY_PATH=../../../sw/runtime VORTEX_DRIVER=simx \
  VORTEX_FGC_STATS=stats.json VORTEX_FGC_TAGS=tags.txt \
  ./fgc_hs -c1 -m16 -l16          # prints KERNEL: cycles=... and PASSED!

# 3. charge the counters to the tagged steps of the source
python3 ../../../../tests/regression/fgc_common/fgc_map.py stats.json kernel.elf
```

`-c0` runs the same problem without actors (a plain atomic or gather kernel),
so every actor run has a reference on the same machine.

## Applications

Two applications for each communication motif:

| Motif | Applications | Messages |
|---|---|---|
| Move updates to data (fire-and-forget) | histogram (`fgc_hs`), PageRank push (`fgc_ps`) | one-way; the owner applies the update |
| Move fetches to data (active fetch) | index gather (`fgc_ig`), PageRank pull (`fgc_pl`) | request and response; the owner's state is unchanged |
| Move continuations to data (stateful expansion) | random permutation (`fgc_rp`), topological sort (`fgc_ts`) | a message changes the owner's state, which produces more messages |

Each application checks its result on the host and has a non-actor version
(`-c0`). Options and sizes are in
[`tests/regression/fgc_common/README.md`](tests/regression/fgc_common/README.md).

## Base machines

| Profile | Cores x warps x threads |
|---|---|
| `base_1c` | 1 x 4 x 4 |
| `base_4c` | 4 x 4 x 16 |
| `base_16c` | 16 x 4 x 16 |
| `base_16c32t` | 16 x 16 x 32 |

Each profile uses the stock `VX_config.toml` core, caches and memory, plus:

- the L2 on (more than one core needs it, or atomics lose updates);
- atomics with compare-and-swap (A + Zacas);
- the divergent indirect-call split on;
- a 10M-cycle stall watchdog.

Every profile builds for rv32 and rv64 (`--xlen=64`).

For another shape, copy a profile and change `VX_CFG_NUM_CORES` (under
`chip`) or `VX_CFG_NUM_WARPS` / `VX_CFG_NUM_THREADS` (under `core`). After
any profile change, rebuild the tree from scratch with `rm -rf`, because an
in-place reconfigure leaves stale objects. `third_party` is built inside the
source tree and shared by every build tree, so build it from one tree before
building several trees in parallel.

## Where things are

| Path | What |
|---|---|
| `sim/simx/fgc_stats.h` | counter definitions (per-PC states, memory flows, address tags) |
| `sim/simx/processor.cpp` | `fgc_dump`: writes the counter JSON |
| `sim/simx/mem/memory.cpp`, `mem/cache.cpp` | the two bug fixes |
| `profiles/base_*.json` | the base machines |
| `tests/regression/fgc_common/` | actor runtime (`fgc_actor.h`), host helpers, graph generator, `fgc_map.py`, README |
| `tests/regression/fgc_{hs,ig,pl,ps,ts,rp}` | the six applications |
| `tests/regression/simt_div` | divergence patterns (see below) |
| `tests/regression/fgc_pcmap`, `fgc_memmap` | known-answer tests for the counters |

## Counters

Set `VORTEX_FGC_STATS=<file>` and SimX writes JSON at the end of each run.
The JSON accumulates over all runs in the process. Without the variable the
counters are off and cost nothing.

**Per core, per PC** (`cores[i].pcs`, fields listed in `pc_fields`). Each
cycle, every active warp is charged to exactly one PC and one state, so the
states of a core sum to its warp-cycles.

| Field | Meaning |
|---|---|
| `issue` | warp-cycles issuing this PC (= uops issued) |
| `lanes` | active lanes summed over those uops; `lanes / issue` is the SIMT efficiency of the instruction |
| `scrb` | at the head of the warp, blocked on the scoreboard (waiting for an operand, usually a load) |
| `head` | at the head, blocked on anything else (a full unit queue, a barrier) |
| `fetch` | fetching this PC |
| `wait` | fetch held by this PC until it resolves: a branch or jump, a warp-control instruction (`tmc`, `split`/`join`, `wspawn`, barrier, `wsync`) or a system instruction |
| `idle` | ready to fetch this PC but not picked by the scheduler |
| `commits`, `latency` | uops committed, and their issue-to-commit cycles |
| `disp` | cycles a collected uop waited for its unit's queue |

Each core also reports the stock perf counters, among them `branches` and
`divergence`.

**Memory** (`l1`, `l2`, `l3`, `dram`). Each level counts what arrives from
above (`in`) and what it sends below (`out`) as `[rd, wr, wr_bytes, amo]`.
It counts them two ways:

- **`by_pc`**: by the load or store that caused them. The PC travels with
  the request down to the fills it causes; write-backs carry PC 0.
- **`by_tag`**: by buffer. Tag 0 is everything untagged and tag 1 is the
  thread stacks. With `VORTEX_FGC_TAGS=<file>`, the host writes one
  `name base size` line per buffer before launch (`fgc::Buffers::write_tags`
  in `fgc_host.h`), and SimX names the rest of the tags from that file.

What each memory level sees:

- `l1` is the data caches only. Instruction fetches reach the L2 without
  passing through `l1`.
- Host uploads and readbacks pass through the L2 with PC 0. Leave out PC 0
  to see the kernel's own traffic.
- Uncached (IO) requests bypass the caches, so they appear only at `dram`.
- `l3` stays zero unless the profile turns the L3 on.

**From PCs to source.** `fgc_map.py stats.json kernel.elf [--lines]
[--cores]` charges every PC to a step tagged in the source with comments.
The code itself has no markers.

- `// @step <name>`: the lines from here on belong to step `<name>`.
- `// @step end`: the lines from here on belong to no step; they take the
  step of their caller.
- `// @event <name>`: counts how often that line ran. Put it inside the
  branch you want to count.

The runtime is already tagged (`claim`, `publish`, `recv`, `process`, `term`,
`done`), and so are the applications. Kernels build with
`-gline-tables-only -gdwarf-4`, which adds line tables and leaves the code
unchanged.

## Starting points

**Branch divergence.**

- `lanes / issue` per PC, summed by step, shows where the warps lose lanes.
- `tests/regression/simt_div` has known patterns (`-m <mode>`):

  | Mode | Pattern |
  |---|---|
  | 0 | divergent indirect call |
  | 1 | per-lane while-loop exits |
  | 2 | loop on a per-lane done flag |
  | 3 | endless loop with an early break |
  | 4 | an actor-style drain loop |
  | 6 | uniform indirect call |
  | 8 | divergent jump table |
  | 9 | divergent tail call |

- The compiler emits no split/join around a divergent indirect call.
  `VX_CFG_ICALL_SPLIT_ENABLE` (on in `base_*`) splits it in SimX. Turn it
  off to see the failure.

**Queue overflow.**

- Every receiver queue is a ring. `-q` sets the slots per consumer actor
  (default 32), and `FGC_TOPO` sets how many senders share a ring.
- A sender always gets a ticket (`claim`). It writes the message only when
  that ticket's slot is free (`publish`). While the ring is full, the message
  waits in the sender's out-slot, a new `send` on that mailbox fails, and the
  actor processes its own mail before it retries. A full ring therefore
  shows up as repeated `publish` attempts and cycles in `recv`.
- To count the retries directly, add an `// @event` line on the
  `return false` path of `publish` in `fgc_actor.h`.

## Bug fixes

Both bugs reproduce on unmodified upstream `ccf811e11`.

1. **Uncached atomics hang** (`sim/simx/mem/memory.cpp`).
   - Cause: an atomic to an IO address bypasses every cache and reaches
     memory. Memory treated it as a write: it stored the raw operand and
     sent no response, so the warp's fence never released.
   - Where it shows: `printf` on 16 cores, once the print ring is full
     (`vx_putchar` counts lost characters with an `amoadd`).
   - Fix: memory now runs the atomic itself, with the LLC's semantics and
     its own LR/SC reservations, and answers like a read.
2. **MSHR overflow** (`sim/simx/mem/cache.cpp`, `assert(!mshr_.full())`).
   - Cause: a replay leaving the MSHR freed its slot, and admission gave the
     slot to a new request. If the replay's sector had been evicted
     meanwhile, the replay re-entered as a fresh miss into a full MSHR.
   - Where it shows: `lazy_fetch` and `dogfood` at 16 x 16 x 32, and the
     actor applications at 16 cores.
   - Fix: replays in the pipe keep their slot until they leave it.

## Validation

All runs below were built from this branch, and all pass.

- **Applications.** On every base machine (rv32, plus `base_16c` on rv64),
  the six applications run in the default runtime configuration, both the
  actor version (`-c1`) and the non-actor reference (`-c0`). On `base_1c`
  they also run in all 12 runtime configurations. After the last runtime
  change, every SPSC configuration ran again on `base_4c`. Each run checks
  its result on the host.
- **Upstream regression suite.** On `base_1c`, every test in
  `tests/regression` passes except `probe`, a debug tool, and `spin`, an
  intentional livelock. The other machines passed the same suite on the tree
  this branch was cut from.
- **Counters,** on every base machine:
  - `fgc_pcmap`'s event counts are exact.
  - `fgc_memmap`'s bytes, line fills and atomics per buffer are exact, at
    the L1 and the L2.
  - Each core's per-PC states sum to the stock perf counters: states to
    active warp-cycles, issued uops to `issued_warps`, lanes to
    `issued_threads`, and commits to `instrs`.
  - At every memory level, the per-buffer totals equal the per-PC totals.
  - Every kernel store carries its PC below the L1.

## Known limits

- **SPSC queues and T0 termination grow as N^2.** This is a property of
  those designs, not a defect.
  - SPSC keeps one ring per (sender, receiver) pair. With thread senders,
    16 x 16 x 32 needs 8192 x 8192 rings, which do not fit in device memory.
  - With CTA senders there is one ring per (core, receiver) pair. Every
    receiver polls one ring from each core, and once the rings outgrow the L2,
    every empty poll is a DRAM read.
  - The counters show this on 16 cores, where 91% of warp-cycles are in
    `recv`, waiting on those reads. At 16 x 16 x 32, even the smallest inputs
    take 25M to 40M cycles (histogram and PageRank push), against 0.8M with
    the default queue.
- **Some upstream tests need other sizes on big machines.** These are limits
  of the tests' own design, not of the machine or SimX.
  - `ldst_order` and `raw_order` split their buffer into one group per warp.
    From 16 cores up, the default 1024 words leave warps with nothing to
    test, so pass `-n65536`.
  - `wsync` steps through a 1024-line buffer that wraps at iteration 128,
    so pass `-i127`.
  - At 16 x 16 x 32, `amo`, `sort` and `diverge` pass at their default sizes
    but take hours to a day. Use `-n2` for `amo` and `-n4` for the other
    two.

## Running many simulations

`make run-simx` in one tree races on the runtime stamp. Build each test
once, then launch the binary directly; many can then run in parallel from
one tree:

```bash
LD_LIBRARY_PATH=<tree>/sw/runtime VORTEX_DRIVER=simx ./<test> <options>
```

Do not run an upstream test that sets `CONFIGS` in a shared tree, because
`run-simx` rebuilds the tree's SimX with them.

`VORTEX_CP_POLL_TIMEOUT_S` can fire on slow but correct runs when the host is
loaded. Set it generously.
