# Fine-grained actor messaging on Vortex (SimX)

A SimX platform for studying fine-grained communication on GPUs, and an actor
runtime with HClib bale_actor semantics built on it. It contains:

- simulator additions: per-PC counters and a split for divergent indirect
  calls;
- the base machine profiles (`base_*`, see `ISPASS.md` at the repository
  root);
- an actor runtime (`fgc_actor.h`) covering 4 queue topologies, 3 sender
  granularities and 3 termination methods, chosen at compile time;
- six actor applications, each with a non-actor reference and a host check;
- `fgc_map.py`, which maps the per-PC counters to steps tagged in the source.

## 1. Build a machine

One build tree per profile (`profiles/<name>.json`). After any profile change
rebuild the tree from scratch (`rm -rf`); an in-place reconfigure leaves stale
objects.

```bash
mkdir build_b4c && cd build_b4c
../configure --xlen=32 --tooldir=$HOME/tools_v3 --profile=base_4c
make -s third_party software
make -s -C sim/simx
make -s -C sw/runtime/simx DESTDIR=$PWD/sw/runtime
```

| Profile | Cores x warps x threads |
|---|---|
| `base_1c` | 1 x 4 x 4 |
| `base_4c` | 4 x 4 x 16 |
| `base_16c` | 16 x 4 x 16 |
| `base_16c32t` | 16 x 16 x 32 |

All four have the L2 on, atomics with compare-and-swap (A + Zacas), the
indirect-call split on and a 10M-cycle stall watchdog; each builds for rv32
and rv64.

## 2. Platform rules

Each of these cost real debugging time; the code here follows all of them.

- **The L1 is not coherent across cores.** Read anything another core writes
  with LR (`fgc::l2_load32`, `l2_load64`, `l2_load<T>` in `fgc_actor.h`): the
  L2 answers it like a plain read without touching the L1. A plain load can
  return a stale L1 line even after `vx_fence()`. Data only its owner touches
  can use plain accesses.
- **Divergent indirect calls.** The compiler emits no split/join around an
  indirect call whose target differs across lanes. The `base_*` profiles
  turn on `VX_CFG_ICALL_SPLIT_ENABLE`, which splits such a
  call in SimX and rejoins at the return. It splits linking calls only, so
  kernels build with `-fno-optimize-sibling-calls`.
- **Compare-and-swap needs Zacas.** Without it, CAS becomes an lr/sc loop
  with no split/join, which hangs divergent warps. All `base_*` profiles have
  it.
- **More than one core needs the L2.** Without it, atomics run at each core's
  L1 and lose updates (all `base_*` profiles turn it on).
- **Debug info:** build kernels with `-gline-tables-only -gdwarf-4`. The code
  is unchanged, and lld segfaults on DWARF 5 line tables for some rv64 kernels.
- **Never run upstream tests that set `CONFIGS` in a study tree:** `run-simx`
  rebuilds the tree's SimX with them.
- **Parallel runs:** `make run-simx` in one tree races on the runtime stamp.
  Build once, then launch the binary directly:
  `LD_LIBRARY_PATH=<tree>/sw/runtime VORTEX_DRIVER=simx ./app <options>`.
- **`VORTEX_CP_POLL_TIMEOUT_S`** fires on slow but correct runs when the host
  is loaded; set it generously, or not at all for full-size runs.

## 3. The actor runtime

`fgc_actor.h` (device), `fgc_actor_args.h` (shared), `fgc_host.h` (host),
`fgc_graph.h` (Bale Erdos-Renyi graphs).

**Model (HClib selectors).** One actor is one thread; every hardware thread
is an actor (one CTA of warps x threads per core). An actor has up to 4
mailboxes, each with a handler run by the owner one message at a time.

- `send(mb, msg, dst)` is non-blocking. It fails while the mailbox's out-slot
  still holds an earlier message; the actor then keeps processing its own mail.
- `done(mb)`: this actor sends nothing more on `mb`.
- A mailbox ends when every actor is done with it and every message sent on it
  has been processed; each actor then calls `done` on the next mailbox.
- `run(app)` returns when every mailbox has ended.
- `next_epoch()` starts a new finish scope, for iterative apps.

**An app** provides `main_step(Rt&)`, `process(Rt&, mb, msg, from)` and
`kSendsFrom[NMB]`, a bit mask per mailbox of the mailboxes its handler sends
on. A message is held back only while the out-slot its handler needs is taken,
which keeps request-reply chains free of deadlock. `tests/regression/fgc_hs`
is the smallest example.

**Configurations, compiled in** (`make FGC_TOPO=.. FGC_SENDER=.. FGC_TERM=..`;
the kernel and the host get the same macros; use one build directory per
configuration):

| Make variable | Values |
|---|---|
| `FGC_TOPO` | 0 SPSC: one queue per (producer, receiver); 1 MPSC-actor: one queue per receiver actor (default); 2 MPSC-warp: one per receiver warp; 3 MPSC-CTA: one per receiver CTA |
| `FGC_SENDER` | 0 thread: each lane claims its own slot (default); 1 warp: lanes sending to one queue share a claim; 2 CTA: the CTA's lanes share one claim per queue |
| `FGC_TERM` | 0 T0: a done marker to every actor; 1 T1: global counters (default); 2 T2: counters summed per CTA first |
| `FGC_STAGE` | messages a receiver group takes off its queue per batch (default 256) |

Under SPSC the producer is the sender unit (thread, warp or CTA). SPSC and T0
grow as N^2 and fit only small machines.

**Buffers.** A queue is a ring of slots. A sender takes a ticket on the
queue's tail (an atomic add, or a plain add under SPSC). A per-slot sequence
word marks the slot free (2 x lap) or published (2 x lap + 1), so a zeroed
ring is empty. Ring slots (`-q`) are counted per consumer actor, so a queue
read by a group holds the group's share.

## 4. The applications

Each lives in `tests/regression/fgc_<app>`. `-c0` runs the non-actor
reference, `-c1` the actor version. `-q` ring slots, `-r` messages per receive
pass (R), `-t` idle passes per termination check (K), `-k` kernel file. Each
prints `KERNEL: cycles=` (the first core's start to the last core's end,
uploads excluded) and checks its result on the host.

| App | Mailboxes | Sizes |
|---|---|---|
| `fgc_hs` histogram | 1 | `-m` updates per actor, `-l` buckets per actor |
| `fgc_ig` index gather | REQUEST, RESPONSE | `-m` requests per actor, `-l` table entries per actor |
| `fgc_pl` PageRank pull | REQUEST, RESPONSE, one epoch per iteration | `-l` rows per actor, `-z` nonzeros per row, `-i` iterations |
| `fgc_ps` PageRank push | 1, one epoch per iteration | `-l`, `-z`, `-i` |
| `fgc_ts` topological sort (matching phase) | 1 | `-l` rows per actor, `-z` nonzeros per row |
| `fgc_rp` random permutation (dart throwing) | THROW, REPLY | `-l` darts per actor |

```bash
cd build_b4c/tests/regression && mkdir hs_cta && cd hs_cta
cp ../../../../tests/regression/fgc_hs/Makefile .
make -s fgc_hs kernel.vxbin FGC_TOPO=3 FGC_SENDER=2
LD_LIBRARY_PATH=../../../sw/runtime VORTEX_DRIVER=simx ./fgc_hs -c1 -m16 -l16
```

## 5. Counters

- `VORTEX_FGC_STATS=<file>`: per-PC counters as JSON. Every cycle each active
  warp is charged to one PC and one state (issue, scoreboard, head, fetch,
  wait, idle). Memory requests carry their PC through L1, L2 and DRAM.
- `VORTEX_FGC_TAGS=<file>`: the host writes each buffer's name and range
  (`fgc::Buffers::write_tags`), and requests are also split by buffer.
- `fgc_map.py stats.json kernel.elf [--lines] [--cores]` maps PCs to
  `// @step <name>` regions in the source (no markers in the code). A line in
  no step, before a file's first tag or after `// @step end`, takes its
  caller's step. `// @event <name>` counts how often a line ran.

## 6. Known limits

- SPSC queues and T0 grow as N^2. At 16 x 16 x 32, SPSC with CTA senders
  (`FGC_TOPO=0 FGC_SENDER=2`) costs about 40M cycles per run even at the
  smallest inputs; use it on smaller machines.
