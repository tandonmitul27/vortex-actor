# Actor kernels on Vortex

Kernels for running the actor model on Vortex. Actors talk only by messages, in rounds:
everyone sends, a barrier, then every actor receives and applies its own messages.

There are two kinds of kernels here:
- the **append-queue kernels** (`*_aq`), built on the L2 append described below;
- the **message-passing kernels** from the first study, at the end of this file.

## The L2 append in the simulator (`sim/simx`)

Every message goes to its receiver's queue. The L2 packs the messages from all senders into full
cache lines, so each queue reaches DRAM as whole lines.

Two instructions:

| Instruction | Encoding | What it does |
|---|---|---|
| `AQAPPEND.w` / `.d q, msg` | AMO opcode `0x2F`, funct5 `0x1e`, funct7 `0x78`, `rd = x0` | Sends a 4 or 8 B message to queue `q`. One-way, with no result. It skips the sender's L1 and goes to the L2 slice (bank) that owns `q`. |
| `AQCOUNT q` | AMO opcode `0x2F`, funct5 `0x1f`, funct7 `0x7c` | Returns how many messages `q` received this round. The slice computes the number in one pipeline pass, with no DRAM access. |

What the L2 slice does:
- **Open line:** each queue has one open (part-filled) line, which may use any way of its set. Each line has one open bit.
- **Next slot:** the next message goes to the slot after the line's last written byte. The slice fills slots in order, so the written bytes always form one unbroken run.
- **Full line:** the line's open bit is cleared and it is marked to be evicted first. The queue's next line opens at once, without a DRAM read.
- **Early eviction:** if an open line is evicted early, the queue's count is saved in a pinned area of the L2 (2 B per queue). The next append reopens the line without a DRAM read and continues at the saved slot. A read of a reopened line fetches its earlier bytes from DRAM once and merges them.
- **Count:** `AQCOUNT` = line number × messages per line + slots written in the open line. If the open line was evicted early, it is the saved count; if the queue received nothing, it is 0.

Supporting changes:
- **No L1 for atomics:** atomics, appends and counts skip the L1.
- **Fence:** the fence waits until every write has been acknowledged.
- **`L1INV`:** MISC-MEM funct3 7 invalidates the L1 before the receive.

The intrinsics and the queue layout macros are in `isa2.h`.

This is the simulator's default L2 append, used by the proposal's actor version (`-DACTOR=1`); no setting is needed to
select it. The new hardware is one open bit per L2 line (2 KB for a 1 MB L2) and the queue descriptors (768 B),
about 2.8 KB in all, plus the pinned count area, which takes 2 B per queue from the ordinary L2 capacity.

## Workloads

Each kernel directory holds the actor version and the non-actor version of one workload.

| Kernel | Workload | Non-actor version | Size |
|---|---|---|---|
| `actor_histo_aq` | Histogram | `baseline_histo` | `-DM` updates and `-DL` buckets per actor |
| `actor_ig_aq` | Index gather (request and reply) | `baseline_ig` | `-DM` queries per actor, `-DTABLE_SIZE` entries per actor |
| `transpose_aq` | Sparse matrix transpose | `-DACTOR=0` | env `TR_LROWS` |
| `randperm_aq` | Random permutation | `-DACTOR=0` | `-DRP_L` |
| `topo_aq` | Topological sort | `-DACTOR=0` | `-DTP_L` and env `TP_LROWS` |
| `pr_aq` | PageRank, push | `-DACTOR=0` | env `PR_LROWS` |
| `prp_aq` | PageRank, pull (request and reply) | `-DACTOR=0` | env `PR_LROWS` |
| `sssp_aq` | Single-source shortest paths | `-DACTOR=0` | env `SS_LROWS` |
| `tc_aq` | Triangle counting | `-DACTOR=0` | env `TC_MTX=<graph>.mmio` and `TC_DEGORD=1` |
| `permute_aq` | Sparse matrix permutation (bale's `permute_matrix`) | `-DACTOR=0` | env `PM_LROWS` |

- `*_LROWS` is rows per actor at 256 actors, like `l_numrows` in bale.
- The TC graphs are not in the repository. They are SNAP graphs (`roadNet-PA`, `amazon0312`, `loc-gowalla`, `soc-Slashdot0811`, `soc-Epinions1`, `ca-HepTh`, `email-EuAll`, `ca-GrQc`) and one random Erdős–Rényi graph, stored as `<name>_adj.mmio` adjacency lists.

Shared headers:
- `isa2.h`: the instructions and the queue layout.
- `abl.h`: the switches listed below.
- `swq.h`: software queues.
- `skew.h`: input skew.
- `lmem_check.h`: scratchpad layout checks.
- `probe_apply.h`: measurement-only timers for the motivation figure (`PROBE_APPLYT=1`, `ROUND1=1`). Off by default, and then the binary is unchanged.

### Versions and switches (compile flags)

- **`-DACTOR=1`:** our design, with all four pieces on.
- **`-DACTOR=1 -DBASELINE=1`:** the plain actor version, as it would run on a GPU today. It uses software queues, one actor per lane, no tiles, and one lane per queue at receive.
- **`-DACTOR=0`:** the non-actor version.

Each piece of the design can be switched off by itself:

| Piece | What it does | Off |
|---|---|---|
| A | L2 append: one request per message, and the L2 packs lines | `-DSWQ=1` and env `VORTEX_SWQ=1` (atomic add on the tail, then a store) |
| B | Tiles: queues keyed by (owner, tile), sized so the lines being filled fit in the L2 | `-DNOTILE=1` |
| C | Whole-warp receive: a warp drains one queue a line at a time | `-DLANE_RECV=1` |
| D | Actor placement: the actor count is chosen apart from the lane count | `-DNASSUME=1024 -DSPLIT=1` (one actor per lane) |

The histogram and index gather kernels use their own names for the same pieces; see their `common.h`.
`permute_aq` has only our design and the non-actor version, with no switches. Its row tiles are 128 rows (`RTILE`),
so the output lines being filled at once fit the L2. At `PM_LROWS=10000` it needs `-DAQ_SIZE=0x200000` and
`VORTEX_AQ_SIZE=0x200000` for its 22,784 queues.
Skewed inputs (randperm, shortest paths, TC) also split the sending work by edge rather than by row.

Scratchpad size: the kernels assume 16 KB per core (`-DLMEM_BYTES=16384`). For another size, build the simulator with
`-DLMEM_LOG_SIZE=<log2 bytes>` and pass the same `-DLMEM_LOG_SIZE` and a matching `-DLMEM_BYTES` to the kernel.

## Build and run

The machine shape is fixed when the simulator is compiled. Build it once per shape:

```bash
make -C build/runtime/simx clean
make -C build/runtime/simx CONFIGS="-DNUM_CORES=16 -DNUM_WARPS=4 -DNUM_THREADS=16 -DL2_ENABLE -DL2_WRITEBACK=1"
```

Without `CONFIGS` you silently get the default machine: 1 core, 4 warps, 4 threads and no L2.
The first line of every run prints `[MACHINE]`; check it.

Kernels built with `-DSTACK_COLOR=1` link a second kernel runtime that spreads the lane stacks across
L2 banks. Build that runtime once:

```bash
mkdir -p build/kernel_color
sed 's/^CFLAGS += -DXLEN_$(XLEN)$/CFLAGS += -DXLEN_$(XLEN) -DSTACK_COLOR=1/' build/kernel/Makefile > build/kernel_color/Makefile
make -C build/kernel_color
```

Build and run a kernel, here topological sort with our design:

```bash
M="-DNUM_CORES=16 -DNUM_WARPS=4 -DNUM_THREADS=16 -DL2_ENABLE"
make -C build/tests/actor/topo_aq CONFIGS="$M -DACTOR=1 -DNASSUME=256 -DSPLIT=4 -DSTACK_COLOR=1 -DHWBAR=1 -DWC_U=4 -DTP_L=1000" \
     VORTEX_KN_PATH=$PWD/build/kernel_color
cd build/tests/actor/topo_aq
LD_LIBRARY_PATH=../../../runtime:$LD_LIBRARY_PATH VORTEX_DRIVER=simx VORTEX_PROFILING=2 \
  VORTEX_L2_NOFETCH=1 VORTEX_L2_NOFETCH_ALL=1 VORTEX_AMO_RT=1 VORTEX_AQ_CHARGE=4 \
  VORTEX_AQ_BASE=0xD0000000 VORTEX_AQ_SIZE=0x100000 TP_LROWS=1000 ./topo_aq
```

Runtime settings:
- **`VORTEX_L2_NOFETCH=1` and `VORTEX_L2_NOFETCH_ALL=1`:** a write miss allocates its line without reading it, as GPU L2s do with byte masks. This applies to every kernel alike.
- **`VORTEX_AMO_RT=1`:** an atomic that returns a value is timed as a round trip.
- **`VORTEX_AQ_CHARGE=4`:** runs the L2 append.
- **`VORTEX_AQ_BASE` and `VORTEX_AQ_SIZE`:** the queue region.

A run ends with `PASSED` or `FAILED` and `PERF:` lines. The total is in `PERF: instrs=..., cycles=...`.

Other simulator changes:
- **`-DXBAR_RR`** (simulator `CONFIGS`): round-robin arbitration in the crossbar. The default fixed priority can starve a
  core and hang polling kernels on 32 cores.
- **Timeout:** the runtime waits up to 30 days for the device (it was 24 h, and longer runs read back an unfinished device).
- **Diagnostics,** off unless set: `VORTEX_MEM_PAGEHIST=<file>` writes DRAM reads and writes per 4 KB page;
  `VORTEX_EPOCH_RANGES=lo-hi[,lo-hi...]` counts, between global barriers, the distinct 64 B lines of those hex
  ranges that cores read and write; `VORTEX_ALLOC_LOG=1` prints each device allocation.

## Message-passing kernels (first study)

Sizes are compile-time macros, set with `make CONFIGS="-DM=1000 -DCAP=64"`:

| Macro | Meaning |
|---|---|
| `M` | updates (histogram) or queries (index gather) per actor |
| `L` | buckets per actor (histogram) |
| `TABLE_SIZE` | table entries per actor (index gather) |
| `CAP` | slots per buffer |

### Histogram

Each actor makes `M` random updates. Bucket `G` belongs to actor `G mod N`.

| Kernel | Description |
|---|---|
| `actor_histo` | **Actor baseline.** One message per update to the bucket's owner, then a done message to every actor, then drain the mailbox until all N dones arrive. |
| `actor_histo_2t` | Two threads per actor, split by warp: one sends while the other drains, instead of one thread doing both in turn. |
| `actor_histo_tree` | A tree barrier replaces the N × N done messages. After the barrier, each actor drains every buffer once. |
| `actor_histo_2t_tree` | Two threads per actor plus the tree barrier: the sender sends and joins the barrier, the receiver waits for release and drains once. |
| `actor_histo_kmux` | Interleaves sending and draining: send `K` updates, drain the mailbox, repeat. Terminates with the tree barrier. |
| `actor_histo_mpsc` | One shared inbox per receiver instead of N buffers. Senders claim a slot with an atomic ticket, and the receiver reads one inbox instead of scanning N. |
| `actor_histo_mpsc_tree` | The shared inbox with tree termination instead of done messages. |
| `actor_histo_agg_amo` | **Sender-side pre-aggregation.** Folds its updates into one count vector per owner and sends at most one message per owner, instead of one per update. |
| `actor_histo_agg_amo_lmem` | **On-chip pre-aggregation.** As sender-side pre-aggregation, but a vector for an owner on the same core stays in that core's scratchpad. Only cross-core vectors go through global memory. |
| `actor_histo_agg_comb` | **Per-core combining.** No messages: all actors on a core add into one shared table in the scratchpad, the core writes it out once, and each owner sums its buckets across the cores. The whole histogram must fit in the scratchpad. |

### Index gather

Each actor asks `M` random owners for one entry of their table. The owner answers by
loading from its own table.

| Kernel | Description |
|---|---|
| `actor_ig` | **Actor baseline.** Requests travel on one mailbox and replies on a second. The request handler sends the reply. An actor announces done on replies only after it has a done on requests from everyone. |
| `actor_ig_2t` | Two threads per actor: one issues requests, the other serves requests and lands replies. It takes a request only once the reply buffer has room, so it never blocks. |
| `actor_ig_mpsc` | One shared inbox per receiver for requests and one for replies, instead of N buffers each. |

### Infrastructure tests

| Kernel | Description |
|---|---|
| `ring_for_actors` | One buffer: a producer sends N messages and a consumer checks their order. |
| `grid_for_actors` | The N × N buffer grid: each actor sends to its right neighbour and receives from its left. |
