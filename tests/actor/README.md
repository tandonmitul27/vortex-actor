# Actor kernels on Vortex

## Sizes

Sizes are compile-time macros, set with `make CONFIGS="-DM=1000 -DCAP=64"`:

| Macro | Meaning |
|---|---|
| `M` | updates (histogram) or queries (index gather) per actor |
| `L` | buckets per actor (histogram) |
| `TABLE_SIZE` | table entries per actor (index gather) |
| `CAP` | slots per buffer |

## Histogram

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
| `baseline_histo` | **Atomic baseline**, not an actor kernel. Every thread adds straight into one shared table with an atomic add. |

## Index gather

Each actor asks `M` random owners for one entry of their table. The owner answers by
loading from its own table.

| Kernel | Description |
|---|---|
| `actor_ig` | **Actor baseline.** Requests travel on one mailbox and replies on a second. The request handler sends the reply. An actor announces done on replies only after it has a done on requests from everyone. |
| `actor_ig_2t` | Two threads per actor: one issues requests, the other serves requests and lands replies. It takes a request only once the reply buffer has room, so it never blocks. |
| `actor_ig_mpsc` | One shared inbox per receiver for requests and one for replies, instead of N buffers each. |
| `baseline_ig` | **Non-actor baseline.** Every thread reads the remote entry directly from the shared table. |

## Infrastructure tests

| Kernel | Description |
|---|---|
| `ring_for_actors` | One buffer: a producer sends N messages and a consumer checks their order. |
| `grid_for_actors` | The N × N buffer grid: each actor sends to its right neighbour and receives from its left. |
