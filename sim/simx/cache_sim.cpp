// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cache_sim.h"
#include "debug.h"
#include "types.h"
#include <util.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <list>
#include <queue>

using namespace vortex;

struct params_t {
	uint32_t sets_per_bank;
	uint32_t lines_per_set;
	uint32_t words_per_line;
	uint32_t log2_num_inputs;

	int32_t word_select_addr_start;
	int32_t word_select_addr_end;

	int32_t bank_select_addr_start;
	int32_t bank_select_addr_end;

	int32_t set_select_addr_start;
	int32_t set_select_addr_end;

	int32_t tag_select_addr_start;
	int32_t tag_select_addr_end;

	params_t(const CacheSim::Config& config) {
		int32_t offset_bits = config.L - config.W;
		int32_t index_bits = config.C - (config.L + config.A + config.B);
		assert(offset_bits >= 0);
		assert(index_bits >= 0);

		this->log2_num_inputs = log2ceil(config.num_inputs);

		this->sets_per_bank  = 1 << index_bits;
		this->lines_per_set  = 1 << config.A;
		this->words_per_line = 1 << offset_bits;

		// Word select
		this->word_select_addr_start = config.W;
		this->word_select_addr_end = (this->word_select_addr_start+offset_bits-1);

		// Bank select
		this->bank_select_addr_start = (1+this->word_select_addr_end);
		this->bank_select_addr_end = (this->bank_select_addr_start+config.B-1);

		// Set select
		this->set_select_addr_start = (1+this->bank_select_addr_end);
		this->set_select_addr_end = (this->set_select_addr_start+index_bits-1);

		// Tag select
		this->tag_select_addr_start = (1+this->set_select_addr_end);
		this->tag_select_addr_end = (config.addr_width-1);
	}

	uint32_t addr_bank_id(uint64_t addr) const {
		if (bank_select_addr_end >= bank_select_addr_start)
			return (uint32_t)bit_getw(addr, bank_select_addr_start, bank_select_addr_end);
		else
			return 0;
	}

	uint32_t addr_set_id(uint64_t addr) const {
		if (set_select_addr_end >= set_select_addr_start)
			return (uint32_t)bit_getw(addr, set_select_addr_start, set_select_addr_end);
		else
			return 0;
	}

	uint64_t addr_tag(uint64_t addr) const {
		if (tag_select_addr_end >= tag_select_addr_start)
			return bit_getw(addr, tag_select_addr_start, tag_select_addr_end);
		else
			return 0;
	}

	uint64_t mem_addr(uint32_t bank_id, uint32_t set_id, uint64_t tag) const {
		uint64_t addr(0);
		if (bank_select_addr_end >= bank_select_addr_start)
			addr = bit_setw(addr, bank_select_addr_start, bank_select_addr_end, bank_id);
		if (set_select_addr_end >= set_select_addr_start)
			addr = bit_setw(addr, set_select_addr_start, set_select_addr_end, set_id);
		if (tag_select_addr_end >= tag_select_addr_start)
			addr = bit_setw(addr, tag_select_addr_start, tag_select_addr_end, tag);
		return addr;
	}
};

struct line_t {
	uint64_t tag;
	uint32_t lru_ctr;
	bool     valid;
	bool     dirty;
	bool     pending;   // way reserved for an in-flight fill (VORTEX_CACHE_RESV)

	void reset() {
		valid = false;
		dirty = false;
		pending = false;
	}
};

// VORTEX_CACHE_RESV (default 1): a miss RESERVES its victim way until the fill lands, and
// the victim is clean once written back. Without it, every miss to a set while a fill is in
// flight picked the same LRU way: fills overwrote each other (lines lost, fetched again) and
// a dirty victim was written back once per colliding miss. 0 = the old model.
// VORTEX_L2_NOFETCH_ALL=1 (with VORTEX_L2_NOFETCH=1): EVERY write miss allocates without a
// fetch, not only queue storage. Real GPU L2s (NVIDIA) write-allocate with per-sector byte
// masks and do not read the line on a write miss. Applies to every kernel alike (fair).
static const bool g_nofetch_all = getenv("VORTEX_L2_NOFETCH_ALL") && getenv("VORTEX_L2_NOFETCH_ALL")[0] == '1';
// VORTEX_AQ_TAILTAB=N (2026-11-03, L2 append realism): an L2 bank that accepts an append (a write
// into queue storage) takes no input for the next N cycles, charging the tail-table
// read-modify-write against the slice's own request slots. Off (0) by default.
static const bool g_aq_spill_l2 = getenv("VORTEX_AQ_SPILL") ? std::string(getenv("VORTEX_AQ_SPILL")) == "l2" : aq_final_on();
static const bool g_aq_protect = getenv("VORTEX_AQ_PROTECT") && getenv("VORTEX_AQ_PROTECT")[0] == '1';
static const uint32_t g_aq_tailtab = getenv("VORTEX_AQ_TAILTAB") ? (uint32_t)atoi(getenv("VORTEX_AQ_TAILTAB")) : 0;
static const bool g_cache_resv = !(getenv("VORTEX_CACHE_RESV") && getenv("VORTEX_CACHE_RESV")[0] == '0');

struct set_t {
	std::vector<line_t> lines;

	set_t(uint32_t num_ways)
		: lines(num_ways)
	{}

	void reset() {
		for (auto& line : lines) {
			line.reset();
		}
	}

	int tag_lookup(uint64_t tag, int* free_line_id, int* repl_line_id) {
		uint32_t max_cnt = 0;
		int hit_line_id = -1;
		*free_line_id = -1;
		*repl_line_id = -1;
		for (uint32_t i = 0, n = lines.size(); i < n; ++i) {
			auto& line = lines.at(i);
			const bool resv = g_cache_resv && line.pending;
			if (!resv && max_cnt < line.lru_ctr) {
				max_cnt = line.lru_ctr;
				*repl_line_id = i;
			}
			if (line.valid) {
				if (line.tag == tag) {
					hit_line_id = i;
					line.lru_ctr = 0;
				} else {
					++line.lru_ctr;
				}
			} else if (!resv) {
				*free_line_id = i;
			}
		}
		if (*repl_line_id == -1) {   // every candidate at lru 0 or reserved: first unreserved, else way 0
			*repl_line_id = 0;
			for (uint32_t i = 0, n = lines.size(); i < n; ++i)
				if (!(g_cache_resv && lines.at(i).pending)) { *repl_line_id = i; break; }
		}
		return hit_line_id;
	}
};

struct bank_req_t {

  using Ptr = std::shared_ptr<bank_req_t>;

	enum ReqType {
		None   = 0,
		Replay = 2,
		Core   = 3
	};

	uint64_t addr_tag;
	uint32_t set_id;
	uint32_t cid;
	uint64_t req_tag;
	uint64_t uuid;
	ReqType  type;
	bool     write;
	bool     amo = false;
	uint8_t  aqop = 0;   // ISA2: 1 append, 2 count

	bank_req_t() {
		this->reset();
	}

	void reset() {
		type = ReqType::None;
	}

	friend std::ostream &operator<<(std::ostream &os, const bank_req_t& req) {
		os << "set=" << req.set_id << ", rw=" << req.write;
		os << ", type=" << req.type;
		os << ", addr_tag=0x" << std::hex << req.addr_tag;
		os << ", req_tag=" << req.req_tag;
		os << ", cid=" << std::dec << req.cid;
		os << " (#" << req.uuid << ")";
		return os;
	}
};

struct mshr_entry_t {
	bank_req_t bank_req;
	uint32_t line_id;

	mshr_entry_t() {}

	void reset() {
		bank_req.reset();
	}
};

class MSHR {
public:
	MSHR(uint32_t size)
		: entries_(size)
		, ready_reqs_(0)
		, size_(0)
	{}

	uint32_t capacity() const {
		return entries_.size();
	}

	uint32_t size() const {
		return size_;
	}

	bool empty() const {
		return (0 == size_);
	}

	bool full() const {
		assert(size_ <= entries_.size());
		return (size_ == entries_.size());
	}

	bool has_ready_reqs() const {
		return (ready_reqs_ != 0);
	}

	bool lookup(const bank_req_t& bank_req) {
		for (auto& entry : entries_) {;
			if (entry.bank_req.type != bank_req_t::None
		 	 && entry.bank_req.set_id == bank_req.set_id
		   && entry.bank_req.addr_tag == bank_req.addr_tag) {
				return true;
			}
		}
		return false;
	}

	int enqueue(const bank_req_t& bank_req, uint32_t line_id) {
		assert(bank_req.type == bank_req_t::Core);
		for (uint32_t i = 0, n = entries_.size(); i < n; ++i) {
			auto& entry = entries_.at(i);
			if (entry.bank_req.type == bank_req_t::None) {
				entry.bank_req = bank_req;
				entry.line_id = line_id;
				++size_;
				return i;
			}
		}
		std::abort();
		return -1;
	}

	mshr_entry_t& replay(uint32_t id) {
		auto& root_entry = entries_.at(id);
		assert(root_entry.bank_req.type == bank_req_t::Core);
		assert(ready_reqs_ == 0);
		// mark all related mshr entries for replay
		for (auto& entry : entries_) {
			if (entry.bank_req.type == bank_req_t::Core
			 && entry.bank_req.set_id == root_entry.bank_req.set_id
			 && entry.bank_req.addr_tag == root_entry.bank_req.addr_tag) {
				entry.bank_req.type = bank_req_t::Replay;
				++ready_reqs_;
			}
		}
		return root_entry;
	}

	void dequeue(bank_req_t* out) {
		assert(ready_reqs_ > 0);
		for (auto& entry : entries_) {
			if (entry.bank_req.type == bank_req_t::Replay) {
				*out = entry.bank_req;
				entry.bank_req.type = bank_req_t::None;
				--ready_reqs_;
				--size_;
				break;
			}
		}
	}

	void reset() {
		for (auto& entry : entries_) {
			entry.reset();
		}
		size_ = 0;
	}

private:
	std::vector<mshr_entry_t> entries_;
	uint32_t ready_reqs_;
	uint32_t size_;
};

class CacheBank : public SimObject<CacheBank> {
public:
	SimPort<MemReq> core_req_port;
  SimPort<MemRsp> core_rsp_port;

  SimPort<MemReq> mem_req_port;
  SimPort<MemRsp> mem_rsp_port;

  CacheBank(const SimContext& ctx,
	          const char* name,
	          const CacheSim::Config& config,
				    const params_t& params,
						uint32_t bank_id)
    : SimObject<CacheBank>(ctx, name)
		, core_req_port(this)
		, core_rsp_port(this)
		, mem_req_port(this)
		, mem_rsp_port(this)
		, config_(config)
	  , params_(params)
		, bank_id_(bank_id)
		, sets_(params.sets_per_bank, params.lines_per_set)
		, mshr_(config.mshr_size)
		, pipe_req_(TFifo<bank_req_t>::Create("", config.latency-1))
	{
		this->reset();
		// VORTEX_AQ_PIN_BYTES=N (design A, pinned counter array): N bytes of the L2 are set aside
		// for the spilled tail counters. Modelled as capacity only: N/64 lines, split evenly over
		// the banks, spread over the sets, are never allocated (held reserved for the whole run).
		const char* pb = getenv("VORTEX_AQ_PIN_BYTES");
		if (pb && config_.write_nofetch) {
			uint64_t lines = (strtoull(pb, nullptr, 0) + 63) / 64;
			uint64_t nb = 1ull << config_.B, per = (lines + nb - 1) / nb;
			uint64_t ns = sets_.size(), nw = params_.lines_per_set;
			if (per > ns * (nw - 1)) per = ns * (nw - 1);   // always leave one usable way per set
			for (uint64_t i = 0; i < per; ++i) {
				uint64_t s = (i * 167) % ns, w = nw - 1 - i / ns;   // 167 odd: a bijection over the sets
				sets_.at(s).lines.at(w).pending = true;               // reserved: never a victim, never free
			}
			if (bank_id_ == 0) fprintf(stderr, "[L2 POLICY] pinned tail-counter area %llu bytes (%llu lines per bank)\n",
			                           (unsigned long long)strtoull(pb, nullptr, 0), (unsigned long long)per);
		}
		// VORTEX_AQ_FINAL default (no explicit PIN_BYTES): the area grows to 2 B per queue used as queues
		// appear (pin_grow), with the same line order as above.
		pin_auto_ = !pb && config_.write_nofetch && aq_final_on();
	}

	// Pins lines up to 2 B x (queues used), spread as the explicit area. A line in flight (pending)
	// waits for a later cycle. A valid line is dropped from the timing model when pinned: its data is
	// functional in RAM, and a dirty one's write-back is not charged (a few lines, once per new queue range).
	void pin_grow() {
		uint64_t nb = 1ull << config_.B, ns = sets_.size(), nw = params_.lines_per_set;
		uint64_t need = ((aq_pin_entry_bytes() * g_aq_qhw + 63) / 64 + nb - 1) / nb;
		if (need > ns * (nw - 1)) need = ns * (nw - 1);
		while (pinned_ < need) {
			auto& ln = sets_.at((pinned_ * 167) % ns).lines.at(nw - 1 - pinned_ / ns);
			if (ln.pending) return;
			ln.reset();
			ln.pending = true;
			++pinned_;
		}
	}

  void reset() {
		perf_stats_ = CacheSim::PerfStats();
		pending_mshr_size_ = 0;
    pending_read_reqs_ = 0;
		pending_write_reqs_ = 0;
		pending_fill_reqs_ = 0;
  }

  void tick() {
		// process input requests
		this->processInputs();

		// process pipeline requests
		this->processRequests();

		// calculate memory latency
		perf_stats_.mem_latency += pending_fill_reqs_;
	}

	const CacheSim::PerfStats& perf_stats() const {
		return perf_stats_;
	}

	// ISA2 L1INV: drop every valid line; a line reserved for a fill in flight is kept (the fill is fresh data)
	void invalidate_all() {
		for (auto& set : sets_)
			for (auto& line : set.lines)
				if (!line.pending) line.valid = false;
	}
	// ISA2: an atomic or append performed at the L2 leaves no stale copy in this L1
	void invalidate_line(uint64_t addr) {
		auto& set = sets_.at(params_.addr_set_id(addr));
		uint64_t tag = params_.addr_tag(addr);
		for (auto& line : set.lines)
			if (line.valid && line.tag == tag) line.valid = false;
	}

private:

	void processInputs() {
		if (pin_auto_) pin_grow();
		if (tt_busy_ > 0) { --tt_busy_; return; }   // VORTEX_AQ_TAILTAB: tail-table RMW
		// proces inputs in prioroty order
		do {
			bank_req_t bank_req;

			// first: schedule MSHR replay
			if (mshr_.has_ready_reqs()) {
				mshr_.dequeue(&bank_req);
				--pending_mshr_size_;
				pipe_req_->push(bank_req);
				break;
			}

			// second: schedule memory fill
			if (!this->mem_rsp_port.empty()) {
				auto& mem_rsp = mem_rsp_port.front();
				DT(3, this->name() << "-fill-rsp: " << mem_rsp);
				// update MSHR
				auto& entry = mshr_.replay(mem_rsp.tag);
				auto& set   = sets_.at(entry.bank_req.set_id);
				auto& line  = set.lines.at(entry.line_id);
				line.valid  = true;
				line.tag    = entry.bank_req.addr_tag;
				// A line just filled from memory matches memory. Without this the
				// victim's dirty bit survived into its replacement, and the write
				// that MISSED was never marked dirty at all, so a write-back L2
				// silently dropped every write miss instead of writing it back.
				// The replay below marks it dirty again if the request was a write.
				line.dirty  = false;
				// LASTBYTE: a merge fill keeps the messages the reopened line already holds
				if (aq_lastbyte_filled(params_.mem_addr(bank_id_, entry.bank_req.set_id, entry.bank_req.addr_tag))) line.dirty = true;
				if (g_cache_resv) { line.pending = false; line.lru_ctr = 0; }
				mshr_.dequeue(&bank_req);
				--pending_mshr_size_;
				pipe_req_->push(bank_req);
				mem_rsp_port.pop();
				--pending_fill_reqs_;
				break;
			}

			// third: schedule core request
			if (!this->core_req_port.empty()) {
				auto& core_req = core_req_port.front();
				// check MSHR capacity
				if ((!core_req.write || config_.write_back)
				 && (pending_mshr_size_ >= mshr_.capacity())) {
					++perf_stats_.mshr_stalls;
					break;
				}
				++pending_mshr_size_;
				DT(3, this->name() << "-core-req: " << core_req);
				bank_req.type = bank_req_t::Core;
				bank_req.cid = core_req.cid;
				bank_req.uuid = core_req.uuid;
				bank_req.set_id = params_.addr_set_id(core_req.addr);
				bank_req.addr_tag = params_.addr_tag(core_req.addr);
				bank_req.req_tag = core_req.tag;
				bank_req.write = core_req.write;
				bank_req.amo = core_req.amo;
				bank_req.aqop = core_req.aqop;
				pipe_req_->push(bank_req);
				if (g_aq_tailtab && config_.write_nofetch && core_req.write && aq_in_qregion(core_req.addr))
					tt_busy_ = g_aq_tailtab;
				if (core_req.write)
					++perf_stats_.writes;
				else
					++perf_stats_.reads;
				core_req_port.pop();
				break;
			}
		} while (false);
	}

	void processRequests() {
		if (pipe_req_->empty())
			return;
		auto bank_req = pipe_req_->front();

		switch (bank_req.type) {
		case bank_req_t::None:
			break;
		case bank_req_t::Replay: {
			// A write that missed is only now landing in the line the fill brought
			// in, so this is where it becomes dirty. Marking it on the write HIT
			// alone lost every write miss.
			if ((bank_req.write || bank_req.amo) && config_.write_back) {
				auto& set = sets_.at(bank_req.set_id);
				int32_t free_line_id = -1, repl_line_id = 0;
				int hit_line_id = set.tag_lookup(bank_req.addr_tag, &free_line_id, &repl_line_id);
				if (hit_line_id != -1)
					set.lines.at(hit_line_id).dirty = true;
			}
			if (bank_req.write && config_.wack_sink) wack_done(bank_req.cid);   // ISA2: a write that missed is done now
			// send core response
			if (!bank_req.write || config_.write_reponse) {
				MemRsp core_rsp{bank_req.req_tag, bank_req.cid, bank_req.uuid};
				this->core_rsp_port.push(core_rsp);
				DT(3, this->name() << "-replay: " << core_rsp);
			}
		} break;
		case bank_req_t::Core: {
			// ISA2 COUNT: the queue's slice answers from its open line (number + byte mask) or the saved counter, one
			// access in this bank's pipeline; it allocates nothing and never reaches DRAM.
			if (bank_req.aqop == 2) {
				MemRsp core_rsp{bank_req.req_tag, bank_req.cid, bank_req.uuid};
				this->core_rsp_port.push(core_rsp);
				--pending_mshr_size_;
				break;
			}
			// ORACLE PROBE (T38), not a machine. Queue storage is answered here
			// unconditionally, so the transport never misses, never allocates a
			// line, never evicts anything and never reaches DRAM. This is the
			// CEILING of P4 with no capacity assumption: everything else in the
			// kernel still competes for the real cache. Off unless asked for.
			if (config_.q_oracle
			 && aq_is_qstorage(params_.mem_addr(bank_id_, bank_req.set_id, bank_req.addr_tag))) {
				++perf_stats_.q_oracle_hits;
				if (bank_req.write && config_.wack_sink) wack_done(bank_req.cid);
				if (!bank_req.write || config_.write_reponse) {
					MemRsp core_rsp{bank_req.req_tag, bank_req.cid, bank_req.uuid};
					this->core_rsp_port.push(core_rsp);
				}
				--pending_mshr_size_;
				break;
			}
			int32_t free_line_id = -1;
			int32_t repl_line_id = 0;
			auto& set = sets_.at(bank_req.set_id);
			// tag lookup
			int hit_line_id = set.tag_lookup(bank_req.addr_tag, &free_line_id, &repl_line_id);
			// LASTBYTE: a reopened queue line holds only the messages appended since it was reopened, the earlier ones
			// are in DRAM. A read of it fetches the line and merges before answering (a read arriving while that merge
			// is in flight waits for it in the MSHR).
			if (hit_line_id != -1 && !bank_req.write && !bank_req.amo && aq_lastbyte_on()) {
				const bool waits_ = g_cache_resv && set.lines.at(hit_line_id).pending && mshr_.lookup(bank_req);
				if (waits_ || aq_lastbyte_read(params_.mem_addr(bank_id_, bank_req.set_id, bank_req.addr_tag))) {
					++perf_stats_.read_misses;
					auto mshr_id = mshr_.enqueue(bank_req, hit_line_id);
					if (!waits_) {
						if (g_cache_resv) set.lines.at(hit_line_id).pending = true;
						MemReq mem_req;
						mem_req.addr  = params_.mem_addr(bank_id_, bank_req.set_id, bank_req.addr_tag);
						mem_req.write = false;
						mem_req.tag   = mshr_id;
						mem_req.cid   = bank_req.cid;
						mem_req.uuid  = bank_req.uuid;
						this->mem_req_port.push(mem_req);
						++pending_fill_reqs_;
					}
					break;
				}
			}
			if (hit_line_id != -1) {
				// Hit handling
				if (bank_req.write) {
					// handle write has_hit
					auto& hit_line = set.lines.at(hit_line_id);
					if (config_.wack_sink) wack_done(bank_req.cid);   // ISA2: performed here
					if (!config_.write_back) {
						// forward write request to memory
						MemReq mem_req;
						mem_req.addr  = params_.mem_addr(bank_id_, bank_req.set_id, bank_req.addr_tag);
						mem_req.write = true;
						mem_req.cid   = bank_req.cid;
						mem_req.uuid  = bank_req.uuid;
						this->mem_req_port.push(mem_req);
						DT(3, this->name() << "-writethrough: " << mem_req);
					} else {
						// mark line as dirty
						hit_line.dirty = true;
						if (config_.write_nofetch && aq_set_on()
						 && aq_reopenbit_hit(params_.mem_addr(bank_id_, bank_req.set_id, bank_req.addr_tag)))
							tt_busy_ += 1;   // E2b: slot = saved base + popcount, the base is read from the pinned area
						if (config_.write_nofetch && aq_set_on()
						 && aq_take_done(params_.mem_addr(bank_id_, bank_req.set_id, bank_req.addr_tag))) {
							hit_line.lru_ctr = 1u << 30;   // design A: a full queue line leaves first
							++g_aq_evict_first;
							aq_reopenbit_full(params_.mem_addr(bank_id_, bank_req.set_id, bank_req.addr_tag));
						}
					}
				} else if (bank_req.amo && config_.write_back) {
					// an AMO executes at the L2: read-modify-write leaves the line dirty
					set.lines.at(hit_line_id).dirty = true;
				}
				// send core response
				if (!bank_req.write || config_.write_reponse) {
					MemRsp core_rsp{bank_req.req_tag, bank_req.cid, bank_req.uuid};
					this->core_rsp_port.push(core_rsp);
					DT(3, this->name() << "-core-rsp: " << core_rsp);
				}
				--pending_mshr_size_;
			} else {
				// Miss handling
				if (bank_req.write)
					++perf_stats_.write_misses;
				else
					++perf_stats_.read_misses;

				// a miss that merges into a fill already in flight replaces nothing
				// (a no-fetch write to a line whose fill is in flight also merges: allocating a
				// second copy of the line would leave two ways with the same tag)
				const bool merged_ = g_cache_resv && mshr_.lookup(bank_req);
				// design A + VORTEX_AQ_PROTECT=1: the replacement skips open (part-filled) queue lines while
				// any other way can go; the open bit the design already keeps is the only state used.
				if (g_aq_protect && free_line_id == -1 && config_.write_nofetch && aq_set_on()) {
					auto open_ = [&](int w) { return aq_open_tail_addr(params_.mem_addr(bank_id_, bank_req.set_id, set.lines.at(w).tag)) != 0; };
					if (open_(repl_line_id)) {
						int best = -1; uint32_t bc = 0;
						for (int w = 0, n = (int)set.lines.size(); w < n; ++w) {
							auto& l = set.lines.at(w);
							if ((g_cache_resv && l.pending) || open_(w)) continue;
							if (best == -1 || l.lru_ctr > bc) { best = w; bc = l.lru_ctr; }
						}
						if (best != -1) { repl_line_id = best; ++g_aq_protect_saves; }
					}
				}
				// E1, VORTEX_AQ_OPENWAYS=W: a queue line (append, or the refetch of a reopened line) is placed only
				// in ways 0..W-1: an invalid one there, else the oldest unreserved one there.
				// ISA2: a queue line is one an APPEND allocates (the request says so); legacy: by address
				const bool qline_ = isa2_on() ? (bank_req.aqop == 1)
				                              : aq_is_qstorage(params_.mem_addr(bank_id_, bank_req.set_id, bank_req.addr_tag));
				if (aq_openways() && bank_req.write && config_.write_nofetch && !merged_ && qline_) {
					int fl = -1, best = -1; uint32_t bc = 0;
					for (int w = 0, n = std::min<int>((int)aq_openways(), (int)set.lines.size()); w < n; ++w) {
						auto& l = set.lines.at(w);
						if (g_cache_resv && l.pending) continue;
						if (!l.valid) { fl = w; break; }
						if (best == -1 || l.lru_ctr > bc) { best = w; bc = l.lru_ctr; }
					}
					if (fl != -1) { free_line_id = fl; ++g_aq_openways_allocs; }
					else if (best != -1) { free_line_id = -1; repl_line_id = best; ++g_aq_openways_allocs; }
				}
				if (free_line_id == -1 && config_.write_back && !merged_) {
					// write back dirty line
					auto& repl_line = set.lines.at(repl_line_id);
					if (repl_line.dirty) {
						MemReq mem_req;
						mem_req.addr  = params_.mem_addr(bank_id_, bank_req.set_id, repl_line.tag);
						mem_req.write = true;
						mem_req.cid   = bank_req.cid;
						this->mem_req_port.push(mem_req);
						DT(3, this->name() << "-writeback: " << mem_req);
						++perf_stats_.evictions;
						aq_lastbyte_evicted(mem_req.addr);
						if (g_cache_resv) repl_line.dirty = false;   // written back once
						if (config_.write_nofetch && aq_set_on()) {   // design A: open line evicted -> tail spill + refill
							uint64_t ta = aq_open_tail_addr(mem_req.addr);
							if (ta != 0) {
								++g_aq_open_evicts;
								aq_reopen_note(mem_req.addr);
								if (g_aq_spill_l2) tt_busy_ += 2;   // VORTEX_AQ_SPILL=l2: tail word spill + refill in the L2
								else for (int s = 0; s < 2; ++s) {
									MemReq tr; tr.addr = ta & ~(uint64_t)63; tr.write = true; tr.cid = bank_req.cid;
									this->mem_req_port.push(tr);
								}
							}
						}
					}
				}

				if (bank_req.write && config_.write_back && config_.write_nofetch && !merged_
				 && (g_nofetch_all || qline_)
				 && !aq_reopen_take(params_.mem_addr(bank_id_, bank_req.set_id, bank_req.addr_tag))) {
					// P4, allocate without fetch. The line is being written, so
					// fetching it from memory only to overwrite it is pure waste.
					// Take the victim, evict it if dirty (done above), and make
					// the line present and dirty now. No fill request, so the
					// write never touches memory on the way in and the reader
					// that follows finds it here.
					int32_t line_id = (free_line_id != -1) ? free_line_id : repl_line_id;
					auto& line = set.lines.at(line_id);
					line.valid = true;
					line.tag   = bank_req.addr_tag;
					line.dirty = true;
					if (g_cache_resv) line.lru_ctr = 0;   // just used: not the victim's age
					if (aq_set_on() && aq_reopenbit_alloc(params_.mem_addr(bank_id_, bank_req.set_id, bank_req.addr_tag)))
						tt_busy_ += 1;   // E2b: reopened without a fetch, the append reads the saved base
					if (aq_set_on()) aq_lastbyte_alloc(params_.mem_addr(bank_id_, bank_req.set_id, bank_req.addr_tag));
					if (aq_set_on() && aq_take_done(params_.mem_addr(bank_id_, bank_req.set_id, bank_req.addr_tag))) {
						line.lru_ctr = 1u << 30;   // design A: a full queue line leaves first
						++g_aq_evict_first;
						aq_reopenbit_full(params_.mem_addr(bank_id_, bank_req.set_id, bank_req.addr_tag));
					}
					++perf_stats_.write_nofetch_allocs;
					if (config_.wack_sink) wack_done(bank_req.cid);   // ISA2: performed here
					if (config_.write_reponse) {
						MemRsp core_rsp{bank_req.req_tag, bank_req.cid, bank_req.uuid};
						this->core_rsp_port.push(core_rsp);
					}
					--pending_mshr_size_;
				} else if (bank_req.write && !config_.write_back) {
					if (config_.wack_sink) wack_done(bank_req.cid);   // ISA2: performed (written through) here
					// forward write request to memory
					{
						MemReq mem_req;
						mem_req.addr  = params_.mem_addr(bank_id_, bank_req.set_id, bank_req.addr_tag);
						mem_req.write = true;
						mem_req.cid   = bank_req.cid;
						mem_req.uuid  = bank_req.uuid;
						this->mem_req_port.push(mem_req);
						DT(3, this->name() << "-writethrough: " << mem_req);
					}
					// send core response
					if (config_.write_reponse) {
						MemRsp core_rsp{bank_req.req_tag, bank_req.cid, bank_req.uuid};
						this->core_rsp_port.push(core_rsp);
						DT(3, this->name() << "-core-rsp: " << core_rsp);
					}
					--pending_mshr_size_;
				} else {
					// MSHR lookup
					auto mshr_pending = mshr_.lookup(bank_req);

					// allocate MSHR
					auto mshr_id = mshr_.enqueue(bank_req, (free_line_id != -1) ? free_line_id : repl_line_id);
					DT(3, this->name() << "-mshr-enqueue: " << bank_req);

					// send fill request
					if (!mshr_pending) {
						if (g_cache_resv) set.lines.at((free_line_id != -1) ? free_line_id : repl_line_id).pending = true;
						MemReq mem_req;
						mem_req.addr  = params_.mem_addr(bank_id_, bank_req.set_id, bank_req.addr_tag);
						mem_req.write = false;
						mem_req.amo   = bank_req.amo;
						mem_req.tag   = mshr_id;
						mem_req.cid   = bank_req.cid;
						mem_req.uuid  = bank_req.uuid;
						this->mem_req_port.push(mem_req);
						DT(3, this->name() << "-fill-req: " << mem_req);
						++pending_fill_reqs_;
					}
				}
			}
		} break;
		default:
			std::abort();
		}

		pipe_req_->pop();
	}

	CacheSim::Config config_;
	params_t params_;
	uint32_t bank_id_;

  std::vector<set_t> sets_;
	MSHR mshr_;
	uint32_t pending_mshr_size_;
	uint32_t tt_busy_ = 0;
	bool pin_auto_ = false;
	uint64_t pinned_ = 0;
	TFifo<bank_req_t>::Ptr pipe_req_;

	CacheSim::PerfStats perf_stats_;

	uint64_t pending_read_reqs_;
	uint64_t pending_write_reqs_;
	uint64_t pending_fill_reqs_;
};

///////////////////////////////////////////////////////////////////////////////

class CacheSim::Impl {
public:
	Impl(CacheSim* simobject, const Config& config)
		: simobject_(simobject)
		, config_(config)
		, params_(config)
		, banks_(1 << config.B)
		, nc_mem_arbs_(config.mem_ports)
	{
		char sname[100];

		uint32_t num_banks = (1 << config.B);

		if (config_.bypass) {
			snprintf(sname, 100, "%s-bypass_arb", simobject->name().c_str());
			auto bypass_arb = MemArbiter::Create(sname, ArbiterType::RoundRobin, config_.num_inputs, config_.mem_ports);
			for (uint32_t i = 0; i < config_.num_inputs; ++i) {
				simobject->CoreReqPorts.at(i).bind(&bypass_arb->ReqIn.at(i));
				bypass_arb->RspIn.at(i).bind(&simobject->CoreRspPorts.at(i));
			}
			for (uint32_t i = 0; i < config_.mem_ports; ++i) {
				bypass_arb->ReqOut.at(i).bind(&simobject->MemReqPorts.at(i));
				simobject->MemRspPorts.at(i).bind(&bypass_arb->RspOut.at(i));
			}
			return;
		}

		// create non-cacheable arbiter
		for (uint32_t i = 0; i < config_.mem_ports; ++i) {
			snprintf(sname, 100, "%s-nc_arb%d", simobject->name().c_str(), i);
			nc_mem_arbs_.at(i) = MemArbiter::Create(sname, ArbiterType::Priority, 2, 1);
		}

		// Connect non-cacheable arbiter output port 0 to outgoing memory ports
		for (uint32_t i = 0; i < config_.mem_ports; ++i) {
			nc_mem_arbs_.at(i)->ReqOut.at(0).bind(&simobject->MemReqPorts.at(i));
			simobject->MemRspPorts.at(i).bind(&nc_mem_arbs_.at(i)->RspOut.at(0));
		}

		// Create bank's memory arbiter
		snprintf(sname, 100, "%s-mem_arb", simobject->name().c_str());
		auto bank_mem_arb = MemArbiter::Create(sname, ArbiterType::RoundRobin, num_banks, config_.mem_ports);

		// Connect bank's memory arbiter to non-cacheable arbiter's input 0
		for (uint32_t i = 0; i < config_.mem_ports; ++i) {
			bank_mem_arb->ReqOut.at(i).bind(&nc_mem_arbs_.at(i)->ReqIn.at(0));
			nc_mem_arbs_.at(i)->RspIn.at(0).bind(&bank_mem_arb->RspOut.at(i));
		}

		// Create bank's core crossbar
		snprintf(sname, 100, "%s-core_xbar", simobject->name().c_str());
		bank_core_xbar_ = MemCrossBar::Create(sname, ArbiterType::RoundRobin, config_.num_inputs, num_banks,
			[&](const MemCrossBar::ReqType& req) {
				return params_.addr_bank_id(req.addr);
			});

		// Create cache banks
		for (uint32_t i = 0, n = num_banks; i < n; ++i) {
			snprintf(sname, 100, "%s-bank%d", simobject->name().c_str(), i);
			banks_.at(i) = CacheBank::Create(sname, config, params_, i);

			// bind core ports
			bank_core_xbar_->ReqOut.at(i).bind(&banks_.at(i)->core_req_port);
			banks_.at(i)->core_rsp_port.bind(&bank_core_xbar_->RspOut.at(i));

			// bind memory ports
			banks_.at(i)->mem_req_port.bind(&bank_mem_arb->ReqIn.at(i));
			bank_mem_arb->RspIn.at(i).bind(&banks_.at(i)->mem_rsp_port);
		}
	}

  void reset() {
		if (config_.bypass)
			return;

		// calculate cache initialization cycles
		init_cycles_ = params_.sets_per_bank;
	}

  void tick() {
		if (config_.bypass)
			return;

		// wait on cache initialization cycles
		if (init_cycles_ != 0) {
			--init_cycles_;
			DT(3, simobject_->name() << "-init: line=" << init_cycles_);
			return;
		}

		// handle cache bypasss responses
		for (uint32_t i = 0, n = config_.mem_ports; i < n; ++i) {
			// Forward non-cacheable arbiter's output 1 to core response ports
			auto& bypass_port = nc_mem_arbs_.at(i)->RspIn.at(1);
			if (!bypass_port.empty()) {
				auto& mem_rsp = bypass_port.front();
				this->processBypassResponse(mem_rsp);
				bypass_port.pop();
			}
		}

		// schedule core responses
		for (uint32_t req_id = 0, n = config_.num_inputs; req_id < n; ++req_id) {
			auto& bank_rsp_port = bank_core_xbar_->RspIn.at(req_id);
			if (bank_rsp_port.empty())
				continue;
			auto& core_rsp = bank_rsp_port.front();
			simobject_->CoreRspPorts.at(req_id).push(core_rsp, 0);
			DT(3, simobject_->name() << "-core-rsp: " << core_rsp);
			bank_rsp_port.pop();
		}

		// schedule core requests
		for (uint32_t req_id = 0, n = config_.num_inputs; req_id < n; ++req_id) {
			auto& core_req_port = simobject_->CoreReqPorts.at(req_id);
			if (core_req_port.empty())
				continue;
			auto& core_req = core_req_port.front();
			if (core_req.type == AddrType::IO) {
				this->processBypassRequest(core_req, req_id);
			} else if (config_.l1skip_ops && core_req.l1skip) {
				// ISA2: atomics, appends and counts are performed at the L2; this L1 forwards them uncached
				if (core_req.write || core_req.amo)
					banks_.at(params_.addr_bank_id(core_req.addr))->invalidate_line(core_req.addr);
				this->processBypassRequest(core_req, req_id);
			} else {
				bank_core_xbar_->ReqIn.at(req_id).push(core_req, 0);
			}
			core_req_port.pop();
		}
	}

	void invalidate_all() {
		if (config_.bypass) return;
		for (auto& bank : banks_) bank->invalidate_all();
	}

	PerfStats perf_stats() const {
		PerfStats perf_stats;
		if (!config_.bypass) {
			for (const auto& bank : banks_) {
				perf_stats += bank->perf_stats();
			}
			perf_stats.bank_stalls = bank_core_xbar_->collisions();
		}
		return perf_stats;
	}

private:

	void processBypassResponse(const MemRsp& mem_rsp) {
		uint32_t req_id = mem_rsp.tag & ((1 << params_.log2_num_inputs)-1);
		uint64_t tag = mem_rsp.tag >> params_.log2_num_inputs;
		MemRsp core_rsp{tag, mem_rsp.cid, mem_rsp.uuid};
		simobject_->CoreRspPorts.at(req_id).push(core_rsp, 0);
		DT(3, simobject_->name() << "-bypass-core-rsp: " << core_rsp);
	}

	void processBypassRequest(const MemReq& core_req, uint32_t req_id) {
		{
			// Push core request to non-cacheable arbiter's input 1
			MemReq mem_req(core_req);
			mem_req.tag = (core_req.tag << params_.log2_num_inputs) + req_id;
			uint32_t mem_port = req_id % config_.mem_ports;
			nc_mem_arbs_.at(mem_port)->ReqIn.at(1).push(mem_req, 0);
			DT(3, simobject_->name() << "-bypass-dram-req: " << mem_req);
		}

		if (core_req.write && config_.write_reponse) {
			MemRsp core_rsp{core_req.tag, core_req.cid, core_req.uuid};
			simobject_->CoreRspPorts.at(req_id).push(core_rsp, 0);
			DT(3, simobject_->name() << "-bypass-core-rsp: " << core_rsp);
		}
	}

	CacheSim* const simobject_;
	Config config_;
	params_t params_;
	std::vector<CacheBank::Ptr> banks_;
	MemArbiter::Ptr bank_arb_;
	std::vector<MemArbiter::Ptr> nc_mem_arbs_;
	MemCrossBar::Ptr bank_core_xbar_;
	uint32_t init_cycles_;
};

///////////////////////////////////////////////////////////////////////////////

CacheSim::CacheSim(const SimContext& ctx, const char* name, const Config& config)
	: SimObject<CacheSim>(ctx, name)
	, CoreReqPorts(config.num_inputs, this)
	, CoreRspPorts(config.num_inputs, this)
	, MemReqPorts(config.mem_ports, this)
	, MemRspPorts(config.mem_ports, this)
	, impl_(new Impl(this, config))
{}

CacheSim::~CacheSim() {
  delete impl_;
}

void CacheSim::reset() {
  impl_->reset();
}

void CacheSim::tick() {
  impl_->tick();
}

CacheSim::PerfStats CacheSim::perf_stats() const {
  return impl_->perf_stats();
}

void CacheSim::invalidate_all() {
  impl_->invalidate_all();
}