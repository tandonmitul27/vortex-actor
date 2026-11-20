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

#pragma once

#include <stdint.h>
#include <bitset>
#include <queue>
#include <vector>
#include <unordered_map>
#include <variant>
#include <util.h>
#include <stringutil.h>
#include <VX_config.h>
#include <VX_types.h>
#include <simobject.h>
#include <bitvector.h>
#include <iostream>
#include "debug.h"
#include "constants.h"

namespace vortex {

typedef uint8_t Byte;

#if (XLEN == 32)
typedef uint32_t Word;
typedef int32_t  WordI;
typedef uint64_t DWord;
typedef int64_t  DWordI;
#elif (XLEN == 64)
typedef uint64_t Word;
typedef int64_t  WordI;
typedef __uint128_t DWord;
typedef __int128_t DWordI;
#else
#error unsupported XLEN
#endif

typedef std::bitset<MAX_NUM_CORES>   CoreMask;
typedef std::bitset<MAX_NUM_REGS>    RegMask;
typedef BitVector<Word>              ThreadMask;
typedef std::bitset<MAX_NUM_WARPS>   WarpMask;

///////////////////////////////////////////////////////////////////////////////

union reg_data_t {
  uint8_t  u8;
  uint16_t u16;
  Word     u;
  WordI    i;
  float    f32;
  double   f64;
  uint32_t u32;
  uint64_t u64;
  int32_t  i32;
  int64_t  i64;
};

///////////////////////////////////////////////////////////////////////////////

struct op_string_t {
  std::string op;
  std::string arg;
};

///////////////////////////////////////////////////////////////////////////////

enum class RegType {
  None,
  Integer,
  Float,
#ifdef EXT_V_ENABLE
  Vector,
#endif
  Count
};

inline std::ostream &operator<<(std::ostream &os, const RegType& type) {
  switch (type) {
  case RegType::None: break;
  case RegType::Integer: os << "x"; break;
  case RegType::Float:   os << "f"; break;
#ifdef EXT_V_ENABLE
  case RegType::Vector:  os << "v"; break;
#endif
  default: assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

struct RegOpd {
  RegType type = RegType::None;
  uint32_t idx = 0;

  uint32_t id() const {
    if (type == RegType::None)
      return 0;
    // unique register id embedding the type
    return (((int)(type)-1) << LOG_NUM_REGS) | idx;
  }

  friend std::ostream &operator<<(std::ostream &os, const RegOpd& reg) {
    os << reg.type << reg.idx;
    return os;
  }

  constexpr static uint32_t ID_BITS = log2ceil((int)RegType::Count) + LOG_NUM_REGS;
};

///////////////////////////////////////////////////////////////////////////////

enum class FUType {
  ALU,
  LSU,
  FPU,
  SFU,
#ifdef EXT_V_ENABLE
  VPU,
#endif
#ifdef EXT_TCU_ENABLE
  TCU,
#endif
  Count
};

inline std::ostream &operator<<(std::ostream &os, const FUType& type) {
  switch (type) {
  case FUType::ALU: os << "ALU"; break;
  case FUType::LSU: os << "LSU"; break;
  case FUType::FPU: os << "FPU"; break;
  case FUType::SFU: os << "SFU"; break;
#ifdef EXT_V_ENABLE
  case FUType::VPU: os << "VPU"; break;
#endif
#ifdef EXT_TCU_ENABLE
  case FUType::TCU: os << "TCU"; break;
#endif
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class AluType {
  LUI,
  AUIPC,
  ADD,
  SUB,
  SLL,
  SRL,
  SRA,
  SLT,
  SLTU,
  AND,
  OR,
  XOR,
  CZERO
};

struct IntrAluArgs {
  uint32_t is_imm : 1;
  uint32_t is_w : 1;
  uint32_t imm;
};

inline std::ostream &operator<<(std::ostream &os, const AluType& type) {
  switch (type) {
  case AluType::LUI:     os << "LUI"; break;
  case AluType::AUIPC:   os << "AUIPC"; break;
  case AluType::ADD:     os << "ADD"; break;
  case AluType::SUB:     os << "SUB"; break;
  case AluType::SLL:     os << "SLL"; break;
  case AluType::SRL:     os << "SRL"; break;
  case AluType::SRA:     os << "SRA"; break;
  case AluType::SLT:     os << "SLT"; break;
  case AluType::SLTU:    os << "SLTU"; break;
  case AluType::AND:     os << "AND"; break;
  case AluType::OR:      os << "OR"; break;
  case AluType::XOR:     os << "XOR"; break;
  case AluType::CZERO:   os << "CZERO"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class BrType {
  BR,
  JAL,
  JALR,
  SYS,
};

struct IntrBrArgs {
  uint32_t cmp : 3;
  uint32_t offset;
};

inline std::ostream &operator<<(std::ostream &os, const BrType& type) {
  switch (type) {
  case BrType::BR:   os << "BR"; break;
  case BrType::JAL:  os << "JAL"; break;
  case BrType::JALR: os << "JALR"; break;
  case BrType::SYS:  os << "SYS"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class VoteType {
  ALL,
  ANY,
  UNI,
  BAL
};

inline std::ostream &operator<<(std::ostream &os, const VoteType& vote) {
  switch (vote) {
  case VoteType::ALL: os << "ALL"; break;
  case VoteType::ANY: os << "ANY"; break;
  case VoteType::UNI: os << "UNI"; break;
  case VoteType::BAL: os << "BAL"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class ShflType {
  UP,
  DOWN,
  BFLY,
  IDX
};

inline std::ostream &operator<<(std::ostream &os, const ShflType& shfl) {
  switch (shfl) {
  case ShflType::UP:   os << "UP"; break;
  case ShflType::DOWN: os << "DOWN"; break;
  case ShflType::BFLY: os << "BFLY"; break;
  case ShflType::IDX:  os << "IDX"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class MdvType {
  MUL,
  MULHU,
  MULH,
  MULHSU,
  DIV,
  DIVU,
  REM,
  REMU
};

struct IntrMdvArgs {
  uint32_t is_w : 1;
};

inline std::ostream &operator<<(std::ostream &os, const MdvType& type) {
  switch (type) {
  case MdvType::MUL:    os << "MUL"; break;
  case MdvType::MULHU:  os << "MULHU"; break;
  case MdvType::MULH:   os << "MULH"; break;
  case MdvType::MULHSU: os << "MULHSU"; break;
  case MdvType::DIV:    os << "DIV"; break;
  case MdvType::DIVU:   os << "DIVU"; break;
  case MdvType::REM:    os << "REM"; break;
  case MdvType::REMU:   os << "REMU"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class LsuType {
  LOAD,
  STORE,
  FENCE,
  L1INV      // ISA2: invalidate the core's L1 data cache (whole cache, lines in flight kept)
};

struct IntrLsuArgs {
  uint32_t width : 3;
  uint32_t is_float : 1;
  uint32_t offset;
};

inline std::ostream &operator<<(std::ostream &os, const LsuType& type) {
  switch (type) {
  case LsuType::LOAD:   os << "LOAD"; break;
  case LsuType::STORE:  os << "STORE"; break;
  case LsuType::FENCE:  os << "FENCE"; break;
  case LsuType::L1INV:  os << "L1INV"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class AmoType {
  LR,
  SC,
  AMOADD,
  AMOSWAP,
  AMOAND,
  AMOOR,
  AMOXOR,
  AMOMIN,
  AMOMAX,
  AMOMINU,
  AMOMAXU,
  AMOCAS,    // Zacas amocas.w/.d: rd holds the expected value on entry
  AQAPPEND,  // ISA2: one-way append(q = rs1, message = rs2), 4 B (.w) or 8 B (.d); no result
  AQCOUNT    // ISA2: rd = messages in queue q = rs1 this round, answered by the queue's L2 slice
};

struct IntrAmoArgs {
  uint32_t width : 3;
  uint32_t aq : 1;
  uint32_t rl : 1;
};

inline std::ostream &operator<<(std::ostream &os, const AmoType& type) {
  switch (type) {
  case AmoType::LR:      os << "LR"; break;
  case AmoType::SC:      os << "SC"; break;
  case AmoType::AMOADD:  os << "AMOADD"; break;
  case AmoType::AMOSWAP: os << "AMOSWAP"; break;
  case AmoType::AMOAND:  os << "AMOAND"; break;
  case AmoType::AMOOR:   os << "AMOOR"; break;
  case AmoType::AMOXOR:  os << "AMOXOR"; break;
  case AmoType::AMOMIN:  os << "AMOMIN"; break;
  case AmoType::AMOMAX:  os << "AMOMAX"; break;
  case AmoType::AMOMINU: os << "AMOMINU"; break;
  case AmoType::AMOMAXU: os << "AMOMAXU"; break;
  case AmoType::AMOCAS:  os << "AMOCAS"; break;
  case AmoType::AQAPPEND: os << "AQAPPEND"; break;
  case AmoType::AQCOUNT: os << "AQCOUNT"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class FpuType {
  FADD,
  FSUB,
  FMUL,
  FDIV,
  FSQRT,
  FMADD,
  FMSUB,
  FNMADD,
  FNMSUB,
  F2I,
  I2F,
  F2F,
  FCMP,
  FSGNJ,
  FCLASS,
  FMVXW,
  FMVWX,
  FMINMAX,
};

struct IntrFpuArgs {
  uint32_t frm : 3;
  uint32_t cvt : 2;
  uint32_t is_f64 : 1;
};

inline std::ostream &operator<<(std::ostream &os, const FpuType& type) {
  switch (type) {
  case FpuType::FADD:   os << "FADD"; break;
  case FpuType::FSUB:   os << "FSUB"; break;
  case FpuType::FMUL:   os << "FMUL"; break;
  case FpuType::FDIV:   os << "FDIV"; break;
  case FpuType::FSQRT:  os << "FSQRT"; break;
  case FpuType::FMADD:  os << "FMADD"; break;
  case FpuType::FMSUB:  os << "FMSUB"; break;
  case FpuType::FNMADD: os << "FNMADD"; break;
  case FpuType::FNMSUB: os << "FNMSUB"; break;
  case FpuType::F2I:    os << "F2I"; break;
  case FpuType::I2F:    os << "I2F"; break;
  case FpuType::F2F:    os << "F2F"; break;
  case FpuType::FCMP:   os << "FCMP"; break;
  case FpuType::FSGNJ:  os << "FSGNJ"; break;
  case FpuType::FCLASS: os << "FCLASS"; break;
  case FpuType::FMVXW:  os << "FMVXW"; break;
  case FpuType::FMVWX:  os << "FMVWX"; break;
  case FpuType::FMINMAX: os << "FMIN_MAX"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class WctlType {
  TMC,
  WSPAWN,
  SPLIT,
  JOIN,
  BAR,
  PRED
};

struct IntrWctlArgs {
  uint32_t is_neg : 1;
};

inline std::ostream &operator<<(std::ostream &os, const WctlType& type) {
  switch (type) {
  case WctlType::TMC:    os << "TMC"; break;
  case WctlType::WSPAWN: os << "WSPAWN"; break;
  case WctlType::SPLIT:  os << "SPLIT"; break;
  case WctlType::JOIN:   os << "JOIN"; break;
  case WctlType::BAR:    os << "BAR"; break;
  case WctlType::PRED:   os << "PRED"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class CsrType {
  CSRRW,
  CSRRS,
  CSRRC
};

struct IntrCsrArgs {
  uint32_t is_imm: 1;
  uint32_t imm : 5;
  uint32_t csr : 12;
};

inline std::ostream &operator<<(std::ostream &os, const CsrType& type) {
  switch (type) {
  case CsrType::CSRRW: os << "CSRRW"; break;
  case CsrType::CSRRS: os << "CSRRS"; break;
  case CsrType::CSRRC: os << "CSRRC"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class VsetType {
  VSETVLI,
  VSETIVLI,
  VSETVL
};

struct IntrVsetArgs {
  uint32_t zimm: 11;
  uint32_t uimm: 5;

  std::string to_string(VsetType type) const {
    std::string str;
    if (type != VsetType::VSETVL) {
      str = "zimm=" + to_hex_str(zimm);
      if (type == VsetType::VSETIVLI) {
        str += ", uimm=" + to_hex_str(uimm);
      }
    }
    return str;
  }
};

inline std::ostream &operator<<(std::ostream &os, const VsetType& type) {
  switch (type) {
  case VsetType::VSETVLI:  os << "VSETVLI"; break;
  case VsetType::VSETIVLI: os << "VSETIVLI"; break;
  case VsetType::VSETVL:   os << "VSETVL"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class VlsType {
  VL,
  VLS,
  VLX,
  VS,
  VSS,
  VSX
};

struct IntrVlsArgs {
  uint32_t width:2;
  uint32_t umop: 5;
  uint32_t vm: 1;
  uint32_t mew: 1;
  uint32_t nf: 3;

  std::string to_string(VlsType type) const {
    std::string str = "width=" + std::to_string(width);
    if (type == VlsType::VL) {
      str +=  ", umop=" + std::to_string(umop);
    }
    str += ", vm=" + std::to_string(vm) +
           ", mew=" + std::to_string(mew) +
           ", nf=" + std::to_string(nf);
    return str;
  }
};

inline std::ostream &operator<<(std::ostream &os, const VlsType& type) {
  switch (type) {
  case VlsType::VL:  os << "VL"; break;
  case VlsType::VLS: os << "VLS"; break;
  case VlsType::VLX: os << "VLX"; break;
  case VlsType::VS:  os << "VS"; break;
  case VlsType::VSS: os << "VSS"; break;
  case VlsType::VSX: os << "VSX"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class VopType {
  OPIVV,
  OPFVV,
  OPMVV,
  OPIVI,
  OPIVX,
  OPFVF,
  OPMVX
};

struct IntrVopArgs {
  uint32_t vm: 1;
  uint32_t funct6: 6;
  uint32_t imm: 5;

  std::string to_string(VopType type) const {
    std::string str = "vm=" + std::to_string(vm) +
                      ", funct6=" + std::to_string(funct6);
    if (type == VopType::OPIVI || type == VopType::OPIVX) {
      str += ", imm=" + to_hex_str(imm);
    }
    return str;
  }
};

inline std::ostream &operator<<(std::ostream &os, const VopType& type) {
  switch (type) {
  case VopType::OPIVV:    os << "OPIVV"; break;
  case VopType::OPFVV:    os << "OPFVV"; break;
  case VopType::OPMVV:    os << "OPMVV"; break;
  case VopType::OPIVI:    os << "OPIVI"; break;
  case VopType::OPIVX:    os << "OPIVX"; break;
  case VopType::OPFVF:    os << "OPFVF"; break;
  case VopType::OPMVX:    os << "OPMVX"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class VpuOpType {
  VSET    = 0,

  ARITH   = 1,
  IMUL    = 2,
  IDIV    = 3,

  FMA     = 4,
  FDIV    = 5,
  FSQRT   = 6,
  FCVT    = 7,
  FNCP    = 8,

  // reduction
  ARITH_R = 9,
  FMA_R   = 10,
  FNCP_R  = 11
};

inline std::ostream &operator<<(std::ostream &os, const VpuOpType& type) {
  switch (type) {
  case VpuOpType::VSET:    os << "VSET"; break;
  case VpuOpType::ARITH:   os << "ARITH"; break;
  case VpuOpType::IMUL:    os << "IMUL"; break;
  case VpuOpType::IDIV:    os << "IDIV"; break;
  case VpuOpType::FMA:     os << "FMA"; break;
  case VpuOpType::FDIV:    os << "FDIV"; break;
  case VpuOpType::FSQRT:   os << "FSQRT"; break;
  case VpuOpType::FCVT:    os << "FCVT"; break;
  case VpuOpType::FNCP:    os << "FNCP"; break;
  case VpuOpType::ARITH_R: os << "ARITH_R"; break;
  case VpuOpType::FMA_R:   os << "FMA_R"; break;
  case VpuOpType::FNCP_R:  os << "FNCP_R"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

enum class TcuType {
  WMMA,
};

struct IntrTcuArgs {
  uint32_t fmt_s  : 4;
  uint32_t fmt_d  : 4;
  uint32_t step_m : 4;
  uint32_t step_n : 4;
};

inline std::ostream &operator<<(std::ostream &os, const TcuType& type) {
  switch (type) {
  case TcuType::WMMA: os << "WMMA"; break;
  default:
    assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

using OpType = std::variant<
  AluType
, BrType
, MdvType
, LsuType
, AmoType
, FpuType
, CsrType
, VoteType
, ShflType
, WctlType
#ifdef EXT_V_ENABLE
, VsetType
, VlsType
, VopType
#endif
#ifdef EXT_TCU_ENABLE
, TcuType
#endif
>;

using IntrArgs = std::variant<
  IntrAluArgs
, IntrBrArgs
, IntrMdvArgs
, IntrLsuArgs
, IntrAmoArgs
, IntrFpuArgs
, IntrCsrArgs
, IntrWctlArgs
#ifdef EXT_V_ENABLE
, IntrVsetArgs
, IntrVlsArgs
, IntrVopArgs
#endif
#ifdef EXT_TCU_ENABLE
, IntrTcuArgs
#endif
>;

///////////////////////////////////////////////////////////////////////////////

enum class AddrType {
  Global,
  Shared,
  MsgMem,     // on-chip actor message memory: global/coherent data, on-chip timing
  RemoteLmem, // remote-push window: a store deposits into ANOTHER core's LMEM
  AppendQ,    // combining append queue window: a store is an append(q, word)
  IO
};

// Modeled on-chip message memory. Set from the environment
// (VORTEX_MSGMEM_BASE / VORTEX_MSGMEM_SIZE, both in bytes) so a kernel can place
// its actor inboxes there. Size 0 disables the model, restoring stock behaviour.
// The data stays in the global memory image, so it remains coherent across cores;
// only the TIMING is served by the on-chip structure. This models a hardware
// queue's storage without needing per-core private semantics.
extern uint64_t g_msgmem_base;
extern uint64_t g_msgmem_size;

// Modeled remote scratchpad push (see remote_push_plan.md). The window is laid
// out as base + dst_core * (1 << LMEM_LOG_SIZE) + offset: a store into it
// deposits directly into the DESTINATION core's LMEM, bypassing the memory
// hierarchy. Set from the environment (VORTEX_RPUSH_BASE / VORTEX_RPUSH_SIZE,
// bytes); size 0 disables the model, restoring stock behaviour. v0 timing is an
// optimistic ceiling: the store is charged like a local scratchpad access, with
// no NoC hop or destination-bank contention. Kernels should treat the window as
// write-only (push model); reads are supported for debugging only.
extern uint64_t g_rpush_base;
extern uint64_t g_rpush_size;

inline uint32_t rpush_core(uint64_t addr) {
  return (uint32_t)((addr - g_rpush_base) >> LMEM_LOG_SIZE);
}

// Modeled combining append queue (see docs/append_queue_plan.md). The window
// starts at VORTEX_AQ_BASE and spans VORTEX_AQ_SIZE bytes; size 0 disables the
// model and restores stock behaviour. Line 0 of the window is a control block
// the KERNEL writes before its first append: {qbase, qstride, tailbase, wsize}
// as four 8 byte words. Line q+1 is queue q: a store to any byte of that line
// is append(q, word). Functionally the word lands at
// qbase + q*qstride + tail[q]*wsize in the global image and tail[q] is
// mirrored to tailbase + 4q, so the consumer reads the queue and its length
// with ordinary loads after a barrier. Timing: the lane is charged a
// scratchpad access on the queue's line (LocalMem banks by line, so appends to
// different queues take different banks and appends to one queue serialise on
// its combining port), and every (64/wsize)'th append to a queue issues one
// full line write down the ordinary dcache path. v0 does not charge L2 tail
// contention, the NoC hop, or combining line capacity; it reports the peak
// number of queues touched instead.
extern uint64_t g_aq_base;
extern uint64_t g_aq_size;
struct AqCtrl { uint64_t qbase; uint64_t qstride; uint64_t tailbase; uint64_t wsize; uint64_t cap; };
extern AqCtrl g_aq_ctrl;
extern std::vector<uint32_t> g_aq_tail;   // functional: next slot per queue
extern std::vector<uint32_t> g_aq_fill;   // timing: appends seen per queue
extern uint64_t g_aq_appends;
extern uint64_t g_aq_dropped;   // appends refused because a queue was full
extern uint64_t g_aq_flushes;
extern uint64_t g_aq_tail_flushes;         // partial line writes charged at an aq_flush or a geometry change
extern std::vector<uint32_t> g_aq_flushed;  // timing: fill count per queue at its last flush
extern uint64_t g_aq_used_hw;              // queues ever used, across chunk resets
extern uint64_t g_aq_maxfill_hw;           // deepest queue ever, across chunk resets
constexpr uint32_t AQ_FLUSH_OFF = 40;       // control word 5: any store flushes the round's partial lines
// Control word 6: END OF ROUND. Flush the partial lines as word 5 does, then
// reset every queue's tail to zero, in the model and in the tails array the
// kernel reads. P3 sends its mail in K chunks with a receive between, so a
// queue is FILLED AND DRAINED K times, and its storage is a small buffer reused
// each chunk instead of one sized for the whole run. That is what bounds the
// live footprint, and it is also what makes the tail charge exact per chunk.
constexpr uint32_t AQ_ROUND_OFF = 48;
// Control word 7: OVERFLOW POOL (skewed traffic). Tagged stores, value = (field << 60) | payload:
// 1 pool base, 2 chunk table base (0 = overflow off: a full queue drops, as before), 3 (chunk words << 32) |
// max chunks per queue, 4 pool size in chunks. When a queue's region is full, the L2 append unit takes the
// next chunk from the pool (one atomic per chunk) and writes its address to ctab[q * maxc + c] (64 bit).
constexpr uint32_t AQ_POOL_OFF = 56;
struct AqPool { uint64_t base; uint64_t ctab; uint64_t ch; uint64_t maxc; uint64_t nch; uint64_t next; };
extern AqPool g_aq_pool;
extern std::unordered_map<uint32_t, std::vector<uint64_t>> g_aq_ovf;   // functional: chunk bases per queue
extern uint64_t g_aq_ovf_chunks;                                        // chunks taken (all geometries)
inline uint64_t aq_slot_addr(uint32_t q, uint64_t slot) {               // slot -> address, region or chunk
  if (g_aq_ctrl.cap == 0 || slot < g_aq_ctrl.cap || g_aq_pool.ctab == 0)
    return g_aq_ctrl.qbase + (uint64_t)q * g_aq_ctrl.qstride + slot * g_aq_ctrl.wsize;
  uint64_t e = slot - g_aq_ctrl.cap, c = e / g_aq_pool.ch;
  auto it = g_aq_ovf.find(q);
  if (it == g_aq_ovf.end() || c >= it->second.size()) return g_aq_ctrl.qbase + (uint64_t)q * g_aq_ctrl.qstride;
  return it->second[c] + (e % g_aq_pool.ch) * g_aq_ctrl.wsize;
}
void aq_tail_flush();                       // charge one line write per queue whose tail sits mid line
constexpr uint32_t AQ_LINE = 64;

// queue index of a window address; -1 is the control block
inline int64_t aq_queue(uint64_t addr) {
  return (int64_t)((addr - g_aq_base) / AQ_LINE) - 1;
}

// ---- apply engine (typed drains): docs/apply_engine_plan.md ----------------
// One engine per core on the core's memory port. A lane writes an 8-word
// command line (line = lane) in its window, the last word triggering; the
// engine streams the named queues and scatters STORE/ADD through a small
// combining cache. Line AP_DONE_LINE word 0 reads as the core's done flag.
// One command slot per hardware lane of the core: warp id x threads + lane.
// Several warps share one engine and one done flag; a warp that polls while
// another's command is active simply waits for it too, which is conservative.
constexpr uint32_t AP_NLANES    = (NUM_WARPS) * (NUM_THREADS);
constexpr uint32_t AP_DONE_LINE = AP_NLANES;
constexpr uint32_t AP_WINDOW    = (AP_NLANES + 16) * AQ_LINE;
constexpr uint32_t AP_TRIG_OFF  = 56;          // byte offset of word 7 in the line
inline uint64_t ap_base() { return g_aq_base + g_aq_size; }
// P4: is this address inside the queue's storage array, i.e. a line the transport
// writes WHOLE? Only such a line may be allocated without fetching it. An atomic
// read modify write, or any partial write, still needs the old bytes.
//
// A kernel installs SEVERAL geometries in one run: the index gather appends to
// its tile queues under one, then rewrites the control block and appends replies
// under another, while STILL READING the tile queues it filled. Testing only the
// installed geometry therefore misses the reads of every earlier one, so each
// geometry is remembered as a region and the test is against all of them. The
// region is bounded by the highest queue index actually appended, which keeps it
// inside the host's storage buffer: the window always has more queue slots than
// a kernel uses, and taking the whole window would spill into the next
// allocation and wrongly exempt it.
struct AqRegion { uint64_t lo; uint64_t stride; uint64_t cap_bytes; uint64_t maxq; };
extern std::vector<AqRegion> g_aq_regions;
void aq_note_queue(int64_t q);            // record the highest queue index touched
bool aq_is_qstorage(uint64_t addr);
bool aq_in_qregion(uint64_t addr);
bool aq_set_on();
uint32_t aq_pin_entry_bytes();             // pinned-area entry: 2 B, or 8 B with VORTEX_AQ_OVFCHG and a pool                         // VORTEX_AQ_SET (design A)
bool aq_final_on();                       // VORTEX_AQ_FINAL: the decided L2 append is the default
extern uint64_t g_aq_qhw;                 // highest queue appended to + 1 (auto pinned area)
bool aq_take_done(uint64_t line_addr);     // line made full by an append: mark evict-first once
void aq_reopen_note(uint64_t line_addr);   // VORTEX_AQ_CNTMASK: open line evicted early
bool aq_reopen_take(uint64_t line_addr);   // VORTEX_AQ_CNTMASK: next append to it refetches the line
void aq_reopen_clear();                    // VORTEX_AQ_CNTMASK / REOPENBIT: new round or geometry
bool aq_lastbyte_alloc(uint64_t line_addr);    // VORTEX_AQ_LASTBYTE: reopened without a fetch (line now partial)
bool aq_lastbyte_read(uint64_t line_addr);     // VORTEX_AQ_LASTBYTE: a read of a partial line fetches and merges
bool aq_lastbyte_filled(uint64_t line_addr);   // VORTEX_AQ_LASTBYTE: that merge fill landed (line stays dirty)
void aq_lastbyte_evicted(uint64_t line_addr);  // VORTEX_AQ_LASTBYTE: a line written back is no longer partial
bool aq_lastbyte_on();
bool aq_reopenbit_alloc(uint64_t line_addr);   // VORTEX_AQ_REOPENBIT (E2b)
bool aq_reopenbit_hit(uint64_t line_addr);
void aq_reopenbit_full(uint64_t line_addr);
uint32_t aq_openways();                    // VORTEX_AQ_OPENWAYS (E1), 0 = off
extern uint64_t g_aq_openways_allocs;
uint64_t aq_open_tail_addr(uint64_t line_addr);   // tail word of an open line, 0 if not open
// ---- ISA2 (2026-11-10, the final mechanism in the paper): two new instructions, the append (one-way, sender
// names the queue) and the count (returns a queue's message count from its L2 slice); every atomic, append and
// count passes the L1 and is performed at the L2; a fence waits until the core's writes are acknowledged by the
// L2; L1INV invalidates the L1 at a barrier; queue storage is laid out by the program at the addresses the L2's
// ordinary mapping places in the queue's slice and set (no remap of ordinary accesses). Default ON;
// VORTEX_LEGACY=1 restores the old model exactly (window appends, tails mirrored, atomics via the L1,
// loads-only fence, remapped queue regions, e1w4a knobs off unless set).
bool isa2_on();
uint64_t aq_home_slot(uint32_t q, uint64_t slot);   // ISA2: address of a queue's slot in the program's layout
void wack_add(uint32_t cid, int64_t n);             // ISA2: core cid has n more writes on their way to the L2
void wack_done(uint32_t cid);                       // ISA2: the L2 performed one of core cid's writes
bool wack_idle(uint32_t cid);                       // ISA2: every write of core cid acknowledged
void wack_enable();                                 // ISA2: an L2 that acknowledges writes exists
extern uint64_t g_isa2_appends, g_isa2_counts, g_isa2_l1skips, g_isa2_l1invs, g_isa2_fence_wait;
extern uint64_t g_aq_open_evicts, g_aq_evict_first, g_aq_protect_saves;       // any byte of a queue region or the pool (appends only write there)
inline bool ap_is_window(uint64_t addr) {
  return g_aq_size != 0 && addr >= ap_base() && (addr - ap_base()) < AP_WINDOW;
}
inline uint32_t ap_line(uint64_t addr) { return (uint32_t)((addr - ap_base()) / AQ_LINE); }
struct ApTrace {
  std::vector<uint64_t> qlines;   // queue (and tail) lines to read, in order
  std::vector<uint32_t> qidx;     // per word: index into qlines it needs
  std::vector<uint64_t> tlines;   // per word: target line
  uint32_t op = 0;                // 0 STORE, 1 ADD
  bool pending = false;           // traced functionally, not yet activated
};
ApTrace&  ap_trace(uint32_t core, uint32_t lane);
uint64_t* ap_cmdwords(uint32_t core, uint32_t lane);   // 8-word staging
uint32_t& ap_done(uint32_t core);
uint32_t& ap_pending(uint32_t core);

inline AddrType get_addr_type(uint64_t addr) {
  if (addr >= IO_BASE_ADDR && addr < IO_END_ADDR) {
     return AddrType::IO;
  }
  if (LMEM_ENABLED) {
    if (addr >= LMEM_BASE_ADDR && (addr-LMEM_BASE_ADDR) < (1 << LMEM_LOG_SIZE)) {
        return AddrType::Shared;
    }
  }
  if (g_msgmem_size != 0 && addr >= g_msgmem_base && (addr - g_msgmem_base) < g_msgmem_size) {
    return AddrType::MsgMem;
  }
  if (g_rpush_size != 0 && addr >= g_rpush_base && (addr - g_rpush_base) < g_rpush_size) {
    return AddrType::RemoteLmem;
  }
  if (g_aq_size != 0 && addr >= g_aq_base && (addr - g_aq_base) < g_aq_size + AP_WINDOW) {
    return AddrType::AppendQ;   // the queue window and, right after it, the apply engine window
  }
  return AddrType::Global;
}

inline std::ostream &operator<<(std::ostream &os, const AddrType& type) {
  switch (type) {
  case AddrType::Global: os << "Global"; break;
  case AddrType::Shared: os << "Shared"; break;
  case AddrType::MsgMem: os << "MsgMem"; break;
  case AddrType::RemoteLmem: os << "RemoteLmem"; break;
  case AddrType::AppendQ: os << "AppendQ"; break;
  case AddrType::IO:     os << "IO"; break;
  default: assert(false);
  }
  return os;
}

///////////////////////////////////////////////////////////////////////////////

struct mem_addr_size_t {
  uint64_t addr;
  uint32_t size;
};

///////////////////////////////////////////////////////////////////////////////

enum class ArbiterType {
  Priority,
  RoundRobin,
  Matrix
};

inline std::ostream &operator<<(std::ostream &os, const ArbiterType& type) {
  switch (type) {
  case ArbiterType::Priority:   os << "Priority"; break;
  case ArbiterType::RoundRobin: os << "RoundRobin"; break;
  case ArbiterType::Matrix:     os << "Matrix"; break;
  default: assert(false);
  }
  return os;
}

class IArbiterImpl {
public:
  IArbiterImpl() {}
  virtual ~IArbiterImpl() {}
  virtual uint32_t grant(const BitVector<>& requests) = 0;
  virtual void reset() = 0;
};

class PriorityArbiter : public IArbiterImpl {
public:
  PriorityArbiter(uint32_t size) : size_(size) {
    this->reset();
  }

  uint32_t grant(const BitVector<>& requests) override {
    assert(requests.size() == size_);
    for (uint32_t i = 0; i < size_; ++i) {
      if (requests.test(i)) {
        return i;
      }
    }
    return -1;
  }

  void reset() override {
    //--
  }
private:
  uint32_t size_;
};

class RoundRobinArbiter : public IArbiterImpl {
public:
  RoundRobinArbiter(uint32_t size) : size_(size) {
    this->reset();
  }

  uint32_t grant(const BitVector<>& requests) override {
    assert(requests.size() == size_);
    uint32_t start = (last_grant_ + 1) % size_;
    for (uint32_t i = 0; i < size_; ++i) {
      uint32_t idx = (start + i) % size_;
      if (requests.test(idx)) {
        last_grant_ = idx;
        return idx;
      }
    }
    return -1;
  }

  void reset() override {
    last_grant_ = 0;
  }

private:
  uint32_t size_;
  uint32_t last_grant_;
};

class MatrixArbiter : public IArbiterImpl {
public:
  MatrixArbiter(uint32_t size)
    : size_(size)
    , priority_matrix_(size, std::vector<bool>(size)) {
    this->reset();
  }

  uint32_t grant(const BitVector<>& requests) override {
    assert(requests.size() == size_);
    for (uint32_t i = 0; i < size_; ++i) {
      if (requests[i]) {
        // Check if this request has the highest priority by comparing it
        bool highest_priority = true;
        for (uint32_t j = 0; j < size_; ++j) {
          if (requests[j] && priority_matrix_[i][j]) {
            // If there is any active request with higher priority, this is not the highest
            highest_priority = false;
            break;
          }
        }

        if (highest_priority) {
          // Update the priority matrix: clear the row and set the column
          for (uint32_t j = 0; j < size_; ++j) {
            if (i != j) {
              priority_matrix_[i][j] = false;
              priority_matrix_[j][i] = true;
            }
          }
          return i; // Return the granted request index
        }
      }
    }
    return -1;
  }

  void reset() override {
    // Initialize the priority matrix
    for (uint32_t i = 0; i < size_; ++i) {
      priority_matrix_[i].resize(size_);
      // Initialize only the upper triangle to true
      for (uint32_t j = i + 1; j < size_; ++j) {
        priority_matrix_[i][j] = true;
      }
    }
  }

private:
  uint32_t size_;
  std::vector<std::vector<bool>> priority_matrix_;
};

class Arbiter {
public:
  Arbiter(ArbiterType type = ArbiterType::Priority, uint32_t size = 0) {
    switch (type) {
    case ArbiterType::Priority:
      impl_ = std::make_shared<PriorityArbiter>(size);
      break;
    case ArbiterType::RoundRobin:
      impl_ = std::make_shared<RoundRobinArbiter>(size);
      break;
    case ArbiterType::Matrix:
      impl_ = std::make_shared<MatrixArbiter>(size);
      break;
    default:
      assert(false); // Should never reach here
    }
  }

  virtual ~Arbiter() {}

  uint32_t grant(const BitVector<>& requests) {
    return impl_->grant(requests);
  }

  void reset() {
    impl_->reset();
  }

private:
  std::shared_ptr<IArbiterImpl> impl_;
};

///////////////////////////////////////////////////////////////////////////////

struct LsuReq {
  BitVector<> mask;
  std::vector<uint64_t> addrs;
  bool     write;
  bool     amo;   // AMO timed as a read (VORTEX_AMO_RT=1): the L2 still marks its line dirty
  uint8_t  aqop = 0;     // ISA2: 1 append (addrs = queue numbers at the LSU, slot addresses after the switch), 2 count
  bool     l1skip = false;   // ISA2: atomics, appends and counts pass the L1 untouched and are performed at the L2
  uint32_t tag;
  uint32_t cid;
  uint64_t uuid;

  LsuReq(uint32_t size)
    : mask(size)
    , addrs(size, 0)
    , write(false)
    , amo(false)
    , tag(0)
    , cid(0)
    , uuid(0)
  {}

  friend std::ostream &operator<<(std::ostream &os, const LsuReq& req) {
    os << "rw=" << req.write << ", mask=" << req.mask << ", addr={";
    bool first_addr = true;
    for (size_t i = 0; i < req.mask.size(); ++i) {
      if (!first_addr) os << ", ";
      first_addr = false;
      if (req.mask.test(i)) {
        os << "0x" << std::hex << req.addrs.at(i) << std::dec;
      } else {
        os << "-";
      }
    }
    os << "}, tag=0x" << std::hex << req.tag << std::dec << ", cid=" << req.cid;
    os << " (#" << req.uuid << ")";
    return os;
  }
};

///////////////////////////////////////////////////////////////////////////////

struct LsuRsp {
  BitVector<> mask;
  uint64_t tag;
  uint32_t cid;
  uint64_t uuid;

 LsuRsp(uint32_t size)
    : mask(size)
    , tag (0)
    , cid(0)
    , uuid(0)
  {}

  friend std::ostream &operator<<(std::ostream &os, const LsuRsp& rsp) {
    os << "mask=" << rsp.mask << ", tag=0x" << std::hex << rsp.tag << std::dec << ", cid=" << rsp.cid;
    os << " (#" << rsp.uuid << ")";
    return os;
  }
};

///////////////////////////////////////////////////////////////////////////////

struct MemReq {
  uint64_t addr;
  bool     write;
  bool     amo = false;
  uint8_t  aqop = 0;         // ISA2: 1 append, 2 count (see LsuReq)
  bool     l1skip = false;   // ISA2: the L1 forwards it on its uncached path
  AddrType type;
  uint32_t tag;
  uint32_t cid;
  uint64_t uuid;

  MemReq(uint64_t _addr = 0,
          bool _write = false,
          AddrType _type = AddrType::Global,
          uint64_t _tag = 0,
          uint32_t _cid = 0,
          uint64_t _uuid = 0
  ) : addr(_addr)
    , write(_write)
    , type(_type)
    , tag(_tag)
    , cid(_cid)
    , uuid(_uuid)
  {}

  friend std::ostream &operator<<(std::ostream &os, const MemReq& req) {
    os << "rw=" << req.write << ", ";
    os << "addr=0x" << std::hex << req.addr << std::dec << ", type=" << req.type;
    os << ", tag=0x" << std::hex << req.tag << std::dec << ", cid=" << req.cid;
    os << " (#" << req.uuid << ")";
    return os;
  }
};

///////////////////////////////////////////////////////////////////////////////

struct MemRsp {
  uint64_t tag;
  uint32_t cid;
  uint64_t uuid;

  MemRsp(uint64_t _tag = 0, uint32_t _cid = 0, uint64_t _uuid = 0)
    : tag (_tag)
    , cid(_cid)
    , uuid(_uuid)
  {}

  friend std::ostream &operator<<(std::ostream &os, const MemRsp& rsp) {
    os << "tag=0x" << std::hex << rsp.tag << std::dec << ", cid=" << rsp.cid;
    os << " (#" << rsp.uuid << ")";
    return os;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
class HashTable {
public:
  typedef T DataType;

  HashTable(uint32_t capacity)
    : entries_(capacity)
    , size_(0)
  {}

  bool empty() const {
    return (0 == size_);
  }

  bool full() const {
    return (size_ == entries_.size());
  }

  uint32_t size() const {
    return size_;
  }

  bool contains(uint32_t index) const {
    return entries_.at(index).first;
  }

  const T& at(uint32_t index) const {
    auto& entry = entries_.at(index);
    assert(entry.first);
    return entry.second;
  }

  T& at(uint32_t index) {
    auto& entry = entries_.at(index);
    assert(entry.first);
    return entry.second;
  }

  uint32_t allocate(const T& value) {
    for (uint32_t i = 0, n = entries_.size(); i < n; ++i) {
      auto& entry = entries_.at(i);
      if (!entry.first) {
        entry.first = true;
        entry.second = value;
        ++size_;
        return i;
      }
    }
    assert(false);
    return -1;
  }

  void release(uint32_t index) {
    auto& entry = entries_.at(index);
    assert(entry.first);
    entry.first = false;
    --size_;
  }

  void clear() {
    for (uint32_t i = 0, n = entries_.size(); i < n; ++i) {
      auto& entry = entries_.at(i);
      entry.first = false;
    }
    size_ = 0;
  }

private:
  std::vector<std::pair<bool, T>> entries_;
  uint32_t size_;
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
class TFifo : public SimObject<TFifo<T>> {
public:
  using Type = T;

  TFifo(const SimContext& ctx, const char* name, uint32_t delay, uint32_t capacity = 0)
    : SimObject<TFifo<T>>(ctx, name)
    , bus_(this, capacity)
    , delay_(delay) {
  }

  void reset() {
    //--
  }

  void tick() {
    //--
  }

  bool empty() const {
    return bus_.empty();
  }

  bool full() const {
    return bus_.full();
  }

  uint32_t size() const {
    return bus_.size();
  }

  void push(const Type& value) {
    if (bus_.full()) {
      throw std::runtime_error("FIFO is full");
    }
    bus_.push(value, delay_);
  }

  Type& front() {
    if (bus_.empty()) {
      throw std::runtime_error("FIFO is empty");
    }
    return bus_.front();
  }

  const Type& front() const {
    if (bus_.empty()) {
      throw std::runtime_error("FIFO is empty");
    }
    return bus_.front();
  }

  void pop() {
    if (bus_.empty()) {
      throw std::runtime_error("FIFO is empty");
    }
    bus_.pop();
  }

private:
  SimPort<Type> bus_;
  uint32_t delay_;
};

///////////////////////////////////////////////////////////////////////////////
template <typename Type>
class TxArbiter : public SimObject<TxArbiter<Type>> {
public:
  typedef Type ReqType;

  struct RspType {
    Type     data;
    uint32_t input;

    RspType(const Type& _data, uint32_t _input = 0)
      : data(_data)
      , input(_input)
    {}

    operator Type() const {
      return data;
    }
  };

  std::vector<SimPort<ReqType>> Inputs;
  std::vector<SimPort<RspType>> Outputs;

  TxArbiter(
    const SimContext& ctx,
    const char* name,
    ArbiterType type,
    uint32_t num_inputs,
    uint32_t num_outputs = 1,
    uint32_t delay = 1
  ) : SimObject<TxArbiter<Type>>(ctx, name)
    , Inputs(num_inputs, this)
    , Outputs(num_outputs, this)
    , delay_(delay)
    , lg2_num_reqs_(log2ceil(num_inputs / num_outputs))
    , arbiters_(num_outputs, {type, 1u << lg2_num_reqs_})
  {
    assert(delay != 0);
    assert(num_inputs <= 64);
    assert(num_outputs <= 64);
    assert(num_inputs >= num_outputs);

    // bypass mode
    if (num_inputs == num_outputs) {
      for (uint32_t i = 0; i < num_inputs; ++i) {
        Inputs.at(i).bind(&Outputs.at(i));
      }
    }
  }

  void reset() {
    for (auto& arb : arbiters_) {
      arb.reset();
    }
  }

  void tick() {
    uint32_t I = Inputs.size();
    uint32_t O = Outputs.size();
    uint32_t R = 1 << lg2_num_reqs_;

    // skip bypass mode
    if (I == O)
      return;

    // process inputs
    for (uint32_t o = 0; o < O; ++o) {
      BitVector<> requests(R);
      for (uint32_t r = 0; r < R; ++r) {
        uint32_t i = o * R + r;
        if (i >= I)
          continue;
        requests.set(r, !Inputs.at(i).empty());
      }
      if (requests.any()) {
        uint32_t g = arbiters_.at(o).grant(requests);
        uint32_t i = o * R + g;
        auto& req_in = Inputs.at(i);
        auto& req = req_in.front();
        DT(4, this->name() << "-req" << i << "_" << o << ": " << req);
        Outputs.at(o).push(RspType(req, i), delay_);
        req_in.pop();
      }
    }
  }

protected:

  uint32_t delay_;
  uint32_t lg2_num_reqs_;
  std::vector<Arbiter> arbiters_;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class TxCrossBar : public SimObject<TxCrossBar<Type>> {
public:
  typedef Type ReqType;

  struct RspType {
    Type     data;
    uint32_t input;

    RspType(const Type& _data, uint32_t _input= 0)
      : data(_data)
      , input(_input)
    {}

    operator Type() const {
      return data;
    }
  };

  std::vector<SimPort<ReqType>> Inputs;
  std::vector<SimPort<RspType>> Outputs;

  TxCrossBar(
    const SimContext& ctx,
    const char* name,
    uint32_t num_inputs,
    uint32_t num_outputs,
    std::function<uint32_t(const Type& req)> output_sel,
    uint32_t delay = 1
  )
    : SimObject<TxCrossBar<Type>>(ctx, name)
    , Inputs(num_inputs, this)
    , Outputs(num_outputs, this)
    , delay_(delay)
    , lg2_inputs_(log2ceil(num_inputs))
    , lg2_outputs_(log2ceil(num_outputs))
    , output_sel_(output_sel)
    , collisions_(0)
#ifdef XBAR_RR
    , rr_next_(num_outputs, 0)
#endif
  {
    assert(delay != 0);
    assert(num_inputs <= 64);
    assert(num_outputs <= 64);
    assert(ispow2(num_outputs));
    assert(output_sel != nullptr);

    // bypass mode
    if (num_inputs == 1 && num_outputs == 1) {
      Inputs.at(0).bind(&Outputs.at(0));
    }
  }

  void reset() {
    //--
  }

  void tick() {
    uint32_t I = Inputs.size();
    uint32_t O = Outputs.size();
    if (I == 1 && O == 1)
      return;

    // process incoming requests
    for (uint32_t o = 0; o < O; ++o) {
      int32_t input_idx = -1;
      bool has_collision = false;
#ifdef XBAR_RR
      // round-robin per output: the scan starts after the input this output granted last
      // (the fixed lowest-index-first scan starved the last sockets of a 32-core machine)
      for (uint32_t k = 0; k < I; ++k) {
        uint32_t i = (rr_next_[o] + k) % I;
#else
      for (uint32_t i = 0; i < I; ++i) {
#endif
        auto& req_in = Inputs.at(i);
        if (req_in.empty())
          continue;
        auto& req = req_in.front();
        uint32_t output_idx = 0;
        if (lg2_outputs_ != 0) {
          // select output index
          output_idx = output_sel_(req);
          // skip if input is not going to current output
          if (output_idx != o)
            continue;
        }
        if (input_idx != -1) {
          has_collision = true;
          break;
        }
        input_idx = i;
      }
      if (input_idx != -1) {
        auto& req_in = Inputs.at(input_idx);
        auto& req = req_in.front();
        DT(4, this->name() << "-req" << input_idx << "_" << o << ": " << req);
        Outputs.at(o).push(RspType(req, input_idx), delay_);
        req_in.pop();
        collisions_ += has_collision;
#ifdef XBAR_RR
        rr_next_[o] = (input_idx + 1) % I;
#endif
      }
    }
  }

  uint64_t collisions() const {
    return collisions_;
  }

protected:

  uint32_t delay_;
  uint32_t lg2_inputs_;
  uint32_t lg2_outputs_;
  std::function<uint32_t(const Type& req)> output_sel_;
  uint64_t collisions_;
#ifdef XBAR_RR
  std::vector<uint32_t> rr_next_;
#endif
};

///////////////////////////////////////////////////////////////////////////////

template <typename Req, typename Rsp>
class TxRxArbiter : public SimObject<TxRxArbiter<Req, Rsp>> {
public:
  typedef Req ReqType;
  typedef Rsp RspType;

  std::vector<SimPort<Req>>  ReqIn;
  std::vector<SimPort<Rsp>>  RspIn;

  std::vector<SimPort<Req>>  ReqOut;
  std::vector<SimPort<Rsp>>  RspOut;

  TxRxArbiter(
    const SimContext& ctx,
    const char* name,
    ArbiterType type,
    uint32_t num_inputs,
    uint32_t num_outputs = 1,
    uint32_t req_delay = 1,
    uint32_t rsp_delay = 1
  )
    : SimObject<TxRxArbiter<Req, Rsp>>(ctx, name)
    , ReqIn(num_inputs, this)
    , RspIn(num_inputs, this)
    , ReqOut(num_outputs, this)
    , RspOut(num_outputs, this)
    , arbiter_(nullptr)
    , rsp_delay_(rsp_delay)
    , lg2_num_reqs_(log2ceil(num_inputs / num_outputs))
  {
    if (num_inputs != num_outputs) {
      // allocate arbiter
      arbiter_ = ReqArb::Create(name, type, num_inputs, num_outputs, req_delay);
      // bind arbiter inputs and outputs
      for (uint32_t i = 0; i < num_inputs; ++i) {
        ReqIn.at(i).bind(&arbiter_->Inputs.at(i));
      }
      for (uint32_t o = 0; o < num_outputs; ++o) {
        arbiter_->Outputs.at(o).bind(&ReqOut.at(o),
          [lg2_num_reqs = lg2_num_reqs_](const typename ReqArb::RspType& arb_rsp) {
            Req req(arb_rsp.data);
            if (lg2_num_reqs != 0) {
              uint32_t r = arb_rsp.input & ((1 << lg2_num_reqs) - 1);
              req.tag = (req.tag << lg2_num_reqs) | r;
            }
            return req;
          });
      }
    } else {
      // bypass mode
      for (uint32_t i = 0; i < num_inputs; ++i) {
        ReqIn.at(i).bind(&ReqOut.at(i));
        RspOut.at(i).bind(&RspIn.at(i));
      }
    }
  }

  void reset() {
    //--
  }

  void tick() {
    if (!arbiter_)
      return;

    uint32_t O = ReqOut.size();
    uint32_t R = 1 << lg2_num_reqs_;

    // process outgoing responses
    for (uint32_t o = 0; o < O; ++o) {
      auto& rsp_out = RspOut.at(o);
      if (!rsp_out.empty()) {
        auto& rsp = rsp_out.front();
        uint32_t r = 0;
        Rsp in_rsp(rsp);
        if (lg2_num_reqs_ != 0) {
          r = rsp.tag & (R-1);
          in_rsp.tag = rsp.tag >> lg2_num_reqs_;
        }
        uint32_t i = o * R + r;
        DT(4, this->name() << "-rsp" << o << "_" << i << ": " << in_rsp);
        RspIn.at(i).push(in_rsp, rsp_delay_);
        rsp_out.pop();
      }
    }
  }

protected:
  typedef TxArbiter<Req> ReqArb;

  typename ReqArb::Ptr arbiter_;
  uint32_t rsp_delay_;
  uint32_t lg2_num_reqs_;
};

///////////////////////////////////////////////////////////////////////////////

template <typename Req, typename Rsp>
class TxRxCrossBar : public SimObject<TxRxCrossBar<Req, Rsp>> {
public:
  typedef Req ReqType;
  typedef Rsp RspType;

  std::vector<SimPort<Req>> ReqIn;
  std::vector<SimPort<Rsp>> RspIn;

  std::vector<SimPort<Req>> ReqOut;
  std::vector<SimPort<Rsp>> RspOut;

  TxRxCrossBar(
    const SimContext& ctx,
    const char* name,
    ArbiterType type,
    uint32_t num_inputs,
    uint32_t num_outputs,
    std::function<uint32_t(const Req& req)> output_sel,
    uint32_t req_delay = 1,
    uint32_t rsp_delay = 1
  )
    : SimObject<TxRxCrossBar<Req, Rsp>>(ctx, name)
    , ReqIn(num_inputs, this)
    , RspIn(num_inputs, this)
    , ReqOut(num_outputs, this)
    , RspOut(num_outputs, this)
    , crossbar_(nullptr)
    , arbiter_(type, num_outputs)
    , rsp_delay_(rsp_delay)
    , lg2_inputs_(log2ceil(num_inputs)) {

    if (num_inputs != 1 || num_outputs != 1) {
      // allocate crossbar
      crossbar_ = ReqXbar::Create(name, num_inputs, num_outputs, output_sel, req_delay);
      // bind crossbar inputs and outputs
      for (uint32_t i = 0; i < num_inputs; ++i) {
        ReqIn.at(i).bind(&crossbar_->Inputs.at(i));
      }
      for (uint32_t o = 0; o < num_outputs; ++o) {
        crossbar_->Outputs.at(o).bind(&ReqOut.at(o),
          [lg2_inputs = lg2_inputs_](const typename ReqXbar::RspType& xbar_rsp) {
            Req req(xbar_rsp.data);
            if (lg2_inputs != 0) {
              req.tag = (req.tag << lg2_inputs) | xbar_rsp.input;
            }
            return req;
          });
      }
    } else {
      // bypass mode
      ReqIn.at(0).bind(&ReqOut.at(0));
      RspOut.at(0).bind(&RspIn.at(0));
    }
  }

  void reset() {
    arbiter_.reset();
  }

  void tick() {
    if (!crossbar_)
      return;

    uint32_t I = ReqIn.size();
    uint32_t O = ReqOut.size();
    uint32_t R = 1 << lg2_inputs_;

    // process outgoing responses
    for (uint32_t i = 0; i < I; ++i) {
      BitVector<> requests(O);
      for (uint32_t o = 0; o < O; ++o) {
        auto& rsp_out = RspOut.at(o);
        if (rsp_out.empty())
          continue;
        auto& rsp = rsp_out.front();
        // skip if response is not going to current input
        if (lg2_inputs_ != 0) {
          uint32_t input_idx = rsp.tag & (R-1);
          if (input_idx != i)
            continue;
        }
        requests.set(o);
      }
      if (requests.any()) {
        uint32_t g = arbiter_.grant(requests);
        auto& rsp_out = RspOut.at(g);
        auto& rsp = rsp_out.front();
        Rsp in_rsp(rsp);
        if (lg2_inputs_ != 0) {
          in_rsp.tag = rsp.tag >> lg2_inputs_;
        }
        DT(4, this->name() << "-rsp" << g << "_" << i << ": " << in_rsp);
        RspIn.at(i).push(in_rsp, rsp_delay_);
        rsp_out.pop();
      }
    }
  }

  uint64_t collisions() const {
    if (crossbar_) {
      return crossbar_->collisions();
    }
    return 0;
  }

protected:
  typedef TxCrossBar<Req> ReqXbar;

  typename ReqXbar::Ptr crossbar_;
  Arbiter arbiter_;
  uint32_t rsp_delay_;
  uint32_t lg2_inputs_;
};

///////////////////////////////////////////////////////////////////////////////

class LocalMemSwitch : public SimObject<LocalMemSwitch> {
public:
  SimPort<LsuReq> ReqIn;
  SimPort<LsuRsp> RspIn;

  SimPort<LsuReq> ReqLmem;
  SimPort<LsuRsp> RspLmem;

  SimPort<LsuReq> ReqDC;
  SimPort<LsuRsp> RspDC;

  LocalMemSwitch(
    const SimContext& ctx,
    const char* name,
    uint32_t delay,
    uint32_t core_id = 0
  );

  void reset();

  void tick();

private:
  uint32_t delay_;
  uint32_t core_id_;
};

///////////////////////////////////////////////////////////////////////////////

class LsuMemAdapter : public SimObject<LsuMemAdapter> {
public:
  SimPort<LsuReq> ReqIn;
  SimPort<LsuRsp> RspIn;

  std::vector<SimPort<MemReq>> ReqOut;
  std::vector<SimPort<MemRsp>> RspOut;

  LsuMemAdapter(
    const SimContext& ctx,
    const char* name,
    uint32_t num_inputs,
    uint32_t delay
  );

  void reset();

  void tick();

private:
  uint32_t delay_;
};

using LsuArbiter  = TxRxArbiter<LsuReq, LsuRsp>;
using MemArbiter  = TxRxArbiter<MemReq, MemRsp>;
using MemCrossBar = TxRxCrossBar<MemReq, MemRsp>;

}
