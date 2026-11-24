ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := fgc_hs

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# rebuild when the app or the actor runtime headers change
HDRS    := $(wildcard $(SRC_DIR)/*.h $(VORTEX_HOME)/tests/regression/fgc_common/*.h)
VX_HDRS := $(HDRS)

OPTS ?= -c1 -m16 -l16

KERNEL_LIB := vortex2

# line tables only: the code is the same as without them. DWARF 4: lld
# segfaults linking this kernel's DWARF 5 line tables on rv64.
VX_CFLAGS += -gline-tables-only -gdwarf-4

# a divergent tail call would jump to one lane's target (simx ICALL_SPLIT
# splits linking calls only)
VX_CFLAGS += -fno-optimize-sibling-calls

# the buffer configuration is compiled into the kernel and the host
# (fgc_actor_args.h): topology, sender granularity, termination
FGC_TOPO   ?= 1
FGC_SENDER ?= 0
FGC_TERM   ?= 1
FGC_DEFS   := -DFGC_TOPO=$(FGC_TOPO) -DFGC_SENDER=$(FGC_SENDER) -DFGC_TERM=$(FGC_TERM)
VX_CFLAGS  += $(FGC_DEFS)
CXXFLAGS   += $(FGC_DEFS)

include ../common.mk
