ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := fgc_pcmap

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# rebuild when the app or the actor runtime headers change
HDRS    := $(wildcard $(SRC_DIR)/*.h $(VORTEX_HOME)/tests/regression/fgc_common/*.h)
VX_HDRS := $(HDRS)

OPTS ?= -n256 -i100

KERNEL_LIB := vortex2

# line tables only: the code is the same as without them
VX_CFLAGS += -gline-tables-only

include ../common.mk
