ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := simt_div

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?= -n64 -m0

KERNEL_LIB := vortex2

# extra kernel flags for divergence experiments, e.g. -fno-jump-tables. The
# default keeps calls out of tail position: a divergent tail call jumps to one
# lane's target (simx ICALL_SPLIT splits linking calls only).
DIV_CFLAGS ?= -fno-optimize-sibling-calls
VX_CFLAGS += $(DIV_CFLAGS)

include ../common.mk
