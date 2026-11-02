# Part 4(a): dot-product driver + HAL + stand-alone test, cross-compiled for the simulated ARM.
#   source the aarch64 toolchain first (LRC):
#     module load xilinx/2022 ; source /usr/local/packages/Xilinx_2022.2/Vivado/2022.2/settings64.sh
#   make            kernel module (driver/dotprod_drv.ko) and test (test/hal_test)
#   make module     only the kernel module
#   make test       only the test
#   make STATIC=    if the toolchain has no static libc
CROSS  ?= aarch64-linux-gnu-
CC      = $(CROSS)gcc
STATIC ?= -static
CFLAGS ?= -O2 -Wall -Wextra
# golden_data.h lives in part3/test
GOLDEN ?= ../part3/test

all: module test

module:
	$(MAKE) -C driver CROSS=$(CROSS)

test: test/hal_test

test/hal_test: test/hal_test.c test/ref_vec_dot.c hal/dot_hal.c hal/dot_hal.h hal/dot_hal_internal.h driver/dotprod_ioctl.h
	$(CC) $(CFLAGS) $(STATIC) -I $(GOLDEN) -o $@ test/hal_test.c test/ref_vec_dot.c hal/dot_hal.c -lm -lpthread
	-$(CROSS)strip $@

clean:
	$(MAKE) -C driver clean
	$(RM) test/hal_test

.PHONY: all module test clean
