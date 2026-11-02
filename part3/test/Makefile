# Cross-compile the golden test for the simulated ARM (aarch64 Linux).
#   make                               aarch64-linux-gnu-gcc (what the QEMU_SystemC_app examples use)
#   make CROSS=aarch64-none-linux-gnu- different toolchain prefix
#   make host                          native build (only useful for syntax checks; needs /dev/mem + the device to run)
CROSS ?= aarch64-linux-gnu-
CC     = $(CROSS)gcc
STATIC ?= -static          # if the toolchain has no static libc: make STATIC=
CFLAGS ?= -O2 -Wall -Wextra

all: dot_golden_test dot_dma_test

dot_golden_test: dot_golden_test.c golden_data.h
	$(CC) $(CFLAGS) $(STATIC) -o $@ dot_golden_test.c -lm
	-$(CROSS)strip $@

# extra credit: bus-mastering (DMA) mode
dot_dma_test: dot_dma_test.c golden_data.h
	$(CC) $(CFLAGS) $(STATIC) -o $@ dot_dma_test.c
	-$(CROSS)strip $@

host: dot_golden_test.c dot_dma_test.c golden_data.h
	$(or $(HOSTCC),cc) $(CFLAGS) -o dot_golden_test_host dot_golden_test.c -lm
	$(or $(HOSTCC),cc) $(CFLAGS) -o dot_dma_test_host dot_dma_test.c

# Regenerate the golden vectors (deterministic)
golden_data.h: gen_golden.py
	python3 gen_golden.py

clean:
	$(RM) dot_golden_test dot_golden_test_host dot_dma_test dot_dma_test_host

.PHONY: all host clean
