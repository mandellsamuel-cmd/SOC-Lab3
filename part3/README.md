# Lab 3, Part 3: Dot-product accelerator in the QEMU/SystemC virtual platform

A SystemC TLM-2.0 model of the Lab 2 fixed-point Q8_0 dot-product accelerator, attached as a
memory-mapped slave to the ZynqMP co-simulation platform (`zynqmp_demo`). The ARM (Linux on QEMU)
loads operands into the accelerator's scratchpads, starts it, polls for completion and reads the result.

## Files

| File | Purpose |
|---|---|
| `dot_product.h` | SystemC module `dot_product` (TLM target socket, scratchpads, registers, latency model) |
| `hls_q8_dot_vec.h` | Function prototypes and sizes (`QWEN_N` = 896, `QWEN_NB` = 28, `QK8_0` = 32) |
| `hls_q8_dot_vec_fixed.cpp` | Lab 2 fixed-point kernel. Called by the model for the numerical result |
| `hls_q8_dot_vec.cpp` | Lab 2 floating-point version (reference only, **not built** into the platform) |
| `zynqmp_demo.cc` | Top-level platform, based on Xilinx's demo. Instantiates and maps the accelerator |
| `Makefile` | Xilinx demo Makefile with `hls_q8_dot_vec_fixed.o` and the Vitis include path added |
| `test/dot_golden_test.c` | ARM-side test program (runs under Linux in QEMU) |
| `test/golden_data.h` | Golden inputs and bit-exact expected outputs (18 cases) |
| `test/gen_golden.py` | Generates `golden_data.h` (deterministic) |
| `test/Makefile` | Cross-compiles the test for aarch64 |

## Changes relative to the stock `zynqmp_demo`

In `zynqmp_demo.cc`:

1. `#include "dot_product.h"` and a `dot_product dot_accel;` member, constructed as `dot_accel("dot_accel")`.
2. `bus.memmap(0xa0400000ULL, 0x10000 - 1, ADDRMODE_RELATIVE, -1, dot_accel.target_socket);`
   This must stay **before** the catch-all `memmap(0x0, 0xffffffff - 1, ... s_axi_hpc_fpd[0])`.
3. `NR_DEVICES` changed from `6 + NR_DEMODMA` to `(7 + NR_DEMODMA)`. The accelerator is one more target
   on the interconnect than the stock demo has. `NR_MASTERS` is only parenthesized.
4. VCD tracing is off (the two `trace_fp` lines in `sc_main` are commented out).

In the `Makefile`: `SC_OBJS += hls_q8_dot_vec_fixed.o` and `CPPFLAGS += -I <Vitis_HLS>/include`
(needed for `hls_half.h`, `ap_int.h` and `utils/x_hls_utils.h`). Adjust the Vitis path if yours differs.

## Accelerator register map

Base address `0xa0400000`, 64 KB window. Addresses below are offsets from the base. The bus uses
relative addressing, so the model sees the offsets directly.

| Offset | Name | Access | Description |
|---|---|---|---|
| `0x0000` | CTRL | W | bit 0 = start. Ignored (SystemC warning) if the accelerator is busy |
| `0x0004` | STATUS | R | `1` = ready/done, `0` = busy |
| `0x0008` | RESULT | R | FP32 result, valid when STATUS = 1 |
| `0x0010` | TIME_LO | R | Simulated time in ns, low 32 bits. Reading it latches TIME_HI |
| `0x0014` | TIME_HI | R | Simulated time in ns, high 32 bits (read TIME_LO first) |
| `0x1000` | QX | W/R | 896 x `int8` (quantized x) |
| `0x2000` | DX | W/R | 28 x FP16 raw bits (x block scales), 56 bytes |
| `0x3000` | QY | W/R | 896 x `int8` |
| `0x4000` | DY | W/R | 28 x FP16 raw bits |

Behavior:
- Operands are sampled at the moment of the start write. Scratchpad writes while busy are ignored with a warning.
- STATUS goes to `0` immediately inside the start write, so a read right after the start cannot see a stale "ready".
- RESULT is published only after the modeled latency: **457 cycles x 10 ns = 4570 ns** (Lab 2 HLS report design point).
  Every bus access also adds 10 ns to the transaction delay.
- Out-of-range or unmapped offsets return `TLM_ADDRESS_ERROR_RESPONSE`.
- Accesses of 1 to 8 bytes are supported. **Use aligned 32-bit accesses from software** (see notes below).
- Local memory: 2 x 896 B + 2 x 56 B = 1904 B, far below the ZU3EG limit (216 x 36 Kb BRAM = 7.6 Mb).

## Build the SystemC platform

Environment as in the QEMU/SystemC tutorial (LRC machine or the course Docker image):

```
module load systemc/2.3.4
cd <this directory>       # must contain libsystemctlm-soc (git submodule update --init libsystemctlm-soc)
make zynqmp_demo          # build only this target; plain 'make' also builds the unrelated Versal demos
```

Notes:
- `make` expects `libsystemctlm-soc` next to the Makefile. It is **not** copied into this repo. Fetch it with the
  submodule command above, or copy this directory's files into your `systemctlm-cosim-demo` checkout.
- The Makefile uses lowercase `libsystemctlm-soc/soc/xilinx/...` paths. If your checkout uses a different case, match it.
- If the link fails with undefined `fpo`/Xilinx FP symbols, add `-DHLS_NO_XIL_FPO_LIB` to the HLS object's `CPPFLAGS`.

## Run the co-simulation

Two terminals on the same machine.

Terminal 1 (QEMU):
```
/home/projects/gerstl/ece382m/qemu-boot -c
```
QEMU starts and then waits for the SystemC side. Add the icount/quantum options if you use them (`qemu-boot --help`).

Terminal 2 (SystemC):
```
module load systemc/2.3.4
./zynqmp_demo unix:<tmp path>/qemu-rport-_amba@0_cosim@0 <sync quantum in ns>
```
The quantum must match the one QEMU uses (the tutorial uses `1000000`). Linux then finishes booting.
For timing measurements, use a small quantum such as `1000` or `100` on **both** sides. With a large quantum
the ARM and the model run loosely synchronized, and busy-poll counts and times are less meaningful.

Stop the simulation with Ctrl-C on the SystemC side (or a write to `0xa0000008`, the debug device's exit
register). Restart both sides for each run.

## Quick manual check (busybox `peek`/`poke`)

```
sudo peek 0xa0400004            # 0x00000001 (ready)
sudo poke 0xa0401000 0x01010101 # qx[0..3] = 1
sudo poke 0xa0403000 0x01010101 # qy[0..3] = 1
sudo poke 0xa0402000 0x00003c00 # dx[0] = 1.0 (fp16)
sudo poke 0xa0404000 0x00003c00 # dy[0] = 1.0
sudo poke 0xa0400000 1          # start
sudo peek 0xa0400008            # 0x40800000 = 4.0f
```
(Each shell command takes milliseconds of simulated time, so the accelerator is always done by the next
command. STATUS will read 1. Use the test program below to observe the busy phase.)

## Golden test

The test checks the full-size path: 896-element vectors, 28 scales, all edge cases below. Each result is compared
**bit-exactly** with the golden FP32.

Golden values: the accelerator accumulates all 28 block terms exactly as an integer and rounds once to FP32
(round-to-nearest-even). So the golden value is the correctly rounded FP32 of the exact real-number result.
`gen_golden.py` computes it with Python big integers and cross-checks with a second, independently written
rounding routine.

Cases (18): all-ones; maximum positive and negative sums; zero quants; zero scales; exact cancellation (result `+0.0`);
single non-zero element; moderate random; scales at the FP16 normal/subnormal boundary; FP16 subnormal scales;
smallest subnormal `2^-24`; large scales (+/-65504) and maximum magnitude; mixed subnormal/normal; 4 wide-dynamic-range random
cases (stress the rounding path). Infinity/NaN scales are not tested because the Lab 2 kernel does not handle them.

Build the test (on the host, cross-compiling for the simulated ARM):
```
cd test
make                           # uses aarch64-linux-gnu-gcc, produces a static binary 'dot_golden_test'
# different toolchain prefix:  make CROSS=aarch64-none-linux-gnu-
```

Get `dot_golden_test` into the simulated Linux by whatever mechanism the QEMU/SystemC tutorial uses for
adding files to the guest (it is a single static binary and needs no other files).

Run it in the guest, as root because it opens `/dev/mem`:
```
sudo ./dot_golden_test         # add -v for a double-precision cross-check per case
```
Expected output ends with `ALL PASSED: 18/18 cases passed`. Each line also shows `busy-polls` (status reads that
returned 0) and `sim-time` (ns between the start write and STATUS = 1, from the TIME registers). Exit code is 0 only if all pass.

Reference results from a host SystemC run (this model plus `hls_q8_dot_vec_fixed.cpp`, no QEMU): all 18 cases pass and
the measured time is about 4600 ns (4570 ns compute plus bus access delays). The co-simulated numbers depend on the sync quantum.

## Notes for Part 4 (HAL and driver)

- Use aligned 32-bit volatile loads/stores to the device window, as `dot_golden_test.c` does. `memcpy` into an
  `mmap`ed `/dev/mem` region can issue unaligned or wide accesses, which fault on real hardware (device memory).
- Pass FP16 scales as raw 16-bit patterns (`ggml_fp16_t` is already that).
- ggml's `block_q8_0` interleaves each 2-byte scale with 32 quants (34 bytes). The HAL must split these into the
  separate QX / DX arrays above.
- Poll STATUS; there is no interrupt in this model.
- Each call moves about 1.9 KB in and 4 bytes out. For GEMV the same `y` operand is reused across many rows, so
  keeping it resident would avoid reloading it (extra-credit 4(c)).

## Limitations

- Slave-only accelerator: the ARM copies all data (extra-credit 3(f) bus-master version not implemented).
- Fixed latency (single HLS design point), polling only, one operation at a time.
- Verification of the model was done with Xilinx's open-source `ap_int` headers and a small stand-in for the
  Vitis `fp_struct`/`half` types. Your real Vitis build is the authoritative one.
