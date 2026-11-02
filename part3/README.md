# Lab 3, Part 3: Dot-product accelerator in the QEMU/SystemC virtual platform

A SystemC TLM-2.0 model of the Lab 2 fixed-point Q8_0 dot-product accelerator, attached as a
memory-mapped slave to the ZynqMP co-simulation platform (`zynqmp_demo`). The ARM (Linux on QEMU)
loads operands into the accelerator's scratchpads, starts it, polls for completion and reads the result.

## Files

```
part3/
  dot_product.h              SystemC module `dot_product` (TLM target socket, scratchpads, registers, latency model)
  hls_q8_dot_vec.h           Prototypes and sizes (QWEN_N = 896, QWEN_NB = 28, QK8_0 = 32)
  hls_q8_dot_vec_fixed.cpp   Lab 2 fixed-point kernel; the model calls it for the numerical result
  hls_q8_dot_vec.cpp         Lab 2 floating-point version (reference only, NOT built into the platform)
  q8_dot_vec.h               Lab 2 plain-C Q8_0 types/prototype (not used by Part 3; kept for the Part 4 software check)
  zynqmp_demo.cc             Top-level platform: Xilinx's zynqmp_demo + our accelerator
  debugdev.{cc,h}            }  unmodified copies of Xilinx's systemctlm-cosim-demo sources
  demo-dma.{cc,h}            }  that zynqmp_demo links against
  xilinx-axidma.{cc,h}       }
  Makefile                   Xilinx demo Makefile + our kernel object and the Vitis include path
  libsystemctlm-soc/         Xilinx co-simulation library (git submodule, fetched by `git submodule update --init`)
  test/
    dot_golden_test.c        ARM-side test program (runs under Linux in QEMU)
    dot_dma_test.c           ARM-side test of the bus-mastering (DMA) mode, extra credit 3(f)
    golden_data.h            Golden inputs and bit-exact expected outputs (18 cases)
    gen_golden.py            Regenerates golden_data.h (deterministic)
    Makefile                 Cross-compiles the test for aarch64
    host_sim/                Host-only testbench of the SystemC model (no QEMU/ARM needed)
    host_result.txt          Output of that testbench (host SystemC only)
    qemu_result.txt          Output of dot_golden_test inside QEMU + SystemC on LRC (the full co-simulation)
```

Only the `zynqmp_demo` target is supported (the Makefile still lists Xilinx's other demos, whose sources are not included).

## Changes relative to the stock `zynqmp_demo`

In `zynqmp_demo.cc`:

1. `#include "dot_product.h"` and a `dot_product dot_accel;` member, constructed as `dot_accel("dot_accel")`.
2. `bus.memmap(0xa0400000ULL, 0x10000 - 1, ADDRMODE_RELATIVE, -1, dot_accel.target_socket);`
   This must stay **before** the catch-all `memmap(0x0, 0xffffffff - 1, ... s_axi_hpc_fpd[0])`.
3. `NR_DEVICES` changed from `6 + NR_DEMODMA` to `(7 + NR_DEMODMA)`. The accelerator is one more target
   on the interconnect than the stock demo has. `NR_MASTERS` is only parenthesized.
4. VCD tracing is off (the two `trace_fp` lines in `sc_main` are commented out).
5. Extra credit 3(f): a second interconnect `iconnect<1,1> *mbus` maps `0x0 .. 0xffffffff` onto `zynq.s_axi_hp_fpd[0]`
   (S_AXI_HP0_FPD) and `dot_accel.master_socket` is bound to it, so the accelerator can read system memory itself.

In the `Makefile` (stock Xilinx Makefile plus): `SYSTEMC` set to the LRC SystemC 2.3.4 (tutorial step 3c); the `mcdma.o` and `mrmac.o`
lines commented out (tutorial step 3c); `SC_OBJS += hls_q8_dot_vec_fixed.o`; `CPPFLAGS += -I $(VITIS_HLS_INCLUDE)` (needed for
`hls_half.h`, `ap_int.h` and `utils/x_hls_utils.h`; override the variable if your Vitis path differs).

## Accelerator register map

Base address `0xa0400000`, 64 KB window. Addresses below are offsets from the base. The bus uses
relative addressing, so the model sees the offsets directly.

| Offset | Name | Access | Description |
|---|---|---|---|
| `0x0000` | CTRL | W | bit 0 = start, bit 1 = DMA mode (see below; 0 = copy mode). Ignored (SystemC warning) if busy |
| `0x0004` | STATUS | R | `1` = ready/done, `0` = busy |
| `0x0008` | RESULT | R | FP32 result, valid when STATUS = 1 (and ERROR = 0) |
| `0x000C` | ERROR | R | `0` ok, `1` DMA address outside DRAM/OCM, `2` DMA bus error. Cleared by each start |
| `0x0010` | TIME_LO | R | Simulated time in ns, low 32 bits. Reading it latches TIME_HI |
| `0x0014` | TIME_HI | R | Simulated time in ns, high 32 bits (read TIME_LO first) |
| `0x0018` | BUSY_NS | R | Start-to-done time of the last operation in ns, measured inside the accelerator (not inflated by the sync quantum) |
| `0x0020` | X_ADDR | W/R | DMA mode: physical address of x (28 llama.cpp `block_q8_0`, 952 bytes) |
| `0x0024` | Y_ADDR | W/R | DMA mode: physical address of y |
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

## Where everything runs

Nothing in this lab uses the real Ultra96/FPGA. Two ordinary programs run on one host machine (an LRC server or a Docker container):
**QEMU** emulates the Zynq ARM and boots Linux, and **`zynqmp_demo`** (SystemC) models the FPGA fabric including our accelerator.
They are connected by a socket file; every ARM load/store to `0xa0400000...` is forwarded to the SystemC `dot_product` module.

## Quick start on LRC (copy and paste)

These are the commands that were run and verified on `peach.ece.utexas.edu` (QEMU and SystemC must be on the SAME machine).
`<eid>` is your LRC user name. Your login loads the Synopsys VCS module (`VCS_HOME`); the Makefile ignores it, nothing to do.

**1. Copy the code to scratch and get the Linux image** (run on your own computer; the Xilinx library is a submodule, so run
`git submodule update --init` first, or clone with `--recurse-submodules`):
```
ssh peach 'mkdir -p /misc/scratch/<eid>/Lab3'
rsync -az --exclude='.git' --exclude='*.o' --exclude='*.d' part3/ peach:/misc/scratch/<eid>/Lab3/part3/
ssh peach 'cd /misc/scratch/<eid>/Lab3 && cp -a /home/projects/gerstl/ece382m/images .'      # 461 MB, once
```

**2. Build** (on peach):
```
cd /misc/scratch/<eid>/Lab3/part3
module load systemc/2.3.4
make -j8 zynqmp_demo                                   # SystemC platform with the accelerator
module load xilinx/2022
source /usr/local/packages/Xilinx_2022.2/Vivado/2022.2/settings64.sh
make -C test dot_golden_test                           # ARM binary (static aarch64); 'make STATIC=' if the static link fails
```

**3. Run** (three terminals on peach, or three tmux windows; start them in this order):
```
# terminal 1: QEMU (waits for SystemC)
cd /misc/scratch/<eid>/Lab3 && /home/projects/gerstl/ece382m/qemu-boot -c 1000000

# terminal 2: SystemC (after ./tmp/qemu-rport-_amba@0_cosim@0 exists). Quantum must match QEMU's.
cd /misc/scratch/<eid>/Lab3 && part3/zynqmp_demo unix:/misc/scratch/<eid>/Lab3/tmp/qemu-rport-_amba@0_cosim@0 1000000

# terminal 3: file server for getting the test binary into the guest
mkdir -p /misc/scratch/<eid>/Lab3/share && cp /misc/scratch/<eid>/Lab3/part3/test/dot_golden_test /misc/scratch/<eid>/Lab3/share/
cd /misc/scratch/<eid>/Lab3/share && python3 -m http.server 8765 --bind 127.0.0.1
```

**4. In the QEMU terminal** (Linux takes a few minutes to boot). Login `petalinux`; the first time it forces you to set a password.
`10.0.2.2` is how the guest reaches the host machine (QEMU user networking), so no scp or password is needed:
```
wget -O dot_golden_test http://10.0.2.2:8765/dot_golden_test && chmod +x dot_golden_test
sudo ./dot_golden_test
```
Expected last line: `ALL PASSED: 18/18 cases passed`. Real output is in `test/qemu_result.txt`.

**5. Shut down:** `sudo poweroff` in the guest, then Ctrl-A then X in the QEMU terminal (Ctrl-C does not work), then Ctrl-C the SystemC and http terminals.
Restart both sides for every run.

The `sim-time` column in that run (4,000,040 ns, ~89,000 ns, busy-polls = 0) is dominated by the 1,000,000 ns sync quantum, not by the
modeled 4570 ns latency: the ARM and SystemC only synchronize once per quantum. Use `qemu-boot -c 1000` and `1000` as the last argument of
`zynqmp_demo` (same value on both) for timing measurements. The host testbench below shows the un-quantized figure.

## Notes on the build

- The Makefile points at the LRC SystemC (`/usr/local/packages/systemc-2.3.4/`) and Vitis HLS headers
  (`VITIS_HLS_INCLUDE`, default `/usr/local/packages/Xilinx_2022.2/Vitis_HLS/2022.2/include`). Override on the command line if yours differ.
- The two lines that need a newer compiler (`mcdma.o`, `mrmac.o`) are commented out, as the tutorial requires.
- `libsystemctlm-soc` is Linux-only: `make zynqmp_demo` does not link on macOS (`off64_t`). Build on LRC or in the course Docker image
  (`gerstla/qemu-systemc`, which already contains the Xilinx demo: copy this directory's files over it and rebuild).
- Only the `zynqmp_demo` target is supported; plain `make` also lists Xilinx demos whose sources are not included here.

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

Build the test (cross-compiles for the simulated ARM; built in step 2 of the Quick start):
```
cd test
make                           # aarch64-linux-gnu-gcc, static binary 'dot_golden_test'
make CROSS=aarch64-none-linux-gnu-   # different toolchain prefix
make STATIC=                   # if the toolchain has no static libc
```

Get `dot_golden_test` into the simulated Linux as shown in the Quick start. It is a single static binary.

Run it in the guest, as root because it opens `/dev/mem`:
```
sudo ./dot_golden_test         # add -v for a double-precision cross-check per case
```
Expected output ends with `ALL PASSED: 18/18 cases passed`. Each line also shows `busy-polls` (status reads that
returned 0) and `sim-time` (ns between the start write and STATUS = 1, from the TIME registers). Exit code is 0 only if all pass.

### Host-only check (no QEMU needed)

`test/host_sim` drives the real `dot_product.h` and the real Lab 2 kernel from a SystemC testbench, using the same 18 golden cases and
the same register protocol as the ARM test. It uses the open-source Xilinx `ap_int` headers plus two tiny stand-ins for the Vitis-only
`hls_half.h` / `x_hls_utils.h` (in `host_sim/stubs`), so it runs anywhere SystemC 2.3.x is installed:

```
cd test/host_sim
make fetch                                        # once: clones the ap_int headers
make run SYSTEMC=/usr/local/packages/systemc-2.3.4
```

Result (`test/host_result.txt`, macOS, SystemC 2.3.4): **18/18 pass, bit-exact**. The modeled time from the start write to
STATUS = 1 is **4580 ns** (4570 ns compute + one 10 ns bus access); a testbench that polls on every bus access sees 456 busy polls.
This checks the model and kernel on their own. The full QEMU/ARM path was also run on LRC: 18/18 pass (`test/qemu_result.txt`).

## Extra credit 3(f): bus-mastering (DMA) mode

In copy mode the accelerator is a pure slave: the ARM writes all 1904 bytes of operands (476 32-bit stores) into the
scratchpads. In DMA mode the accelerator also has a **master socket** (`master_socket`, bound to S_AXI_HP0_FPD in `zynqmp_demo.cc`) and an
active load/store process inside `compute_thread`. The ARM only writes X_ADDR, Y_ADDR and CTRL = 3 (start + DMA).

Sequence inside the accelerator: read x (952 B) then y (952 B) from system memory in bursts, unpack each llama.cpp
`block_q8_0` `{fp16 d; int8 qs[32]}` into the existing scratchpads (`d` into DX/DY, `qs` into QX/QY), run the same Lab 2 kernel,
wait the same 457 cycles, then STATUS = 1. Total time = load time + 4570 ns. The ARM no longer touches the scratchpads, so
llama.cpp's own arrays can be passed straight through (no splitting into QX/DX in software).

- **Addresses are physical.** The accelerator accepts DRAM (`0x0 - 0x7FFFFFFF`, 2 GB on the boards) and OCM (`0xFFFC0000 - 0xFFFFFFFF`);
  anything else sets ERROR = 1 and the result is not computed. `dot_dma_test` uses OCM because Linux lets a user program map it
  through `/dev/mem` (it refuses RAM). For DRAM buffers a kernel driver is needed to learn the physical address (Part 4).
- **Load timing model (our assumption, not from a report):** a 128-bit port (16 B per beat) at the 10 ns clock, bursts of at most 16 beats
  that do not cross a 4 KB boundary, plus the delay the bus reports for each burst. x and y = 8 bursts, about 1.2 us before the 4.57 us compute.
- **Coherence:** HP0 is non-coherent. On the real board the CPU must clean its data cache for the buffers before the start write
  (or use the coherent HPC port). QEMU does not model caches, so the simulation cannot show a missing flush.
- **Timing:** BUSY_NS is measured on the SystemC clock, so it excludes the ARM's own bus accesses. In copy mode it is the modeled 4570 ns.
  In DMA mode it includes the time of the reads through the master socket, which depends on what is behind that socket (see results).

Test: `test/dot_dma_test.c` runs every golden case in both modes, checks the result bit-exactly, checks that a bad address gives ERROR = 1,
and prints BUSY_NS plus the number of ARM accesses to the accelerator for both modes. `test/host_sim` runs the same checks against a fake OCM (37 checks).

### Results

Full QEMU + SystemC co-simulation on LRC, sync quantum 1,000,000 ns (`test/qemu_dma_result.txt`): `dot_golden_test` 18/18 and
`dot_dma_test` 19/19 (18 cases in DMA mode and copy mode, plus the bad-address check). OCM at `0xFFFC0000` maps and reads back through `/dev/mem`.

| | Copy mode | DMA mode |
|---|---|---|
| ARM accesses to the accelerator per dot product | 485 | 26 to 27 (including the TIME/RESULT/ERROR reads and the STATUS polls) |
| BUSY_NS, host testbench (fake 1 ns memory) | 4570 ns | 5778 ns |
| BUSY_NS, QEMU co-simulation | 4570 ns | about 316,000 to 406,000 ns |

The ARM-side work drops by a factor of about 18, as intended. **The co-simulation time does not improve**: in DMA mode the 8 read bursts
each go through the remote-port link into QEMU, and about 40 us of simulated time per burst is charged, far above the 10 ns/beat model
(the host testbench shows the model alone costs 1208 ns). We did not investigate the cause; it is probably the co-simulation
synchronization (quantum and remote-port round trips) and not a property of the accelerator, so the DMA-mode time from this setup should
not be read as a hardware estimate. The host-testbench figure is the one that follows our load-timing model.

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

- Copy mode is slave-only (the ARM copies all data); the bus-master DMA mode (extra credit 3(f)) removes that. DMA mode needs physical addresses
  and, on real hardware, cache maintenance (see above).
- Fixed latency (single HLS design point), polling only, one operation at a time.
- The host testbench uses stand-ins for the Vitis `half`/`fp_struct`; the Vitis build on LRC is authoritative.
