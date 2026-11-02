# Lab 3, Part 4: HAL, kernel-module driver and llama.cpp on the accelerator

The accelerator from Part 3 (`part3/dot_product.h`, mapped at `0xa0400000` in the QEMU/SystemC platform) is made usable from
ordinary Linux programs:

```
llama.cpp matrix multiply
  -> ggml_vec_dot_q8_0_q8_0(n, &s, ..., x, ..., y, ...)         hal/dot_hal.c     (HAL: same name and arguments as llama.cpp's function)
       -> ioctl(/dev/dotprod, DOTPROD_IOC_DOT, &req)              one system call per <= 28 blocks
            -> dotprod_drv.ko                                     driver/dotprod_drv.c
                 -> registers/scratchpads at 0xa0400000           QEMU -> SystemC -> accelerator (Part 3)
```

Part 4(a) is the driver, the HAL and a stand-alone test (`driver/`, `hal/`, `test/hal_test.c`). Part 4(b) builds llama.cpp with this HAL and runs
Tiny-Stories on the simulated ARM + accelerator (`llama.cpp/`, see "Part 4(b)" below).

## Files

```
part4/
  driver/dotprod_drv.c     kernel module (adapted from the tutorial's application.irq/fpga_drv.c)
  driver/dotprod_ioctl.h   request format shared by driver and HAL
  driver/Makefile          builds dotprod_drv.ko against the course kernel tree on LRC
  hal/dot_hal.h, dot_hal.c the HAL: ggml_vec_dot_q8_0_q8_0() on the accelerator, with CPU fallback and optional checking
  hal/dot_hal_internal.h   Q8_0 block layout and fp16 conversion (no dependency on ggml headers)
  test/hal_test.c          stand-alone test run on the simulated ARM
  test/ref_vec_dot.c       stand-in for llama.cpp's original CPU function (only for the stand-alone test)
  test/qemu_result.txt     hal_test output from the QEMU + SystemC co-simulation (copy and DMA mode)
  test/llama_result.txt    llama.cpp output from the co-simulation (CPU reference and accelerator run)
  Makefile                 builds the module and the test
  llama.cpp/lab3-fpga-hal.patch   the llama.cpp change (4 files, 25 lines) on the course fork ECE382M-20/llama.cpp, commit a7fb71f
  llama.cpp/build_llama_fpga.sh   clones the fork, applies the patch, cross-compiles llama-completion
```

## How it works

**Driver** (`/dev/dotprod`, a misc device registered by a platform driver, as in the tutorial's `fpga_drv.c`):
- The tutorial driver needs an `ece382m,fpga` node in the device tree. Our boot image has none for the accelerator, so the driver is built with
  `-DNO_DTS` (the default in `driver/Makefile`): the address is compiled in and `insmod dotprod_drv.ko install=1` registers the device.
  Device-tree matching (`ece382m,dotprod`) is still in the code.
- `DOTPROD_IOC_DOT` performs a whole dot product: copy the operands in from user space, load them into the accelerator, start it,
  poll STATUS, read RESULT/BUSY_NS/ERROR. The tutorial's one-ioctl-per-register interface is kept as `DOTPROD_IOC_RDREG/WRREG` for debugging.
- The accelerator has no interrupt line, so completion is detected by polling STATUS (10 s timeout); an interrupt is optional in `probe`.
- A mutex serializes callers: llama.cpp runs several threads but there is one accelerator.
- Two data paths, chosen at load time:
  - **copy mode** (default): the driver unpacks the llama.cpp blocks into the accelerator's QX/DX/QY/DY scratchpads with `writel` (Part 3 protocol, CTRL = 1).
  - **DMA mode** (`dma=1`, uses the Part 3 extra-credit bus-master mode): the driver copies the blocks, unchanged, into two `dma_alloc_coherent` buffers,
    writes their physical addresses to X_ADDR/Y_ADDR and CTRL = 3; the accelerator fetches and unpacks them itself.
- `/proc/dotprod` shows the mode, number of dot products, average/max BUSY_NS, timeouts and hardware errors.
- The device node is created with mode 0666, so llama.cpp does not need root.

**Vector length.** The accelerator always computes 28 blocks (896 elements). The HAL splits a vector of `n` elements into chunks of up to 28 blocks
and adds the chunk results in float. The driver pads the last, shorter chunk with blocks whose scale is 0; such a block contributes exactly 0
(the driver only rewrites the scales that changed since the previous call). So any `n` that is a multiple of 32 works: the Tiny-Stories model
(`n_embd` = 64, feed-forward width 256) needs 2 and 8 blocks, Qwen2.5-0.5B needs 28 or more.

**HAL** (`hal/dot_hal.c`): `ggml_vec_dot_q8_0_q8_0()` has llama.cpp's exact name and signature. It falls back to the original CPU code
(`ggml_vec_dot_q8_0_q8_0_cpu`, which is the original function renamed) when `/dev/dotprod` cannot be opened, when `nrc != 1`, or when an accelerator call fails.
Environment variables:

| Variable | Effect |
|---|---|
| `DOT_HAL_CHECK=1` | also check every call: against the CPU code, and (calls of <= 28 blocks) against the exact result; mismatches are counted and reported |
| `DOT_HAL_CHECK=2` | same, and print the operands of the first mismatching calls in hex (for off-line analysis) |
| `DOT_HAL_VERBOSE=1` | print call/fallback/error/mismatch counts at exit |
| `DOT_HAL_DISABLE=1` | do not use the accelerator (CPU only), for comparison runs |
| `DOT_HAL_DEVICE=path` | use another device node |

The check cannot demand bit-exact equality with the CPU code. The accelerator adds all block products exactly and rounds once to FP32 (Lab 2 design),
while llama.cpp's ARM code rounds in float after every multiply-add, in 4 lanes per block whose positive and negative values can cancel. Its error therefore follows
the size of the individual products, not of the (possibly tiny) final sum. Two checks are made:
- accelerator vs CPU code: allowed difference `(blocks + 4) * 2^-23 * sum(|qx*qy| * |dx*dy|)` (`dot_hal_tolerance`);
- accelerator vs exact result (`dot_hal_exact`, computed in `long double`): for one accelerator operation (<= 28 blocks) the result must be the correctly rounded
  FP32 value, i.e. at most half an ulp from the exact one (`dot_hal_matches_exact`). This is the stricter and more meaningful test.

Why both: a first version bounded the CPU difference by the size of the *cancelled block totals*. That was too small, and 62 of 561,514 calls were flagged in the llama.cpp run
(see below). Analysis of the 10 dumped operand sets showed the accelerator equal to the exact result in all 10, and llama.cpp's own ARM code up to 60 ulp off.

On the target (cross-compiled, no `dotprod`/`i8mm` instructions) llama.cpp's faster Q8_0 paths (weight repacking and llamafile) are compiled out, so every Q8_0
row dot product goes through `ggml_vec_dot_q8_0_q8_0` with `nrc == 1`. That is the call the HAL replaces.

## Build (on LRC, copy and paste)

Same layout as Part 3: `/misc/scratch/<eid>/Lab3/` contains `images/`, `part3/` and `part4/`.

```
# from the repo root on your machine
rsync -az --exclude='.git' part4/ peach:/misc/scratch/<eid>/Lab3/part4/

# on peach
cd /misc/scratch/<eid>/Lab3/part4
module load xilinx/2022
source /usr/local/packages/Xilinx_2022.2/Vivado/2022.2/settings64.sh
make                     # driver/dotprod_drv.ko and test/hal_test   ('make STATIC=' if the static link fails)
```

The module is built against `/home/projects/gerstl/ece382m/linux-xlnx` (kernel 5.15.36-xilinx-v2022.2, the kernel in the boot image), as in the tutorial.
`hal_test` includes `golden_data.h` from `part3/test`.

## Run

Start the platform exactly as in `part3/README.md` (QEMU with `qemu-boot -c 1000000`, `part3/zynqmp_demo` with the same quantum, a file server), then in the guest:

```
wget -O dotprod_drv.ko http://10.0.2.2:8765/dotprod_drv.ko
wget -O hal_test       http://10.0.2.2:8765/hal_test && chmod +x hal_test
sudo insmod ./dotprod_drv.ko install=1          # add dma=1 for the bus-master data path
cat /proc/dotprod ; ls -l /dev/dotprod
./hal_test                                      # DOT_HAL_CHECK=1 DOT_HAL_VERBOSE=1 ./hal_test  to see the HAL's own cross-check
sudo rmmod dotprod_drv                          # before loading it again with the other mode
```

(copy the two files into `/misc/scratch/<eid>/Lab3/share/` first if your file server serves that directory).

## Test

`test/hal_test.c` calls the HAL the way llama.cpp does, with llama.cpp's block layout:
1. the 18 Part 3 golden cases (n = 896): bit-exact FP32 result;
2. random vectors with n = 32, 64, 256, 896, 1024, 2048 (1 to 64 blocks, so padding and chunking), up to 28 blocks the result must be the exact result correctly rounded to FP32, always within the HAL tolerance of the CPU code;
3. 4 threads x 40 calls at the same time;
4. no CPU fallbacks and no hardware errors, and the accelerator was used.

### Results

Full QEMU + SystemC co-simulation on LRC, sync quantum 1,000,000 ns (`test/qemu_result.txt`). `hal_test` was run as the unprivileged `petalinux` user
(`/dev/dotprod` is mode 0666) after loading the module in each mode:

| | copy mode | DMA mode (`dma=1`) |
|---|---|---|
| golden cases (n = 896, bit-exact) | 18/18 | 18/18 |
| random vectors, n = 32 to 2048 | pass (up to 28 blocks: exact result correctly rounded) | same |
| 4 threads x 40 calls | 0 wrong | 0 wrong |
| CPU fallbacks / hardware errors | 0 / 0 | 0 / 0 |
| `DOT_HAL_CHECK=1` (HAL checks every call) | 209 calls: 0 mismatches vs the CPU code (worst 0.066 of the bound); 199 single-operation calls: 0 not correctly rounded | same |
| BUSY_NS from `/proc/dotprod` (average / max) | 4570 / 4570 ns | 362,745 / 8,004,780 ns |

Each `hal_test` run makes 223 accelerator operations (the second run in `/proc/dotprod` doubles it to 448). The DMA-mode BUSY_NS is much larger than in
the host testbench because each fetch goes through the QEMU link (see "Timing" in `part3/README.md`, section on 3(f)); it is an artifact of the co-simulation.

## Notes and limitations

- The device is registered with `install=1` and no device tree; a device-tree entry would be needed to load it without the parameter.
- DMA mode needs the buffers below 2 GB (the accelerator accepts DRAM `0x0 - 0x7FFFFFFF` and OCM); the driver checks this.
  The platform has no IOMMU in this model and QEMU does not model caches; on real hardware the CPU would have to clean its caches
  (`dma_alloc_coherent` memory avoids that).
- Every dot product costs one system call plus (copy mode) 8 quant words per block per vector and one word per two scales, plus the start write and the STATUS/RESULT/BUSY_NS/ERROR reads.
  Timing in the co-simulation is dominated by the sync quantum (see `part3/README.md`), so the simulated time of a dot product is not a hardware estimate.
- One token of a real model needs a very large number of dot products (the output layer alone has one per vocabulary entry), and each one is a round trip through the co-simulation,
  so Part 4(b) runs are limited to a few tokens (`-n 10`).

## Part 4(b): llama.cpp with the HAL

**The change in llama.cpp** (`llama.cpp/lab3-fpga-hal.patch`, branch `lab3-fpga-hal` of the course fork https://github.com/ECE382M-20/llama.cpp at commit a7fb71f; default OFF, so a normal build is unchanged):

| File | Change |
|---|---|
| `ggml/CMakeLists.txt` | option `GGML_FPGA_HAL` and path `GGML_FPGA_HAL_DIR` (this `part4` directory) |
| `ggml/src/ggml-cpu/CMakeLists.txt` | in the ARM branch, adds `${GGML_FPGA_HAL_DIR}/hal/dot_hal.c` and defines `GGML_FPGA_HAL` |
| `ggml/src/ggml-cpu/arch/arm/quants.c` | renames the original `ggml_vec_dot_q8_0_q8_0` to `ggml_vec_dot_q8_0_q8_0_cpu` (two preprocessor lines) |
| `ggml/src/ggml-cpu/quants.h` | declares `ggml_vec_dot_q8_0_q8_0_cpu` |

The HAL then defines `ggml_vec_dot_q8_0_q8_0` itself, with the same signature, so nothing else in llama.cpp changes (the function is reached through
`type_traits_cpu[GGML_TYPE_Q8_0].vec_dot`). A cross build has no `dotprod`/`i8mm` flags, so llama.cpp's other Q8_0 paths are compiled out and every Q8_0 row goes through this call.

**Build** (on LRC; same layout as above, with the llama.cpp checkout next to `part3/` and `part4/`):
```
cd /misc/scratch/<eid>/Lab3
part4/llama.cpp/build_llama_fpga.sh            # clone, apply the patch, cross-compile a static aarch64 llama-completion, copy it to share/
```
It uses the course toolchain file (`/home/projects/gerstl/ece382m/ultra96_toolchain.cmake`, Vitis aarch64 GCC 11.2) and `-DGGML_NATIVE=OFF -DGGML_OPENMP=OFF
-DLLAMA_OPENSSL=OFF -DCMAKE_EXE_LINKER_FLAGS=-static -DGGML_FPGA_HAL=ON -DGGML_FPGA_HAL_DIR=<part4>`. OpenMP is off because the toolchain has no static `libgomp`.

**Model.** `tinystories-gpt-0.1-3m.Q8_0.gguf` (9.6 MB, from https://huggingface.co/afrideva/Tinystories-gpt-0.1-3m-GGUF): GPT-2, 8 layers, `n_embd` 64, feed-forward 256.
All its large tensors are Q8_0, so its dot products have 64 elements (2 blocks) or 256 elements (8 blocks). Download it into `Lab3/share/` on LRC (`curl -L -o ... <url>/resolve/main/tinystories-gpt-0.1-3m.Q8_0.gguf`).

**Run** (platform started as in `part3/README.md`; the files are served from `Lab3/share/`):
```
wget -O llama-completion http://10.0.2.2:8765/llama-completion && chmod +x llama-completion
wget -O tinystories-gpt-0.1-3m.Q8_0.gguf http://10.0.2.2:8765/tinystories-gpt-0.1-3m.Q8_0.gguf
wget -O dotprod_drv.ko http://10.0.2.2:8765/dotprod_drv.ko

ARGS='-m tinystories-gpt-0.1-3m.Q8_0.gguf -p "Once upon a time" -n 10 --temp 0 -no-cnv --no-warmup -c 128 -t 4'
DOT_HAL_DISABLE=1 ./llama-completion $ARGS                      # reference: llama.cpp's own ARM code
sudo insmod ./dotprod_drv.ko install=1
DOT_HAL_CHECK=1 DOT_HAL_VERBOSE=1 ./llama-completion $ARGS      # accelerator, every call checked
cat /proc/dotprod
```
(`--temp 0` makes generation deterministic, so the two texts can be compared. Use `-n 1` first for a quick check.)

### Results (`test/llama_result.txt`)

Full QEMU + SystemC co-simulation on LRC, sync quantum 1,000,000 ns, prompt "Once upon a time", `-n 10`:

| | CPU only (`DOT_HAL_DISABLE=1`) | Accelerator (copy mode, `DOT_HAL_CHECK`) |
|---|---|---|
| Generated text | `Once upon a time, there was a little girl named Lily. She` | **identical** |
| HAL calls | 561,514 (all on the CPU) | 561,514, all on the accelerator; 0 CPU fallbacks, 0 hardware errors |
| Driver `dot products` (`/proc/dotprod`) | - | 561,738 = 561,514 + the 224 operations of the `hal_test` run before it |
| vs the CPU code | - | 0 of 561,514 outside the allowed bound (worst difference 0.17 of the bound) |
| vs the exact result | - | 0 of 561,514 not correctly rounded |
| Accelerator time per operation (`BUSY_NS`) | - | 4570 ns average and maximum, 0 timeouts |
| llama.cpp generation time (real time) | 0.14 s eval for 9 tokens (62 tokens/s) | 467 s eval for 9 tokens (0.02 tokens/s) |

The accelerator run is about 3000 times slower in real time because every dot product is several round trips through the QEMU <-> SystemC link; the simulated
platform, not the accelerator, sets that speed.

Where 561,514 comes from: 13 tokens are evaluated (4 prompt tokens + 9 more), each needing 8 layers x (192 + 64 + 256 + 64 weight rows) = 4,608 dot products
(attention QKV, attention output, feed-forward up and down), i.e. 59,904; 10 sets of logits (the last prompt token and the 9 generated ones) need 50,257 dot products each
(output layer) = 502,570; and 960 fewer, because in the 4-token prompt batch llama.cpp computes the last layer's feed-forward only for the one token whose logits are needed
(3 x (256 + 64)). 59,904 + 502,570 - 960 = 561,514, exactly the number the HAL counted.

The same run first flagged 62 mismatches against the CPU code with a too-tight bound (see "HAL" above). The analysis of the 10 dumped cases (`DOT_HAL_CHECK=2`) is the reason for the
two checks now in `dot_hal.c`; the generated text was identical in that run too.
