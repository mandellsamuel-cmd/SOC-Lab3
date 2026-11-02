#!/bin/bash
# Build llama-completion for the simulated ARM with the Lab 3 FPGA HAL (run on LRC).
#   part4/llama.cpp/build_llama_fpga.sh [<work dir>]        default work dir: /misc/scratch/$USER/Lab3
# Clones the course fork of llama.cpp (if not there yet), puts lab3-fpga-hal.patch on a branch 'lab3-fpga-hal',
# and cross-compiles a static llama-completion that takes its HAL from part4/hal.
set -e
W=${1:-/misc/scratch/$USER/Lab3}
PART4=$(cd "$(dirname "$0")/.." && pwd)
cd "$W"

if [ ! -d llama.cpp ]; then
  git clone https://github.com/ECE382M-20/llama.cpp.git llama.cpp
  git -C llama.cpp checkout -b lab3-fpga-hal a7fb71fab83b474a0892b9a05aaa3a8ddca2729b
  git -C llama.cpp am "$PART4/llama.cpp/lab3-fpga-hal.patch"
fi
cd llama.cpp

# cross toolchain from the course (Vitis aarch64 GCC + sysroot); /usr/bin/cmake because 'module load xilinx' puts an old cmake first
# static: no dependence on the guest's libraries; no OpenMP (the sysroot has no static libgomp), no OpenSSL (not needed)
/usr/bin/cmake -S . -B build-fpga \
  -DCMAKE_TOOLCHAIN_FILE=/home/projects/gerstl/ece382m/ultra96_toolchain.cmake -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_EXE_LINKER_FLAGS=-static \
  -DGGML_FPGA_HAL=ON -DGGML_FPGA_HAL_DIR="$PART4"
make -C build-fpga -j16 llama-completion

# stripped copy in share/ for the guest to download (wget http://10.0.2.2:8765/llama-completion)
mkdir -p "$W/share"
/usr/local/packages/Xilinx_2022.2/Vitis/2022.2/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-strip \
  -o "$W/share/llama-completion" build-fpga/bin/llama-completion
ls -l build-fpga/bin/llama-completion "$W/share/llama-completion"
