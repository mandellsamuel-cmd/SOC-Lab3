# Host-only functional test of dot_product.h + the Lab 2 kernel against the golden vectors.
# No QEMU, no ARM, no Vitis. Needs SystemC 2.3.x and the open-source Xilinx ap_int headers.
#
#   make fetch                      # clones Xilinx/HLS_arbitrary_Precision_Types next to this Makefile (once)
#   make run SYSTEMC=/path/to/systemc
#   (LRC: SYSTEMC=/usr/local/packages/systemc-2.3.4   macOS: see lab3/env.sh)
SYSTEMC     ?= /usr/local/packages/systemc-2.3.4
SYSTEMC_LIB ?= $(firstword $(wildcard $(SYSTEMC)/lib-linux64 $(SYSTEMC)/lib-macosxarm64 $(SYSTEMC)/lib-macosx64 $(SYSTEMC)/lib))
AP_INC      ?= HLS_arbitrary_Precision_Types/include
PART3       := ../..
# -D...SPECIAL_H__: skip ap_int's optional std::complex helpers (clash with libc++, unused here)
CXXFLAGS    += -std=c++14 -O2 -D__AP_INT_SPECIAL_H__ -D__AP_FIXED_SPECIAL_H__ -Wno-unknown-pragmas -Wno-deprecated-declarations \
               -I stubs -I $(AP_INC) -I $(PART3) -I .. -I $(SYSTEMC)/include

tb_dot_product: tb_dot_product.cpp $(PART3)/dot_product.h $(PART3)/hls_q8_dot_vec_fixed.cpp
	$(CXX) $(CXXFLAGS) -o $@ tb_dot_product.cpp $(PART3)/hls_q8_dot_vec_fixed.cpp \
	    -L $(SYSTEMC_LIB) -lsystemc -Wl,-rpath,$(SYSTEMC_LIB)

run: tb_dot_product
	./tb_dot_product

fetch:
	git clone --depth 1 https://github.com/Xilinx/HLS_arbitrary_Precision_Types.git

clean:
	$(RM) tb_dot_product
.PHONY: run fetch clean
