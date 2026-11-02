// Minimal stand-in for Vitis HLS <utils/x_hls_utils.h>: fp_struct<half>/fp_struct<float> as used by
// hls_q8_dot_vec_fixed.cpp. Host testbench only.
#ifndef STUB_X_HLS_UTILS_H
#define STUB_X_HLS_UTILS_H
#include <ap_int.h>
#include <string.h>
#include <hls_half.h>

template <typename T> struct fp_struct;

template <> struct fp_struct<half> {
    ap_uint<16> bits;
    explicit fp_struct(half h) : bits(h.bits) {}
    ap_uint<16> data() const { return bits; }
};

template <> struct fp_struct<float> {
    ap_uint<32> bits;
    explicit fp_struct(ap_uint<32> b) : bits(b) {}
    float to_ieee() const { uint32_t u = (uint32_t)bits; float f; memcpy(&f, &u, 4); return f; }
};
#endif
