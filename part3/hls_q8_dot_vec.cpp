#include "hls_q8_dot_vec.h"
#include <utils/x_hls_utils.h>   // fp_struct: raw-bit access to half/float in both csim and synthesis

// FP16 -> FP32 built from integer logic instead of the floating-point conversion IP.
// The Xilinx FP IP flushes subnormal inputs to zero, which would zero out an entire block
// whenever its scale is below 2^-14 (~6.1e-5). Every FP16 subnormal is a normal FP32, so
// renormalizing here keeps the full value. Same algorithm as fp16_to_fp32 in part3/q8_dot_vec.h.
static float half_to_float(half h) {
    ap_uint<16> hb = fp_struct<half>(h).data();
    ap_uint<32> sign = hb[15];
    ap_uint<5>  exp  = hb(14, 10);
    ap_uint<10> mant = hb(9, 0);
    ap_uint<32> bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;                                   // +/- 0
        } else {
            // subnormal: value = mant * 2^-24. Shift the leading 1 up to bit 10.
            int shift = 0;
        NORM_LOOP:
            for (int i = 0; i < 10; ++i) {                       // highest set bit wins -> priority encoder
                if (mant[i]) shift = 10 - i;
            }
            ap_uint<11> m = (ap_uint<11>)mant << shift;
            ap_uint<32> e = 113 - shift;                         // (1 - shift) - 15 + 127
            bits = (sign << 31) | (e << 23) | ((ap_uint<32>)m(9, 0) << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | ((ap_uint<32>)0xff << 23) | ((ap_uint<32>)mant << 13);  // inf / nan
    } else {
        bits = (sign << 31) | ((ap_uint<32>)(exp + 112) << 23) | ((ap_uint<32>)mant << 13);
    }

    fp_struct<float> f(bits);
    return f.to_ieee();
}

float hls_vec_dot_q8_0_q8_0(const int8_t qx[N], const half dx[NB],
                       const int8_t qy[N], const half dy[NB]) {

//defining the inputs as ports that are readable by HLS
#pragma HLS INTERFACE mode=bram port=qx depth=896 // 8 bit quants of vector x
#pragma HLS INTERFACE mode=bram port=dx depth=28 // 1 fp 16 scale per block of x
#pragma HLS INTERFACE mode=bram port=qy depth=896 // 8 bit quants of vector y
#pragma HLS INTERFACE mode=bram port=dy depth=28 // 1 fp 16 scale per block of y

    float sumf = 0.0f;

//had to redesign the loops from original ggml vector dot product function to account for flat storage of loops as opposed to struct
BLOCK_LOOP:
    for (int ib = 0; ib < NB; ++ib){
        int sumi = 0;

    DOT_LOOP:
        for(int j = 0; j < QK8_0; ++j) {
            sumi += qx[ib * QK8_0 + j] * qy[ib * QK8_0 + j];
        }

        float scale = half_to_float(dx[ib]) * half_to_float(dy[ib]);
        sumf += (float)sumi * scale;
    }

    return sumf;
}
