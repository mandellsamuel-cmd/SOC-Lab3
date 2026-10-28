// Every FP16 scale is an 11-bit integer times a power of two (M * 2^E), so every block term
//     sumi * dx * dy = (sumi * Mx * My) * 2^(Ex + Ey)
// is an exact integer shifted by a known amount. All 28 terms are added exactly into one wide
// integer accumulator whose LSB is 2^-48, and the only rounding happens in the single conversion
// to FP32 at the end. No floating-point operators are generated, and FP16 subnormals need no
// special handling (they are just small M with E = -24).

#include "hls_q8_dot_vec.h"
#include <ap_int.h>
#include <utils/x_hls_utils.h>

#define ACC_BITS  108
#define ACC_FRAC  48             // accumulator LSB = 2^-48 (smallest possible dx*dy is 2^-24 * 2^-24)

typedef ap_int<ACC_BITS>  acc_t;
typedef ap_uint<ACC_BITS> mag_t;

// FP16 -> signed integer significand M and exponent E with value = M * 2^E
static void split_half(half h, ap_int<12> & M, ap_int<7> & E) {
    ap_uint<16> hb = fp_struct<half>(h).data();
    ap_uint<1>  sign = hb[15];
    ap_uint<5>  exp  = hb(14, 10);
    ap_uint<10> mant = hb(9, 0);
    ap_uint<11> m;
    if (exp == 0) {                 // zero / subnormal: mant * 2^-24
        m = mant;
        E = -24;
    } else {                        // normal: (1024 + mant) * 2^(exp - 25)
        m = (ap_uint<11>)mant | (ap_uint<11>)0x400;
        E = (ap_int<7>)exp - 25;
    }
    M = sign ? (ap_int<12>)(-(ap_int<12>)m) : (ap_int<12>)m;
}

// value = acc * 2^-48, rounded to nearest-even FP32
static float acc_to_float(acc_t acc) {
    if (acc == 0) return 0.0f;
    ap_uint<1> sign = acc < 0;
    mag_t mag = sign ? (mag_t)(-acc) : (mag_t)acc;

    // position of the leading 1 (fixed bound, fully unrolled -> priority encoder)
    int p = 0;
LEAD_LOOP:
    for (int i = 0; i < ACC_BITS; ++i) {
#pragma HLS UNROLL
        if (mag[i]) p = i;
    }

    // keep 24 significant bits (implicit 1 + 23 mantissa bits)
    ap_uint<25> m;
    if (p <= 23) {
        m = mag << (23 - p);                                 // exact, nothing dropped
    } else {
        int drop = p - 23;                                   // bits below the kept 24
        m = mag >> drop;
        bool guard  = mag[drop - 1];
        mag_t below = mag & ((((mag_t)1) << (drop - 1)) - 1);
        bool sticky = below != 0;
        if (guard && (sticky || m[0])) m = m + 1;           // round to nearest, ties to even
    }
    if (m[24]) { m = m >> 1; p = p + 1; }                    // rounding carried into a new bit

    ap_uint<32> exp  = p - ACC_FRAC + 127;                   // always a normal FP32 (p in 0..104)
    ap_uint<32> bits = ((ap_uint<32>)sign << 31) | (exp << 23) | ((ap_uint<32>)m(22, 0));
    fp_struct<float> f(bits);
    return f.to_ieee();
}

float hls_vec_dot_q8_0_q8_0_fixed(const int8_t qx[QWEN_N], const half dx[QWEN_NB],
                                  const int8_t qy[QWEN_N], const half dy[QWEN_NB]) {

#pragma HLS INTERFACE mode=bram port=qx depth=896 // 8 bit quants of vector x
#pragma HLS INTERFACE mode=bram port=dx depth=28 // 1 fp 16 scale per block of x
#pragma HLS INTERFACE mode=bram port=qy depth=896 // 8 bit quants of vector y
#pragma HLS INTERFACE mode=bram port=dy depth=28 // 1 fp 16 scale per block of y

    acc_t acc = 0;

BLOCK_LOOP:
    for (int ib = 0; ib < QWEN_NB; ++ib) {
        ap_int<21> sumi = 0;

    DOT_LOOP:
        for (int j = 0; j < QK8_0; ++j) {
            sumi += qx[ib * QK8_0 + j] * qy[ib * QK8_0 + j];
        }

        ap_int<12>  mx, my;
        ap_int<7>   ex, ey;
        split_half(dx[ib], mx, ex);
        split_half(dy[ib], my, ey);

        ap_int<23>  mxy   = mx * my;                          // |mx*my| < 2^22
        ap_int<43>  term  = sumi * mxy;                       // exact
        ap_uint<6>  shift = ex + ey + ACC_FRAC;               // 0..58
        acc += (acc_t)term << shift;
    }

    return acc_to_float(acc);
}
