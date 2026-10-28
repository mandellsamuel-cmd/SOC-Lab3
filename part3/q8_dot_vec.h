// q8_dot.h - standalone pieces needed by ggml_vec_dot_q8_0_q8_0
#ifndef Q8_DOT_VEC_H
#define Q8_DOT_VEC_H

#include <stdint.h>   
#include <stddef.h>   
#include <string.h>   

#define QK8_0 32      // elements per Q8_0 block

// Names used by the original llama.cpp code
#define UNUSED(x) (void)(x)
#define GGML_CPU_FP16_TO_FP32(x) fp16_to_fp32(x) //defining our own version of the conversion instead of using original look up table

typedef uint16_t ggml_half;         

// One Q8_0 block: FP16 scale + 32 int8 values (34 bytes total)
typedef struct {
    ggml_half d;
    int8_t    qs[QK8_0];
} block_q8_0;

#ifdef __cplusplus
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes (no padding)");
#else
_Static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes (no padding)");
#endif

// Portable FP16 -> FP32 
static inline float fp16_to_fp32(ggml_half h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    int32_t  exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                                  // +/- 0
        } else {                                          // subnormal: renormalize
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            bits = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);         // inf / nan
    } else {
        bits = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// extern "C" so the Part 4 C++ testbench can link against this C implementation
#ifdef __cplusplus
extern "C" {
#endif
void ggml_vec_dot_q8_0_q8_0(int n, float * s, size_t bs,
                            const void * vx, size_t bx,
                            const void * vy, size_t by, int nrc);
#ifdef __cplusplus
}
#endif

#endif