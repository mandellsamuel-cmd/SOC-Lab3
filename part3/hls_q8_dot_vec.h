#ifndef HLS_Q8_DOT_VEC_H
#define HLS_Q8_DOT_VEC_H

#include <hls_half.h>
#include <stdint.h>

#define QK8_0 32             //number of elements per block
#define QWEN_NB    28             //number of blocks per vector (Qwen2.5-0.5B hidden size 896 = 28*32)
#define QWEN_N     (QWEN_NB * QK8_0)   //number of elements

float hls_vec_dot_q8_0_q8_0(const int8_t qx[QWEN_N], const half dx[QWEN_NB],
                            const int8_t qy[QWEN_N], const half dy[QWEN_NB]);

float hls_vec_dot_q8_0_q8_0_fixed(const int8_t qx[QWEN_N], const half dx[QWEN_NB],
                                  const int8_t qy[QWEN_N], const half dy[QWEN_NB]);

#endif
