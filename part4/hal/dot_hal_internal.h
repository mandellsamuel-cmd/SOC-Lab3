/*
 * dot_hal_internal.h - Q8_0 block layout and fp16 conversion used by the HAL and its stand-alone reference.
 * Independent of ggml's headers so the HAL builds with or without llama.cpp.
 */
#ifndef DOT_HAL_INTERNAL_H
#define DOT_HAL_INTERNAL_H

#include <stdint.h>
#include <string.h>

#define DOT_QK8_0 32   /* QK8_0 in llama.cpp: elements per block */

/* Same layout as ggml's block_q8_0: 34 bytes, no padding (the scale is a raw fp16 bit pattern) */
#pragma pack(push, 1)
typedef struct {
	uint16_t d;
	int8_t   qs[DOT_QK8_0];
} dot_block_q8_0;
#pragma pack(pop)

_Static_assert(sizeof(dot_block_q8_0) == 34, "dot_block_q8_0 must match ggml's block_q8_0 (34 bytes)");

/* IEEE half -> float (exact; includes subnormals, inf and NaN) */
static inline float dot_fp16_to_fp32(uint16_t h)
{
	uint32_t sign = (uint32_t)(h >> 15) << 31;
	uint32_t exp = (h >> 10) & 0x1f;
	uint32_t man = h & 0x3ff;
	uint32_t bits;
	float f;

	if (exp == 0) {
		if (man == 0) {
			bits = sign;
		} else {                            /* subnormal: normalize */
			exp = 127 - 15 + 1;
			while (!(man & 0x400)) { man <<= 1; exp--; }
			bits = sign | (exp << 23) | ((man & 0x3ff) << 13);
		}
	} else if (exp == 0x1f) {
		bits = sign | 0x7f800000u | (man << 13);
	} else {
		bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
	}
	memcpy(&f, &bits, 4);
	return f;
}

#endif
