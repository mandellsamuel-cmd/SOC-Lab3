/*
 * ref_vec_dot.c - stand-alone stand-in for llama.cpp's original dot product, for the HAL test.
 * This is the generic C version from ggml/src/ggml-cpu/quants.c (ggml_vec_dot_q8_0_q8_0_generic), same arithmetic:
 * integer sum per block, scaled by the product of the two fp16 scales in float, accumulated in float.
 * In llama.cpp itself the real function is used (renamed ggml_vec_dot_q8_0_q8_0_cpu).
 */
#include <assert.h>

#include "../hal/dot_hal.h"
#include "../hal/dot_hal_internal.h"

void ggml_vec_dot_q8_0_q8_0_cpu(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc)
{
	const int nb = n / DOT_QK8_0;
	const dot_block_q8_0 *x = vx, *y = vy;
	float sumf = 0;

	(void)bs; (void)bx; (void)by;
	assert(n % DOT_QK8_0 == 0);
	assert(nrc == 1);

	for (int ib = 0; ib < nb; ++ib) {
		int sumi = 0;

		for (int j = 0; j < DOT_QK8_0; j++)
			sumi += x[ib].qs[j] * y[ib].qs[j];

		sumf += sumi * (dot_fp16_to_fp32(x[ib].d) * dot_fp16_to_fp32(y[ib].d));
	}
	*s = sumf;
}
