/*
 * dot_hal.h - hardware abstraction layer for the Lab 3 dot-product accelerator.
 *
 * ggml_vec_dot_q8_0_q8_0() here has the exact name and arguments of llama.cpp's function
 * (ggml/src/ggml-cpu/quants.h), so it is a drop-in substitute: the HAL computes the same dot product
 * on the accelerator through the dotprod driver (/dev/dotprod).
 *
 * The original CPU implementation must be available as ggml_vec_dot_q8_0_q8_0_cpu() (in llama.cpp: the
 * original function renamed; in the stand-alone test: test/ref_vec_dot.c). The HAL uses it
 *   - as a fallback when /dev/dotprod cannot be opened or a call fails, and
 *   - to check every result when the environment variable DOT_HAL_CHECK=1 is set.
 * DOT_HAL_VERBOSE=1 prints a summary line at exit.
 */
#ifndef DOT_HAL_H
#define DOT_HAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_vec_dot_q8_0_q8_0(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0_cpu(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);

struct dot_hal_stats {
	unsigned long calls;        /* HAL calls */
	unsigned long hw_ops;       /* ioctls, i.e. accelerator operations (one per <= 28 blocks) */
	unsigned long fallbacks;    /* calls computed on the CPU instead */
	unsigned long hw_errors;    /* failed ioctls / ERROR != 0 (those calls fell back to the CPU) */
	unsigned long checked;      /* calls compared against the CPU code (DOT_HAL_CHECK=1) */
	unsigned long mismatches;   /* of those: outside dot_hal_tolerance */
	double max_rel_err;         /* largest |hw - cpu| / dot_hal_tolerance seen while checking */
	unsigned long exact_checked;    /* calls of <= 28 blocks also compared against the exact result */
	unsigned long exact_mismatches; /* of those: not the correctly rounded FP32 value (more than 0.5 ulp off) */
};

void dot_hal_get_stats(struct dot_hal_stats *st);

/* allowed |accelerator - CPU code|. The CPU code rounds in float after every multiply-add, in several lanes that can cancel
 * each other, so its error scales with the size of the individual products, not with the (possibly tiny) final sum. */
double dot_hal_tolerance(int n, const void *vx, const void *vy);

/* the mathematically exact dot product (long double: 113-bit mantissa on aarch64, enough to hold it exactly for real inputs) */
long double dot_hal_exact(int n, const void *vx, const void *vy);

/* 1 if hw is the correctly rounded FP32 value of exact, i.e. at most half an ulp away (what the Lab 2 design promises
 * for a single operation of up to 28 blocks) */
int dot_hal_matches_exact(float hw, long double exact);

#ifdef __cplusplus
}
#endif

#endif
