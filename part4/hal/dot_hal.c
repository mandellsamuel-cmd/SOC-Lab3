/*
 * dot_hal.c - HAL: llama.cpp's ggml_vec_dot_q8_0_q8_0() computed on the dot-product accelerator.
 *
 * Call flow:   ggml_vec_dot_q8_0_q8_0()  ->  ioctl(/dev/dotprod, DOTPROD_IOC_DOT)  ->  dotprod_drv.ko  ->  accelerator
 *
 * The accelerator always computes DOTPROD_HW_BLOCKS (28) blocks; the driver pads shorter operations with
 * zero scales. Vectors longer than 28 blocks are split into chunks and the chunk results are added.
 *
 * Environment variables:
 *   DOT_HAL_CHECK=1    also run the original CPU code for every call and compare (tolerance: dot_hal_tolerance); for
 *                      calls of <= 28 blocks also check the result is the exact result correctly rounded to FP32
 *   DOT_HAL_CHECK=2    same, and print the operands of the first mismatching calls (hex) for off-line analysis
 *   DOT_HAL_VERBOSE=1  print a statistics line at exit (also printed when a check found a mismatch)
 *   DOT_HAL_DISABLE=1  do not use the accelerator at all (CPU code only), for comparison runs
 *   DOT_HAL_DEVICE=... device node to open (default /dev/dotprod)
 */
#ifndef _GNU_SOURCE        /* ggml already defines it for every file */
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "dot_hal.h"
#include "dot_hal_internal.h"
#include "../driver/dotprod_ioctl.h"

static pthread_once_t hal_once = PTHREAD_ONCE_INIT;
static int hal_fd = -1;
static int hal_check;
static int hal_verbose;

static struct dot_hal_stats stats;
static pthread_mutex_t stats_lock = PTHREAD_MUTEX_INITIALIZER;

#define COUNT(field) __atomic_fetch_add(&stats.field, 1, __ATOMIC_RELAXED)

void dot_hal_get_stats(struct dot_hal_stats *st)
{
	pthread_mutex_lock(&stats_lock);
	*st = stats;
	pthread_mutex_unlock(&stats_lock);
}

static void hal_report(void)
{
	struct dot_hal_stats st;

	dot_hal_get_stats(&st);
	if (!hal_verbose && !st.mismatches && !st.exact_mismatches)
		return;
	fprintf(stderr, "dot_hal: %lu calls, %lu accelerator ops, %lu CPU fallbacks, %lu hw errors; "
			"vs CPU code: %lu checked, %lu mismatches, worst error %.3g of the allowed; "
			"vs exact result: %lu checked, %lu not correctly rounded\n",
		st.calls, st.hw_ops, st.fallbacks, st.hw_errors, st.checked, st.mismatches, st.max_rel_err,
		st.exact_checked, st.exact_mismatches);
}

static void hal_init(void)
{
	const char *e, *dev = getenv("DOT_HAL_DEVICE");

	e = getenv("DOT_HAL_CHECK");
	hal_check = e ? atoi(e) : 0;
	e = getenv("DOT_HAL_VERBOSE");
	hal_verbose = e && *e && *e != '0';
	atexit(hal_report);
	e = getenv("DOT_HAL_DISABLE");
	if (e && *e && *e != '0')
		return;

	hal_fd = open(dev ? dev : DOTPROD_DEV, O_RDWR | O_CLOEXEC);
	if (hal_fd < 0)
		fprintf(stderr, "dot_hal: cannot open %s (%s); computing dot products on the CPU. "
				"Is dotprod_drv.ko loaded ('insmod dotprod_drv.ko install=1')?\n",
			dev ? dev : DOTPROD_DEV, strerror(errno));
}

/* Allowed |hw - cpu|. The accelerator adds all block products exactly and rounds once. The CPU code (NEON path) rounds in float after
 * every multiply-add in 4 lanes per block, and positive and negative lane values can cancel in the final sum, so its error is about
 * (blocks + 4) * 2^-23 times the sum of the sizes of the individual products |qx*qy| * |dx*dy| (not of the cancelled block totals). */
double dot_hal_tolerance(int n, const void *vx, const void *vy)
{
	const dot_block_q8_0 *x = vx, *y = vy;
	int nb = n / DOT_QK8_0;
	double sumabs = 0.0;

	for (int b = 0; b < nb; b++) {
		long sumq = 0;
		for (int j = 0; j < DOT_QK8_0; j++)
			sumq += labs((long)x[b].qs[j] * y[b].qs[j]);
		sumabs += (double)sumq * fabs((double)dot_fp16_to_fp32(x[b].d)) * fabs((double)dot_fp16_to_fp32(y[b].d));
	}
	return (nb + 4) * ldexp(1.0, -23) * sumabs;
}

long double dot_hal_exact(int n, const void *vx, const void *vy)
{
	const dot_block_q8_0 *x = vx, *y = vy;
	long double sum = 0.0L;

	for (int b = 0; b < n / DOT_QK8_0; b++) {
		long sumi = 0;
		for (int j = 0; j < DOT_QK8_0; j++)
			sumi += (long)x[b].qs[j] * y[b].qs[j];
		/* scale product: 11 bit x 11 bit significands, exact; times sumi (19 bit): exact; the sum needs at most ~110 bits */
		sum += (long double)sumi * ((long double)dot_fp16_to_fp32(x[b].d) * (long double)dot_fp16_to_fp32(y[b].d));
	}
	return sum;
}

int dot_hal_matches_exact(float hw, long double exact)
{
	float mag = fabsf(hw);
	long double half_ulp = 0.5L * ((long double)nextafterf(mag, INFINITY) - (long double)mag);   /* spacing above hw */

	return fabsl((long double)hw - exact) <= half_ulp;      /* false for NaN/inf (half_ulp is NaN) */
}

/* DOT_HAL_CHECK=2: print the operands of a mismatching call (raw block bytes in hex) so it can be reproduced off-line */
static void hal_dump(int n, float hw, float cpu, const void *vx, const void *vy)
{
	const unsigned char *p[2] = { vx, vy };
	uint32_t hb, cb;

	memcpy(&hb, &hw, 4);
	memcpy(&cb, &cpu, 4);
	fprintf(stderr, "dot_hal: DUMP n=%d hw=0x%08x cpu=0x%08x", n, hb, cb);
	for (int v = 0; v < 2; v++) {
		fprintf(stderr, " %c=", v ? 'y' : 'x');
		for (int i = 0; i < n / DOT_QK8_0 * DOTPROD_BLOCK_BYTES; i++)
			fprintf(stderr, "%02x", p[v][i]);
	}
	fprintf(stderr, "\n");
}

static int hw_dot(int n, float *s, const dot_block_q8_0 *x, const dot_block_q8_0 *y)
{
	int nb = n / DOT_QK8_0;
	float sum = 0.0f;

	for (int b = 0; b < nb; b += DOTPROD_HW_BLOCKS) {
		struct dotprod_req r;
		float part;

		memset(&r, 0, sizeof r);
		r.x = (uintptr_t)(x + b);
		r.y = (uintptr_t)(y + b);
		r.nblocks = nb - b < DOTPROD_HW_BLOCKS ? nb - b : DOTPROD_HW_BLOCKS;

		if (ioctl(hal_fd, DOTPROD_IOC_DOT, &r) < 0 || r.error) {
			COUNT(hw_errors);
			return -1;
		}
		COUNT(hw_ops);
		memcpy(&part, &r.result, sizeof part);
		sum += part;
	}
	*s = sum;
	return 0;
}

void ggml_vec_dot_q8_0_q8_0(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc)
{
	pthread_once(&hal_once, hal_init);
	COUNT(calls);

	/* the accelerator computes single dot products of whole blocks; anything else is left to the CPU */
	if (hal_fd < 0 || nrc != 1 || n <= 0 || n % DOT_QK8_0 != 0 || hw_dot(n, s, vx, vy) != 0) {
		COUNT(fallbacks);
		ggml_vec_dot_q8_0_q8_0_cpu(n, s, bs, vx, bx, vy, by, nrc);
		return;
	}

	if (hal_check) {
		float ref;
		double tol, err;
		int single = n / DOT_QK8_0 <= DOTPROD_HW_BLOCKS;      /* one accelerator operation: the result is a single correct rounding */
		int exact_bad = single && !dot_hal_matches_exact(*s, dot_hal_exact(n, vx, vy));

		ggml_vec_dot_q8_0_q8_0_cpu(n, &ref, bs, vx, bx, vy, by, nrc);
		tol = dot_hal_tolerance(n, vx, vy);
		err = fabs((double)*s - (double)ref);

		pthread_mutex_lock(&stats_lock);
		stats.checked++;
		if (single) stats.exact_checked++;
		if (exact_bad) {
			stats.exact_mismatches++;
			if (stats.exact_mismatches <= 10)
				fprintf(stderr, "dot_hal: NOT CORRECTLY ROUNDED n=%d hw=%.9g exact=%.12Lg\n", n, (double)*s, dot_hal_exact(n, vx, vy));
		}
		if (tol > 0.0 && err / tol > stats.max_rel_err)
			stats.max_rel_err = err / tol;     /* fraction of the allowed error that was used */
		if (!(err <= tol)) {                       /* also catches NaN */
			stats.mismatches++;
			if (stats.mismatches <= 10) {
				fprintf(stderr, "dot_hal: MISMATCH n=%d hw=%.9g cpu=%.9g (allowed %.3g)\n", n, (double)*s, (double)ref, tol);
				if (hal_check >= 2)
					hal_dump(n, *s, ref, vx, vy);
			}
		}
		pthread_mutex_unlock(&stats_lock);
	}
}
