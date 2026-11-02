/*
 * hal_test.c - stand-alone test of the HAL + driver, run on the simulated ARM (Linux under QEMU).
 * Calls ggml_vec_dot_q8_0_q8_0() exactly as llama.cpp would, with llama.cpp's block layout, and checks:
 *   1. the 18 Part 3 golden cases (n = 896): bit-exact FP32 result from the accelerator,
 *   2. random vectors of several lengths (n = 32 .. 2048, i.e. padding and chunking): up to 28 blocks the result must be the exact
 *      result correctly rounded to FP32; longer vectors (several operations) within the HAL tolerance; always within the
 *      HAL tolerance of the original CPU code (whose own float rounding is the larger error),
 *   3. 4 threads calling the HAL at the same time (the driver serializes the single accelerator),
 *   4. that every call really went to the accelerator (no CPU fallbacks, no hardware errors).
 * Exit code is 0 only if everything passes.
 *
 * Needs /dev/dotprod:  sudo insmod dotprod_drv.ko install=1  [dma=1]   (then run as any user, or with sudo)
 */
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../hal/dot_hal.h"
#include "../hal/dot_hal_internal.h"
#include "golden_data.h"      /* from part3/test (-I ../../part3/test) */

static int failures;

static void check(int ok, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void check(int ok, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	printf("[%s] ", ok ? "PASS" : "FAIL");
	vprintf(fmt, ap);
	printf("\n");
	va_end(ap);
	if (!ok) failures++;
}

/* ---- helpers ------------------------------------------------------------------------------------------------ */

static uint32_t f2b(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }

/* small xorshift PRNG: every thread has its own state */
static uint32_t rnd(uint32_t *st)
{
	uint32_t v = *st;
	v ^= v << 13; v ^= v >> 17; v ^= v << 5;
	return *st = v;
}

/* random operand: full int8 range, scales 2^-8 .. 2^4 with random mantissa (keeps results finite) */
static void make_random(uint32_t *st, int nb, dot_block_q8_0 *v)
{
	for (int b = 0; b < nb; b++) {
		unsigned e = 15 - 8 + rnd(st) % 13;
		v[b].d = (uint16_t)(((rnd(st) & 1) << 15) | (e << 10) | (rnd(st) & 0x3ff));
		for (int j = 0; j < DOT_QK8_0; j++)
			v[b].qs[j] = (int8_t)(rnd(st) & 0xff);
	}
}

/* run one HAL call and compare with the exact double reference and the CPU code; returns 1 if ok */
static int run_random(uint32_t *st, int n, double *rel_used)
{
	int nb = n / DOT_QK8_0;
	dot_block_q8_0 *x = malloc(nb * sizeof *x), *y = malloc(nb * sizeof *y);
	float hw = 0, cpu = 0;
	int ok;

	make_random(st, nb, x);
	make_random(st, nb, y);
	ggml_vec_dot_q8_0_q8_0(n, &hw, 0, x, 0, y, 0, 1);
	ggml_vec_dot_q8_0_q8_0_cpu(n, &cpu, 0, x, 0, y, 0, 1);

	/* one accelerator operation (<= 28 blocks): the result must be the exact result correctly rounded to FP32;
	 * longer vectors add chunk results in float, so only the cancellation-aware bound applies */
	long double exact = dot_hal_exact(n, x, y);
	double tol = dot_hal_tolerance(n, x, y);
	double e_cpu = fabs((double)hw - (double)cpu), e_ref = fabs((double)((long double)hw - exact));
	int exact_ok = nb <= 28 ? dot_hal_matches_exact(hw, exact) : e_ref <= tol;
	ok = exact_ok && e_cpu <= tol;
	*rel_used = tol > 0 ? e_cpu / tol : 0;
	free(x); free(y);
	return ok;
}

/* ---- 1. golden cases ---------------------------------------------------------------------------------------- */

static void test_golden(void)
{
	printf("-- golden cases (n = %d, bit-exact)\n", GOLDEN_N);
	for (int t = 0; t < GOLDEN_NUM_CASES; t++) {
		const golden_case_t *c = &golden_cases[t];
		dot_block_q8_0 x[GOLDEN_NB], y[GOLDEN_NB];
		float s = 0;

		for (int b = 0; b < GOLDEN_NB; b++) {
			x[b].d = c->dx[b]; memcpy(x[b].qs, c->qx + b * 32, 32);
			y[b].d = c->dy[b]; memcpy(y[b].qs, c->qy + b * 32, 32);
		}
		ggml_vec_dot_q8_0_q8_0(GOLDEN_N, &s, 0, x, 0, y, 0, 1);
		check(f2b(s) == c->expect, "%-26s got 0x%08x exp 0x%08x", c->name, f2b(s), c->expect);
	}
}

/* ---- 2. random lengths -------------------------------------------------------------------------------------- */

static void test_random(void)
{
	static const int lens[] = { 32, 64, 256, 896, 1024, 2048 };   /* 1, 2, 8, 28, 32, 64 blocks */
	uint32_t st = 0x1234567u;

	printf("-- random vectors (HW vs exact result and vs CPU code)\n");
	for (unsigned i = 0; i < sizeof lens / sizeof lens[0]; i++) {
		int bad = 0;
		double worst = 0;

		for (int k = 0; k < 5; k++) {
			double used;
			if (!run_random(&st, lens[i], &used)) bad++;
			if (used > worst) worst = used;
		}
		check(!bad, "n = %-4d (%2d blocks, %d HW op%s)  5 vectors, worst difference to the CPU code = %.3f of the allowed bound",
		      lens[i], lens[i] / 32, (lens[i] / 32 + 27) / 28, (lens[i] / 32 + 27) / 28 > 1 ? "s" : "", worst);
	}
}

/* ---- 3. threads --------------------------------------------------------------------------------------------- */

#define NTHREADS 4
#define CALLS_PER_THREAD 40

static void *worker(void *arg)
{
	uint32_t st = 0x9e3779b9u * (uint32_t)(uintptr_t)(arg + 1);
	long bad = 0;

	for (int k = 0; k < CALLS_PER_THREAD; k++) {
		double used;
		if (!run_random(&st, k % 2 ? 64 : 256, &used)) bad++;
	}
	return (void *)bad;
}

static void test_threads(void)
{
	pthread_t th[NTHREADS];
	long bad = 0;

	printf("-- %d threads x %d calls at the same time\n", NTHREADS, CALLS_PER_THREAD);
	for (long i = 0; i < NTHREADS; i++)
		pthread_create(&th[i], NULL, worker, (void *)i);
	for (int i = 0; i < NTHREADS; i++) {
		void *r;
		pthread_join(th[i], &r);
		bad += (long)r;
	}
	check(!bad, "%d concurrent calls, %ld wrong", NTHREADS * CALLS_PER_THREAD, bad);
}

/* ------------------------------------------------------------------------------------------------------------- */

int main(void)
{
	struct dot_hal_stats before, after;

	setvbuf(stdout, NULL, _IOLBF, 0);
	printf("dot-product HAL test\n");

	/* one throwaway call so the HAL opens the device (and prints a warning if it cannot) */
	{
		dot_block_q8_0 z[1];
		float s;
		memset(z, 0, sizeof z);
		ggml_vec_dot_q8_0_q8_0(32, &s, 0, z, 0, z, 0, 1);
	}
	dot_hal_get_stats(&before);

	test_golden();
	test_random();
	test_threads();

	dot_hal_get_stats(&after);
	printf("-- accelerator usage\n");
	check(after.fallbacks == before.fallbacks && after.hw_errors == 0,
	      "no CPU fallbacks, no hardware errors (calls %lu, accelerator ops %lu, fallbacks %lu, hw errors %lu)",
	      after.calls - before.calls, after.hw_ops - before.hw_ops, after.fallbacks - before.fallbacks, after.hw_errors);
	check(after.hw_ops > before.hw_ops, "accelerator was used (%lu operations)", after.hw_ops - before.hw_ops);

	printf("\n%s: %d check%s failed\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
	return failures ? 1 : 0;
}
