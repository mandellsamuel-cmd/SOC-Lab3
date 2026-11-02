/*
 * Golden test for the dot-product accelerator (Lab 3, Part 3).
 *
 * Runs on the simulated ARM (Linux under QEMU). For every case in golden_data.h it
 *   1. loads qx/dx/qy/dy into the accelerator scratchpads,
 *   2. starts it and polls STATUS,
 *   3. reads RESULT and compares the raw FP32 bits to the golden value (bit-exact),
 *   4. reports the simulated time the operation took (TIME_LO/HI registers).
 * Exit code is 0 only if every case passes.
 *
 * Needs root (opens /dev/mem):   sudo ./dot_golden_test
 *
 * All device accesses are aligned 32-bit volatile loads/stores. /dev/mem is mapped as device
 * memory, where a memcpy() may issue unaligned or wide accesses that fault on real hardware.
 */
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "golden_data.h"

#define DOT_BASE   0xa0400000UL
#define DOT_SPAN   0x10000UL

#define REG_CTRL   0x0000
#define REG_STATUS 0x0004
#define REG_RESULT 0x0008
#define REG_TIME_LO 0x0010
#define REG_TIME_HI 0x0014
#define QX_OFF     0x1000
#define DX_OFF     0x2000
#define QY_OFF     0x3000
#define DY_OFF     0x4000

#define POLL_LIMIT 50000000UL

static volatile uint8_t *dev;

static inline uint32_t rd32(unsigned off) { return *(volatile uint32_t *)(dev + off); }
static inline void wr32(unsigned off, uint32_t v) { *(volatile uint32_t *)(dev + off) = v; }

/* copy a little-endian byte buffer to the device, one aligned 32-bit word at a time */
static void put_bytes(unsigned off, const void *src, unsigned nbytes)
{
    const uint8_t *s = (const uint8_t *)src;
    for (unsigned i = 0; i < nbytes; i += 4) {
        uint32_t w;
        memcpy(&w, s + i, 4);          /* host-side memcpy only, not to device memory */
        wr32(off + i, w);
    }
}

static uint64_t read_time_ns(void)
{
    uint32_t lo = rd32(REG_TIME_LO);   /* reading LO latches HI */
    uint32_t hi = rd32(REG_TIME_HI);
    return ((uint64_t)hi << 32) | lo;
}

static double half_to_double(uint16_t h)
{
    int sign = h >> 15, exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
    double v = (exp == 0) ? ldexp((double)mant, -24) : ldexp((double)(mant | 0x400), exp - 25);
    return sign ? -v : v;
}

/* informational only: straightforward double-precision reference */
static double reference_double(const golden_case_t *c)
{
    double sum = 0.0;
    for (int ib = 0; ib < GOLDEN_NB; ib++) {
        long sumi = 0;
        for (int j = 0; j < 32; j++)
            sumi += (long)c->qx[ib * 32 + j] * (long)c->qy[ib * 32 + j];
        sum += (double)sumi * half_to_double(c->dx[ib]) * half_to_double(c->dy[ib]);
    }
    return sum;
}

static float bits_to_float(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }

int main(int argc, char **argv)
{
    int verbose = (argc > 1 && strcmp(argv[1], "-v") == 0);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "open /dev/mem: %s (run with sudo)\n", strerror(errno));
        return 2;
    }
    void *m = mmap(NULL, DOT_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, DOT_BASE);
    if (m == MAP_FAILED) {
        fprintf(stderr, "mmap 0x%lx: %s\n", DOT_BASE, strerror(errno));
        return 2;
    }
    dev = (volatile uint8_t *)m;

    printf("dot-product accelerator golden test, %d cases\n", GOLDEN_NUM_CASES);

    if (rd32(REG_STATUS) != 1) {
        fprintf(stderr, "FAIL: STATUS is %u before the first start (expected 1 = ready)\n", rd32(REG_STATUS));
        return 1;
    }

    int failures = 0;
    for (int t = 0; t < GOLDEN_NUM_CASES; t++) {
        const golden_case_t *c = &golden_cases[t];

        put_bytes(QX_OFF, c->qx, GOLDEN_N);
        put_bytes(DX_OFF, c->dx, GOLDEN_NB * 2);
        put_bytes(QY_OFF, c->qy, GOLDEN_N);
        put_bytes(DY_OFF, c->dy, GOLDEN_NB * 2);

        uint64_t t0 = read_time_ns();
        wr32(REG_CTRL, 1);

        unsigned long polls = 0, busy_polls = 0;
        uint32_t st;
        while ((st = rd32(REG_STATUS)) == 0 && polls < POLL_LIMIT) { polls++; busy_polls++; }
        uint64_t t1 = read_time_ns();

        if (st != 1) {
            printf("[FAIL] %-26s timeout waiting for STATUS=1\n", c->name);
            failures++;
            continue;
        }

        uint32_t got = rd32(REG_RESULT);
        double ref = reference_double(c);
        double gotf = (double)bits_to_float(got);
        double rel = (ref != 0.0) ? fabs((gotf - ref) / ref) : fabs(gotf);
        int ok = (got == c->expect);

        printf("[%s] %-26s got 0x%08x exp 0x%08x  (%.9g)  busy-polls=%lu  sim-time=%llu ns\n",
               ok ? "PASS" : "FAIL", c->name, got, c->expect, gotf, busy_polls,
               (unsigned long long)(t1 - t0));
        if (verbose || !ok)
            printf("        double-precision reference %.9g, relative diff %.3g\n", ref, rel);
        if (!ok) failures++;
    }

    printf("\n%s: %d/%d cases passed\n", failures ? "FAILED" : "ALL PASSED",
           GOLDEN_NUM_CASES - failures, GOLDEN_NUM_CASES);
    munmap(m, DOT_SPAN);
    close(fd);
    return failures ? 1 : 0;
}
