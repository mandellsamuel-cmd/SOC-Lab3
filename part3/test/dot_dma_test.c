/*
 * Bus-mastering (DMA mode) test for the dot-product accelerator (Lab 3, Part 3 extra credit).
 *
 * Runs on the simulated ARM (Linux under QEMU). For every case in golden_data.h it
 *   1. writes x and y as llama.cpp block_q8_0 arrays {fp16 d; int8 qs[32]} into OCM (0xFFFC0000),
 *   2. gives the accelerator their physical addresses (X_ADDR / Y_ADDR) and starts it with CTRL = 3,
 *      so the accelerator fetches the data itself through S_AXI_HP0_FPD,
 *   3. compares RESULT bit-exactly to the golden value,
 *   4. runs the same case in copy mode (ARM writes the scratchpads, CTRL = 1) for comparison.
 * It prints the accelerator's own start->done time (BUSY_NS) and how many accesses the ARM made to the
 * accelerator in each mode. Exit code is 0 only if every check passes.
 *
 * OCM is used because a user program can map it through /dev/mem (Linux does not own it as RAM);
 * the accelerator accepts DRAM addresses too (0x0-0x7FFFFFFF), e.g. from a kernel driver's DMA buffer.
 * Only the first 4 KB of OCM are used (ARM trusted firmware normally sits near the top of OCM).
 *
 * Needs root (opens /dev/mem):   sudo ./dot_dma_test
 */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "golden_data.h"

#define DOT_BASE   0xa0400000UL
#define DOT_SPAN   0x10000UL
#define OCM_BASE   0xfffc0000UL
#define OCM_SPAN   0x1000UL
#define X_PHYS     (OCM_BASE + 0x000)
#define Y_PHYS     (OCM_BASE + 0x400)

#define REG_CTRL    0x0000
#define REG_STATUS  0x0004
#define REG_RESULT  0x0008
#define REG_ERROR   0x000C
#define REG_TIME_LO 0x0010
#define REG_TIME_HI 0x0014
#define REG_BUSY_NS 0x0018
#define REG_X_ADDR  0x0020
#define REG_Y_ADDR  0x0024
#define QX_OFF      0x1000
#define DX_OFF      0x2000
#define QY_OFF      0x3000
#define DY_OFF      0x4000

#define CTRL_START  1u
#define CTRL_DMA    2u
#define BLOCK_BYTES 34                       /* sizeof(block_q8_0) */
#define VEC_BYTES   (GOLDEN_NB * BLOCK_BYTES)

#define POLL_LIMIT 50000000UL

static volatile uint8_t *dev, *ocm;
static unsigned long dev_accesses;           /* ARM loads/stores that went to the accelerator */

static inline uint32_t rd32(unsigned off) { dev_accesses++; return *(volatile uint32_t *)(dev + off); }
static inline void wr32(unsigned off, uint32_t v) { dev_accesses++; *(volatile uint32_t *)(dev + off) = v; }

/* aligned 32-bit stores only: both mappings are device memory */
static void put_words(volatile uint8_t *base, unsigned off, const void *src, unsigned nbytes, int to_dev)
{
    const uint8_t *s = (const uint8_t *)src;
    for (unsigned i = 0; i < nbytes; i += 4) {
        uint32_t w;
        memcpy(&w, s + i, 4);
        if (to_dev) wr32(off + i, w);
        else *(volatile uint32_t *)(base + off + i) = w;
    }
}

static void pack_blocks(uint8_t *dst, const int8_t *q, const uint16_t *d)
{
    for (int b = 0; b < GOLDEN_NB; b++) {
        memcpy(dst + b * BLOCK_BYTES, &d[b], 2);
        memcpy(dst + b * BLOCK_BYTES + 2, q + b * 32, 32);
    }
}

static uint64_t read_time_ns(void)
{
    uint32_t lo = rd32(REG_TIME_LO);   /* reading LO latches HI */
    uint32_t hi = rd32(REG_TIME_HI);
    return ((uint64_t)hi << 32) | lo;
}

typedef struct { uint32_t result, error, busy_ns; unsigned long busy_polls, accesses; uint64_t sim_ns; int timeout; } run_t;

/* start with the given CTRL value, poll until ready, collect result and timing */
static run_t run_once(uint32_t ctrl, unsigned long accesses_before)
{
    run_t r = {0};
    uint64_t t0 = read_time_ns();
    wr32(REG_CTRL, ctrl);
    uint32_t st;
    unsigned long polls = 0;
    while ((st = rd32(REG_STATUS)) == 0 && polls < POLL_LIMIT) polls++;
    uint64_t t1 = read_time_ns();
    r.timeout = (st != 1);
    r.busy_polls = polls;
    r.sim_ns = t1 - t0;
    r.result = rd32(REG_RESULT);
    r.error = rd32(REG_ERROR);
    r.busy_ns = rd32(REG_BUSY_NS);
    r.accesses = dev_accesses - accesses_before;
    return r;
}

int main(void)
{
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "open /dev/mem: %s (run with sudo)\n", strerror(errno));
        return 2;
    }
    void *m = mmap(NULL, DOT_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, DOT_BASE);
    void *o = mmap(NULL, OCM_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, OCM_BASE);
    if (m == MAP_FAILED || o == MAP_FAILED) {
        fprintf(stderr, "mmap: %s\n", strerror(errno));
        return 2;
    }
    dev = (volatile uint8_t *)m;
    ocm = (volatile uint8_t *)o;

    printf("dot-product accelerator DMA-mode test, %d cases (x at 0x%lx, y at 0x%lx)\n",
           GOLDEN_NUM_CASES, X_PHYS, Y_PHYS);

    /* OCM sanity check: what we write must read back */
    *(volatile uint32_t *)ocm = 0xa5a5c3c3u;
    if (*(volatile uint32_t *)ocm != 0xa5a5c3c3u) {
        fprintf(stderr, "FAIL: OCM at 0x%lx does not read back\n", OCM_BASE);
        return 1;
    }
    if (rd32(REG_STATUS) != 1) {
        fprintf(stderr, "FAIL: STATUS is not 1 (ready) before the first start\n");
        return 1;
    }

    int failures = 0;
    for (int t = 0; t < GOLDEN_NUM_CASES; t++) {
        const golden_case_t *c = &golden_cases[t];
        uint8_t buf[VEC_BYTES];

        /* DMA mode: data goes to OCM (plain memory), the accelerator only gets two addresses */
        pack_blocks(buf, c->qx, c->dx);
        put_words(ocm, X_PHYS - OCM_BASE, buf, VEC_BYTES, 0);
        pack_blocks(buf, c->qy, c->dy);
        put_words(ocm, Y_PHYS - OCM_BASE, buf, VEC_BYTES, 0);
        /* poison block 0 of the scratchpads so a DMA that loads nothing cannot pass with stale data */
        wr32(QX_OFF, 0x7f7f7f7f); wr32(DX_OFF, 0x3c003c00); wr32(QY_OFF, 0x7f7f7f7f); wr32(DY_OFF, 0x3c003c00);

        unsigned long a0 = dev_accesses;
        wr32(REG_X_ADDR, (uint32_t)X_PHYS);
        wr32(REG_Y_ADDR, (uint32_t)Y_PHYS);
        run_t d = run_once(CTRL_START | CTRL_DMA, a0);

        /* copy mode, same case: the ARM writes all 1904 bytes into the scratchpads */
        a0 = dev_accesses;
        put_words(dev, QX_OFF, c->qx, GOLDEN_N, 1);
        put_words(dev, DX_OFF, c->dx, GOLDEN_NB * 2, 1);
        put_words(dev, QY_OFF, c->qy, GOLDEN_N, 1);
        put_words(dev, DY_OFF, c->dy, GOLDEN_NB * 2, 1);
        run_t k = run_once(CTRL_START, a0);

        int ok = !d.timeout && !k.timeout && d.error == 0 && d.result == c->expect && k.result == c->expect;
        printf("[%s] %-26s got 0x%08x exp 0x%08x | dma: err=%u busy-ns=%u arm-accesses=%lu sim-time=%llu ns"
               " | copy: busy-ns=%u arm-accesses=%lu sim-time=%llu ns\n",
               ok ? "PASS" : "FAIL", c->name, d.result, c->expect, d.error, d.busy_ns, d.accesses,
               (unsigned long long)d.sim_ns, k.busy_ns, k.accesses, (unsigned long long)k.sim_ns);
        if (!ok) failures++;
    }

    /* error path: an address outside DRAM/OCM must be refused, not read */
    wr32(REG_X_ADDR, 0x90000000u);
    run_t e = run_once(CTRL_START | CTRL_DMA, dev_accesses);
    int eok = !e.timeout && e.error == 1;
    printf("[%s] bad X_ADDR 0x90000000 -> ERROR=%u (expect 1)\n", eok ? "PASS" : "FAIL", e.error);
    if (!eok) failures++;

    int total = GOLDEN_NUM_CASES + 1;
    printf("\n%s: %d/%d checks passed\n", failures ? "FAILED" : "ALL PASSED", total - failures, total);
    munmap(m, DOT_SPAN);
    munmap(o, OCM_SPAN);
    close(fd);
    return failures ? 1 : 0;
}
