/*
 * dotprod_ioctl.h - interface between the dot-product driver (dotprod_drv.ko) and user space (the HAL).
 * Included by both the kernel module and user programs.
 */
#ifndef DOTPROD_IOCTL_H
#define DOTPROD_IOCTL_H

#include <linux/types.h>
#include <linux/ioctl.h>

#define DOTPROD_DEV          "/dev/dotprod"
#define DOTPROD_HW_BLOCKS    28   /* blocks per hardware operation (QWEN_NB in part3/hls_q8_dot_vec.h) */
#define DOTPROD_BLOCK_BYTES  34   /* sizeof(block_q8_0) in llama.cpp: fp16 d + int8 qs[32] */

/* One dot product of up to DOTPROD_HW_BLOCKS blocks. x and y point to llama.cpp block_q8_0 arrays. */
struct dotprod_req {
	__u64 x;         /* in:  user address of nblocks x blocks */
	__u64 y;         /* in:  user address of nblocks y blocks */
	__u32 nblocks;   /* in:  1 .. DOTPROD_HW_BLOCKS */
	__u32 result;    /* out: raw FP32 bits of RESULT */
	__u32 busy_ns;   /* out: BUSY_NS (accelerator start -> done, simulated ns) */
	__u32 error;     /* out: ERROR register (DMA mode; 0 = ok) */
};

/* Single register access (debugging), like the tutorial driver's register ioctls */
struct dotprod_reg {
	__u32 offset;    /* byte offset in the 64 KB device window, 4-byte aligned */
	__u32 value;
};

#define DOTPROD_IOC_MAGIC  'D'
#define DOTPROD_IOC_DOT    _IOWR(DOTPROD_IOC_MAGIC, 1, struct dotprod_req)
#define DOTPROD_IOC_RDREG  _IOWR(DOTPROD_IOC_MAGIC, 2, struct dotprod_reg)
#define DOTPROD_IOC_WRREG  _IOW(DOTPROD_IOC_MAGIC, 3, struct dotprod_reg)

#endif
