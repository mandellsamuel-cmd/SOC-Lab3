/*
 *  dotprod_drv.c - Linux driver for the Lab 3 dot-product accelerator (part3/dot_product.h)
 *
 *  Adapted from the tutorial's application.irq/fpga_drv.c (platform driver, misc device, /proc entry,
 *  NO_DTS resources). Changes:
 *   - /dev/dotprod; one DOTPROD_IOC_DOT ioctl does a whole dot product (load operands, start, wait, read result)
 *     instead of one ioctl per register access. Register ioctls are kept for debugging.
 *   - The accelerator has no interrupt: completion is detected by polling STATUS (with a timeout).
 *     The interrupt is optional in probe.
 *   - A mutex serializes callers (llama.cpp calls from several threads; there is one accelerator).
 *   - Module parameter dma=1 uses the accelerator's bus-master mode (part 3 extra credit): operands are
 *     copied into a DMA-coherent buffer and the accelerator fetches them itself.
 */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/interrupt.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/io.h>
#include <linux/ioport.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/miscdevice.h>
#include <linux/uaccess.h>
#include <linux/mutex.h>
#include <linux/jiffies.h>
#include <linux/dma-mapping.h>
#include <linux/platform_device.h>

#include <linux/of_address.h>
#include <linux/of_device.h>
#include <linux/of_platform.h>

#include "dotprod_ioctl.h"

#ifndef CONFIG_OF  // If no device-tree support in kernel, turn it off here
#define NO_DTS
#endif

#define dotprod_VERSION "1.0"
#define dotprod_NAME    "dotprod_drv"

/* Without device-tree support, compile in hardcoded resource info */
#ifdef NO_DTS
#define DOT_BASE     0xa0400000    // Base address of the accelerator (bus.memmap in part3/zynqmp_demo.cc)
#endif
#define DOT_SIZE     0x00010000

/* Register map (part3/dot_product.h) */
#define REG_CTRL     0x0000
#define REG_STATUS   0x0004
#define REG_RESULT   0x0008
#define REG_ERROR    0x000C
#define REG_BUSY_NS  0x0018
#define REG_X_ADDR   0x0020
#define REG_Y_ADDR   0x0024
#define QX_OFF       0x1000
#define DX_OFF       0x2000
#define QY_OFF       0x3000
#define DY_OFF       0x4000

#define CTRL_START   0x1
#define CTRL_DMA     0x2

#define POLL_TIMEOUT_MS 10000      // generous: every STATUS read is a QEMU <-> SystemC round trip

#define VEC_BYTES    (DOTPROD_HW_BLOCKS * DOTPROD_BLOCK_BYTES)

MODULE_AUTHOR("ECE382M Lab 3 (based on fpga_drv.c by gerstl@ece.utexas.edu)");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Dot-product accelerator driver");

#define DRIVER_NAME "dotprod"

#ifdef NO_DTS
/* command line parameters */
static unsigned install = 0;
module_param(install, uint, S_IRUGO);
MODULE_PARM_DESC(install, "register the (hardcoded) device on load");
#endif

static bool dma = false;
module_param(dma, bool, S_IRUGO);
MODULE_PARM_DESC(dma, "use the accelerator's bus-master (DMA) mode instead of copying into its scratchpads");

/* instance-specific driver-internal data structure */
static struct dotprod_drv_local {
	int irq;
	unsigned long reg_start;
	unsigned long reg_end;
	void __iomem *reg_ptr;
	struct mutex lock;               // one dot product at a time
	unsigned hw_blocks;              // copy mode: scales in DX/DY beyond this block are known to be 0
	u8 *kx, *ky;                     // copy mode: operands copied in from user space
	u8 *dma_x, *dma_y;               // DMA mode: coherent buffers the accelerator reads
	dma_addr_t dma_x_handle, dma_y_handle;
	unsigned dma_blocks;             // DMA mode: blocks beyond this are known to be 0 in the buffers
	struct proc_dir_entry *proc_file;
	/* statistics for /proc/dotprod */
	unsigned long calls, timeouts, hw_errors, interrupts;
	u64 busy_ns_total;
	u32 busy_ns_max;
} l;


/* =========================================================================
 * Device access
 */

static inline u32 dot_rd(unsigned off) { return readl(l.reg_ptr + off); }
static inline void dot_wr(unsigned off, u32 v) { writel(v, l.reg_ptr + off); }

/*
 * Copy mode: unpack nblocks block_q8_0 {fp16 d; int8 qs[32]} from buf into the accelerator's
 * quant (q_off) and scale (d_off) scratchpads. The hardware always computes DOTPROD_HW_BLOCKS blocks, so the
 * scales of the unused blocks must be 0 (a 0 scale makes the block contribute exactly 0); 'stale' is how many
 * leading scales may still be non-zero from the previous call.
 */
static void load_vector(const u8 *buf, unsigned q_off, unsigned d_off, unsigned nblocks, unsigned stale)
{
	unsigned b, i, nscale = max(nblocks, stale);

	for (b = 0; b < nblocks; b++) {
		const u8 *qs = buf + b * DOTPROD_BLOCK_BYTES + 2;
		for (i = 0; i < 32; i += 4) {
			u32 w;
			memcpy(&w, qs + i, 4);           // qs is not 4-byte aligned in the block layout
			dot_wr(q_off + b * 32 + i, w);
		}
	}
	/* two fp16 scales per 32-bit word */
	for (b = 0; b < nscale; b += 2) {
		u16 lo = 0, hi = 0;
		if (b < nblocks)     memcpy(&lo, buf + b * DOTPROD_BLOCK_BYTES, 2);
		if (b + 1 < nblocks) memcpy(&hi, buf + (b + 1) * DOTPROD_BLOCK_BYTES, 2);
		dot_wr(d_off + b * 2, (u32)lo | ((u32)hi << 16));
	}
}

static long dotprod_run(struct dotprod_req *r)
{
	size_t len;
	unsigned long deadline;
	long rv = 0;

	if (r->nblocks == 0 || r->nblocks > DOTPROD_HW_BLOCKS)
		return -EINVAL;
	len = r->nblocks * DOTPROD_BLOCK_BYTES;

	if (mutex_lock_interruptible(&l.lock))
		return -ERESTARTSYS;

	if (dma) {
		/* operands keep the llama.cpp block layout; the accelerator unpacks them itself */
		if (copy_from_user(l.dma_x, u64_to_user_ptr(r->x), len) ||
		    copy_from_user(l.dma_y, u64_to_user_ptr(r->y), len)) {
			rv = -EFAULT;
			goto out;
		}
		if (l.dma_blocks > r->nblocks) {
			memset(l.dma_x + len, 0, (l.dma_blocks - r->nblocks) * DOTPROD_BLOCK_BYTES);
			memset(l.dma_y + len, 0, (l.dma_blocks - r->nblocks) * DOTPROD_BLOCK_BYTES);
		}
		l.dma_blocks = r->nblocks;
		dot_wr(REG_X_ADDR, (u32)l.dma_x_handle);
		dot_wr(REG_Y_ADDR, (u32)l.dma_y_handle);
		dot_wr(REG_CTRL, CTRL_START | CTRL_DMA);  // writel orders the buffer writes before the start
	} else {
		if (copy_from_user(l.kx, u64_to_user_ptr(r->x), len) ||
		    copy_from_user(l.ky, u64_to_user_ptr(r->y), len)) {
			rv = -EFAULT;
			goto out;
		}
		load_vector(l.kx, QX_OFF, DX_OFF, r->nblocks, l.hw_blocks);
		load_vector(l.ky, QY_OFF, DY_OFF, r->nblocks, l.hw_blocks);
		l.hw_blocks = r->nblocks;
		dot_wr(REG_CTRL, CTRL_START);
	}

	/* the accelerator has no interrupt: poll STATUS (1 = done) */
	deadline = jiffies + msecs_to_jiffies(POLL_TIMEOUT_MS);
	while (dot_rd(REG_STATUS) == 0) {
		if (time_after(jiffies, deadline)) {
			l.timeouts++;
			rv = -ETIMEDOUT;
			goto out;
		}
		cpu_relax();
	}

	r->result  = dot_rd(REG_RESULT);
	r->busy_ns = dot_rd(REG_BUSY_NS);
	r->error   = dot_rd(REG_ERROR);

	l.calls++;
	l.busy_ns_total += r->busy_ns;
	if (r->busy_ns > l.busy_ns_max) l.busy_ns_max = r->busy_ns;
	if (r->error) l.hw_errors++;

#ifdef DEBUG
	printk(KERN_INFO "dotprod_drv: %u blocks -> 0x%08x (busy %u ns, error %u)\n",
	       r->nblocks, r->result, r->busy_ns, r->error);
#endif
out:
	mutex_unlock(&l.lock);
	return rv;
}


/* =========================================================================
 * Interrupt handling (optional: the current accelerator model has no interrupt line)
 */

static irqreturn_t dotprod_int_handler(int irq, void *lp)
{
	l.interrupts++;
	return IRQ_HANDLED;
}


/* =========================================================================
 * Driver access methods
 */

static int dotprod_open(struct inode *inode, struct file *file)
{
	return 0;
}

static int dotprod_release(struct inode *inode, struct file *file)
{
	return 0;
}

static long dotprod_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	void __user *argp = (void __user *)arg;
	struct dotprod_req req;
	struct dotprod_reg reg;
	long rv;

	switch (cmd) {
	case DOTPROD_IOC_DOT:
		if (copy_from_user(&req, argp, sizeof req))
			return -EFAULT;
		rv = dotprod_run(&req);
		if (rv)
			return rv;
		if (copy_to_user(argp, &req, sizeof req))
			return -EFAULT;
		return 0;

	case DOTPROD_IOC_RDREG:
	case DOTPROD_IOC_WRREG:
		if (copy_from_user(&reg, argp, sizeof reg))
			return -EFAULT;
		if (reg.offset >= DOT_SIZE || (reg.offset & 3))
			return -EINVAL;
		mutex_lock(&l.lock);
		if (cmd == DOTPROD_IOC_RDREG)
			reg.value = dot_rd(reg.offset);
		else
			dot_wr(reg.offset, reg.value);
		mutex_unlock(&l.lock);
#ifdef DEBUG
		printk(KERN_INFO "dotprod_drv: %s 0x%04x = 0x%08x\n",
		       cmd == DOTPROD_IOC_RDREG ? "read" : "write", reg.offset, reg.value);
#endif
		if (cmd == DOTPROD_IOC_RDREG && copy_to_user(argp, &reg, sizeof reg))
			return -EFAULT;
		return 0;
	}
	return -ENOTTY;
}

/* define which file operations are supported by the driver */
static const struct file_operations dotprod_fops = {
	.owner          = THIS_MODULE,
	.unlocked_ioctl = dotprod_ioctl,
	.open           = dotprod_open,
	.release        = dotprod_release,
};


/* =========================================================================
 * /proc entry
 */

static int proc_read_dotprod(struct seq_file *f, void *v)
{
	seq_printf(f, "mode                 %s\n", dma ? "dma (accelerator fetches operands)" : "copy (driver writes scratchpads)");
	seq_printf(f, "registers            0x%08lx - 0x%08lx\n", l.reg_start, l.reg_end);
	if (dma)
		seq_printf(f, "dma buffers          x 0x%08llx  y 0x%08llx\n",
			   (unsigned long long)l.dma_x_handle, (unsigned long long)l.dma_y_handle);
	seq_printf(f, "dot products         %lu\n", l.calls);
	seq_printf(f, "busy ns (avg / max)  %llu / %u\n",
		   l.calls ? l.busy_ns_total / l.calls : 0ULL, l.busy_ns_max);
	seq_printf(f, "timeouts             %lu\n", l.timeouts);
	seq_printf(f, "hardware errors      %lu\n", l.hw_errors);
	seq_printf(f, "interrupts           %lu\n", l.interrupts);
	return 0;
}

static int proc_open_dotprod(struct inode *inode, struct file *file)
{
	return single_open(file, proc_read_dotprod, NULL);
}

/* operations supported on the /proc entry */
static const struct proc_ops proc_ops = {
	.proc_open    = proc_open_dotprod,
	.proc_read    = seq_read,
	.proc_lseek   = seq_lseek,
	.proc_release = single_release,
};


/* =========================================================================
 * Device handling
 */

/* /dev entry; mode 0666 so llama.cpp does not have to run as root */
static struct miscdevice dotprod_miscdev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name  = DRIVER_NAME,
	.fops  = &dotprod_fops,
	.mode  = 0666,
};

static void dotprod_free_dma(struct device *dev)
{
	if (l.dma_x) dma_free_coherent(dev, VEC_BYTES, l.dma_x, l.dma_x_handle);
	if (l.dma_y) dma_free_coherent(dev, VEC_BYTES, l.dma_y, l.dma_y_handle);
	l.dma_x = l.dma_y = NULL;
}

/* probe and install module instance for device */
static int dotprod_drv_probe(struct platform_device *pdev)
{
	void __iomem *base;
	struct resource *r_reg; /* IO register resources */
	struct device *dev = &pdev->dev;
	int rv;

	dev_info(dev, "dotprod: probing\n");

	mutex_init(&l.lock);
	l.hw_blocks = DOTPROD_HW_BLOCKS;   // unknown scratchpad contents: zero every unused scale on the first call
	l.dma_blocks = 0;                  // dma_alloc_coherent memory starts zeroed

	// get, register and remap address region assigned to the device registers and scratchpads
	base = devm_platform_get_and_ioremap_resource(pdev, 0, &r_reg);
	if (IS_ERR(base)) {
		dev_err(dev, "dotprod: unable to map device registers\n");
		return PTR_ERR(base);
	}
	l.reg_ptr = base;
	l.reg_start = r_reg->start;
	l.reg_end = r_reg->end;

	l.kx = devm_kmalloc(dev, VEC_BYTES, GFP_KERNEL);
	l.ky = devm_kmalloc(dev, VEC_BYTES, GFP_KERNEL);
	if (!l.kx || !l.ky)
		return -ENOMEM;

	if (dma) {
		// a device registered without a device tree has no DMA mask; the accelerator takes 32-bit addresses
		if (!dev->dma_mask)
			dev->dma_mask = &dev->coherent_dma_mask;
		rv = dma_set_mask_and_coherent(dev, DMA_BIT_MASK(32));
		if (rv) {
			dev_err(dev, "dotprod: no usable 32-bit DMA mask (%d)\n", rv);
			return rv;
		}
		l.dma_x = dma_alloc_coherent(dev, VEC_BYTES, &l.dma_x_handle, GFP_KERNEL);
		l.dma_y = dma_alloc_coherent(dev, VEC_BYTES, &l.dma_y_handle, GFP_KERNEL);
		if (!l.dma_x || !l.dma_y) {
			dev_err(dev, "dotprod: cannot allocate DMA buffers\n");
			dotprod_free_dma(dev);
			return -ENOMEM;
		}
		// the accelerator only accepts DRAM 0x0 - 0x7FFFFFFF (or OCM)
		if (l.dma_x_handle + VEC_BYTES > 0x80000000ULL || l.dma_y_handle + VEC_BYTES > 0x80000000ULL) {
			dev_err(dev, "dotprod: DMA buffers outside the accelerator's address range\n");
			dotprod_free_dma(dev);
			return -ENOMEM;
		}
		dev_info(dev, "dotprod: DMA buffers x 0x%08llx y 0x%08llx\n",
			 (unsigned long long)l.dma_x_handle, (unsigned long long)l.dma_y_handle);
	}

	// interrupt is optional (none in the current accelerator model)
	l.irq = platform_get_irq_optional(pdev, 0);
	if (l.irq > 0) {
		rv = devm_request_irq(dev, l.irq, &dotprod_int_handler, 0, DRIVER_NAME, NULL);
		if (rv) {
			dev_err(dev, "dotprod: can't get interrupt %d: %d\n", l.irq, rv);
			dotprod_free_dma(dev);
			return rv;
		}
		dev_info(dev, "dotprod: using interrupt %d\n", l.irq);
	}

	// create /proc file system entry
	l.proc_file = proc_create(DRIVER_NAME, 0444, NULL, &proc_ops);
	if (l.proc_file == NULL) {
		dev_err(dev, "dotprod: create /proc entry failed\n");
		dotprod_free_dma(dev);
		return -ENOMEM;
	}

	// register /dev/dotprod last, when everything it uses is ready
	rv = misc_register(&dotprod_miscdev);
	if (rv) {
		dev_err(dev, "dotprod: unable to register device\n");
		remove_proc_entry(DRIVER_NAME, NULL);
		dotprod_free_dma(dev);
		return rv;
	}

	dev_info(dev, "dotprod: 0x%08lx size 0x%08lx mapped to 0x%px, %s mode, /dev/%s (10, %d)\n",
		 l.reg_start, l.reg_end - l.reg_start + 1, l.reg_ptr, dma ? "dma" : "copy",
		 DRIVER_NAME, dotprod_miscdev.minor);
	dev_info(dev, "dotprod: %s %s initialized\n", dotprod_NAME, dotprod_VERSION);
	return 0;
}

/* remove driver from kernel */
static int dotprod_drv_remove(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;

	misc_deregister(&dotprod_miscdev);
	remove_proc_entry(DRIVER_NAME, NULL);
	dotprod_free_dma(dev);

	dev_info(dev, "dotprod: %s %s removed\n", dotprod_NAME, dotprod_VERSION);
	return 0;
}

#ifdef CONFIG_OF
static struct of_device_id dotprod_drv_of_match[] = {
	{ .compatible = "ece382m,dotprod", },
	{ /* end of list */ },
};
MODULE_DEVICE_TABLE(of, dotprod_drv_of_match);
#else
# define dotprod_drv_of_match
#endif

static struct platform_device *dotprod_dev = NULL;

/* Kernel driver data structure */
static struct platform_driver dotprod_driver = {
	.driver = {
		.name = DRIVER_NAME,
		.owner = THIS_MODULE,
		.of_match_table = dotprod_drv_of_match,
	},
	.probe  = dotprod_drv_probe,
	.remove = dotprod_drv_remove,
};

#ifdef NO_DTS
/* Resources assigned to device: registers + scratchpads in one 64 KB window, no interrupt */
static const struct resource dotprod_resources[] = {
	{
		.start = DOT_BASE,
		.end   = DOT_BASE + DOT_SIZE - 1,
		.flags = IORESOURCE_MEM,
		.name  = "io-regs",
	},
};
#endif

/* =========================================================================
 * Module handling
 */

static int __init dotprod_init_module(void)
{
	int rv;

	rv = platform_driver_register(&dotprod_driver);
	if (rv) return rv;

#ifdef NO_DTS
	// if we are asked to install the device, register (and hence probe) it
	if (install) {
		dotprod_dev = platform_device_register_simple(DRIVER_NAME, -1, dotprod_resources,
							      ARRAY_SIZE(dotprod_resources));
		if (IS_ERR(dotprod_dev)) {
			rv = PTR_ERR(dotprod_dev);
			dotprod_dev = NULL;
			platform_driver_unregister(&dotprod_driver);
			return rv;
		}
	}
#endif
	return 0;
}

static void __exit dotprod_cleanup_module(void)
{
	if (dotprod_dev) platform_device_unregister(dotprod_dev);
	platform_driver_unregister(&dotprod_driver);
}

module_init(dotprod_init_module);
module_exit(dotprod_cleanup_module);
