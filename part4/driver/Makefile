# --- Kernel module for the dot-product accelerator (based on the tutorial's application.irq/Makefile)
#   make                 build dotprod_drv.ko against the course kernel tree on LRC
#   make KFLAGS="-DNO_DTS -DDEBUG"   also printk every access

# ARM cross-compiler
ARCH	= arm64
CROSS	?= aarch64-linux-gnu-

KDIR	?= /home/projects/gerstl/ece382m/linux-xlnx
# NO_DTS: our boot image's device tree has no node for the accelerator, so its address is compiled in
# and the module is loaded with 'insmod dotprod_drv.ko install=1' (see fpga_drv.c in the tutorial)
KFLAGS	?= -DNO_DTS

obj-m	= dotprod_drv.o

all: dotprod_drv.ko

dotprod_drv.ko: dotprod_drv.c dotprod_ioctl.h
	$(MAKE) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS) EXTRA_CFLAGS="$(KFLAGS)" -C $(KDIR) M=$(CURDIR) modules

clean:
	$(RM) .*.cmd *.o *.ko *.mod.c *.mod
	$(RM) modules.order Module.symvers
	$(RM) -R .tmp*

.PHONY: all clean
