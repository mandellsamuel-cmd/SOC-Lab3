#ifndef DOT_PRODUCT_H
#define DOT_PRODUCT_H

#include <systemc.h>
#include "tlm.h"
#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/simple_target_socket.h"
#include <cstring>
#include <stdint.h>
#include "hls_q8_dot_vec.h" // QWEN_N, QWEN_NB, half, function prototypes

/*
 * Register / memory map (offsets from the base address the bus maps this module at,
 * e.g. 0xa0400000 in zynqmp_demo.cc; the iconnect uses ADDRMODE_RELATIVE).
 *
 *   0x0000  CTRL    (W)  bit0 = start, bit1 = mode (0 = copy: ARM filled the scratchpads,
 *                        1 = DMA: fetch x/y from system memory first). Ignored (with a warning) while busy.
 *   0x0004  STATUS  (R)  1 = ready/done, 0 = busy
 *   0x0008  RESULT  (R)  FP32 result, valid when STATUS == 1 and ERROR == 0
 *   0x000C  ERROR   (R)  0 = ok, 1 = DMA address outside DRAM/OCM, 2 = DMA bus error (cleared on start)
 *   0x0010  TIME_LO (R)  simulated time in ns, low 32 bits  (for latency measurement)
 *   0x0014  TIME_HI (R)  simulated time in ns, high 32 bits (read LO first; HI is latched)
 *   0x0018  BUSY_NS (R)  start -> done time of the last operation, measured inside the accelerator
 *   0x0020  X_ADDR  (RW) DMA mode: physical address of x as QWEN_NB llama.cpp block_q8_0 {fp16 d; int8 qs[32]}
 *   0x0024  Y_ADDR  (RW) DMA mode: physical address of y, same layout
 *   0x1000  QX      (W)  int8  x QWEN_N   (896 B)
 *   0x2000  DX      (W)  fp16  x QWEN_NB  (56 B)
 *   0x3000  QY      (W)  int8  x QWEN_N
 *   0x4000  DY      (W)  fp16  x QWEN_NB
 *
 * Accesses may be 1..8 bytes (and bulk memcpy from the ARM turns into 8/16 byte
 * bursts), so everything below is length-generic and uses memcpy, not casts.
 *
 * DMA mode (Part 3 extra credit): master_socket is bound to S_AXI_HP0_FPD (non-coherent), so on real
 * hardware the CPU must flush x/y out of its caches before the start write. QEMU does not model caches.
 */
SC_MODULE(dot_product) {
    tlm_utils::simple_target_socket<dot_product> target_socket;
    tlm_utils::simple_initiator_socket<dot_product> master_socket;

    enum : uint64_t {
        REG_BASE  = 0x0000, REG_SIZE = 0x40,
        QX_BASE   = 0x1000,
        DX_BASE   = 0x2000,
        QY_BASE   = 0x3000,
        DY_BASE   = 0x4000
    };
    enum : uint32_t { CTRL_START = 1u << 0, CTRL_DMA = 1u << 1 };
    enum : uint32_t { ERR_OK = 0, ERR_ADDR = 1, ERR_BUS = 2 };

    // Timing model: 457 cycles @ 10 ns from the Lab 2 HLS report (change if you pick another design point)
    static const unsigned LATENCY_CYCLES = 457;
    static const unsigned CLK_PERIOD_NS  = 10;
    static const unsigned ACCESS_NS      = 10;   // per bus access, added to the transaction delay

    // DMA model: 128-bit (16 B) data beats at the 10 ns clock, bursts of at most 16 beats that
    // never cross a 4 KB boundary (AXI rule)
    static const unsigned BEAT_BYTES  = 16;
    static const unsigned BURST_BYTES = 16 * BEAT_BYTES;
    static const unsigned BLOCK_BYTES = 2 + QK8_0;              // sizeof(block_q8_0) in llama.cpp
    static const unsigned VEC_BYTES   = QWEN_NB * BLOCK_BYTES;  // 952 B per vector

    // Local scratchpads (~1.9 KB total, far below the 7.6 Mb of BRAM)
    int8_t scratch_qx[QWEN_N];
    half   scratch_dx[QWEN_NB];
    int8_t scratch_qy[QWEN_N];
    half   scratch_dy[QWEN_NB];

    float    result_mem;
    uint32_t reg_control;
    uint32_t reg_status;     // 1 = ready, 0 = busy
    uint32_t reg_error;
    uint32_t reg_busy_ns;
    uint32_t reg_x_addr, reg_y_addr;
    uint32_t time_hi_latch;  // TIME_HI captured when TIME_LO is read

    sc_event start_ev;

    SC_CTOR(dot_product) : target_socket("target_socket"), master_socket("master_socket") {
        static_assert(sizeof(half) == 2, "half must be 2 bytes for raw copies");
        target_socket.register_b_transport(this, &dot_product::b_transport);
        SC_THREAD(compute_thread);

        memset(scratch_qx, 0, sizeof scratch_qx);
        memset(scratch_qy, 0, sizeof scratch_qy);
        memset((void*)scratch_dx, 0, sizeof scratch_dx);
        memset((void*)scratch_dy, 0, sizeof scratch_dy);
        reg_control = 0;
        reg_status  = 1;
        reg_error   = ERR_OK;
        reg_busy_ns = 0;
        reg_x_addr  = reg_y_addr = 0;
        result_mem  = 0.0f;
        time_hi_latch = 0;
    }

    void b_transport(tlm::tlm_generic_payload& trans, sc_time& delay) {
        const tlm::tlm_command cmd = trans.get_command();
        const uint64_t addr = trans.get_address();
        const unsigned len  = trans.get_data_length();
        unsigned char* ptr  = trans.get_data_ptr();

        delay += sc_time(ACCESS_NS, SC_NS);
        trans.set_dmi_allowed(false);

        if (cmd == tlm::TLM_IGNORE_COMMAND) {
            trans.set_response_status(tlm::TLM_OK_RESPONSE);
            return;
        }
        if (trans.get_byte_enable_ptr() != NULL) {
            trans.set_response_status(tlm::TLM_BYTE_ENABLE_ERROR_RESPONSE);
            return;
        }

        bool ok;
        if (addr < REG_SIZE)
            ok = reg_access(cmd, addr, len, ptr, delay);
        else if (!(ok = mem_access(cmd, addr, len, ptr, QX_BASE, scratch_qx, sizeof scratch_qx)) &&
                 !(ok = mem_access(cmd, addr, len, ptr, DX_BASE, (void*)scratch_dx, sizeof scratch_dx)) &&
                 !(ok = mem_access(cmd, addr, len, ptr, QY_BASE, scratch_qy, sizeof scratch_qy)))
            ok = mem_access(cmd, addr, len, ptr, DY_BASE, (void*)scratch_dy, sizeof scratch_dy);

        trans.set_response_status(ok ? tlm::TLM_OK_RESPONSE : tlm::TLM_ADDRESS_ERROR_RESPONSE);
    }

    // Hardware thread: event-driven (no polling), result published only after the modeled latency
    void compute_thread() {
        while (true) {
            wait(start_ev);                       // reg_status already set to 0 in b_transport
            const sc_time t_start = sc_time_stamp();

            // load/store unit: in DMA mode fill the scratchpads from system memory before computing
            if (reg_control & CTRL_DMA)
                reg_error = dma_load_vector(reg_x_addr, scratch_qx, scratch_dx);
            if ((reg_control & CTRL_DMA) && reg_error == ERR_OK)
                reg_error = dma_load_vector(reg_y_addr, scratch_qy, scratch_dy);

            if (reg_error == ERR_OK) {
                // inputs are sampled at start; scratchpad writes are blocked while busy
                float r = hls_vec_dot_q8_0_q8_0_fixed(scratch_qx, scratch_dx, scratch_qy, scratch_dy);

                wait(sc_time(LATENCY_CYCLES * CLK_PERIOD_NS, SC_NS));
                result_mem = r;
            }

            reg_busy_ns = (uint32_t)((sc_time_stamp() - t_start).to_double() / 1000.0);
            reg_control = 0;
            reg_status  = 1;                      // done
        }
    }

private:
    // DRAM 0x0-0x7FFFFFFF (2 GB on our boards) or OCM 0xFFFC0000-0xFFFFFFFF
    static bool dma_addr_ok(uint64_t a, uint64_t n) {
        return (a + n <= 0x80000000ULL) || (a >= 0xFFFC0000ULL && a + n <= 0x100000000ULL);
    }

    // read one vector of QWEN_NB block_q8_0 from system memory and unpack it into q[] / d[]
    uint32_t dma_load_vector(uint64_t addr, int8_t* q, half* d) {
        unsigned char buf[VEC_BYTES];
        if (!dma_addr_ok(addr, VEC_BYTES)) return ERR_ADDR;

        for (unsigned off = 0; off < VEC_BYTES; ) {
            const uint64_t a = addr + off;
            unsigned n = VEC_BYTES - off;
            if (n > BURST_BYTES) n = BURST_BYTES;
            if (n > 4096 - (a & 4095)) n = 4096 - (a & 4095);
            if (!dma_read(a, buf + off, n)) return ERR_BUS;
            off += n;
        }
        for (unsigned b = 0; b < QWEN_NB; b++) {
            memcpy((void*)&d[b], buf + b * BLOCK_BYTES, 2);
            memcpy(q + b * QK8_0, buf + b * BLOCK_BYTES + 2, QK8_0);
        }
        return ERR_OK;
    }

    // one read burst on the master socket (same pattern as demodma::do_dma_trans in demo-dma.cc)
    bool dma_read(uint64_t addr, unsigned char* buf, unsigned len) {
        tlm::tlm_generic_payload tr;
        sc_time delay = SC_ZERO_TIME;

        tr.set_command(tlm::TLM_READ_COMMAND);
        tr.set_address(addr);
        tr.set_data_ptr(buf);
        tr.set_data_length(len);
        tr.set_streaming_width(len);
        tr.set_byte_enable_ptr(NULL);
        tr.set_dmi_allowed(false);
        tr.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);

        master_socket->b_transport(tr, delay);

        // sync after every burst: the remote-port bridge time-stamps each request with the current SystemC time
        delay += sc_time(((len + BEAT_BYTES - 1) / BEAT_BYTES) * CLK_PERIOD_NS, SC_NS);
        wait(delay);
        return tr.get_response_status() == tlm::TLM_OK_RESPONSE;
    }

    bool reg_access(tlm::tlm_command cmd, uint64_t addr, unsigned len, unsigned char* ptr, const sc_time& delay) {
        if (addr + len > REG_SIZE) return false;

        uint32_t img[REG_SIZE / 4];
        memset(img, 0, sizeof img);
        img[0] = reg_control;
        img[1] = reg_status;
        memcpy(&img[2], &result_mem, sizeof(float));
        img[3] = reg_error;
        {
            // time as seen by the initiator: global SystemC time plus its local (temporally decoupled) offset
            uint64_t ns = (uint64_t)((sc_time_stamp() + delay).to_double() / 1000.0);   // resolution is 1 ps
            img[4] = (uint32_t)ns;
            img[5] = time_hi_latch;
            if (cmd == tlm::TLM_READ_COMMAND && addr <= 0x10 && addr + len > 0x10)
                time_hi_latch = img[5] = (uint32_t)(ns >> 32);
        }
        img[6] = reg_busy_ns;
        img[8] = reg_x_addr;
        img[9] = reg_y_addr;

        if (cmd == tlm::TLM_READ_COMMAND) {
            memcpy(ptr, (unsigned char*)img + addr, len);
            return true;
        }

        // writable: CTRL (bytes 0..3), X_ADDR / Y_ADDR (0x20..0x27); everything else is ignored
        memcpy((unsigned char*)img + addr, ptr, len);
        if (addr < 0x28 && addr + len > 0x20) {
            if (reg_status == 0) {
                SC_REPORT_WARNING("dot_product", "address register write while busy, ignored");
            } else {
                reg_x_addr = img[8];
                reg_y_addr = img[9];
            }
        }
        if (addr < 4 && (img[0] & CTRL_START)) {
            if (reg_status == 0) {
                SC_REPORT_WARNING("dot_product", "start written while busy, ignored");
            } else {
                reg_control = img[0];
                reg_status  = 0;          // busy immediately, so an ARM read right after the write can't see stale 'ready'
                reg_error   = ERR_OK;
                start_ev.notify(SC_ZERO_TIME);
            }
        }
        return true;
    }

    bool mem_access(tlm::tlm_command cmd, uint64_t addr, unsigned len, unsigned char* ptr,
                    uint64_t base, void* mem, size_t size) {
        if (addr < base || addr + len > base + size) return false;
        if (cmd == tlm::TLM_WRITE_COMMAND) {
            if (reg_status == 0) {
                SC_REPORT_WARNING("dot_product", "scratchpad write while busy, ignored");
                return true;
            }
            memcpy((unsigned char*)mem + (addr - base), ptr, len);
        } else {
            memcpy(ptr, (unsigned char*)mem + (addr - base), len);
        }
        return true;
    }
};

#endif
