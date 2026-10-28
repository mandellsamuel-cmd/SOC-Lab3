#ifndef DOT_PRODUCT_H
#define DOT_PRODUCT_H

#include <systemc.h>
#include "tlm.h"
#include "tlm_utils/simple_target_socket.h"
#include <cstring>
#include <stdint.h>
#include "hls_q8_dot_vec.h" // QWEN_N, QWEN_NB, half, function prototypes

/*
 * Register / memory map (offsets from the base address the bus maps this module at,
 * e.g. 0xa0400000 in zynqmp_demo.cc; the iconnect uses ADDRMODE_RELATIVE).
 *
 *   0x0000  CTRL    (W)  bit0 = start. Ignored (with a warning) while busy.
 *   0x0004  STATUS  (R)  1 = ready/done, 0 = busy
 *   0x0008  RESULT  (R)  FP32 result, valid when STATUS == 1
 *   0x0010  TIME_LO (R)  simulated time in ns, low 32 bits  (for latency measurement)
 *   0x0014  TIME_HI (R)  simulated time in ns, high 32 bits (read LO first; HI is latched)
 *   0x1000  QX      (W)  int8  x QWEN_N   (896 B)
 *   0x2000  DX      (W)  fp16  x QWEN_NB  (56 B)
 *   0x3000  QY      (W)  int8  x QWEN_N
 *   0x4000  DY      (W)  fp16  x QWEN_NB
 *
 * Accesses may be 1..8 bytes (and bulk memcpy from the ARM turns into 8/16 byte
 * bursts), so everything below is length-generic and uses memcpy, not casts.
 */
SC_MODULE(dot_product) {
    tlm_utils::simple_target_socket<dot_product> target_socket;

    enum : uint64_t {
        REG_BASE  = 0x0000, REG_SIZE = 0x20,
        QX_BASE   = 0x1000,
        DX_BASE   = 0x2000,
        QY_BASE   = 0x3000,
        DY_BASE   = 0x4000
    };

    // Timing model: 457 cycles @ 10 ns from the Lab 2 HLS report (change if you pick another design point)
    static const unsigned LATENCY_CYCLES = 457;
    static const unsigned CLK_PERIOD_NS  = 10;
    static const unsigned ACCESS_NS      = 10;   // per bus access, added to the transaction delay

    // Local scratchpads (~1.9 KB total, far below the 7.6 Mb of BRAM)
    int8_t scratch_qx[QWEN_N];
    half   scratch_dx[QWEN_NB];
    int8_t scratch_qy[QWEN_N];
    half   scratch_dy[QWEN_NB];

    float    result_mem;
    uint32_t reg_control;
    uint32_t reg_status;     // 1 = ready, 0 = busy
    uint32_t time_hi_latch;  // TIME_HI captured when TIME_LO is read

    sc_event start_ev;

    SC_CTOR(dot_product) : target_socket("target_socket") {
        static_assert(sizeof(half) == 2, "half must be 2 bytes for raw copies");
        target_socket.register_b_transport(this, &dot_product::b_transport);
        SC_THREAD(compute_thread);

        memset(scratch_qx, 0, sizeof scratch_qx);
        memset(scratch_qy, 0, sizeof scratch_qy);
        memset((void*)scratch_dx, 0, sizeof scratch_dx);
        memset((void*)scratch_dy, 0, sizeof scratch_dy);
        reg_control = 0;
        reg_status  = 1;
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

            // inputs are sampled at start; scratchpad writes are blocked while busy
            float r = hls_vec_dot_q8_0_q8_0_fixed(scratch_qx, scratch_dx, scratch_qy, scratch_dy);

            wait(sc_time(LATENCY_CYCLES * CLK_PERIOD_NS, SC_NS));

            result_mem  = r;
            reg_control = 0;
            reg_status  = 1;                      // done
        }
    }

private:
    bool reg_access(tlm::tlm_command cmd, uint64_t addr, unsigned len, unsigned char* ptr, const sc_time& delay) {
        if (addr + len > REG_SIZE) return false;

        uint32_t img[REG_SIZE / 4];
        memset(img, 0, sizeof img);
        img[0] = reg_control;
        img[1] = reg_status;
        memcpy(&img[2], &result_mem, sizeof(float));
        {
            // time as seen by the initiator: global SystemC time plus its local (temporally decoupled) offset
            uint64_t ns = (uint64_t)((sc_time_stamp() + delay).to_double() / 1000.0);   // resolution is 1 ps
            img[4] = (uint32_t)ns;
            img[5] = time_hi_latch;
            if (cmd == tlm::TLM_READ_COMMAND && addr <= 0x10 && addr + len > 0x10)
                time_hi_latch = img[5] = (uint32_t)(ns >> 32);
        }

        if (cmd == tlm::TLM_READ_COMMAND) {
            memcpy(ptr, (unsigned char*)img + addr, len);
        } else {
            // only CTRL (bytes 0..3) is writable; STATUS/RESULT writes are ignored
            if (addr < 4) {
                unsigned n = (addr + len > 4) ? (4 - addr) : len;
                memcpy((unsigned char*)img + addr, ptr, n);
                if (img[0] & 1) {
                    if (reg_status == 0) {
                        SC_REPORT_WARNING("dot_product", "start written while busy, ignored");
                    } else {
                        reg_control = img[0];
                        reg_status  = 0;          // busy immediately, so an ARM read right after the write can't see stale 'ready'
                        start_ev.notify(SC_ZERO_TIME);
                    }
                }
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
