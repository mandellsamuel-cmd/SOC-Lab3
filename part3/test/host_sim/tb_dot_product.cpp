/*
 * Host testbench for the dot_product SystemC module (no QEMU, no ARM).
 * An initiator drives the module through its TLM target socket exactly like the ARM test does:
 * aligned 32-bit writes of the operands, write CTRL=1, poll STATUS, read RESULT, compare bit-exactly with
 * the golden values from ../golden_data.h. Reports modeled time per operation.
 * A second pass uses DMA mode (CTRL=3): the operands are put in a fake OCM as llama.cpp block_q8_0 arrays
 * and the module fetches them itself through its master socket.
 */
#include <systemc.h>
#include <tlm.h>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>
#include <stdio.h>
#include <string.h>
#include "dot_product.h"
#include "golden_data.h"

// stands in for system memory behind S_AXI_HP0_FPD: 256 KB of OCM at 0xFFFC0000, 1 ns per access
SC_MODULE(fake_ocm) {
    static const uint64_t BASE = 0xFFFC0000ULL, SIZE = 0x40000;
    tlm_utils::simple_target_socket<fake_ocm> socket;
    unsigned char mem[SIZE];
    unsigned reads = 0;

    SC_CTOR(fake_ocm) : socket("socket") {
        socket.register_b_transport(this, &fake_ocm::b_transport);
        memset(mem, 0, sizeof mem);
    }
    void b_transport(tlm::tlm_generic_payload& p, sc_time& d) {
        uint64_t a = p.get_address();
        unsigned n = p.get_data_length();
        d += sc_time(1, SC_NS);
        if (a < BASE || a + n > BASE + SIZE) { p.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE); return; }
        if (p.is_read()) { memcpy(p.get_data_ptr(), mem + (a - BASE), n); reads++; }
        else memcpy(mem + (a - BASE), p.get_data_ptr(), n);
        p.set_response_status(tlm::TLM_OK_RESPONSE);
    }
};

SC_MODULE(tb) {
    tlm_utils::simple_initiator_socket<tb> init;
    dot_product dut;
    fake_ocm ocm;
    int failures = 0;

    SC_CTOR(tb) : init("init"), dut("dut"), ocm("ocm") {
        init.bind(dut.target_socket);
        dut.master_socket.bind(ocm.socket);
        SC_THREAD(run);
    }

    // one blocking access, then let simulated time catch up with the delay the target reported
    // (the ARM side behaves the same way: each bus access takes simulated time)
    uint32_t access(tlm::tlm_command cmd, uint64_t addr, uint32_t wdata = 0) {
        tlm::tlm_generic_payload p;
        sc_time d = SC_ZERO_TIME;
        uint32_t data = wdata;
        p.set_command(cmd);
        p.set_address(addr);
        p.set_data_ptr((unsigned char*)&data);
        p.set_data_length(4);
        p.set_streaming_width(4);
        p.set_byte_enable_ptr(NULL);
        p.set_dmi_allowed(false);
        p.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
        init->b_transport(p, d);
        if (p.is_response_error()) { printf("bus error at 0x%llx\n", (unsigned long long)addr); failures++; }
        wait(d);
        return data;
    }
    uint32_t rd(uint64_t a) { return access(tlm::TLM_READ_COMMAND, a); }
    void wr(uint64_t a, uint32_t v) { access(tlm::TLM_WRITE_COMMAND, a, v); }

    void put(uint64_t off, const void* src, unsigned n) {
        for (unsigned i = 0; i < n; i += 4) { uint32_t w; memcpy(&w, (const char*)src + i, 4); wr(off + i, w); }
    }

    // llama.cpp block_q8_0 layout {fp16 d; int8 qs[32]} written straight into the fake OCM
    void put_blocks(uint64_t addr, const int8_t* q, const uint16_t* d) {
        unsigned char* p = ocm.mem + (addr - fake_ocm::BASE);
        for (int b = 0; b < GOLDEN_NB; b++) {
            memcpy(p + b * 34, &d[b], 2);
            memcpy(p + b * 34 + 2, q + b * 32, 32);
        }
    }

    // write CTRL, poll STATUS until ready, return the number of busy polls
    unsigned start_and_wait(uint32_t ctrl) {
        wr(0x0000, ctrl);
        unsigned busy = 0;
        while (rd(0x0004) == 0) busy++;
        return busy;
    }

    void run() {
        const uint64_t X = fake_ocm::BASE, Y = fake_ocm::BASE + 0x400;

        printf("host testbench, %d golden cases\n", GOLDEN_NUM_CASES);
        if (rd(0x0004) != 1) { printf("FAIL: not ready at reset\n"); failures++; }

        printf("-- copy mode (ARM writes the scratchpads, CTRL=1)\n");
        for (int t = 0; t < GOLDEN_NUM_CASES; t++) {
            const golden_case_t* c = &golden_cases[t];
            put(0x1000, c->qx, GOLDEN_N);
            put(0x2000, c->dx, GOLDEN_NB * 2);
            put(0x3000, c->qy, GOLDEN_N);
            put(0x4000, c->dy, GOLDEN_NB * 2);

            sc_time t0 = sc_time_stamp();
            unsigned busy = start_and_wait(1);
            sc_time t1 = sc_time_stamp();

            uint32_t got = rd(0x0008);
            bool ok = (got == c->expect);
            if (!ok) failures++;
            printf("[%s] %-26s got 0x%08x exp 0x%08x busy-polls=%u start->done=%.0f ns busy-ns=%u\n",
                   ok ? "PASS" : "FAIL", c->name, got, c->expect, busy, (t1 - t0).to_double() / 1000.0, rd(0x0018));
        }

        printf("-- DMA mode (accelerator fetches block_q8_0 x/y from OCM, CTRL=3)\n");
        for (int t = 0; t < GOLDEN_NUM_CASES; t++) {
            const golden_case_t* c = &golden_cases[t];
            put_blocks(X, c->qx, c->dx);
            put_blocks(Y, c->qy, c->dy);
            // poison block 0 of the scratchpads so a DMA that loads nothing cannot pass with stale data
            wr(0x1000, 0x7f7f7f7f); wr(0x2000, 0x3c003c00); wr(0x3000, 0x7f7f7f7f); wr(0x4000, 0x3c003c00);
            wr(0x0020, (uint32_t)X);
            wr(0x0024, (uint32_t)Y);

            unsigned reads0 = ocm.reads;
            sc_time t0 = sc_time_stamp();
            unsigned busy = start_and_wait(3);
            sc_time t1 = sc_time_stamp();

            uint32_t got = rd(0x0008), err = rd(0x000C);
            bool ok = (err == 0 && got == c->expect);
            if (!ok) failures++;
            printf("[%s] %-26s got 0x%08x exp 0x%08x err=%u bursts=%u busy-polls=%u start->done=%.0f ns busy-ns=%u\n",
                   ok ? "PASS" : "FAIL", c->name, got, c->expect, err, ocm.reads - reads0, busy,
                   (t1 - t0).to_double() / 1000.0, rd(0x0018));
        }

        printf("-- DMA error path (X_ADDR outside DRAM/OCM)\n");
        wr(0x0020, 0x90000000u);
        start_and_wait(3);
        uint32_t err = rd(0x000C);
        if (err != 1) failures++;
        printf("[%s] bad address -> ERROR=%u (expect 1)\n", err == 1 ? "PASS" : "FAIL", err);

        int total = 2 * GOLDEN_NUM_CASES + 1;
        printf("\n%s: %d/%d passed\n", failures ? "FAILED" : "ALL PASSED", total - failures, total);
        sc_stop();
    }
};

int sc_main(int, char**) {
    tb top("tb");
    sc_start();
    return top.failures ? 1 : 0;
}
