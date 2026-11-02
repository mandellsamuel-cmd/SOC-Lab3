// Minimal stand-in for Vitis HLS <hls_half.h>, ONLY for the host testbench (no Vitis install needed).
// The real build (LRC) uses the Vitis header instead; this one provides just what the Lab 2 kernel touches.
#ifndef STUB_HLS_HALF_H
#define STUB_HLS_HALF_H
#include <stdint.h>
struct half { uint16_t bits; };   // raw IEEE binary16 pattern, same size as Vitis' half
#endif
