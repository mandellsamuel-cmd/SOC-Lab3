#!/usr/bin/env python3
"""
Generates golden_data.h: input vectors + bit-exact expected FP32 results for the
Q8_0 dot-product accelerator (hls_vec_dot_q8_0_q8_0_fixed).

The accelerator sums every block term exactly as an integer (LSB = 2^-48) and rounds
ONCE to FP32 (round-to-nearest-even). So the golden value is the exactly-rounded FP32 of
the exact real-number result. It is computed here with Python big integers, and checked
with a second, independently formulated rounding routine (divmod based).

Usage: python3 gen_golden.py            (rewrites golden_data.h; deterministic)
"""
import os, random, sys

NB, QK = 28, 32
N = NB * QK
ACC_FRAC = 48

def split_half(h):
    """fp16 raw bits -> (signed significand M, exponent E), value = M * 2^E. Same as split_half() in HLS."""
    sign, exp, mant = (h >> 15) & 1, (h >> 10) & 0x1f, h & 0x3ff
    assert exp != 31, "Inf/NaN scales are not supported by the accelerator"
    if exp == 0:
        m, e = mant, -24
    else:
        m, e = mant | 0x400, exp - 25
    return (-m if sign else m), e

def exact_acc(qx, dx, qy, dy):
    """Exact integer accumulator, value = acc * 2^-48."""
    acc = 0
    for ib in range(NB):
        sumi = sum(qx[ib*QK+j] * qy[ib*QK+j] for j in range(QK))
        mx, ex = split_half(dx[ib]); my, ey = split_half(dy[ib])
        acc += (sumi * mx * my) << (ex + ey + ACC_FRAC)    # shift is always >= 0
    return acc

def acc_to_f32_bits(acc):
    """Mirrors acc_to_float() in the HLS file: RNE to FP32, returns raw bits."""
    if acc == 0:
        return 0
    sign = 1 if acc < 0 else 0
    mag = -acc if sign else acc
    p = mag.bit_length() - 1
    if p <= 23:
        m = mag << (23 - p)
    else:
        drop = p - 23
        m = mag >> drop
        guard = (mag >> (drop - 1)) & 1
        sticky = (mag & ((1 << (drop - 1)) - 1)) != 0
        if guard and (sticky or (m & 1)):
            m += 1
    if m >> 24:
        m >>= 1; p += 1
    exp = p - ACC_FRAC + 127
    assert 0 < exp < 255
    return (sign << 31) | (exp << 23) | (m & 0x7fffff)

def acc_to_f32_bits_divmod(acc):
    """Independent formulation: scale to a 24-bit integer with divmod and round half to even."""
    if acc == 0:
        return 0
    sign = 1 if acc < 0 else 0
    mag = abs(acc)
    e = mag.bit_length() - 24                 # want 2^23 <= mag / 2^e < 2^24
    if e <= 0:
        q = mag << -e
    else:
        q, r = divmod(mag, 1 << e)
        half = 1 << (e - 1)
        if r > half or (r == half and (q & 1)):
            q += 1
        if q == 1 << 24:
            q >>= 1; e += 1
    # value = q * 2^e * 2^-48, q in [2^23, 2^24]
    exp = (e - ACC_FRAC) + 23 + 127
    return (sign << 31) | (exp << 23) | (q & 0x7fffff)

# ---------------------------------------------------------------- test case construction
ONE, NEG_ONE = 0x3c00, 0xbc00
MAXH, NEG_MAXH = 0x7bff, 0xfbff            # +/-65504

def rnd_half(rng, lo_exp, hi_exp, allow_sub=False):
    exp = rng.randint(lo_exp, hi_exp)
    if allow_sub and rng.random() < 0.3:
        exp = 0
    return (rng.getrandbits(1) << 15) | (exp << 10) | rng.getrandbits(10)

def rnd_q(rng):  return [rng.randint(-128, 127) for _ in range(N)]

cases = []
def add(name, qx, dx, qy, dy):
    assert len(qx) == N and len(qy) == N and len(dx) == NB and len(dy) == NB
    assert all(-128 <= v <= 127 for v in qx + qy)
    acc = exact_acc(qx, dx, qy, dy)
    a, b = acc_to_f32_bits(acc), acc_to_f32_bits_divmod(acc)
    assert a == b, f"{name}: rounding routines disagree {a:08x} {b:08x}"
    cases.append((name, qx, dx, qy, dy, a))

rng = random.Random(382)

add("ones_unit_scale",        [1]*N,    [ONE]*NB,     [1]*N,    [ONE]*NB)
add("max_positive_sum",       [-128]*N, [ONE]*NB,     [-128]*N, [ONE]*NB)
add("max_negative_sum",       [127]*N,  [ONE]*NB,     [-128]*N, [ONE]*NB)
add("all_zero_quants",        [0]*N,    [ONE]*NB,     [0]*N,    [ONE]*NB)
add("all_zero_scales",        rnd_q(rng), [0]*NB,     rnd_q(rng), [0]*NB)

# exact cancellation -> +0.0f: block pairs identical except for scale sign
q = rnd_q(rng); qy = rnd_q(rng); d = [rnd_half(rng, 10, 20) & 0x7fff for _ in range(NB // 2)]
qx2 = []; qy2 = []; dx2 = []
for k in range(NB // 2):
    for sgn in (0, 0x8000):
        qx2 += q[2*k*QK:(2*k+1)*QK]; qy2 += qy[2*k*QK:(2*k+1)*QK]; dx2.append(d[k] | sgn)
add("exact_cancellation_zero", qx2, dx2, qy2, [ONE]*NB)

# single nonzero element
qx = [0]*N; qy = [0]*N; qx[500] = -77; qy[500] = 100
add("single_element",         qx, [0x3555]*NB, qy, [0x4a00]*NB)

add("moderate_random",        rnd_q(rng), [rnd_half(rng, 11, 18) for _ in range(NB)],
                              rnd_q(rng), [rnd_half(rng, 11, 18) for _ in range(NB)])
add("near_2^-14_boundary",    rnd_q(rng), [0x0400 | (rng.getrandbits(1) << 15) for _ in range(NB)],
                              rnd_q(rng), [0x03ff | (rng.getrandbits(1) << 15) for _ in range(NB)])
add("subnormal_scales",       rnd_q(rng), [(rng.getrandbits(1) << 15) | rng.randint(1, 0x3ff) for _ in range(NB)],
                              rnd_q(rng), [(rng.getrandbits(1) << 15) | rng.randint(1, 0x3ff) for _ in range(NB)])
add("min_subnormal_2^-24",    [127]*N, [0x0001]*NB, [127]*N, [0x0001]*NB)
add("large_scales",           rnd_q(rng), [MAXH if i % 2 else NEG_MAXH for i in range(NB)],
                              rnd_q(rng), [MAXH]*NB)
add("max_magnitude",          [127]*N, [MAXH]*NB, [127]*N, [MAXH]*NB)
add("mixed_subnormal_normal", rnd_q(rng), [rnd_half(rng, 0, 30, True) for _ in range(NB)],
                              rnd_q(rng), [rnd_half(rng, 0, 30, True) for _ in range(NB)])
for s in range(4):             # wide dynamic range: exercises the 24-bit rounding / ties paths
    add(f"wide_range_random_{s}", rnd_q(rng), [rnd_half(rng, 0, 30, True) for _ in range(NB)],
                                  rnd_q(rng), [rnd_half(rng, 0, 30, True) for _ in range(NB)])

# ---------------------------------------------------------------- emit header
def fmt(vals, per, hexa=False):
    f = (lambda v: f"0x{v:04x}") if hexa else (lambda v: str(v))
    lines = []
    for i in range(0, len(vals), per):
        lines.append("      " + ", ".join(f(v) for v in vals[i:i+per]))
    return ",\n".join(lines)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_data.h")
with open(OUT, "w") as f:
    f.write("/* AUTO-GENERATED by gen_golden.py -- do not edit. */\n")
    f.write("#ifndef GOLDEN_DATA_H\n#define GOLDEN_DATA_H\n#include <stdint.h>\n\n")
    f.write(f"#define GOLDEN_N  {N}\n#define GOLDEN_NB {NB}\n#define GOLDEN_NUM_CASES {len(cases)}\n\n")
    f.write("typedef struct {\n  const char *name;\n  int8_t   qx[GOLDEN_N];\n  uint16_t dx[GOLDEN_NB];\n"
            "  int8_t   qy[GOLDEN_N];\n  uint16_t dy[GOLDEN_NB];\n  uint32_t expect;   /* raw FP32 bits */\n} golden_case_t;\n\n")
    f.write("static const golden_case_t golden_cases[GOLDEN_NUM_CASES] = {\n")
    for name, qx, dx, qy, dy, e in cases:
        f.write(f"  {{ \"{name}\",\n    {{\n{fmt(qx, 32)}\n    }},\n    {{\n{fmt(dx, 14, True)}\n    }},\n"
                f"    {{\n{fmt(qy, 32)}\n    }},\n    {{\n{fmt(dy, 14, True)}\n    }},\n    0x{e:08x}u }},\n")
    f.write("};\n\n#endif\n")

print(f"{len(cases)} cases written to {OUT}")
for name, *_ , e in cases:
    import struct
    print(f"  {name:28s} 0x{e:08x}  {struct.unpack('<f', struct.pack('<I', e))[0]!r}")
