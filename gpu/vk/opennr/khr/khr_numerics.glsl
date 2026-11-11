// (hookshot) The three places the network's arithmetic leans on what a GPU's ALU happens to do, made exact for the khr
// route, so it gives the same halves on any conforming Vulkan driver.
//
// * The half fma (the window attention's square sums, the ViT's exponential). A GPU with f16 FMA hardware rounds
//   once; one that promotes to f32 rounds twice, and f32 has too few bits for that to be harmless when the addend is
//   far from the product (llvmpipe did this: 166 of 393,216 window-normalize outputs a code off, an RTX 5060 Ti none).
// * The cosine norm's inverse square root and the softmax's reciprocal. Upstream spells them inversesqrt and 1.0 / x
//   in GLSL, rsqrt.approx / rcp.rn in its PTX, and both are rounded to half. Vulkan allows them 2 and 2.5 ulp of f32,
//   and 1 / sqrt(x) of a half comes within 2^-21.4 of a half's rounding midpoint, so a looser driver may round the
//   other way. Here they are the correctly rounded halves of the exact values -- which, checked for every positive
//   half (the totals and square sums are halves), are what upstream's reference computes too: RN16(RN32(1 / RN32(
//   sqrt(x)))) and RN16(RN32(1 / x)) never differ from RN16 of the exact 1 / sqrt(x) and 1 / x.
// All of it in f32 arithmetic whose every product is exact, so not even an fma is assumed to be fused.
#ifndef KHR_NUMERICS_GLSL
#define KHR_NUMERICS_GLSL

// The midpoint between a positive finite half and its upper / lower neighbour, as an f32 (12 bits: exact). Above the
// largest half the midpoint is 65520, where RNE goes to infinity.
float halfMidpointAbove(float16_t h) {
  uint bits = f16Bits(h);
  return bits >= 0x7bffu ? 65520.0 : 0.5 * (float(h) + float(f16FromBits(bits + 1u)));
}
float halfMidpointBelow(float16_t h) { return 0.5 * (float(h) + float(f16FromBits(f16Bits(h) - 1u))); }
float16_t halfAbove(float16_t h) { return f16FromBits(f16Bits(h) + 1u); }
float16_t halfBelow(float16_t h) { return f16FromBits(f16Bits(h) - 1u); }

// RN16(1 / x) of a positive finite half. y = 1 / x lies above a midpoint m exactly when m x < 1, and m x (12 by 11
// bits) is exact in f32. A midpoint never equals y (that would need m and x both powers of two).
float16_t reciprocalHalf(float16_t x) {
  const float xf = float(x);
  if (!(xf > 0.0) || isinf(xf)) return float16_t(1.0 / xf);
  float16_t h = float16_t(roundF16(1.0 / xf));
  if (isinf(h) || h == float16_t(0.0)) return h;
  for (int i = 0; i < 2; ++i) {   // (the quotient is within an ulp of f32: one step is plenty)
    precise float up = halfMidpointAbove(h) * xf;
    if (up < 1.0) h = halfAbove(h);
    precise float down = halfMidpointBelow(h) * xf;
    if (h > float16_t(0.0) && down > 1.0) h = halfBelow(h);
  }
  return h;
}

// Whether m m x < 1, exactly, for an m of at most 12 significant bits and a half x: m m is exact (24 bits); split into
// halves of 12 bits, each times x is exact (23 bits); x hi - 1 is exact where it matters (x hi near 1: Sterbenz) and
// the sum's sign is right whatever its rounding.
bool squareTimesBelowOne(float m, float x) {
  precise float m2 = m * m;
  const float hi = uintBitsToFloat(floatBitsToUint(m2) & 0xfffff000u);
  precise float lo = m2 - hi;
  precise float d = x * hi - 1.0;
  precise float s = d + x * lo;
  return s < 0.0;
}

// RN16(1 / sqrt(x)) of a positive finite half: y lies above a midpoint m exactly when m^2 x < 1
float16_t inverseSqrtHalf(float16_t x) {
  const float xf = float(x);
  if (!(xf > 0.0) || isinf(xf)) return float16_t(inversesqrt(xf));
  float16_t h = float16_t(roundF16(inversesqrt(xf)));
  for (int i = 0; i < 2; ++i) {
    if (squareTimesBelowOne(halfMidpointAbove(h), xf)) h = halfAbove(h);
    if (!squareTimesBelowOne(halfMidpointBelow(h), xf)) h = halfBelow(h);
  }
  return h;
}

// RN16(a b + c) of three halves, rounded once. a b is exact in f32 (22 bits); t = a b + c is rounded to f32 and e is
// its exact error (Knuth's two-sum). Rounding t to half is then right unless t sits exactly on a half's midpoint with
// e != 0 -- t is an f32, a midpoint is one, and |e| is under half an f32 ulp of t, so nowhere else can e move t across
// one -- and there the exact value is on e's side of it.
float16_t fmaHalf(float16_t a, float16_t b, float16_t c) {
  precise float p = float(a) * float(b);
  precise float t = p + float(c);
  precise float bv = t - p;
  precise float av = t - bv;
  precise float e = (p - av) + (float(c) - bv);
  if (e != 0.0 && !isinf(t)) {
    const int exponent = max(int((floatBitsToUint(abs(t)) >> 23u) & 0xffu) - 127, -14);
    const float ulp = uintBitsToFloat(uint(exponent - 10 + 127) << 23u);   // (the half grid at t; 2^-24 below the
    const float steps = t * uintBitsToFloat(uint(127 - exponent + 10) << 23u);   // normal halves; powers of two: exact)
    if (steps - floor(steps) == 0.5) return float16_t(roundF16((e > 0.0 ? ceil(steps) : floor(steps)) * ulp));
  }
  return float16_t(roundF16(t));
}

#endif
