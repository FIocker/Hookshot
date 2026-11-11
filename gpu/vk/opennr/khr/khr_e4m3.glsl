// (hookshot) Upstream's hardware E4M3 conversions (common.glsl under DLSS_E4M3_HW, which needs VK_EXT_shader_float8),
// spelled in integer arithmetic, for the khr route: the same names and the same results, on any GPU.
//
// Upstream states the equality itself (common.glsl): cvt.rn.satfinite.e4m3 "is equal to e4m3CodeFromF16Bits for every
// non-NaN half", and its e4m3Hw guards NaN to +0 before converting, which e4m3CodeFromF16Bits does too. The decode reads
// NaN codes as 0, as e4m3HwToF32 does. So these are upstream's own functions, not an approximation of them.
//
// scripts/opennr_kernels.py compiles upstream's ops.comp and gemm_reduce.comp with this file in place of their
// `#define DLSS_E4M3_HW` (a build-time copy; third_party stays upstream's), and the khr kernels include it.
#ifndef KHR_E4M3_GLSL
#define KHR_E4M3_GLSL

uint8_t e4m3Hw(float16_t value) { return uint8_t(e4m3CodeFromF16Bits(f16Bits(value))); }

uint16_t e4m3HwPair(f16vec2 value) {
  return uint16_t(e4m3CodeFromF16Bits(f16Bits(value.x)) | (e4m3CodeFromF16Bits(f16Bits(value.y)) << 8u));
}

// (both NaN codes to +0, as the hardware decode's isnan guard gives; e4m3ToF32 alone would make 0xff a -0)
float e4m3HwToF32(uint code) { return (code & 0x7fu) == 0x7fu ? 0.0 : e4m3ToF32(code & 0xffu); }

#endif
