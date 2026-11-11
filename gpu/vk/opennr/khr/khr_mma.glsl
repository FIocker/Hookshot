// (hookshot) OpenDLSS-NR's matrix arithmetic on VK_KHR_cooperative_matrix alone, at subgroup scope: the khr route's
// kernels (gpu/vk/opennr/khr/*.comp) are written against this file and nothing vendor-specific.
//
// The network's contract (upstream's docs/numerics.md) is a chain of k32 steps: each step adds 32 E4M3 x E4M3 products
// to an f16 accumulator and rounds the result to half; the f16 GEMMs (the input adapter, the head) do the same with
// k16 steps of f16 products. Where an accumulator is published between steps is part of the result, so a variant may
// change how a step is computed but never where it rounds.
//
// A variant is a set of defines (scripts/opennr_kernels.py KHR_VARIANTS, gpu/vk/opennr/opennr_khr.h kVariants):
//   KHR_E4      1: the operands are E4M3 cooperative matrices (VK_EXT_shader_float8); 0: E4M3 widened to f16 when it is
//               staged (exact: every E4M3 value is a half, and every product of two of them is exact in f32)
//   KHR_M/N/K   the MMA shape the driver lists for those operands; KHR_HK the K of the f16 GEMMs' MMA (16, or K below 16)
//   KHR_ACC32   0: f16 accumulators, one MMA per step (only with KHR_E4 and K 32: NVIDIA's own mma.sync, so the route
//               computes the bytes of upstream's GLSL route); 1: f32 accumulators seeded with the f16 one, the step's
//               MMAs into them, then an explicit round to half on the bits (roundF16) -- one rounding per step, as the
//               network has it, on any hardware that has f32 accumulation (AMD RDNA 3/4 WMMA, llvmpipe)
//   KHR_SG      the subgroup size the kernels are made for and the pipelines are created with (32, or 8 on llvmpipe)
// The products are exact either way; what an ACC32 variant cannot promise is the hardware's own order of adding them
// inside an MMA, which an f16 accumulator would hide only on the GPU the network was made on. So an ACC32 variant is
// close to the reference, not equal to it: docs/neural-rendering.md says by how much, measured.
//
// Element layout: nothing here, or in the kernels, assumes where an element of a cooperative matrix lives. Every
// operand is staged in shared memory with plain indexing and loaded from there (or loaded straight from a buffer);
// every accumulator goes back to shared memory before any per-element work that depends on its position; the only
// per-element operation on a matrix itself is the rounding, which does not depend on the position (acc[i] = f(acc[i])).
#ifndef KHR_MMA_GLSL
#define KHR_MMA_GLSL

#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_shader_subgroup_basic : require
#if KHR_E4
#extension GL_EXT_float_e4m3 : require
#endif

#if !defined(KHR_E4) || !defined(KHR_M) || !defined(KHR_N) || !defined(KHR_K) || !defined(KHR_HK) || \
    !defined(KHR_ACC32) || !defined(KHR_SG)
#error "a khr variant needs KHR_E4, KHR_M, KHR_N, KHR_K, KHR_HK, KHR_ACC32 and KHR_SG (scripts/opennr_kernels.py)"
#endif
#if !KHR_ACC32 && (!KHR_E4 || KHR_K != 32 || KHR_HK != 16)
#error "f16 accumulators round once per MMA: only an E4M3 16x16x32 (and f16 16x16x16) MMA rounds where the network does"
#endif
#if (32 % KHR_K) != 0 || (16 % KHR_HK) != 0 || (16 % KHR_M) != 0 || (16 % KHR_N) != 0
#error "the MMA shape must divide the network's 16x16 tiles and its k32 / k16 steps"
#endif

#if KHR_E4
#define OPT uint8_t        // an operand element as staged in shared memory
#define OPCT floate4m3_t   // the operand matrices' component type
#define OP_WORDS 2u        // uvec4 words per 32-element operand row
OPT opFromCode(uint code) { return uint8_t(code); }
#else
#define OPT float16_t
#define OPCT float16_t
#define OP_WORDS 4u
// The E4M3 code widened to half. Exact (every E4M3 value is a half), so the conversion's rounding mode cannot matter;
// NaN codes read as 0, as upstream's reference decodes them (no weight is one, and no activation is ever published as
// one: docs/numerics.md).
OPT opFromCode(uint code) { return float16_t(e4m3ToF32(code)); }
#endif

#if KHR_ACC32
#define ACCT float
#else
#define ACCT float16_t
#endif

#define MAT_A coopmat<OPCT, gl_ScopeSubgroup, KHR_M, KHR_K, gl_MatrixUseA>
#define MAT_B coopmat<OPCT, gl_ScopeSubgroup, KHR_K, KHR_N, gl_MatrixUseB>
#define MAT_C coopmat<ACCT, gl_ScopeSubgroup, KHR_M, KHR_N, gl_MatrixUseAccumulator>
#define MAT_C16 coopmat<float16_t, gl_ScopeSubgroup, KHR_M, KHR_N, gl_MatrixUseAccumulator>
#define HMAT_A coopmat<float16_t, gl_ScopeSubgroup, KHR_M, KHR_HK, gl_MatrixUseA>
#define HMAT_B coopmat<float16_t, gl_ScopeSubgroup, KHR_HK, KHR_N, gl_MatrixUseB>

#define ROW_MAJOR gl_CooperativeMatrixLayoutRowMajor
#define COL_MAJOR gl_CooperativeMatrixLayoutColumnMajor

// The end of a step: the accumulator published as half. f16 accumulators already are; f32 ones hold the step's sum
// and are rounded here, RNE on the bit pattern (common.glsl roundF16), so no conversion's rounding mode is involved.
#if KHR_ACC32
#define ROUND_C(c)                                                                                               \
  {                                                                                                              \
    for (int r_ = 0; r_ < c.length(); ++r_) c[r_] = roundF16(c[r_]);                                             \
  }
#else
#define ROUND_C(c)
#endif

// c from / to f16 elements (shared or buffer memory). An f32 accumulator only ever holds values that are halves when
// it is loaded or stored, so both conversions are exact.
#if KHR_ACC32
#define LOAD_C(c, array, offset, stride)                                                                         \
  {                                                                                                              \
    MAT_C16 h_;                                                                                                  \
    coopMatLoad(h_, array, offset, stride, ROW_MAJOR);                                                           \
    c = MAT_C(h_);                                                                                               \
  }
#define STORE_C(c, array, offset, stride)                                                                        \
  {                                                                                                              \
    MAT_C16 h_ = MAT_C16(c);                                                                                     \
    coopMatStore(h_, array, offset, stride, ROW_MAJOR);                                                          \
  }
#else
#define LOAD_C(c, array, offset, stride) coopMatLoad(c, array, offset, stride, ROW_MAJOR);
#define STORE_C(c, array, offset, stride) coopMatStore(c, array, offset, stride, ROW_MAJOR);
#endif

// c += the 32 products of one k32 step, then the step's rounding. A is row-major at aOff (stride aS), B at bOff (stride
// bS, layout bL); aInc and bInc advance them by KHR_K elements of K, in the arrays' own units.
#define MMA_K32(c, A, aOff, aS, aInc, B, bOff, bS, bInc, bL)                                                     \
  {                                                                                                              \
    [[unroll]] for (uint s_ = 0u; s_ < 32u / KHR_K; ++s_) {                                                      \
      MAT_A a_;                                                                                                  \
      coopMatLoad(a_, A, (aOff) + s_ * (aInc), aS, ROW_MAJOR);                                                   \
      MAT_B b_;                                                                                                  \
      coopMatLoad(b_, B, (bOff) + s_ * (bInc), bS, bL);                                                          \
      c = coopMatMulAdd(a_, b_, c);                                                                              \
    }                                                                                                            \
    ROUND_C(c)                                                                                                   \
  }

// The same for one k16 step of the f16 GEMMs (f16 operands).
#define HMMA_K16(c, A, aOff, aS, aInc, B, bOff, bS, bInc, bL)                                                    \
  {                                                                                                              \
    [[unroll]] for (uint s_ = 0u; s_ < 16u / KHR_HK; ++s_) {                                                     \
      HMAT_A a_;                                                                                                 \
      coopMatLoad(a_, A, (aOff) + s_ * (aInc), aS, ROW_MAJOR);                                                   \
      HMAT_B b_;                                                                                                 \
      coopMatLoad(b_, B, (bOff) + s_ * (bInc), bS, bL);                                                          \
      c = coopMatMulAdd(a_, b_, c);                                                                              \
    }                                                                                                            \
    ROUND_C(c)                                                                                                   \
  }

// x + y of two accumulators holding halves, published as half (the split-K partition sums of docs/numerics.md)
#if KHR_ACC32
#define ADD_C(x, y)                                                                                              \
  {                                                                                                              \
    x = x + y;                                                                                                   \
    ROUND_C(x)                                                                                                   \
  }
#else
#define ADD_C(x, y) x = x + y;
#endif

// Sixteen E4M3 codes (a uvec4) as staged operands in a uvec4 array: as they are, or widened to sixteen halves (two
// words). `at` is the uvec4 index of the first word.
#if KHR_E4
#define STAGE16(array, at, codes) array[at] = codes;
#else
uint halfPairOfCodes(uint codes, uint shift) {
  return packFloat2x16(f16vec2(float16_t(e4m3ToF32((codes >> shift) & 0xffu)),
                               float16_t(e4m3ToF32((codes >> (shift + 8u)) & 0xffu))));
}
uvec4 halvesOfCodes(uvec2 codes) {
  return uvec4(halfPairOfCodes(codes.x, 0u), halfPairOfCodes(codes.x, 16u), halfPairOfCodes(codes.y, 0u),
               halfPairOfCodes(codes.y, 16u));
}
#define STAGE16(array, at, codes)                                                                                \
  {                                                                                                              \
    array[at] = halvesOfCodes((codes).xy);                                                                       \
    array[(at) + 1u] = halvesOfCodes((codes).zw);                                                                \
  }
#endif

#endif
