// (hookshot) What the open neural rendering backend's two frame passes share (opennr_pre.comp, opennr_post.comp).
//
// The arithmetic here follows OpenDLSS-NR's demo frame (https://github.com/maanHimself/OpenDLSS-NR, MIT, Copyright
// 2026 maan: demo/shaders/nr_common.glsl, nr_preprocess.comp and nr_composite.comp, and its docs/frame.md): the
// half publications are spelled on the bit pattern because NVIDIA's GLSL compiler removes float(float16_t(x)), the
// Box-Muller noise keeps upstream's instruction order (its approximate transcendentals make it order dependent),
// and the history is truncated to the half grid, not rounded, because the next frame's input reads it.

// The binding set both passes use (gpu/vk/vk_opennr.cpp makes one layout for the two)
layout(binding = 0, rgba8) uniform readonly image2D in_color;      // the frame, display-encoded code values
layout(binding = 1) uniform sampler2D history_prev;                // the previous output, half code values
layout(binding = 2) uniform sampler2D motion;                      // render pixels toward the previous frame
layout(std430, binding = 3) buffer Features { vec4 features[]; };  // f32 [fullHeight][fullWidth][16], 4 vec4 a pixel
layout(std430, binding = 4) readonly buffer Head { float head[]; };    // f32 [fullHeight][fullWidth][4]
layout(binding = 5, rgba8) uniform writeonly image2D out_color;
layout(binding = 6, rgba16f) uniform writeonly image2D history_next;

layout(push_constant) uniform Push {
    uint full_width, full_height, valid_width, valid_height;
    uint seed, history_valid, motion_valid, auto_mask;
    float style, local_tone, local_structure, skin;
    vec2 motion_size;   // the motion target's size: the scene's render size
    float blend_scale, intensity;
} pc;

float roundF16(float value) {
    uint bits = floatBitsToUint(value);
    uint sign = bits & 0x80000000u;
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) return value;
    if (magnitude >= 0x477ff000u) return uintBitsToFloat(sign | 0x7f800000u);
    if (magnitude < 0x38800000u) {
        float scaled = roundEven(uintBitsToFloat(magnitude) * 16777216.0);
        return uintBitsToFloat(sign) + uintBitsToFloat(sign | floatBitsToUint(scaled * 0.000000059604644775390625));
    }
    uint lsb = (magnitude >> 13u) & 1u;
    uint rounded = (magnitude + 0xfffu + lsb) & ~0x1fffu;
    return uintBitsToFloat(sign | rounded);
}

float truncateHalf(float value) {
    uint bits = floatBitsToUint(value);
    uint signBit = (bits >> 16u) & 0x8000u;
    uint exponent = (bits >> 23u) & 0xffu;
    uint mantissa = bits & 0x7fffffu;
    uint halfBits;
    if (exponent == 0xffu) {
        halfBits = signBit | (mantissa != 0u ? 0x7e00u : 0x7c00u);
    } else {
        int halfExponent = int(exponent) - 112;
        if (halfExponent >= 31) halfBits = signBit | 0x7c00u;
        else if (halfExponent <= 0) {
            if (halfExponent < -10) halfBits = signBit;
            else halfBits = signBit | ((mantissa | 0x800000u) >> uint(14 - halfExponent));
        } else halfBits = signBit | (uint(halfExponent) << 10u) | (mantissa >> 13u);
    }
    return unpackHalf2x16(halfBits).x;
}

// A code value as the network's lanes 4-9 take it: f16((f16(c) - 0.5) * 0.125)
vec3 centred(vec3 code) {
    return vec3(roundF16(roundF16(roundF16(code.r) - 0.5) * 0.125), roundF16(roundF16(roundF16(code.g) - 0.5) * 0.125),
                roundF16(roundF16(roundF16(code.b) - 0.5) * 0.125));
}

// Where `pixel` of the frame was in the previous one, in uv. The motion is the scene's, at its render size, which
// the window's frame fills; HUD pixels get the scene's motion under them, which is wrong for them and which the
// network's own blend logit is left to reject (docs/neural-rendering.md says so).
vec2 history_uv(vec2 pixel) {
    vec2 uv = (pixel + 0.5) / vec2(pc.valid_width, pc.valid_height);
    ivec2 size = textureSize(motion, 0);
    ivec2 at = min(ivec2(uv * vec2(size)), size - 1);
    return uv + texelFetch(motion, at, 0).xy / pc.motion_size;
}

bool has_history(vec2 pixel) {
    if (pc.history_valid == 0u) return false;
    if (pc.motion_valid == 0u) return true;   // (no motion this frame: the history is taken at the same pixel)
    vec2 uv = history_uv(pixel);
    return all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0)));
}

// Five-tap Catmull-Rom reconstruction of the history at the reprojected position (upstream's historySample): a
// box or bilinear filter softens the image frame over frame, because the history is fed back into the input.
vec3 history_sample(vec2 pixel) {
    vec2 valid = vec2(pc.valid_width, pc.valid_height);
    vec2 uv = pc.motion_valid != 0u ? history_uv(pixel) : (pixel + 0.5) / valid;
    vec2 position = uv * valid;
    vec2 base = floor(position - 0.5) + 0.5;
    vec2 f = clamp(position - base, vec2(0.0), vec2(1.0));
    vec2 square = f * f; vec2 cube = f * square;
    vec2 w0 = fma(f + cube, vec2(-0.5), square);
    vec2 w1 = (cube * 1.5 - square * 2.5) + 1.0;
    vec2 w3 = (cube - square) * 0.5;
    vec2 w2 = ((1.0 - w0) - w1) - w3;
    vec2 middle = w1 + w2;
    vec2 low = clamp(base - 1.0, vec2(0.5), valid - 0.5) / valid;
    vec2 center = clamp(base + w2 / middle, vec2(0.5), valid - 0.5) / valid;
    vec2 high = clamp(base + 2.0, vec2(0.5), valid - 0.5) / valid;
    vec3 left = textureLod(history_prev, vec2(low.x, center.y), 0.0).rgb;
    vec3 top = textureLod(history_prev, vec2(center.x, low.y), 0.0).rgb;
    vec3 mid = textureLod(history_prev, center, 0.0).rgb;
    vec3 bottom = textureLod(history_prev, vec2(center.x, high.y), 0.0).rgb;
    vec3 right = textureLod(history_prev, vec2(high.x, center.y), 0.0).rgb;
    float a = w0.x * middle.y; float b = w0.y * middle.x;
    float c = middle.x * middle.y; float d = w3.y * middle.x; float e = w3.x * middle.y;
    vec3 value = fma(left, vec3(a), top * b);
    value = fma(mid, vec3(c), value);
    value = fma(bottom, vec3(d), value);
    value = fma(right, vec3(e), value);
    return value * (1.0 / (e + (d + (c + (a + b)))));
}
