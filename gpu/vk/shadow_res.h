// Shadow resolution (the Insert menu's Graphics tab): which depth textures are taken for shadow maps and made larger, the side
// their images get, and the game's rectangles on them. vk_texture.cpp create, vk_draw.cpp scissor_rect, vk_xfer.cpp. No Vulkan
// here, so tests/test_shadow_resolution.py compiles it as it is.
#pragma once

#include <cstdint>

namespace shadow_res {

/// Whether a depth texture of this shape is taken for a shadow map. OoT's are square depth targets of one level: the characters'
/// (768 and less) and the sun's cascades (1600 x 1600 x 3, D16, docked). Square, 2D or a 2D array, one level, single sample,
/// none of its levels left out and not sparse, 256 to 4096 a side and a multiple of 32 (the scene's depth targets that follow
/// dynamic resolution have odd sizes)
inline bool qualifies(bool depth, bool two_d_or_array, uint32_t samples, uint32_t levels, bool skipped_or_sparse, uint32_t w,
                      uint32_t h) {
    return depth && two_d_or_array && samples == 1 && levels == 1 && !skipped_or_sparse && w == h && w >= 256 && w <= 4096 &&
           w % 32 == 0;
}

/// The side the image of a qualifying map of side w is made at, for the row's factor want (1: the game's): w x want, at most
/// max_dim, and stepped down by 256 until its layers of texel bytes take 1 GB at most. w itself when that is no larger
inline uint32_t side(uint32_t w, uint32_t layers, uint32_t texel, float want, uint32_t max_dim) {
    if (want <= 1.0f) return w;
    uint32_t s = (uint32_t)(w * want + 0.5f);
    if (s > max_dim) s = max_dim;
    while (s > w && (uint64_t)s * s * (layers ? layers : 1) * (texel ? texel : 1) > (1ull << 30)) s -= 256;
    return s > w ? s : w;
}

/// The image's size for a texture of the game's size n made rs times larger (vk_texture.h image_width / image_height)
inline uint32_t image_size(uint32_t n, float rs) { return rs == 1.0f ? n : (uint32_t)(n * rs + 0.5f); }

/// A coordinate of the game's (a viewport's, scissor's or region's edge) on a map made scale times larger, to the nearest texel:
/// the same game edge always lands on the same texel, so two rectangles that meet still meet
inline int64_t edge(int64_t v, float scale) {
    return scale == 1.0f ? v : (int64_t)((double)v * scale + (v >= 0 ? 0.5 : -0.5));
}

}  // namespace shadow_res
