// (hookshot) Camera-only motion vectors from the scene's depth, for a temporal upscaler.
//
// Ocarina hands its own upscaler one colour texture and nothing else (docs/upscaling.md), so motion
// has to be derived. This reprojects each pixel through the inverse of this frame's view projection
// and the previous frame's, which is right for a world that does not move and ghosts on anything
// that moves by itself. The shader is gpu/vk/shaders/camera_motion.comp.
#pragma once

#include <cstdint>

#include "vk_internal.h"

namespace mkwvk::camera_motion {

/// A view projection, row major. record() transposes it for the shader, which loads a mat4 column
/// by column.
struct Matrix {
    float m[16]{};
};

/// Records the pass for this frame and returns the motion target, or null when it cannot run: no
/// depth, no matrices yet, or the pipeline could not be made. The caller keeps nothing; the
/// previous frame's matrix is remembered here.
[[nodiscard]] VkImageView record(VkCommandBuffer cmd, VkImageView depth, VkFormat depth_format,
                                 uint32_t width, uint32_t height, const Matrix& view_projection);

/// Forgets the previous frame, so the next pass writes zero motion rather than reprojecting through
/// a matrix from before a cut or a load.
void reset();

/// The motion target this module writes, for a backend that is handed resources rather than views
/// (gpu/vk/vk_dlss.h). Null and VK_FORMAT_UNDEFINED before the first record().
[[nodiscard]] VkImage motion_image();
[[nodiscard]] VkFormat motion_format();

void shutdown();

} // namespace mkwvk::camera_motion
