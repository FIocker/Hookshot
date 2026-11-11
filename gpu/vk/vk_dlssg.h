// (hookshot) NVIDIA DLSS Frame Generation as a source of the frames shown between the game's.
//
// The renderer already shows made frames: vk_framegen.cpp interpolates between the last two
// presented frames and the display thread paces them (gpu/vk/vk_framegen.h). That interpolation is
// made from colour alone, which is all it has. This backend answers the same question with the
// game's motion and depth and NVIDIA's network, and with MultiFrameCount it can make more than one
// frame between two real ones -- which is the only way past a frame rate the game's own simulation
// limits.
//
// It slots in where the hand-written interpolation does, and where it refuses -- no NGX, no runtime,
// a GPU that does not report Frame Generation, inputs the seam could not assemble -- that
// interpolation runs exactly as before. Nothing here changes what the game submits.
//
// Nothing of NVIDIA's is in this repository and nothing of NVIDIA's is needed to build it: the NGX
// half is inside `HOOKSHOT_NGX` and the session is shared (gpu/vk/vk_ngx.h). docs/upscaling.md has
// the licensing decision and the input contract.
#pragma once

#include <cstdint>

#include "vk_internal.h"

namespace mkwvk::dlssg {

/// One evaluation's inputs.
///
/// The colour is the frame the game has just finished, at the size it is presented; the motion and
/// depth are the scene's, at the render size, with `motion_scale` taking a vector to render pixels
/// and `depth_inverted` saying which way the depth runs. The two clip matrices are what let the
/// network place the made frame in time: `clip_to_prev_clip` carries this frame's clip space to the
/// one before it, and `prev_clip_to_clip` is its inverse. Both are row major, as everywhere else in
/// this renderer, and are transposed on the way in.
struct Inputs {
    VkImageView color = VK_NULL_HANDLE;
    VkImage color_image = VK_NULL_HANDLE;
    VkFormat color_format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;

    VkImageView depth = VK_NULL_HANDLE;
    VkImage depth_image = VK_NULL_HANDLE;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    VkImageView motion = VK_NULL_HANDLE;
    VkImage motion_image = VK_NULL_HANDLE;
    VkFormat motion_format = VK_FORMAT_UNDEFINED;
    uint32_t render_width = 0, render_height = 0;

    float motion_scale[2]{1.0f, 1.0f};
    float jitter[2]{};
    float clip_to_prev_clip[16]{};
    float prev_clip_to_clip[16]{};
    /// The rest of the set NGX's own helper sets. An evaluation with only the two above is refused
    /// with NVSDK_NGX_Result_FAIL_MissingInput, which is what a run saw, so these are not optional
    /// however optional the struct they live in is called.
    float camera_view_to_clip[16]{};   ///< the projection
    float clip_to_camera_view[16]{};   ///< its inverse
    float clip_to_lens_clip[16]{};     ///< identity here: this game has no lens distortion
    float camera_pos[3]{};
    float camera_up[3]{};
    float camera_right[3]{};
    float camera_forward[3]{};
    bool have_matrices = false;

    float near_plane = 0.1f;
    float far_plane = 1000.0f;
    float fov_y = 60.0f * 3.14159265358979f / 180.0f;
    float aspect = 16.0f / 9.0f;
    bool depth_inverted = true;
    bool colour_hdr = true;
    /// The motion already includes the camera's movement. Ours does: it *is* the camera's movement
    /// (gpu/vk/vk_fsr3_inputs.h, shaders/fsr3_prep.comp), which is why this defaults true and why
    /// saying otherwise would have the network apply the camera twice.
    bool camera_motion_included = true;
    bool reset = false;

    /// How many frames to make between this one and the last, and which of them this evaluation is
    /// (1-based). 1 and 1 is ordinary 2x generation; a count of 2 or 3 is what DLSS 4 calls
    /// multi-frame generation, and needs an evaluation per made frame.
    uint32_t frames_made = 1;
    uint32_t frame_index = 1;
};

/// Whether frame generation can run here at all: asked for (`RX_VK_DLSSG=1`), built with an NGX SDK,
/// the NGX runtime loaded, and this driver and GPU reporting Frame Generation. Asked once; the one
/// line saying why it is false is logged once.
[[nodiscard]] bool supported();

/// The device extensions frame generation needs here, asked of NGX itself before the device exists.
/// Super resolution's set is not the same set: a device made for one and not the other is where
/// `NVSDK_NGX_VULKAN_CreateFeature` answers FAIL_PlatformError with the feature still reporting
/// itself available, which is exactly what a run saw. Returns how many names were written, at most
/// `max`; zero where frame generation was not asked for, this build has no NGX SDK, or NGX does not
/// answer. The names are NGX's own storage and last as long as the process.
size_t device_extensions(VkInstance instance, VkPhysicalDevice gpu, const char** out, size_t max);

/// How many frames the run asks to be made between two of the game's: `RX_VK_DLSSG=<1-3>`, 1 by
/// default. Zero where frame generation is not asked for at all.
[[nodiscard]] uint32_t frames_wanted();

/// Makes the images the made frames resolve into and remembers these extents, re-making them when
/// the extents change. False when anything could not be made, which leaves generation unavailable
/// rather than half-built.
[[nodiscard]] bool prepare(uint32_t width, uint32_t height, uint32_t render_width,
                           uint32_t render_height, VkFormat present_format);

/// Records one made frame. False when the inputs are not usable or NGX refused, which leaves the
/// caller to show whatever it would otherwise have shown.
[[nodiscard]] bool generate(VkCommandBuffer cmd, const Inputs& in);

/// The made frame of the last `generate`, by index (1-based, as `Inputs::frame_index`), or null.
[[nodiscard]] VkImage made_image(uint32_t index);
[[nodiscard]] VkImageView made_view(uint32_t index);

/// Forgets the sequence, so the next evaluation is told to reset: a cut, a load, a resize.
void reset();

/// Frees everything this backend made. The shared NGX session is not ended here.
void shutdown();

} // namespace mkwvk::dlssg
