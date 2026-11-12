// (hookshot) DLSS 5 neural rendering through NGX: NVIDIA's model over the picture, inside the temporal chain (vk_temporal.cpp)
// before the game draws its HUD, or on the render-size scene before the upscaler (the Neural tab's Stage row), and at the
// present for frames the chain did not make.
//
// NGX knows the feature as NVSDK_NGX_Feature_Reserved18. The public SDK names nothing else about it -- no capability keys, no
// parameters -- so the parameter set below is the one openNR (https://github.com/clshortfuse/openNR, MIT, Copyright 2026
// Carlos Lopez) drives it with in its native benchmark: the names are facts about the feature, written out here; none of
// openNR's code is copied. Nothing of NVIDIA's is in this repository, and nothing here reads or patches NVIDIA's files.
//
// It is a feature on DLSS's one NGX session (vk_dlss.h dlss_ngx_ready: no second NVSDK_NGX_VULKAN_Init), with DLSS's runtime
// folders (its runtime, nvngx_dlssnr.dll / libnvidia-ngx-dlssnr.so, goes where nvngx_dlss.dll does), and its NGX calls are
// made under DLSS Frame Gen's lock (dlss_ngx_feature_mutex), so the two features' calls never overlap. Built only with NVIDIA's
// NGX SDK (HOOKSHOT_NGX); without it, these say it is not there.
#pragma once

#include <cstddef>
#include <cstdint>

#include "vk_internal.h"

namespace mkwvk {

/// The scene's own frame inputs, where the model runs on the jittered render-size scene (Stage: Before upscaler): the temporal
/// chain's depth (R32_SFLOAT; depth_reversed: 1 - d, as FSR 3.1 on the chain has it) and motion (R16G16_SFLOAT, render pixels,
/// as DLSS takes them), the jitter in render pixels (x right, y down) and whether the history is to go (a cut, a gap). Without
/// it (the finished picture, before the HUD or at the present) the model gets a flat depth, no motion and no jitter, which is
/// what a finished picture is
struct NrFrame {
    VkImage depth = VK_NULL_HANDLE;
    VkImageView depth_view = VK_NULL_HANDLE;
    VkImage motion = VK_NULL_HANDLE;
    VkImageView motion_view = VK_NULL_HANDLE;
    float jitter[2]{};
    bool depth_reversed = false;
    bool reset = false;
};

namespace nr {

/// Whether this build can offer neural rendering at all (NVIDIA's NGX SDK, HOOKSHOT_NGX)
[[nodiscard]] bool built();

/// One uplift of `src` (GENERAL, sw x sh), recorded into `cmd`, and the image to use in its place (GENERAL, the same size, this
/// path's own) with its view in `out_view`; VK_NULL_HANDLE when the model did not run, and the caller uses `src` as it would
/// have. frame: the scene's own inputs, where there are some (NrFrame). One caller at a time (the chain's thread and the
/// display thread both call it); when the row is turned off, what the feature held is let go here
[[nodiscard]] VkImage apply(VkCommandBuffer cmd, VkImage src, uint32_t sw, uint32_t sh, VkImageView* out_view,
                            const NrFrame* frame = nullptr);

/// Why it cannot run, for the menu: "" while it can or is not known yet
[[nodiscard]] const char* why();

}  // namespace nr
}  // namespace mkwvk
