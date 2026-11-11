// (hookshot) DLSS 5 neural rendering through NGX directly: the same model as gpu/vk/vk_dlssnr.h, without Streamline.
//
// NGX knows the feature as NVSDK_NGX_Feature_Reserved18. The public SDK names nothing else about it -- no capability keys,
// no parameters -- so the parameter set below is the one openNR (https://github.com/clshortfuse/openNR, MIT, Copyright
// 2026 Carlos Lopez) drives it with in its native benchmark: the names are facts about the feature, written out here; none
// of openNR's code is copied. Nothing of NVIDIA's is in this repository, and nothing here reads or patches NVIDIA's files.
//
// Why a second path to the same model:
//  - Windows without Streamline. sl.dlss_nr's own checks may refuse an RTX 20, 30 or 40 even where the model itself (a
//    community-patched nvngx_dlssnr.dll) would run on it; through NGX, the only gate is the one inside that file.
//  - Linux. Streamline is Windows-only, but NGX is not: the Linux NGX core loads features as libnvidia-ngx-<name>.so* and
//    already answers for DLSS-NR's capability keys. What is missing, as of SDK 310.9.16 and driver 615, is NVIDIA's
//    libnvidia-ngx-dlssnr.so itself. Until it ships this path says so plainly; the day it does, it should simply run.
//
// It shares the one NGX session DLSS super resolution owns (dlss_ngx_session, gpu/vk/vk_dlss.h; gpu/vk/vk_ngx.h says why
// there is one), and it is fed exactly what the Streamline path is fed: the finished frame, a flat depth, no motion.
//
// `nr::` below is which of the two runs: the Insert menu's Neural tab "Backend" row (opts.nr_backend: Auto, Streamline, NGX
// direct), or RX_VK_DLSS_NR_BACKEND=auto|streamline|ngx over it. Auto takes Streamline where it is attached and NGX direct
// otherwise, which on Linux is always.
#pragma once

#include <cstddef>
#include <cstdint>

#include "vk_internal.h"

namespace mkwvk {
/// (super build) The scene's own frame inputs, where the model runs on the jittered render-size scene (the Neural tab's Stage:
/// Before upscaler, vk_temporal.cpp): the temporal chain's depth (R32_SFLOAT) and motion (R16G16_SFLOAT, render pixels, as DLSS
/// takes them), the jitter in render pixels (x right, y down) and whether the history is to go (a cut, a gap). Without it (the
/// finished frame, at the present or before the HUD) the model gets a flat depth, no motion and no jitter, which is what a
/// finished frame is
struct NrFrame {
    VkImage depth = VK_NULL_HANDLE;
    VkImageView depth_view = VK_NULL_HANDLE;
    VkImage motion = VK_NULL_HANDLE;
    VkImageView motion_view = VK_NULL_HANDLE;
    float jitter[2]{};
    bool reset = false;
};
}  // namespace mkwvk

namespace mkwvk::dlssnr_ngx {

/// Whether this build has an NGX SDK (HOOKSHOT_NGX) to compile this path against.
[[nodiscard]] bool built_with_sdk();

/// Whether NGX reports the feature on this GPU and driver. Asked of NGX once, after the device exists; the answer and why
/// are logged once.
[[nodiscard]] bool capable();

/// Why it cannot run, in a line the menu can show: "" while it can, or before it has been asked.
[[nodiscard]] const char* why();

/// The device extensions NGX says feature 18 needs here, asked before the device exists (a device made without a
/// feature's extensions is where CreateFeature answers FAIL_PlatformError). How many names were written, at most `max`;
/// zero in a build without NGX, with RX_VK_DLSS_NR=0, or where NGX does not answer.
size_t device_extensions(VkInstance instance, VkPhysicalDevice gpu, const char** out, size_t max);

/// As dlssnr::apply: one uplift of `src` (GENERAL, sw x sh), recorded into `cmd`, and the image to present in its place
/// (GENERAL, the same size, this path's own) with its view in `out_view`; VK_NULL_HANDLE when the model did not run, and
/// the caller presents `src` as it would have.
[[nodiscard]] VkImage apply(VkCommandBuffer cmd, VkImage src, uint32_t sw, uint32_t sh, VkImageView* out_view,
                            const NrFrame* frame = nullptr);

/// Releases the feature and its images. Before the device goes.
void shutdown();

} // namespace mkwvk::dlssnr_ngx

namespace mkwvk::nr {

enum Backend : int { AUTO = 0, STREAMLINE = 1, NGX = 2, OPEN = 3 };   // (OPEN: OpenDLSS-NR, gpu/vk/vk_opennr.h)

/// Whether this build can offer neural rendering at all: Streamline's headers (Windows) or an NGX SDK.
[[nodiscard]] bool built();

/// The backend asked for (the row, or RX_VK_DLSS_NR_BACKEND over it), and the one that runs for it: STREAMLINE or NGX.
/// Auto is Streamline when Streamline is attached with the feature, NGX direct otherwise.
[[nodiscard]] int asked_backend();
[[nodiscard]] int running_backend();

/// The one call: the running backend's apply (frame: the scene's own inputs, where there are some; NrFrame). One caller at a
/// time; the backend that stops running (the row off, another picked) is released here
[[nodiscard]] VkImage apply(VkCommandBuffer cmd, VkImage src, uint32_t sw, uint32_t sh, VkImageView* out_view,
                            const NrFrame* frame = nullptr);

/// Why the running backend cannot run, for the menu: "" while it can or is not known yet.
[[nodiscard]] const char* why();

} // namespace mkwvk::nr
