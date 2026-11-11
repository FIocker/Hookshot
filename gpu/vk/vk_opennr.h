// (hookshot) Open neural rendering: the DLSS 5 neural rendering network run by OpenDLSS-NR, without NVIDIA's runtime.
//
// The third backend behind the Neural tab, beside Streamline (gpu/vk/vk_dlssnr.cpp) and NGX direct. Streamline runs
// NVIDIA's own model through NVIDIA's own DLLs and is Windows-only; this runs a bit-exact Vulkan reimplementation of
// the same network (https://github.com/maanHimself/OpenDLSS-NR, MIT, vendored in third_party/opendlss-nr) on the
// renderer's own device, so it runs natively on Linux too. On an NVIDIA card with VK_NV_cooperative_matrix2 it runs
// upstream's kernels; anywhere else with VK_KHR_cooperative_matrix -- AMD's RDNA 3 and 4 above all -- it runs the khr
// route (gpu/vk/opennr/khr: upstream's unfused graph on kernels that need only the KHR matrices, docs/neural-rendering.md).
//
// THE WEIGHTS ARE THE PLAYER'S. The network reads a model directory -- manifest.json and its stage files, upstream's
// docs/weights.md -- that nothing public produces, and Hookshot provides no way to get one: nothing here extracts,
// converts or downloads model data. RX_VK_OPENNR_MODEL=<dir> names it, else `nr-model` in the game folder. Without
// one the log and the menu say so and every frame goes to the window as it always did.
//
// Same place and contract as the Streamline backend: present_image (vk_replay.cpp) calls dlssnr::apply on the
// finished frame at display size, before the menu is drawn, and dlssnr::apply hands the frame here when this backend
// is the selected one. Everything is recorded into that frame's command buffer; nothing is allocated per frame.
#pragma once

#include <cstdint>
#include <vector>

#include "vk_internal.h"

namespace mkwvk {
struct NrFrame;   // (vk_dlssnr_ngx.h)
}

namespace mkwvk::opennr {

/// Whether the device is made for this backend this run: the Backend row's Open (opts.nr_backend 3), RX_VK_NR_BACKEND=open,
/// or a model folder in place at the start (so the row's Auto can fall to it). Fixed at device creation.
[[nodiscard]] bool selected();
/// Whether it can run now: selected(), the GPU has what the network needs, and the model was not refused. The Backend
/// row's Auto takes it after Streamline and NGX direct (gpu/vk/vk_dlssnr_ngx.cpp nr::running_backend).
[[nodiscard]] bool ready();

/// Device creation (vk_main.cpp create_device), only when selected(): the extensions the network needs, where the
/// GPU has them...
void device_extensions(std::vector<const char*>& want, const std::vector<VkExtensionProperties>& have);
/// ... their feature structs, chained before vkGetPhysicalDeviceFeatures2 so the query fills them and the device is
/// made with what the GPU supports (the renderer's own convention for its features) ...
void device_features(void**& tail);
/// ... and, once the device exists, whether the network can run on it: what is missing is named in the log.
void device_created();

/// Whether the player has neural rendering on right now (the Neural rendering row, or RX_VK_DLSS_NR), read live.
[[nodiscard]] bool asked();

/// Why it is not running, for the Neural rendering row's hint; empty when it runs or was not asked for.
[[nodiscard]] const char* why();

/// The same contract as dlssnr::apply: `src` (GENERAL, sw x sh) is only read; the result is this backend's own
/// image, GENERAL, the same size, with its view in `out_view`; VK_NULL_HANDLE when the network did not run, and the
/// caller then presents `src` exactly as it would have.
/// frame: the scene's own inputs (vk_dlssnr_ngx.h NrFrame); from them, a reset drops the history
[[nodiscard]] VkImage apply(VkCommandBuffer cmd, VkImage src, uint32_t sw, uint32_t sh, VkImageView* out_view,
                            const NrFrame* frame = nullptr);

/// Releases everything. Before the device goes.
void shutdown();

} // namespace mkwvk::opennr
