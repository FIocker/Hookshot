// (hookshot) DLSS 5 neural rendering (DLSS-NR): NVIDIA's model over the finished frame.
//
// What it is, because the name invites the wrong guess: DLSS-NR is *not* ray reconstruction. Ray
// reconstruction (sl.dlss_d / NGX's DLSSD) denoises a ray-traced image and needs a G-buffer, albedo,
// normals and roughness, which a 2026 recompilation of a 1998 game does not have and never will.
// DLSS-NR is an "uplift" pass: Streamline's own headers name its buffers
// kBufferTypeUpliftInputColor and kBufferTypeUpliftOutputColor, and the only other things it is
// handed are depth and motion vectors. Colour in, colour out. That is why it runs on rasterized
// content at all, and it is why it can sit here, at the end of the frame, over the picture the
// renderer is about to show -- which is where RenoDX and dlss5-anywhere put it too.
//
// So this backend does not touch the 3D scene. It takes the window's image, hands it to the model,
// and gives back the image to present in its place. Depth and motion are tagged because Streamline
// asks for them, with a flat depth and no motion: at this point in the frame the picture is already
// upscaled and already has the HUD on it, so the scene's real depth buffer does not describe it and
// feeding it would be a lie the model would act on.
//
// Windows only -- see gpu/vk/vk_streamline.h for why, and for the licensing: nothing of NVIDIA's is
// in this repository and nothing of NVIDIA's is linked. On every other build each function here
// answers "no" and the frame goes to the window as it always did.
#pragma once

#include <cstdint>

#include "vk_internal.h"

namespace mkwvk::dlssnr {

/// Whether this build has Streamline's headers at all (compile-time). The menu says "this build was
/// made without it" rather than "your card cannot do it", which are different problems.
[[nodiscard]] bool built_with_sdk();

/// Whether the feature could run: Streamline started, the device is attached and both Streamline and
/// the driver report DLSS-NR as supported here. Cached, because none of it changes while the
/// renderer lives.
[[nodiscard]] bool capable();

/// Whether the player has it switched on *right now*. Read live, every frame: the Insert menu's
/// Neural rendering row, or RX_VK_DLSS_NR. A backend that cached this in a `static const` is how the
/// upscaling rows shipped dead once (docs/upscaling.md).
[[nodiscard]] bool asked();

/// Records one uplift of `src` (GENERAL, sw x sh) and returns the image to present in its place --
/// GENERAL, the same size, this backend's own, with its view in `out_view`. Returns VK_NULL_HANDLE
/// when the model did not run,
/// and the caller then presents `src` exactly as it would have: the frame is never lost to this.
///
/// `src` is only read. Writing the model's result back over the game's own image was the other way
/// to do this and it is not done, because the window texture is the game's and a frame the game
/// reads back would come back changed.
[[nodiscard]] VkImage apply(VkCommandBuffer cmd, VkImage src, uint32_t sw, uint32_t sh,
                            VkImageView* out_view);

/// Releases the feature's resources. Before the device goes.
void shutdown();

} // namespace mkwvk::dlssnr
