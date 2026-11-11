// (hookshot) The NVIDIA Streamline session DLSS 5 neural rendering comes through.
//
// Why a second NVIDIA path at all, when gpu/vk/vk_ngx.h already owns an NGX session: DLSS 5 neural
// rendering (DLSS-NR) is not a feature a program can create for itself through NGX. The public NGX
// SDK does not name it, and the only entry point that configures it -- slDLSSNRSetOptions -- lives
// inside NVIDIA's sl.dlss_nr plugin, reached through sl.interposer.dll. So super resolution and
// frame generation keep their direct NGX path, which is measured and works, and neural rendering
// gets this one beside it. Only the features this session asks Streamline to load are loaded, and
// it asks for neural rendering alone, so Streamline's own DLSS and DLSS-G plugins never come up
// against our NGX session over the same device.
//
// Windows only, and that is Streamline's limit rather than ours: the interposer and every sl.*
// plugin ship as DLLs. On Linux every function here answers "no" and the renderer is exactly the
// renderer it was; a player on Linux reaches DLSS-NR by running the Windows build under Proton,
// which is where it was measured (docs/neural-rendering.md).
//
// Nothing of NVIDIA's is in this repository and nothing of NVIDIA's is needed to build it. Only
// Streamline's MIT-licensed headers are compiled in, behind the build-time gate RX_VK_STREAMLINE
// (nx2/build.py streamline_sdk_path), and sl.interposer.dll is the player's own copy, loaded from
// their PC at run time and checked for NVIDIA's signature before it is trusted.
#pragma once

#include <cstddef>
#include <cstdint>

#include "vk_internal.h"

namespace mkwvk::streamline {

/// Whether this build has Streamline's headers at all. Compile-time, so the menu can say "this
/// build was made without it" rather than "your GPU cannot do it", which are different problems
/// with different fixes.
[[nodiscard]] bool built_with_sdk();

/// Loads sl.interposer.dll and starts Streamline with the neural rendering plugin. Must be called
/// *before* vkCreateInstance, because what it loads decides which instance and device extensions
/// the device has to be created with. False, with one line logged, where the player has no
/// interposer, where it is not signed by NVIDIA, or where Streamline refused to start; the renderer
/// then comes up without neural rendering and everything else is unaffected.
///
/// It does *not* ask whether the player has neural rendering switched on, and that is deliberate:
/// the device has to be created for the feature before the menu's options are even read, so a
/// session that waited for the switch could never honour it without a restart. The switch is read
/// per frame instead (dlssnr::asked), and the cost of being wrong here is that a player with
/// NVIDIA's files present and the row off has loaded a plugin they are not using. RX_VK_DLSS_NR=0
/// is the way to not load it at all.
///
/// Asked more than once is free: the answer is kept.
bool start();

/// The extensions neural rendering asks for, as Streamline states them. Both return how many names
/// were written, at most `max`; the names last as long as the process. Zero means Streamline is not
/// running, which leaves the instance and device created exactly as they were. Hard-coding either
/// list would be wrong on the next driver, which is why they are asked for rather than written out.
size_t instance_extensions(const char** out, size_t max);
size_t device_extensions(const char** out, size_t max);

/// Queues of the graphics family Streamline wants beyond the renderer's own, created after it. Zero
/// when Streamline is not running.
[[nodiscard]] uint32_t extra_queues();

/// Hands the created device to Streamline and asks whether neural rendering can actually run on it.
/// `extra_queues_made` is how many of extra_queues() the device was created with: fewer than it
/// asked for is reported and the feature is turned off, rather than letting it fail later inside a
/// frame. False when no feature survives, and the session is stopped.
bool attach_device(VkInstance instance, VkPhysicalDevice gpu, VkDevice device,
                   uint32_t graphics_family, uint32_t extra_queues_made);

/// Whether DLSS 5 neural rendering can run: Streamline started, the device is attached, and both
/// Streamline and the driver report the feature as supported here.
[[nodiscard]] bool nr_available();
/// Why neural rendering is not available, in a line the menu can show: "" while it is, or before the device is
/// attached. The log has the long form (vk_streamline.cpp explain_refusal).
[[nodiscard]] const char* nr_why();

/// Ends the session. Safe when nothing was started; call before the device is destroyed.
void stop();

} // namespace mkwvk::streamline
