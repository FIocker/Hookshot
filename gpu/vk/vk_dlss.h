// NVIDIA DLSS Super Resolution, one of the temporal upscaler's engines (vk_temporal.h; vk_dlss.cpp). Built in only with
// NVIDIA's NGX SDK (HOOKSHOT_NGX, scripts/ngx_sdk.py: Linux and Windows); without it these say DLSS is not there and the
// game's network runs.
#pragma once

#include <string>
#include <vector>

#include "vk_internal.h"

namespace mkwvk {

struct TemporalEngine;

/// Whether this build has DLSS in it (HOOKSHOT_NGX)
bool dlss_built();
/// Why DLSS cannot be chosen here (the Insert menu's Upscaler row says it), DLSS_OK when it can or is not known yet
enum DlssWhy { DLSS_OK, DLSS_NOT_BUILT, DLSS_ENV_OFF, DLSS_NEEDS_RTX, DLSS_NEEDS_DRIVER, DLSS_NO_RUNTIME,
               DLSS_RUNTIME_NOT_ASCII, DLSS_WHY_COUNT };
DlssWhy dlss_why();
/// Asks NGX once whether DLSS runs on this device, so dlss_why knows before the first upscale (the menu, when it opens).
/// Nothing off an NVIDIA device with NGX's extensions, or in a build without DLSS
void dlss_probe();
/// (hookshot) NGX started on this device, by this file, for NGX's other features to use the same session: DLSS frame
/// generation (vk_ngx.cpp, vk_dlssg.cpp) and DLSS 5 neural rendering (vk_dlssnr_ngx.cpp). NGX is initialized once per device
/// and shut down once; a second NVSDK_NGX_VULKAN_Init_with_ProjectID beside this one is what it avoids. True even where super
/// resolution itself is not available, since the session is. Starts it if dlss_probe has not yet (and RX_VK_DLSS is not 0)
bool dlss_ngx_session();
/// (hookshot) Whether a file whose name begins `name` ("nvngx_dlssnr.dll", "libnvidia-ngx-dlssnr.so") is in one of the folders
/// NGX is told to look for its features' runtimes in (ngx_init's: RX_VK_DLSS_DIR, the game folder's ngx, then
/// %LOCALAPPDATA%\Hookshot\ngx or $XDG_DATA_HOME/hookshot/ngx), or the game folder itself, which NGX also searches. `where`
/// gets the folders, in UTF-8 for the log; `found` the first such file's path. False and both empty in a build without DLSS
bool dlss_ngx_runtime(const char* name, std::string* where, std::string* found);
/// The instance's extensions DLSS needs, added to `want` when the loader has them (all: what it has)
void dlss_instance_extensions(std::vector<const char*>& want, const std::vector<VkExtensionProperties>& all);
/// The device's extensions DLSS needs, added to `want` when the device has them
void dlss_device_extensions(std::vector<const char*>& want, const std::vector<VkExtensionProperties>& all);
/// DLSS as the temporal upscaler's engine (DLSS and DLAA on the Upscaler row); nullptr in a build without it
const TemporalEngine* dlss_engine();

} // namespace mkwvk
