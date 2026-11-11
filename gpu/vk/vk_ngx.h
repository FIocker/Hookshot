// (hookshot) The one NGX session the DLSS backends share.
//
// NGX initializes once per process and hands back one capability-parameter block for every feature
// it knows, so three backends that each loaded and initialized it would be three bugs: a second
// NVSDK_NGX_VULKAN_Init over a live session, a second set of parameters, and a shutdown by whichever
// one finished first. Super Resolution (gpu/vk/vk_dlss.h), Frame Generation (gpu/vk/vk_dlssg.h) and
// Ray Reconstruction all come through here instead, and each one owns nothing but its own feature
// handle.
//
// Nothing of NVIDIA's is in this repository and nothing of NVIDIA's is needed to build it: this
// whole file is inside the build-time gate `HOOKSHOT_NGX`, which names the builder's own copy of
// an NGX SDK. A tree without it does not compile this file at all, and the backends' public headers
// stay free of NGX types so that the renderer can call them either way. docs/upscaling.md has the
// licensing decision in full.
#pragma once

#if defined(HOOKSHOT_NGX)

#include <cstdint>

#include "vk_internal.h"

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_vk.h>

namespace mkwvk::ngx {

/// NGX's parameters through its C functions, never NVSDK_NGX_Parameter's virtual Set/Get. The object comes from NVIDIA's
/// MSVC-built library, and its vtable is laid out by Microsoft's ABI, which llvm-mingw's Itanium-ABI calls do not follow:
/// on a Windows build the overloads line up by luck, if at all (gpu/vk/vk_dlss.cpp says the same). One overload per type
/// NGX takes, so a call site reads as it did and the value's type picks the function; any other type does not compile.
inline void set(NVSDK_NGX_Parameter* p, const char* key, unsigned v) { NVSDK_NGX_Parameter_SetUI(p, key, v); }
inline void set(NVSDK_NGX_Parameter* p, const char* key, int v) { NVSDK_NGX_Parameter_SetI(p, key, v); }
inline void set(NVSDK_NGX_Parameter* p, const char* key, float v) { NVSDK_NGX_Parameter_SetF(p, key, v); }
inline void set(NVSDK_NGX_Parameter* p, const char* key, double v) { NVSDK_NGX_Parameter_SetD(p, key, v); }
inline void set(NVSDK_NGX_Parameter* p, const char* key, void* v) { NVSDK_NGX_Parameter_SetVoidPointer(p, key, v); }
/// (a Vulkan resource is handed over by its address, as NGX's own helpers in nvsdk_ngx_helpers_vk.h do)
inline void set(NVSDK_NGX_Parameter* p, const char* key, NVSDK_NGX_Resource_VK* v) {
    NVSDK_NGX_Parameter_SetVoidPointer(p, key, v);
}
template <typename T> void set(NVSDK_NGX_Parameter*, const char*, T) = delete;
inline NVSDK_NGX_Result get(NVSDK_NGX_Parameter* p, const char* key, int* v) { return NVSDK_NGX_Parameter_GetI(p, key, v); }
inline NVSDK_NGX_Result get(NVSDK_NGX_Parameter* p, const char* key, unsigned* v) {
    return NVSDK_NGX_Parameter_GetUI(p, key, v);
}

/// Loads the NGX core the driver ships and resolves what the backends call. False, with one line
/// logged, where there is no NGX runtime here or it is missing an entry point. Asked more than once
/// is free: the answer is kept.
[[nodiscard]] bool load();

/// Initializes the session if it is not up yet. False where NGX refused, which it does on a GPU or
/// driver it does not serve.
[[nodiscard]] bool start();

/// Whether NGX reports this feature as available on this GPU and driver. `available_key` is the
/// feature's own capability key (`NVSDK_NGX_Parameter_SuperSampling_Available` and its siblings) and
/// `name` only names it in the log. A feature that is unavailable because the driver is too old is
/// reported differently from one that failed to initialize, because a user can act on the first.
[[nodiscard]] bool feature_available(const char* available_key, const char* needs_driver_key,
                                     const char* major_key, const char* minor_key, const char* name);

/// A parameter block of this backend's own, to create and evaluate a feature with. Null where NGX
/// would not give one. The caller keeps it for the life of its feature and frees it with
/// `destroy_params`.
[[nodiscard]] NVSDK_NGX_Parameter* allocate_params();
void destroy_params(NVSDK_NGX_Parameter* params);

/// Creates / evaluates / releases one feature. Thin passes to NGX, here only so that the resolved
/// entry points stay in one place.
[[nodiscard]] NVSDK_NGX_Result create_feature(VkCommandBuffer cmd, NVSDK_NGX_Feature feature,
                                              NVSDK_NGX_Parameter* params,
                                              NVSDK_NGX_Handle** out_handle);
[[nodiscard]] NVSDK_NGX_Result evaluate_feature(VkCommandBuffer cmd, NVSDK_NGX_Handle* handle,
                                                NVSDK_NGX_Parameter* params);
void release_feature(NVSDK_NGX_Handle* handle);

/// What a discovery call is asked about, for a feature. Used before the device exists, which is why
/// it takes no device.
[[nodiscard]] NVSDK_NGX_FeatureDiscoveryInfo discovery_info(NVSDK_NGX_Feature feature);

/// The extensions a feature needs here, asked of NGX. Both return how many names were written, at
/// most `max`; the names are NGX's own storage and last as long as the process. Zero means NGX does
/// not answer, which leaves the device created as it would have been.
size_t instance_extensions(NVSDK_NGX_Feature feature, const char** out, size_t max);
size_t device_extensions(VkInstance instance, VkPhysicalDevice gpu, NVSDK_NGX_Feature feature,
                         const char** out, size_t max);

/// One image as NGX takes it: a view *and* its image, range, format and the extent it is read at.
/// NGX is handed resources, not views, which is the whole reason the renderer records what each
/// binding is a view of.
[[nodiscard]] NVSDK_NGX_Resource_VK resource(VkImageView view, VkImage image, VkFormat format,
                                             uint32_t width, uint32_t height, bool read_write);

/// Ends the session. The backends release their own features first; this is safe when nothing was
/// started.
void shutdown();

} // namespace mkwvk::ngx

#endif // HOOKSHOT_NGX
