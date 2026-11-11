// (hookshot) Streamline's API as resolved from the player's sl.interposer.dll.
//
// Only for the Streamline sources: it pulls in <windows.h> and Streamline's headers, so the public
// gpu/vk/vk_streamline.h stays free of both and the renderer can call it on any platform.
#pragma once

#if defined(RX_VK_STREAMLINE) && defined(_WIN32)

#include <windows.h>

#include <sl.h>
#include <sl_consts.h>
#include <sl_core_api.h>
#include <sl_helpers_vk.h>

namespace mkwvk::streamline {

/// What this renderer calls. Resolved once, when the interposer is loaded; every pointer here is
/// non-null for the life of a started session, because start() fails if any one of them is missing.
struct Api {
    PFun_slInit* init{};
    PFun_slShutdown* shutdown{};
    PFun_slIsFeatureSupported* is_feature_supported{};
    PFun_slGetFeatureRequirements* get_feature_requirements{};
    PFun_slGetFeatureVersion* get_feature_version{};
    PFun_slGetFeatureFunction* get_feature_function{};
    PFun_slGetNewFrameToken* get_new_frame_token{};
    PFun_slSetConstants* set_constants{};
    PFun_slEvaluateFeature* evaluate_feature{};
    PFun_slFreeResources* free_resources{};
    PFun_slSetVulkanInfo* set_vulkan_info{};
    PFun_slSetTagForFrame* set_tag_for_frame{};
};

/// The resolved functions. Only meaningful while a session is started.
[[nodiscard]] const Api& api();

/// A feature's own function, by name, or null where the plugin has none by that name. This is the
/// only way to reach slDLSSNRSetOptions: it is not an export of the interposer, it lives inside
/// sl.dlss_nr and comes out through here.
template <typename T>
[[nodiscard]] T* feature_function(sl::Feature feature, const char* name) {
    void* function{};
    if (api().get_feature_function(feature, name, function) != sl::Result::eOk) return nullptr;
    return reinterpret_cast<T*>(function);
}

} // namespace mkwvk::streamline

#endif
