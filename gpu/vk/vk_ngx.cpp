// (hookshot) The shared NGX session: gpu/vk/vk_ngx.h says why there is only one.
#if defined(HOOKSHOT_NGX)

#include "vk_ngx.h"
#include "vk_dlss.h"   // (dlss_ngx_session: the one NGX session)

#include <cstdlib>
#include <cstring>

#include "vk_loader.h"

namespace mkwvk::ngx {
namespace {

/// The session's own state. The NGX entry points are not resolved by name here: on Linux the
/// driver's core (libnvidia-ngx.so) exports none of the public NVSDK_NGX_* API -- it was tried, and
/// `NVSDK_NGX_VULKAN_Init_with_ProjectID` is simply not in it -- so a build that asks for DLSS links
/// the SDK's own static library, which holds those entry points and calls into the core itself.
/// nx2/build.py's `ngx_link` is the other half of that, and both are inside HOOKSHOT_NGX, so a
/// build without an SDK links and runs exactly as it did before.
struct Session {
    bool started = false;
    bool start_failed = false;
};
Session s;

/// A project id of our own. NVIDIA's analytics-key form is for licensed titles; the project-id form
/// exists exactly so a project without one can initialize.
// A UUID of our own, and it has to be a real one: NGX validates the shape and answers
// NVSDK_NGX_Result_FAIL_InvalidParameter (0xbad00005) for anything that is not. The first attempt
// here spelled "hookshot-dlss" into the last two groups, which is not hex, and that is exactly the
// error it got. NVIDIA's analytics-key form of Init is for licensed titles; the project-id form
// exists so a project without one can initialize.
constexpr const char* kProjectId = "7b2f5d0c-6a1e-4c39-9f84-3d5e2a1b0c96";   // (vk_dlss.cpp's: one project, one session)
constexpr const char* kEngineVersion = "1.0";

/// The game folder, wide, for NGX's application data path: where it writes what it keeps and where
/// it looks for the feature libraries (libnvidia-ngx-dlss.so.* and its siblings), which the player
/// supplies beside the binary and which are never ours to ship.
const wchar_t* data_path() {
    static wchar_t wide[MAX_PATH + 1];
    static const bool once = []() {
        char path[MAX_PATH + 1]{};
        const DWORD n = GetModuleFileNameA(nullptr, path, MAX_PATH);
        if (!n) {
            wide[0] = L'.', wide[1] = 0;
            return true;
        }
        char* name = path + n;
        while (name > path && name[-1] != '\\' && name[-1] != '/') name--;
        *name = 0;
        size_t i = 0;
        for (; path[i] && i < MAX_PATH; i++) wide[i] = (wchar_t)(unsigned char)path[i];
        wide[i] = 0;
        if (!i) wide[0] = L'.', wide[1] = 0;
        return true;
    }();
    (void)once;
    return wide;
}

} // namespace

bool load() {
    // Linked, not loaded: there is nothing to look for. Kept as a call of its own because the
    // discovery entry points are used before the session is started, and because a future NGX that
    // does want loading has one place to do it.
    return true;
}

bool start() {
    if (s.started) return true;
    if (s.start_failed) return false;
    // One NGX session per device, and it is super resolution's (vk_dlss.cpp ngx_init): its project
    // id, its search path for the runtimes, its logging. Frame generation is a feature created over
    // that session rather than a second NVSDK_NGX_VULKAN_Init of its own, which NGX does not expect
    // on one device and whose Shutdown1 would take the other's features with it.
    if (!dlss_ngx_session()) {
        logf("vk: ngx: no NGX session on this device; DLSS frame generation is unavailable");
        s.start_failed = true;
        return false;
    }
    s.started = true;
    return true;
}

bool feature_available(const char* available_key, const char* needs_driver_key,
                       const char* major_key, const char* minor_key, const char* name) {
    if (!start()) return false;
    NVSDK_NGX_Parameter* caps = nullptr;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_GetCapabilityParameters(&caps)) || !caps) {
        logf("vk: ngx gave no capability parameters");
        return false;
    }
    int available = 0;
    if (NVSDK_NGX_FAILED(get(caps, available_key, &available)) || !available) {
        // What NGX says about it: an old driver and a feature that failed to initialize are
        // different reports, and a user with the first can act on it.
        int needs_driver = 0, major = 0, minor = 0;
        get(caps, needs_driver_key, &needs_driver);
        get(caps, major_key, &major);
        get(caps, minor_key, &minor);
        if (needs_driver)
            logf("vk: ngx: this driver is too old for %s (%d.%d or newer wanted)", name, major,
                 minor);
        else
            logf("vk: ngx: this GPU and driver do not report %s", name);
        return false;
    }
    return true;
}

NVSDK_NGX_Parameter* allocate_params() {
    if (!start()) return nullptr;
    NVSDK_NGX_Parameter* params = nullptr;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_AllocateParameters(&params))) {
        logf("vk: ngx gave no parameters to evaluate with");
        return nullptr;
    }
    return params;
}

void destroy_params(NVSDK_NGX_Parameter* params) {
    if (params) NVSDK_NGX_VULKAN_DestroyParameters(params);
}

NVSDK_NGX_Result create_feature(VkCommandBuffer cmd, NVSDK_NGX_Feature feature,
                                NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** out_handle) {
    return NVSDK_NGX_VULKAN_CreateFeature(cmd, feature, params, out_handle);
}

NVSDK_NGX_Result evaluate_feature(VkCommandBuffer cmd, NVSDK_NGX_Handle* handle,
                                  NVSDK_NGX_Parameter* params) {
    return NVSDK_NGX_VULKAN_EvaluateFeature(cmd, handle, params, nullptr);
}

void release_feature(NVSDK_NGX_Handle* handle) {
    if (handle) NVSDK_NGX_VULKAN_ReleaseFeature(handle);
}

NVSDK_NGX_FeatureDiscoveryInfo discovery_info(NVSDK_NGX_Feature feature) {
    NVSDK_NGX_FeatureDiscoveryInfo info{};
    info.SDKVersion = NVSDK_NGX_Version_API;
    info.FeatureID = feature;
    info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    info.Identifier.v.ProjectDesc.ProjectId = kProjectId;
    info.Identifier.v.ProjectDesc.EngineType = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
    info.Identifier.v.ProjectDesc.EngineVersion = kEngineVersion;
    info.ApplicationDataPath = data_path();
    return info;
}

size_t instance_extensions(NVSDK_NGX_Feature feature, const char** out, size_t max) {
    if (!load()) return 0;
    const NVSDK_NGX_FeatureDiscoveryInfo info = discovery_info(feature);
    uint32_t n = 0;
    VkExtensionProperties* props = nullptr;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&info, &n, &props)) || !props)
        return 0;
    size_t kept = 0;
    for (uint32_t i = 0; i < n && kept < max; i++) out[kept++] = props[i].extensionName;
    return kept;
}

size_t device_extensions(VkInstance instance, VkPhysicalDevice gpu, NVSDK_NGX_Feature feature,
                         const char** out, size_t max) {
    if (!instance || !gpu || !load()) return 0;
    const NVSDK_NGX_FeatureDiscoveryInfo info = discovery_info(feature);
    uint32_t n = 0;
    VkExtensionProperties* props = nullptr;
    if (NVSDK_NGX_FAILED(
            NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(instance, gpu, &info, &n, &props)) ||
        !props)
        return 0;
    size_t kept = 0;
    for (uint32_t i = 0; i < n && kept < max; i++) out[kept++] = props[i].extensionName;
    return kept;
}

NVSDK_NGX_Resource_VK resource(VkImageView view, VkImage image, VkFormat format, uint32_t width,
                               uint32_t height, bool read_write) {
    VkImageSubresourceRange range{};
    range.aspectMask =
        is_depth_format(format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel = 0, range.levelCount = 1;
    range.baseArrayLayer = 0, range.layerCount = 1;
    return NVSDK_NGX_Create_ImageView_Resource_VK(view, image, range, format, width, height,
                                                  read_write);
}

void shutdown() {
    // (the session is vk_dlss.cpp's, and so is its shutdown)
    s.started = false;
}

} // namespace mkwvk::ngx

#endif // HOOKSHOT_NGX
