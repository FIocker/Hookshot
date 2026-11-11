// (hookshot) AMD FSR 4 through AMD's runtime: gpu/vk/vk_ffx.h says what and why. Everything that
// touches AMD's API is inside _WIN32, because the runtime is Windows DLLs; the stubs at the bottom are
// what a Linux build compiles, so the call sites need no #ifdef.

#include "vk_ffx.h"

#include <cstdlib>

#include "upscaler_rows.h"
#include "vk_fsr4.h"   // (super build: upstream's FSR 4 engine first)
#include "vk_menu.h"

namespace mkwvk {
bool fsr4_wanted() {
    static const bool off = [] {
        const char* e = getenv("RX_VK_FSR");
        return e && *e && atoi(e) == 0;
    }();
    // (the row first: AMD's runtime is not looked for while the row is not on FSR 4. (super build) Upstream's FSR 4 engine on
    // the frame chain (vk_fsr4.cpp, a build with HOOKSHOT_FSR) takes the row where it runs: this one is the fallback)
    if (off || opts.dlss != upscaler_rows::FSR4) return false;
    if (fsr4_built() && fsr4_why() == FSR_OK) return false;
    return ffx::upscale_supported();
}
} // namespace mkwvk

#if defined(_WIN32)

#include <windows.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <string>

#include "../../third_party/ffx-api/api/include/ffx_api.h"
#include "../../third_party/ffx-api/api/include/ffx_api_types.h"
#include "../../third_party/ffx-api/api/include/vk/ffx_api_vk.h"
#include "../../third_party/ffx-api/framegeneration/include/ffx_framegeneration.h"
#include "../../third_party/ffx-api/upscalers/include/ffx_upscale.h"
#include "vk_loader.h"
#include "vk_work.h"   // (last: its vkCmd* go through the recorder, RX_VK_RECORDER; vk_work.h)

namespace mkwvk::ffx {
namespace {

int env_int(const char* name, int fallback) {
    const char* e = getenv(name);
    return e && *e ? atoi(e) : fallback;
}

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation memory = nullptr;
    VkImageView view = VK_NULL_HANDLE;   // (for the pass that squares the upscaler's picture back: vk_fsr3.cpp)
    uint32_t width = 0, height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
};

struct State {
    bool tried = false;
    HMODULE module = nullptr;
    PfnFfxCreateContext create = nullptr;
    PfnFfxDestroyContext destroy = nullptr;
    PfnFfxConfigure configure = nullptr;
    PfnFfxQuery query = nullptr;
    PfnFfxDispatch dispatch = nullptr;

    // upscaling
    int up_supported = -1;
    std::string up_provider;
    ffxContext up = nullptr;
    uint32_t up_rw = 0, up_rh = 0, up_ow = 0, up_oh = 0;
    Image up_out;
    bool up_first = true;
    bool up_infinite = false;   // (the context's FFX_UPSCALE_ENABLE_DEPTH_INFINITE: made again when the camera's differs)
    bool up_failed = false;

    // frame generation
    int fg_supported = -1;
    std::string fg_provider;
    ffxContext fg = nullptr;
    uint32_t fg_w = 0, fg_h = 0, fg_rw = 0, fg_rh = 0;
    VkFormat fg_format = VK_FORMAT_UNDEFINED;
    bool fg_infinite = false;
    Image made[4];
    uint64_t frame_id = 0;
    bool fg_reset = true;
    bool fg_failed = false;
};
State& st() {
    static State s;
    return s;
}

void message(uint32_t type, const wchar_t* text) {
    char b[512];
    WideCharToMultiByte(CP_UTF8, 0, text ? text : L"", -1, b, sizeof b, nullptr, nullptr);
    logf("vk: amd fsr %s: %s", type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning", b);
}

/// Where amd_fidelityfx_loader_vk.dll may be: RX_VK_FFX_DIR, then the game's own folder. A player who
/// drops AMD's DLLs beside the game should not also have to set an environment variable.
std::wstring find_loader() {
    std::wstring dirs[2];
    if (const wchar_t* e = _wgetenv(L"RX_VK_FFX_DIR"); e && *e) dirs[0] = e;
    wchar_t exe[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH);
    if (n && n < MAX_PATH) {
        std::wstring d(exe, n);
        dirs[1] = d.substr(0, d.find_last_of(L"\\/"));
    }
    for (const std::wstring& d : dirs) {
        if (d.empty()) continue;
        const std::wstring p = d + L"\\amd_fidelityfx_loader_vk.dll";
        if (GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES) return p;
    }
    return {};
}

// (hookshot) Every call into AMD's DLLs goes through these: a fault inside the runtime (a driver it does not expect, a
// provider that dereferences what an NVIDIA device does not have) costs FSR 4, not the game. Plain C functions with no
// objects to unwind, which is what __try needs; noinline so the guard stays around the call.
__declspec(noinline) HMODULE guarded_load(const wchar_t* path) {
    __try {
        return LoadLibraryExW(path, nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    } __except (1) {
        return nullptr;
    }
}
constexpr ffxReturnCode_t kFaulted = 0xfa17;   // (not one of the API's codes: the call faulted)
__declspec(noinline) ffxReturnCode_t guarded_query(PfnFfxQuery f, ffxContext* c, ffxQueryDescHeader* d) {
    __try {
        return f(c, d);
    } __except (1) {
        return kFaulted;
    }
}
__declspec(noinline) ffxReturnCode_t guarded_create(PfnFfxCreateContext f, ffxContext* c, ffxCreateContextDescHeader* d) {
    __try {
        return f(c, d, nullptr);
    } __except (1) {
        return kFaulted;
    }
}
__declspec(noinline) ffxReturnCode_t guarded_configure(PfnFfxConfigure f, ffxContext* c, const ffxConfigureDescHeader* d) {
    __try {
        return f(c, d);
    } __except (1) {
        return kFaulted;
    }
}
__declspec(noinline) ffxReturnCode_t guarded_dispatch(PfnFfxDispatch f, ffxContext* c, const ffxDispatchDescHeader* d) {
    __try {
        return f(c, d);
    } __except (1) {
        return kFaulted;
    }
}

bool load() {
    State& s = st();
    if (s.tried) return s.module != nullptr;
    s.tried = true;
    if (env_int("RX_VK_FFX", 1) == 0) {
        logf("vk: amd fsr 4: off for this run (RX_VK_FFX=0)");
        return false;
    }
    const std::wstring path = find_loader();
    if (path.empty()) {
        logf("vk: amd fsr 4: no amd_fidelityfx_loader_vk.dll beside the game or in RX_VK_FFX_DIR (readme.md says "
             "which of AMD's files to put there)");
        return false;
    }
    // (the loader looks for the upscaler and frame generation DLLs in its own folder)
    logf("vk: amd fsr 4: loading AMD's runtime");
    s.module = guarded_load(path.c_str());
    if (!s.module) {
        logf("vk: amd fsr 4: amd_fidelityfx_loader_vk.dll did not load (error %lu)", GetLastError());
        return false;
    }
    s.create = (PfnFfxCreateContext)(void*)GetProcAddress(s.module, "ffxCreateContext");
    s.destroy = (PfnFfxDestroyContext)(void*)GetProcAddress(s.module, "ffxDestroyContext");
    s.configure = (PfnFfxConfigure)(void*)GetProcAddress(s.module, "ffxConfigure");
    s.query = (PfnFfxQuery)(void*)GetProcAddress(s.module, "ffxQuery");
    s.dispatch = (PfnFfxDispatch)(void*)GetProcAddress(s.module, "ffxDispatch");
    if (!s.create || !s.destroy || !s.configure || !s.query || !s.dispatch) {
        logf("vk: amd fsr 4: amd_fidelityfx_loader_vk.dll is missing one of the FSR API's five entry points");
        FreeLibrary(s.module);
        s.module = nullptr;
        return false;
    }
    if (env_int("RX_VK_FFX_LOG", 0) > 0) {
        ffxConfigureDescGlobalDebug d{};
        d.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG;
        d.fpMessage = message;
        d.debugLevel = FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE;
        s.configure(nullptr, &d.header);
    }
    logf("vk: amd fsr 4: AMD's FSR runtime loaded");
    return true;
}

VkDeviceContext device_context() {
    VkDeviceContext c{};
    c.vkDevice = dev.device;
    c.vkPhysicalDevice = dev.physical;
    c.vkDeviceProcAddr = vkGetDeviceProcAddr;
    return c;
}

/// The providers AMD's runtime has for one effect on this device, newest first as it lists them; the
/// first one's name in `name`. Zero when it has none, which is how a device it does not serve answers.
uint64_t providers(uint64_t create_type, std::string& name) {
    State& s = st();
    VkDeviceContext c = device_context();
    uint64_t count = 0;
    ffxQueryDescGetVersions q{};
    q.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    q.createDescType = create_type;
    q.device = &c;
    q.outputCount = &count;
    if (guarded_query(s.query, nullptr, &q.header) != FFX_API_RETURN_OK || count == 0) return 0;
    uint64_t ids[16]{};
    const char* names[16]{};
    count = std::min<uint64_t>(count, 16);
    q.versionIds = ids;
    q.versionNames = names;
    if (guarded_query(s.query, nullptr, &q.header) != FFX_API_RETURN_OK) return 0;
    std::string all;
    for (uint64_t i = 0; i < count; i++) all += (i ? ", " : "") + std::string(names[i] ? names[i] : "?");
    name = names[0] ? names[0] : "";
    logf("vk: amd fsr 4: providers for effect 0x%llx: %s", (unsigned long long)create_type, all.c_str());
    return count;
}

/// Our images are in GENERAL throughout, and AMD's Vulkan backend maps its states to layouts: only
/// COMMON, GENERIC_READ and UNORDERED_ACCESS are GENERAL there (COMPUTE_READ is SHADER_READ_ONLY_OPTIMAL,
/// which would have it transition from a layout the image is not in). So inputs are GENERIC_READ and
/// outputs UNORDERED_ACCESS, and nothing the runtime does to them leaves GENERAL.
FfxApiResource resource(VkImage image, VkFormat format, uint32_t w, uint32_t h, bool write) {
    FfxApiResourceDescription d{};
    d.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    d.format = ffxApiGetSurfaceFormatVK(format);
    d.width = w, d.height = h, d.depth = 1, d.mipCount = 1;
    d.flags = FFX_API_RESOURCE_FLAGS_NONE;
    d.usage = FFX_API_RESOURCE_USAGE_READ_ONLY;
    if (ffxApiIsDepthFormat(format)) d.usage |= FFX_API_RESOURCE_USAGE_DEPTHTARGET;
    if (write) d.usage |= FFX_API_RESOURCE_USAGE_UAV;
    return ffxApiGetResourceVK((void*)image, d,
                               write ? FFX_API_RESOURCE_STATE_UNORDERED_ACCESS : FFX_API_RESOURCE_STATE_GENERIC_READ);
}

/// A far plane the runtime should take as infinite: none given (0), or FLT_MAX (fsr3_math.h: an infinite projection)
bool infinite(float near_plane, float far_plane) { return !(far_plane > near_plane) || far_plane >= FLT_MAX; }

void drop(Image& im) {
    if (im.view) vkDestroyImageView(dev.device, im.view, nullptr);
    if (im.image) destroy_image(im.image, im.memory);
    im = Image{};
}

/// A storage image of ours for the runtime to write, cleared to GENERAL with the command buffer given.
bool make(VkCommandBuffer cmd, Image& im, VkFormat format, uint32_t w, uint32_t h) {
    drop(im);
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = {w, h, 1};
    ci.mipLevels = 1, ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!create_image(ci, im.image, im.memory, false)) {
        logf("vk: amd fsr 4: a %ux%u image (format %d) was not made", w, h, (int)format);
        return false;
    }
    im.width = w, im.height = h, im.format = format;
    VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vc.image = im.image, vc.viewType = VK_IMAGE_VIEW_TYPE_2D, vc.format = format;
    vc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(dev.device, &vc, nullptr, &im.view) != VK_SUCCESS) {
        logf("vk: amd fsr 4: a view of a %ux%u image was not made", w, h);
        drop(im);
        return false;
    }
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im.image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1, di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);
    return true;
}

/// Everything AMD's runtime may still be reading or writing has to be finished before its context or
/// one of our images goes: they are only ever destroyed on a size change or at shutdown.
void idle() { vkDeviceWaitIdle(dev.device); }

void drop_upscaler() {
    State& s = st();
    if (s.up || s.up_out.image) idle();
    if (s.up) s.destroy(&s.up, nullptr);
    s.up = nullptr;
    drop(s.up_out);
    s.up_rw = s.up_rh = s.up_ow = s.up_oh = 0;
}

void drop_framegen() {
    State& s = st();
    if (s.fg || s.made[0].image) idle();
    if (s.fg) s.destroy(&s.fg, nullptr);
    s.fg = nullptr;
    for (Image& m : s.made) drop(m);
    s.fg_w = s.fg_h = s.fg_rw = s.fg_rh = 0;
}

/// A barrier around the runtime's own work: everything before it done, everything after it waiting.
/// The runtime records into our command buffer and does not know what the replay did before it.
void full_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier2 m{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    m.srcStageMask = m.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    m.srcAccessMask = m.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1, di.pMemoryBarriers = &m;
    vkCmdPipelineBarrier2(cmd, &di);
}

} // namespace

bool built() { return true; }

void probe_upscale();
void probe_frames();

// (hookshot) AMD's runtime is asked once, at the start, from the render thread (probe(), from vk_main.cpp once the device
// exists), with a line in the log before each call into it, so a runtime that hangs or faults names itself. The Insert
// menu and the frame only read what it answered: v2.0.13 asked from the menu's first draw, where an answer that never
// came meant a menu that never opened.
void probe() {
    State& s = st();
    if (s.up_supported >= 0 || !dev.device) return;
    s.up_supported = s.fg_supported = 0;
    if (!load()) return;
    logf("vk: amd fsr 4: asking AMD's runtime what it offers on this GPU");
    probe_upscale();
    probe_frames();
    logf("vk: amd fsr 4: AMD's runtime answered: upscaling %s, frame generation %s", s.up_supported ? "yes" : "no",
         s.fg_supported ? "yes" : "no");
}

bool upscale_supported() { return st().up_supported > 0; }

void probe_upscale() {
    State& s = st();
    if (!providers(FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE, s.up_provider)) {
        logf("vk: amd fsr 4: AMD's runtime has no upscaler for this GPU");
        return;
    }
    logf("vk: amd fsr 4: upscaling available (%s)", s.up_provider.c_str());
    s.up_supported = 1;
}

const char* upscale_provider() { return st().up_provider.c_str(); }

bool upscale(VkCommandBuffer cmd, const UpscaleInputs& in) {
    State& s = st();
    if (s.up_failed || !upscale_supported()) return false;
    if (!in.color || !in.depth || !in.motion || !in.render_width || !in.render_height || !in.upscale_width ||
        !in.upscale_height || !(in.near_plane > 0.0f) || !(in.fov_y > 0.0f))
        return false;
    const bool inf = infinite(in.near_plane, in.far_plane);
    if (s.up && (in.render_width > s.up_rw || in.render_height > s.up_rh || in.upscale_width != s.up_ow ||
                 in.upscale_height != s.up_oh || inf != s.up_infinite))
        drop_upscaler();
    if (!s.up) {
        ffxCreateBackendVKDesc backend{};
        backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
        backend.deviceContext = device_context();
        ffxCreateContextDescUpscale c{};
        c.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        c.header.pNext = &backend.header;
        // The colour as FSR 3.1's passes take it (compiled with FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT), the depth
        // reversed as vk_fsr3_inputs.cpp makes it (1 at the near plane), infinite only when the scene's projection is
        // (it is not: its far plane is at 1000.5), and no exposure texture, so the runtime works the exposure out
        // itself. The render size may change under the context as the game's dynamic resolution moves, within the
        // maximum.
        c.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED |
                  (inf ? FFX_UPSCALE_ENABLE_DEPTH_INFINITE : 0u) | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE |
                  FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION;
        if (env_int("RX_VK_FFX_DEBUG", 0) > 0) c.flags |= FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
        c.maxRenderSize = {in.upscale_width, in.upscale_height};
        c.maxUpscaleSize = {in.upscale_width, in.upscale_height};
        c.fpMessage = message;
        const ffxReturnCode_t r = guarded_create(s.create, &s.up, &c.header);
        if (r != FFX_API_RETURN_OK || !s.up) {
            logf("vk: amd fsr 4: the upscaler's context was not made (FSR API result %u): FSR 4 is off", (unsigned)r);
            s.up = nullptr;
            s.up_failed = true;
            return false;
        }
        if (!make(cmd, s.up_out, VK_FORMAT_R16G16B16A16_SFLOAT, in.upscale_width, in.upscale_height)) {
            drop_upscaler();
            s.up_failed = true;
            return false;
        }
        s.up_rw = in.upscale_width, s.up_rh = in.upscale_height;
        s.up_ow = in.upscale_width, s.up_oh = in.upscale_height;
        s.up_first = true;
        s.up_infinite = inf;
        ffxQueryGetProviderVersion v{};
        v.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
        if (guarded_query(s.query, &s.up, &v.header) == FFX_API_RETURN_OK && v.versionName) s.up_provider = v.versionName;
        logf("vk: amd fsr 4: %s from up to %ux%u to %ux%u", s.up_provider.c_str(), s.up_rw, s.up_rh, s.up_ow, s.up_oh);
    }
    ffxDispatchDescUpscale d{};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = (void*)cmd;
    d.color = resource(in.color, in.color_format, in.render_width, in.render_height, false);
    d.depth = resource(in.depth, in.depth_format, in.render_width, in.render_height, false);
    d.motionVectors = resource(in.motion, in.motion_format, in.render_width, in.render_height, false);
    d.output = resource(s.up_out.image, s.up_out.format, s.up_ow, s.up_oh, true);
    // (the jitter and the motion as vk_fsr3.cpp hands them to FSR 3.1's passes: render pixels, x right and y down, and
    //  the vectors already in render pixels, so the scale is 1)
    d.jitterOffset = {in.jitter[0], in.jitter[1]};
    d.motionVectorScale = {in.motion_scale[0], in.motion_scale[1]};
    d.renderSize = {in.render_width, in.render_height};
    d.upscaleSize = {s.up_ow, s.up_oh};
    d.enableSharpening = false;   // (the Sharpness row sharpens the finished picture already)
    d.sharpness = 0.0f;
    d.frameTimeDelta = std::clamp(in.frame_time_ms, 0.0f, 1000.0f);
    d.preExposure = 1.0f;
    d.reset = in.reset || s.up_first;
    d.cameraNear = in.near_plane;
    d.cameraFar = inf ? FLT_MAX : in.far_plane;
    d.cameraFovAngleVertical = in.fov_y;
    d.viewSpaceToMetersFactor = 1.0f;
    d.flags = 0;
    full_barrier(cmd);
    ::mkwvk::rec::boundary(::mkwvk::rec::B_DIRECT);   // (AMD's FidelityFX runtime records into cmd itself: what this thread recorded through vk_work.h reaches cmd first)
    const ffxReturnCode_t r = guarded_dispatch(s.dispatch, &s.up, &d.header);
    full_barrier(cmd);
    s.up_first = false;
    if (r != FFX_API_RETURN_OK) {
        static int told = 0;
        if (told++ < 4) logf("vk: amd fsr 4: an upscale was refused (FSR API result %u)", (unsigned)r);
        return false;
    }
    return true;
}

VkImage output_image() { return st().up_out.image; }
VkImageView output_view() { return st().up_out.view; }

bool frames_supported() { return st().fg_supported > 0; }

void probe_frames() {
    State& s = st();
    if (env_int("RX_VK_FFX_FG", 1) == 0) return;
    if (!providers(FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION, s.fg_provider)) {
        logf("vk: amd fsr 4: AMD's runtime has no frame generation for this GPU");
        return;
    }
    logf("vk: amd fsr 4: frame generation available (%s)", s.fg_provider.c_str());
    s.fg_supported = 1;
}

uint32_t frames_wanted() {
    static const uint32_t cap = (uint32_t)std::clamp(env_int("RX_VK_FFX_FG_MAX", 4), 1, 4);
    return cap;
}

bool generate(VkCommandBuffer cmd, const FrameInputs& in) {
    State& s = st();
    if (s.fg_failed || !frames_supported()) return false;
    if (!in.color || !in.depth || !in.motion || !in.width || !in.height || !in.render_width || !in.render_height ||
        in.frames < 1 || in.frames > 4 || !(in.near_plane > 0.0f) || !(in.fov_y > 0.0f))
        return false;
    const bool inf = infinite(in.near_plane, in.far_plane);
    if (s.fg && (in.width != s.fg_w || in.height != s.fg_h || in.render_width > s.fg_rw ||
                 in.render_height > s.fg_rh || in.color_format != s.fg_format || inf != s.fg_infinite))
        drop_framegen();
    if (!s.fg) {
        ffxCreateBackendVKDesc backend{};
        backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
        backend.deviceContext = device_context();
        ffxCreateContextDescFrameGenerationVersion version{};
        version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION;
        version.header.pNext = &backend.header;
        version.version = FFX_FRAMEGENERATION_VERSION;
        ffxCreateContextDescFrameGeneration c{};
        c.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
        c.header.pNext = &version.header;
        // The presented frame is SDR (RGBA8, already tonemapped by the game), depth as the upscaler's (reversed,
        // infinite only when the scene's projection is). Not ASYNC_WORKLOAD_SUPPORT: the made frames are recorded into
        // the replay's own command buffer.
        c.flags = FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED | (inf ? FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE : 0u);
        if (env_int("RX_VK_FFX_DEBUG", 0) > 0) c.flags |= FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING;
        c.displaySize = {in.width, in.height};
        c.maxRenderSize = {in.width, in.height};
        c.backBufferFormat = ffxApiGetSurfaceFormatVK(in.color_format);
        const ffxReturnCode_t r = guarded_create(s.create, &s.fg, &c.header);
        if (r != FFX_API_RETURN_OK || !s.fg) {
            logf("vk: amd fsr 4: the frame generation context was not made (FSR API result %u): FSR frame "
                 "generation is off",
                 (unsigned)r);
            s.fg = nullptr;
            s.fg_failed = true;
            return false;
        }
        for (Image& m : s.made)
            if (!make(cmd, m, in.color_format, in.width, in.height)) {
                drop_framegen();
                s.fg_failed = true;
                return false;
            }
        s.fg_w = in.width, s.fg_h = in.height, s.fg_rw = in.width, s.fg_rh = in.height;
        s.fg_format = in.color_format;
        s.fg_infinite = inf;
        s.fg_reset = true;
        ffxQueryGetProviderVersion v{};
        v.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
        if (guarded_query(s.query, &s.fg, &v.header) == FFX_API_RETURN_OK && v.versionName) s.fg_provider = v.versionName;
        logf("vk: amd fsr 4: %s frame generation at %ux%u", s.fg_provider.c_str(), s.fg_w, s.fg_h);
    }
    s.frame_id++;
    const bool reset = in.reset || s.fg_reset;
    s.fg_reset = false;

    // Configure, once a frame, with the frame id going up by exactly one each time (anything else resets
    // the runtime's history). No swapchain of AMD's: Hookshot presents and paces its own made frames
    // (vk_framegen.cpp), so the dispatch below is the "manual" one AMD's documentation describes, into
    // our own images on our own command buffer.
    ffxConfigureDescFrameGeneration cfg{};
    cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    cfg.swapChain = nullptr;
    cfg.frameGenerationEnabled = true;
    cfg.allowAsyncWorkloads = false;
    cfg.flags = env_int("RX_VK_FFX_DEBUG", 0) > 1 ? FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_VIEW : 0;
    cfg.generationRect = {0, 0, (int32_t)in.width, (int32_t)in.height};
    cfg.frameID = s.frame_id;
    if (const ffxReturnCode_t r = guarded_configure(s.configure, &s.fg, &cfg.header); r != FFX_API_RETURN_OK) {
        static int told = 0;
        if (told++ < 4) logf("vk: amd fsr 4: frame generation's configuration was refused (FSR API result %u)", (unsigned)r);
        return false;
    }

    ffxDispatchDescFrameGenerationPrepareV2 prep{};
    prep.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2;
    prep.frameID = s.frame_id;
    prep.flags = 0;
    prep.commandList = (void*)cmd;
    prep.renderSize = {in.render_width, in.render_height};
    prep.jitterOffset = {in.jitter[0], in.jitter[1]};
    prep.motionVectorScale = {in.motion_scale[0], in.motion_scale[1]};
    prep.frameTimeDelta = std::clamp(in.frame_time_ms, 0.0f, 1000.0f);
    prep.reset = reset;
    prep.cameraNear = in.near_plane;
    prep.cameraFar = inf ? FLT_MAX : in.far_plane;
    prep.cameraFovAngleVertical = in.fov_y;
    prep.viewSpaceToMetersFactor = 1.0f;
    prep.depth = resource(in.depth, in.depth_format, in.render_width, in.render_height, false);
    prep.motionVectors = resource(in.motion, in.motion_format, in.render_width, in.render_height, false);
    if (in.have_view) {
        // (the view is world -> camera, row major: its rows are the camera's axes in the world, and its position is
        //  minus the transposed rotation times the translation; vk_framegen.cpp's DLSS-G does the same)
        const float* v = in.view;
        for (int c = 0; c < 3; c++) {
            prep.cameraRight[c] = v[0 * 4 + c];
            prep.cameraUp[c] = v[1 * 4 + c];
            prep.cameraForward[c] = v[2 * 4 + c];
            prep.cameraPosition[c] = -(v[0 * 4 + c] * v[0 * 4 + 3] + v[1 * 4 + c] * v[1 * 4 + 3] +
                                       v[2 * 4 + c] * v[2 * 4 + 3]);
        }
    }
    full_barrier(cmd);
    ::mkwvk::rec::boundary(::mkwvk::rec::B_DIRECT);   // (AMD's FidelityFX runtime records into cmd itself: what this thread recorded through vk_work.h reaches cmd first)
    if (const ffxReturnCode_t r = guarded_dispatch(s.dispatch, &s.fg, &prep.header); r != FFX_API_RETURN_OK) {
        static int told = 0;
        if (told++ < 4) logf("vk: amd fsr 4: frame generation's preparation was refused (FSR API result %u)", (unsigned)r);
        return false;
    }

    ffxDispatchDescFrameGeneration d{};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
    d.commandList = (void*)cmd;
    d.presentColor = resource(in.color, in.color_format, in.width, in.height, false);
    for (uint32_t k = 0; k < in.frames; k++)
        d.outputs[k] = resource(s.made[k].image, s.made[k].format, in.width, in.height, true);
    d.numGeneratedFrames = in.frames;
    d.reset = reset;
    d.backbufferTransferFunction = FFX_API_BACKBUFFER_TRANSFER_FUNCTION_SRGB;
    d.minMaxLuminance[0] = 0.0f, d.minMaxLuminance[1] = 1.0f;
    d.generationRect = {0, 0, (int32_t)in.width, (int32_t)in.height};
    d.frameID = s.frame_id;
    ::mkwvk::rec::boundary(::mkwvk::rec::B_DIRECT);   // (AMD's FidelityFX runtime records into cmd itself: what this thread recorded through vk_work.h reaches cmd first)
    const ffxReturnCode_t r = guarded_dispatch(s.dispatch, &s.fg, &d.header);
    full_barrier(cmd);
    if (r != FFX_API_RETURN_OK) {
        static int told = 0;
        if (told++ < 4)
            logf("vk: amd fsr 4: frame generation of %u frames was refused (FSR API result %u)", in.frames, (unsigned)r);
        return false;
    }
    return true;
}

VkImage made_image(uint32_t k) {
    State& s = st();
    return k >= 1 && k <= 4 ? s.made[k - 1].image : VK_NULL_HANDLE;
}

void reset() { st().fg_reset = true; }

void shutdown() {
    State& s = st();
    drop_upscaler();
    drop_framegen();
    if (s.module) FreeLibrary(s.module);
    s = State{};
}

} // namespace mkwvk::ffx

#else // not Windows: AMD's FSR 4 runtime is Windows DLLs

namespace mkwvk::ffx {

bool built() { return false; }
void probe() {}
bool upscale_supported() { return false; }
const char* upscale_provider() { return ""; }
bool upscale(VkCommandBuffer, const UpscaleInputs&) { return false; }
VkImage output_image() { return VK_NULL_HANDLE; }
VkImageView output_view() { return VK_NULL_HANDLE; }
bool frames_supported() { return false; }
uint32_t frames_wanted() { return 0; }
bool generate(VkCommandBuffer, const FrameInputs&) { return false; }
VkImage made_image(uint32_t) { return VK_NULL_HANDLE; }
void reset() {}
void shutdown() {}

} // namespace mkwvk::ffx

#endif
