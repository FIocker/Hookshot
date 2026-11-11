// (hookshot) DLSS 5 neural rendering; see gpu/vk/vk_dlssnr.h for what it is and is not. Everything
// that touches Streamline is inside RX_VK_STREAMLINE and _WIN32; the stubs at the bottom are what
// every other build compiles, so the present path needs no #ifdef.

#include "vk_dlssnr.h"

#include "vk_menu.h"
#include "vk_opennr.h"   // (hookshot: the open backend, which every build has)
#include "vk_streamline.h"

#if defined(RX_VK_STREAMLINE) && defined(_WIN32)

#include <windows.h>

#include <atomic>
#include <mutex>

#include <sl.h>
#include <sl_consts.h>
#include <sl_core_api.h>
#include <sl_dlss.h>

#include "vk_dlssnr_options.h"
#include "vk_streamline_api.h"
#include "vk_work.h"   // (last: its vkCmd* go through the recorder, RX_VK_RECORDER; vk_work.h)

namespace mkwvk::dlssnr {
namespace {

// One viewport: the window's, and there is only ever one.
constexpr uint32_t kViewport = 1;
// The model's own working format. Eight bits a channel is what the window's image is and what the
// uplift was measured against; a float format here would only widen a picture that is already
// quantized by the time it reaches us.
constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kDepthFormat = VK_FORMAT_R32_SFLOAT;
constexpr VkFormat kMotionFormat = VK_FORMAT_R16G16_SFLOAT;
constexpr VkImageUsageFlags kColorUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_DST_BIT;
constexpr VkImageUsageFlags kSceneUsage =
    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
constexpr VkImageSubresourceRange kOneLevel{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

struct Image {
    VkImage image{};
    VkImageView view{};
    VmaAllocation memory{};
};

struct State {
    bool tried{};
    bool capable{};
    std::atomic<bool> failed{};
    // A model, style or preset Streamline will not take is the player's to change, so it is held
    // against those exact settings rather than turning the feature off for the run.
    std::atomic<bool> options_rejected{};
    bool have_applied{}, have_rejected{};
    NrSettings applied{}, rejected{};
    PFun_slDLSSNRSetOptions* set_options{};
    Image in{}, out{}, depth{}, motion{};
    uint32_t width{}, height{};
    bool scene_cleared{};
    uint32_t frame{};
    std::mutex mu;   // (Streamline's calls for one viewport are not thread-safe among themselves)
};

State& state() {
    static State s;
    return s;
}

void drop(Image& t) {
    if (t.view) vkDestroyImageView(dev.device, t.view, nullptr);
    if (t.image) destroy_image(t.image, t.memory);
    t = Image{};
}

bool make(Image& t, VkFormat format, uint32_t w, uint32_t h, VkImageUsageFlags usage) {
    drop(t);
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = {w, h, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!create_image(ci, t.image, t.memory, false)) {
        logf("vk: dlss-nr: a %ux%u image was not made", w, h);
        return false;
    }
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = t.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = kOneLevel;
    if (vkCreateImageView(dev.device, &vi, nullptr, &t.view) != VK_SUCCESS) {
        logf("vk: dlss-nr: an image's view was not made");
        drop(t);
        return false;
    }
    return true;
}

void barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
             VkAccessFlags2 src_access, VkAccessFlags2 dst_access) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = kOneLevel;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);
}

sl::Resource resource(const Image& t, VkFormat format, VkImageLayout layout, uint32_t w, uint32_t h,
                      VkImageUsageFlags usage) {
    sl::Resource r{sl::ResourceType::eTex2d, t.image, nullptr, t.view, (uint32_t)layout};
    r.width = w;
    r.height = h;
    r.nativeFormat = (uint32_t)format;
    r.mipLevels = 1;
    r.arrayLayers = 1;
    r.flags = 0;
    r.usage = usage;
    return r;
}

sl::float4x4 identity() {
    sl::float4x4 m;
    m.setRow(0, {1.0f, 0.0f, 0.0f, 0.0f});
    m.setRow(1, {0.0f, 1.0f, 0.0f, 0.0f});
    m.setRow(2, {0.0f, 0.0f, 1.0f, 0.0f});
    m.setRow(3, {0.0f, 0.0f, 0.0f, 1.0f});
    return m;
}

/// The camera the model is told about. There is none: this runs over a finished, already-upscaled
/// frame with the HUD on it, so an identity camera with no jitter and no motion is the honest
/// description of what it is being handed, and the flat depth below agrees with it. Claiming the
/// scene's real camera here would make the model reproject detail that is not where it thinks.
sl::Constants scene_constants(uint32_t w, uint32_t h, bool reset) {
    const float fw = (float)w, fh = (float)h;
    sl::Constants c{};
    c.cameraViewToClip = identity();
    c.clipToCameraView = identity();
    c.clipToLensClip = identity();
    c.clipToPrevClip = identity();
    c.prevClipToClip = identity();
    c.jitterOffset = {0.0f, 0.0f};
    c.mvecScale = {1.0f / fw, 1.0f / fh};
    c.cameraPinholeOffset = {0.0f, 0.0f};
    c.cameraPos = {0.0f, 0.0f, 0.0f};
    c.cameraUp = {0.0f, 1.0f, 0.0f};
    c.cameraRight = {1.0f, 0.0f, 0.0f};
    c.cameraFwd = {0.0f, 0.0f, 1.0f};
    c.cameraNear = 0.1f;
    c.cameraFar = 1000.0f;
    c.cameraFOV = 1.0f;
    c.cameraAspectRatio = fw / fh;
    c.depthInverted = sl::eFalse;
    c.cameraMotionIncluded = sl::eTrue;
    c.motionVectors3D = sl::eFalse;
    c.reset = reset ? sl::eTrue : sl::eFalse;
    c.motionVectorsJittered = sl::eFalse;
    return c;
}

/// Depth and motion, once per size: one flat depth and no motion at all. See scene_constants.
void clear_scene(VkCommandBuffer cmd, State& s) {
    barrier(cmd, s.depth.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_2_TRANSFER_WRITE_BIT);
    barrier(cmd, s.motion.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_2_TRANSFER_WRITE_BIT);
    // Mid-scene rather than the far plane: a depth the model reads as "everything is at one
    // distance" without reading as "everything is sky", which is what 0 or 1 would say.
    VkClearColorValue flat{};
    flat.float32[0] = 0.5f;
    vkCmdClearColorImage(cmd, s.depth.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &flat, 1,
                         &kOneLevel);
    const VkClearColorValue none{};
    vkCmdClearColorImage(cmd, s.motion.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &none, 1,
                         &kOneLevel);
    barrier(cmd, s.depth.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_ACCESS_2_SHADER_READ_BIT);
    barrier(cmd, s.motion.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_ACCESS_2_SHADER_READ_BIT);
}

bool resize(State& s, uint32_t w, uint32_t h) {
    // The old images may still be in a frame in flight, so the device has to go idle first -- and
    // that is not a bare vkDeviceWaitIdle. Vulkan wants every VkQueue externally synchronized while
    // it idles, and the replay thread submits on dev.queue; device_wait_idle_locked flushes the
    // pending submits and takes the present queue's lock too, which is the only correct way to do
    // this here. apply() runs on the display thread before present_image takes dev.present_mu for
    // its own submit, so taking it here is not a second acquisition.
    {
        std::lock_guard qk(dev.present_mu);
        device_wait_idle_locked();
    }
    if (!make(s.in, kColorFormat, w, h, kColorUsage) ||
        !make(s.out, kColorFormat, w, h, kColorUsage) ||
        !make(s.depth, kDepthFormat, w, h, kSceneUsage) ||
        !make(s.motion, kMotionFormat, w, h, kSceneUsage)) {
        drop(s.in), drop(s.out), drop(s.depth), drop(s.motion);
        return false;
    }
    s.width = w, s.height = h, s.scene_cleared = false;
    logf("vk: DLSS 5 neural rendering on %ux%u frames", w, h);
    return true;
}

void full_blit(VkCommandBuffer cmd, VkImage from, VkImage to, uint32_t w, uint32_t h) {
    VkImageBlit b{};
    b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.dstSubresource = b.srcSubresource;
    b.srcOffsets[1] = {(int32_t)w, (int32_t)h, 1};
    b.dstOffsets[1] = b.srcOffsets[1];
    vkCmdBlitImage(cmd, from, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, to,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_NEAREST);
}

} // namespace

bool built_with_sdk() { return true; }

bool capable() {
    State& s = state();
    if (s.tried) return s.capable && !s.failed.load(std::memory_order_relaxed);
    s.tried = true;
    if (!streamline::nr_available()) return false;
    s.set_options = streamline::feature_function<PFun_slDLSSNRSetOptions>(sl::kFeatureDLSS_NR,
                                                                         "slDLSSNRSetOptions");
    if (!s.set_options) {
        logf("vk: Streamline's DLSS 5 neural rendering has no slDLSSNRSetOptions: it is off");
        return false;
    }
    s.capable = true;
    return true;
}

bool asked() {
    // Live, every frame: the row the player just changed has to reach the next frame.
    if (const char* e = getenv("RX_VK_DLSS_NR")) return atoi(e) != 0;
    return nr_settings().on;
}

VkImage apply(VkCommandBuffer cmd, VkImage src, uint32_t sw, uint32_t sh,
              VkImageView* out_view) {
    State& s = state();
    if (!src || !sw || !sh) return VK_NULL_HANDLE;
    if (s.failed.load(std::memory_order_relaxed) || !capable() || !asked()) return VK_NULL_HANDLE;

    const NrSettings want = nr_settings();
    if (s.options_rejected.exchange(false)) {
        // Streamline turned down what we last sent: hold it against those exact settings, so
        // changing any row tries again.
        s.rejected = s.applied;
        s.have_rejected = s.have_applied;
        s.have_applied = false;
    }
    if (s.have_rejected && s.rejected == want) return VK_NULL_HANDLE;

    const bool recreated = !s.out.image || s.width != sw || s.height != sh;
    if (recreated && !resize(s, sw, sh)) {
        s.failed.store(true, std::memory_order_relaxed);
        return VK_NULL_HANDLE;
    }
    const bool options_changed = recreated || !s.have_applied || !(s.applied == want);
    s.applied = want;
    s.have_applied = true;

    if (!s.scene_cleared) {
        clear_scene(cmd, s);
        s.scene_cleared = true;
    }

    // The window's image into ours. The game's is only read.
    barrier(cmd, src, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
                VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT);
    barrier(cmd, s.in.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_2_TRANSFER_WRITE_BIT);
    full_blit(cmd, src, s.in.image, sw, sh);
    barrier(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT |
                VK_ACCESS_2_SHADER_READ_BIT);
    barrier(cmd, s.in.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_ACCESS_2_SHADER_READ_BIT);
    barrier(cmd, s.out.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0,
            VK_ACCESS_2_SHADER_WRITE_BIT);

    const sl::ViewportHandle viewport{kViewport};
    sl::Result result = sl::Result::eOk;
    const char* stage = "slDLSSNRSetOptions";
    {
        std::lock_guard lk(s.mu);
        if (options_changed) {
            DlssNrOptions o;
            o.mode = DlssNrMode::eOn;
            o.intensity = want.intensity;
            o.localToneStrength = want.local_tone;
            o.localStructureStrength = want.local_structure;
            o.globalToneStrength = want.global_tone;
            o.style = want.style;
            o.preset = want.preset;
            o.useAutoMask = want.auto_mask ? 1u : 0u;
            o.skinStructureStrength = want.skin;
            o.performanceMode = want.quality;
            result = s.set_options(viewport, o);
        }
        sl::FrameToken* token{};
        const uint32_t index = s.frame++;
        if (result == sl::Result::eOk) {
            stage = "slGetNewFrameToken";
            result = streamline::api().get_new_frame_token(token, &index);
        }
        if (result == sl::Result::eOk) {
            stage = "slSetConstants";
            result = streamline::api().set_constants(scene_constants(sw, sh, recreated), *token,
                                                    viewport);
        }
        // The usage each resource is *presented to Streamline as*, which is what Dagor's shipping
        // integration passes: what the model reads is sampled, what it writes is a storage image.
        // Handing it the images' whole usage mask instead told it the input was writable too.
        sl::Resource color_in = resource(s.in, kColorFormat,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sw, sh,
                                         VK_IMAGE_USAGE_SAMPLED_BIT);
        sl::Resource color_out = resource(s.out, kColorFormat, VK_IMAGE_LAYOUT_GENERAL, sw, sh,
                                          VK_IMAGE_USAGE_STORAGE_BIT);
        sl::Resource depth = resource(s.depth, kDepthFormat,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sw, sh,
                                      VK_IMAGE_USAGE_SAMPLED_BIT);
        sl::Resource motion = resource(s.motion, kMotionFormat,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sw, sh,
                                       VK_IMAGE_USAGE_SAMPLED_BIT);
        // The uplift colour pair is tagged as a whole resource, with no extent, and depth and
        // motion with one. That asymmetry is how the shipping integration does it: the colour
        // buffers are the full frame by definition, and an extent on them is one more thing that
        // can disagree with the resource it describes. Ours are exactly sw x sh either way, so this
        // is fidelity to a measured integration rather than a fix for a bug we have seen.
        const sl::Extent area{0, 0, sw, sh};
        const sl::ResourceTag tags[] = {
            sl::ResourceTag{&color_in, sl::kBufferTypeUpliftInputColor,
                            sl::ResourceLifecycle::eValidUntilEvaluate, nullptr},
            sl::ResourceTag{&color_out, sl::kBufferTypeUpliftOutputColor,
                            sl::ResourceLifecycle::eValidUntilEvaluate, nullptr},
            sl::ResourceTag{&depth, sl::kBufferTypeDepth,
                            sl::ResourceLifecycle::eValidUntilEvaluate, &area},
            sl::ResourceTag{&motion, sl::kBufferTypeMotionVectors,
                            sl::ResourceLifecycle::eValidUntilEvaluate, &area},
        };
        const auto command_buffer = reinterpret_cast<sl::CommandBuffer*>(cmd);
        if (result == sl::Result::eOk) {
            stage = "slSetTagForFrame";
            result = streamline::api().set_tag_for_frame(*token, viewport, tags, 4, command_buffer);
        }
        if (result == sl::Result::eOk) {
            stage = "slEvaluateFeature";
            const sl::BaseStructure* inputs[] = {&viewport};
            ::mkwvk::rec::boundary(::mkwvk::rec::B_DIRECT);   // (Streamline records into cmd itself: what this thread recorded through vk_work.h reaches cmd first)
            result = streamline::api().evaluate_feature(sl::kFeatureDLSS_NR, *token, inputs, 1,
                                                        command_buffer);
            // Out of VRAM is the model saying it had to work small, not that it failed.
            if (result == sl::Result::eWarnOutOfVRAM) result = sl::Result::eOk;
        }
    }
    if (result != sl::Result::eOk) {
        const bool only_options = strcmp(stage, "slDLSSNRSetOptions") == 0;
        logf("vk: DLSS 5 neural rendering failed at %s (Streamline result 0x%x); %s", stage,
             (unsigned)result,
             only_options ? "waiting for those settings to change" : "turning it off for this run");
        // The one failure with a known first suspect. Our DlssNrOptions carries a struct GUID
        // recovered from a prerelease plugin and corroborated by nobody who had the real header, so
        // a plugin that does not recognize the struct says exactly this. Named here so the next
        // person does not spend the day the layout already cost.
        if (only_options && result == sl::Result::eErrorInvalidParameter)
            logf("vk: ... an invalid parameter from slDLSSNRSetOptions usually means this "
                 "Streamline does not know our options struct: check the GUID and struct version in "
                 "gpu/vk/vk_dlssnr_options.h against the plugin");
        if (only_options) s.options_rejected.store(true, std::memory_order_relaxed);
        else s.failed.store(true, std::memory_order_relaxed);
        return VK_NULL_HANDLE;
    }
    // The model wrote s.out. The caller presents it instead of the game's image.
    barrier(cmd, s.out.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT);
    if (out_view) *out_view = s.out.view;
    return s.out.image;
}

void shutdown() {
    State& s = state();
    if (!s.out.image && !s.in.image) return;
    {
        std::lock_guard qk(dev.present_mu);
        device_wait_idle_locked();
    }
    if (streamline::nr_available())
        streamline::api().free_resources(sl::kFeatureDLSS_NR, sl::ViewportHandle{kViewport});
    drop(s.in), drop(s.out), drop(s.depth), drop(s.motion);
    s.width = s.height = 0;
    s.scene_cleared = false;
}

} // namespace mkwvk::dlssnr

#else // no Streamline headers, or not Windows

namespace mkwvk::dlssnr {

bool built_with_sdk() { return false; }
bool capable() { return false; }
bool asked() { return false; }
VkImage apply(VkCommandBuffer, VkImage, uint32_t, uint32_t, VkImageView*) { return VK_NULL_HANDLE; }
void shutdown() {}

} // namespace mkwvk::dlssnr

#endif
