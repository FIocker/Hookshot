// (hookshot) Open neural rendering; see gpu/vk/vk_opennr.h for what it is and what the player supplies.
//
// The frame, in the frame's own command buffer (docs/neural-rendering.md has the why of each step):
//   the window's frame -> our RGBA8 copy (a blit: whatever the frame's format, the copy is code values)
//   -> opennr_pre.comp: the network's sixteen input lanes on the padded field
//   -> the network (gpu/vk/opennr: OpenDLSS-NR, pre-recorded once per size into a secondary command buffer)
//   -> opennr_post.comp: the head on the frame, the temporal blend, the next history, style and intensity
//   -> our RGBA8 output, which present_image shows instead of the frame.
// Images stay in VK_IMAGE_LAYOUT_GENERAL; the network is foreign work and is fenced with whole-pipeline barriers on
// both sides; nothing is allocated per frame. Loading the model takes seconds, so its check (the SHA-256 of 141 MiB)
// runs on a thread of its own and the frames go out untouched until it is done.
#define VK_ENABLE_BETA_EXTENSIONS   // (VK_NV_cuda_kernel_launch's feature struct is in vulkan_beta.h)

#include "vk_opennr.h"
#include "vk_dlssnr_ngx.h"   // (NrFrame)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>

#include "opennr/opennr_khr.h"
#include "opennr/opennr_net.h"
#include "opennr_ptx.h"   // (nx2/build.py generates it from third_party/opendlss-nr/scripts/ptx)
#include "shaders/opennr_kernels_spv.h"
#include "shaders/opennr_khr_spv.h"
#include "shaders/opennr_post_comp_spv.h"
#include "shaders/opennr_pre_comp_spv.h"
#include "vk_camera_motion.h"
#include "vk_draw.h"
#include "vk_menu.h"
#include "vk_paths.h"
#include "vk_work.h"   // (last: its vkCmd* go through the recorder, RX_VK_RECORDER; vk_work.h)

namespace mkwvk::opennr {
namespace {

constexpr VkImageSubresourceRange kOneLevel{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
constexpr uint32_t kTimingRing = 4;

struct Image {
    VkImage image{};
    VkImageView view{};
    VmaAllocation memory{};
};

/// opennr_common.glsl's push constants
struct Push {
    uint32_t full_width, full_height, valid_width, valid_height;
    uint32_t seed, history_valid, motion_valid, auto_mask;
    float style, local_tone, local_structure, skin;
    float motion_size[2];
    float blend_scale, intensity;
};
static_assert(sizeof(Push) == 64, "opennr_common.glsl's push block");

enum Phase : int { kIdle, kChecking, kChecked, kReady, kNoModel, kFailed };

struct State {
    // the device (vk_main.cpp create_device)
    bool want_coop2{}, want_fp8{}, want_cuda{}, want_sm{};
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV coop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV};
    VkPhysicalDeviceShaderFloat8FeaturesEXT fp8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    VkPhysicalDeviceCudaKernelLaunchFeaturesNV cuda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV};
    VkPhysicalDeviceShaderSMBuiltinsFeaturesNV sm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES_NV};
    bool device_ok{};
    opennr_net::Route route = opennr_net::Route::Fast;
    const opennr_khr::Variant* khr{};   // (the khr route's kernel variant, when that is the route)
    // the model
    std::atomic<int> phase{kIdle};
    std::atomic<bool> check_done{};
    std::thread checker;
    // (a check still running at exit: let go of, not destroyed joinable, which is std::terminate. It is joined on a later
    // apply with the row on, or in shutdown; the row turned off mid-check left it joinable in this static for good)
    ~State() {
        if (checker.joinable()) checker.detach();
    }
    opennr_net::ModelCheck check;
    std::string model_dir;
    bool last_asked{};
    opennr_net::Network* net{};
    opennr_net::Built built{};
    uint32_t width{}, height{};
    // the frame's own resources
    Image in{}, out{}, history[2]{}, no_motion{};
    bool fresh{};   // (just made: their first transition and clear go in this frame's command buffer)
    VkSampler linear{}, nearest{};
    VkDescriptorSetLayout set_layout{};
    VkPipelineLayout layout{};
    VkPipeline pre{}, post{};
    VkDescriptorPool pool{};
    VkDescriptorSet sets[2]{};
    VkImageView bound_motion{};
    uint32_t parity{};
    bool history_valid{};
    uint64_t frame{};
    float last_view_projection[16]{};
    bool have_view_projection{};
    // the GPU time of the whole pass, read back kTimingRing frames later (never waited for)
    VkQueryPool queries{};
    uint64_t timed{};
    double ms_sum{}, ms_min{1e30};
    uint32_t ms_count{};
    char why[320]{};
};

State& state() {
    static State s;
    return s;
}

bool has(const std::vector<VkExtensionProperties>& all, const char* name) {
    for (const VkExtensionProperties& e : all)
        if (!strcmp(e.extensionName, name)) return true;
    return false;
}

void set_why(State& s, const char* text) {
    snprintf(s.why, sizeof s.why, "%s", text);
}

std::string game_folder() {
    char exe[MAX_PATH];
    const DWORD n = game_module_path(exe, MAX_PATH);
    std::string path(exe, n);
    const size_t slash = path.find_last_of("\\/");
    return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

/// The model directory: RX_VK_OPENNR_MODEL, else nr-model in the game folder
std::string model_directory() {
    if (const char* e = getenv("RX_VK_OPENNR_MODEL"); e && *e) return e;
    return game_folder() + "/nr-model";
}

/// RX_VK_OPENNR_ROUTE=khr: the khr route even on a GPU that runs upstream's (its variant: RX_VK_OPENNR_KHR_VARIANT)
bool khr_forced() {
    const char* r = getenv("RX_VK_OPENNR_ROUTE");
    return r && !strcmp(r, "khr");
}

/// OpenDLSS-NR's host code reads its kernels from a folder (shaders/*.spv, ptx/*.ptx; the khr route's from
/// shaders-khr-<variant>/*.spv). The renderer carries them (the SPIR-V checked in, the PTX generated at build time) and
/// writes them out once, to a folder named by their hash, so a stale folder from another build is never read.
std::string materialize_kernels(std::string* error) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void* data, size_t n) {
        const auto* p = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
    };
    for (const auto& m : opennr_spv::kModules) mix(m.words, m.count * 4u), mix(m.name, strlen(m.name));
    for (const auto& m : opennr_khr_spv::kModules) mix(m.words, m.count * 4u), mix(m.entry, strlen(m.entry));
    for (const auto& f : opennr_ptx::kFiles) mix(f.text, strlen(f.text)), mix(f.name, strlen(f.name));
    char tag[24];
    snprintf(tag, sizeof tag, "%016llx", (unsigned long long)h);
    const std::filesystem::path dir = std::filesystem::path(game_folder()) / "hookshot_opennr_kernels" / tag;
    std::error_code ec;
    if (std::filesystem::exists(dir / "complete", ec)) return dir.string();
    std::filesystem::create_directories(dir / "shaders", ec);
    std::filesystem::create_directories(dir / "ptx", ec);
    std::filesystem::create_directories(dir / "no-ptx", ec);   // (the GLSL routes' empty PTX folder, opennr_net.cpp)
    auto write = [&](const std::filesystem::path& p, const void* data, size_t n) {
        FILE* f = fopen(p.string().c_str(), "wb");   // (narrow paths are UTF-8: nx2/win_manifest.py)
        const bool ok = f && fwrite(data, 1, n, f) == n;
        if (f) fclose(f);
        if (!ok && error) *error = "cannot write " + p.string();
        return ok;
    };
    for (const auto& m : opennr_spv::kModules)
        if (!write(dir / "shaders" / (std::string(m.name) + ".spv"), m.words, m.count * 4u)) return {};
    // (a khr variant's folder: its own matrix kernels, the route's common ones, and upstream's vendor-neutral ones --
    // the same thirteen names; scripts/opennr_kernels.py khr_folder_modules is the same rule)
    for (const opennr_khr::Variant& v : opennr_khr::kVariants) {
        const std::filesystem::path folder = dir / (std::string("shaders-khr-") + v.name);
        std::filesystem::create_directories(folder, ec);
        for (const auto& m : opennr_spv::kModules) {
            const uint32_t* words = m.words;
            unsigned count = m.count;
            for (const char* prefix : {v.name, "common"})
                for (const auto& k : opennr_khr_spv::kModules)
                    if (words == m.words && std::string(k.entry) == std::string(prefix) + "/" + m.name)
                        words = k.words, count = k.count;
            if (!write(folder / (std::string(m.name) + ".spv"), words, count * 4u)) return {};
        }
    }
    for (const auto& f : opennr_ptx::kFiles)
        if (!write(dir / "ptx" / f.name, f.text, strlen(f.text))) return {};
    if (!write(dir / "complete", "", 0)) return {};
    return dir.string();
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
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    if (!create_image(ci, t.image, t.memory, false)) return false;
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = t.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = kOneLevel;
    if (vkCreateImageView(dev.device, &vi, nullptr, &t.view) != VK_SUCCESS) {
        drop(t);
        return false;
    }
    return true;
}

VkPipeline pipeline(const uint32_t* words, size_t bytes, VkPipelineLayout layout) {
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = bytes;
    mi.pCode = words;
    VkShaderModule module{};
    if (vkCreateShaderModule(dev.device, &mi, nullptr, &module) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    ci.layout = layout;
    VkPipeline p{};
    if (vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, &ci, nullptr, &p) != VK_SUCCESS) p = VK_NULL_HANDLE;
    vkDestroyShaderModule(dev.device, module, nullptr);
    return p;
}

/// The two passes' layout, pipelines, samplers and descriptor sets: once, whatever the size
bool make_passes(State& s) {
    if (s.pre) return true;
    const VkDescriptorSetLayoutBinding b[7] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 7;
    li.pBindings = b;
    if (vkCreateDescriptorSetLayout(dev.device, &li, nullptr, &s.set_layout) != VK_SUCCESS) return false;
    const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1;
    pi.pSetLayouts = &s.set_layout;
    pi.pushConstantRangeCount = 1;
    pi.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(dev.device, &pi, nullptr, &s.layout) != VK_SUCCESS) return false;
    s.pre = pipeline(OPENNR_PRE_COMP_SPV, sizeof OPENNR_PRE_COMP_SPV, s.layout);
    s.post = pipeline(OPENNR_POST_COMP_SPV, sizeof OPENNR_POST_COMP_SPV, s.layout);
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    if (vkCreateSampler(dev.device, &si, nullptr, &s.linear) != VK_SUCCESS) return false;
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    if (vkCreateSampler(dev.device, &si, nullptr, &s.nearest) != VK_SUCCESS) return false;
    const VkDescriptorPoolSize sizes[3] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6},
                                           {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
                                           {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 2;
    dpi.poolSizeCount = 3;
    dpi.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(dev.device, &dpi, nullptr, &s.pool) != VK_SUCCESS) return false;
    const VkDescriptorSetLayout layouts[2] = {s.set_layout, s.set_layout};
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = s.pool;
    ai.descriptorSetCount = 2;
    ai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(dev.device, &ai, s.sets) != VK_SUCCESS) return false;
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 2 * kTimingRing;
    if (vkCreateQueryPool(dev.device, &qi, nullptr, &s.queries) != VK_SUCCESS) s.queries = VK_NULL_HANDLE;
    return s.pre && s.post;
}

/// The descriptor sets, for parity p: the history read is history[p], the history written history[1 - p]. Only with
/// none of them in flight (a new size, or the motion target changing, both behind a device wait).
void write_sets(State& s, VkImageView motion) {
    for (uint32_t p = 0; p < 2; p++) {
        const VkDescriptorImageInfo in{VK_NULL_HANDLE, s.in.view, VK_IMAGE_LAYOUT_GENERAL};
        const VkDescriptorImageInfo prev{s.linear, s.history[p].view, VK_IMAGE_LAYOUT_GENERAL};
        const VkDescriptorImageInfo mot{s.nearest, motion ? motion : s.no_motion.view, VK_IMAGE_LAYOUT_GENERAL};
        const VkDescriptorBufferInfo features{s.built.features, 0, s.built.features_bytes};
        const VkDescriptorBufferInfo head{s.built.head, 0, s.built.head_bytes};
        const VkDescriptorImageInfo out{VK_NULL_HANDLE, s.out.view, VK_IMAGE_LAYOUT_GENERAL};
        const VkDescriptorImageInfo next{VK_NULL_HANDLE, s.history[1 - p].view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w[7]{};
        const VkDescriptorType types[7] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                           VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                           VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
        const VkDescriptorImageInfo* images[7] = {&in, &prev, &mot, nullptr, nullptr, &out, &next};
        const VkDescriptorBufferInfo* buffers[7] = {nullptr, nullptr, nullptr, &features, &head, nullptr, nullptr};
        for (uint32_t i = 0; i < 7; i++) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = s.sets[p];
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = types[i];
            w[i].pImageInfo = images[i];
            w[i].pBufferInfo = buffers[i];
        }
        vkUpdateDescriptorSets(dev.device, 7, w, 0, nullptr);
    }
    s.bound_motion = motion;
}

bool make_images(State& s, uint32_t w, uint32_t h) {
    const VkImageUsageFlags color = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const VkImageUsageFlags history = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                      VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const bool ok = make(s.in, VK_FORMAT_R8G8B8A8_UNORM, w, h, color) &&
                    make(s.out, VK_FORMAT_R8G8B8A8_UNORM, w, h, color) &&
                    make(s.history[0], VK_FORMAT_R16G16B16A16_SFLOAT, w, h, history) &&
                    make(s.history[1], VK_FORMAT_R16G16B16A16_SFLOAT, w, h, history) &&
                    (s.no_motion.image || make(s.no_motion, VK_FORMAT_R16G16_SFLOAT, 1, 1,
                                               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
    s.fresh = true;
    return ok;
}

void barrier(VkCommandBuffer cmd, VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
             VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = src_stage, mb.srcAccessMask = src_access;
    mb.dstStageMask = dst_stage, mb.dstAccessMask = dst_access;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
}

void to_general(VkCommandBuffer cmd, VkImage image) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = kOneLevel;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1, di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);
}

/// The model checked (on its thread) -> the network loaded and built for w x h, on the display thread with the
/// queue's lock held (upstream's uploads submit and wait on it). False: s.why says why, and it stays off.
bool load(State& s, uint32_t w, uint32_t h) {
    const auto t0 = std::chrono::steady_clock::now();
    std::string error;
    const std::string kernels = materialize_kernels(&error);
    if (kernels.empty()) {
        logf("vk: open NR: the kernels could not be written out (%s)", error.c_str());
        set_why(s, "Not available: its kernels could not be written to the game folder (the log has more).");
        return false;
    }
    opennr_net::Device d;
    d.instance = dev.instance, d.physical = dev.physical, d.device = dev.device;
    d.queue_family = dev.queue_family, d.queue_index = 0;
    d.get_instance_proc_addr = vkx::instance_proc_addr();
    {
        std::lock_guard qk(dev.present_mu);
        queue_flush_locked();
        s.net = opennr_net::create(d, s.model_dir, kernels, s.route, s.khr ? s.khr->name : nullptr, &error);
        if (s.net && !opennr_net::build(s.net, w, h, &s.built, &error)) {
            device_wait_idle_locked();
            std::lock_guard pq(dev.present_queue_mu);   // (upstream's context waits for the device as it goes)
            opennr_net::destroy(s.net);
            s.net = nullptr;
        }
    }
    if (!s.net) {
        logf("vk: open NR: the network did not start: %s", error.c_str());
        set_why(s, "Not available: the network did not start on this GPU (the log says why).");
        return false;
    }
    if (!make_passes(s) || !make_images(s, w, h)) {
        logf("vk: open NR: the frame's passes or images were not made");
        set_why(s, "Not available: the frame's passes could not be made (the log has more).");
        return false;
    }
    write_sets(s, VK_NULL_HANDLE);
    s.width = w, s.height = h, s.history_valid = false;
    logf("vk: open NR: running on %ux%u frames (field %ux%u), %s: %u dispatches, %s; loaded in %.1f s", w, h,
         s.built.full_width, s.built.full_height, opennr_net::route_name(s.route), s.built.dispatches,
         s.built.chained ? "counter-chained" : "barriers between launches",
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

/// A new frame size: the graph and the images again, with the device idle (frames in flight use the old ones)
bool resize(State& s, uint32_t w, uint32_t h) {
    std::string error;
    bool ok;
    {
        std::lock_guard qk(dev.present_mu);
        device_wait_idle_locked();
        ok = opennr_net::build(s.net, w, h, &s.built, &error);
    }
    if (!ok) {
        logf("vk: open NR: no network for %ux%u frames: %s", w, h, error.c_str());
        set_why(s, "Not available at this window size (the log says why).");
        return false;
    }
    if (!make_images(s, w, h)) return false;
    write_sets(s, VK_NULL_HANDLE);
    s.width = w, s.height = h, s.history_valid = false;
    logf("vk: open NR: now %ux%u frames (field %ux%u)", w, h, s.built.full_width, s.built.full_height);
    return true;
}

void read_timing(State& s, VkCommandBuffer cmd, uint32_t slot) {
    if (!s.queries) return;
    if (s.timed >= kTimingRing) {   // (this slot's last stamps, kTimingRing frames old: read only if they are there)
        uint64_t t[2]{};
        if (vkGetQueryPoolResults(dev.device, s.queries, slot * 2, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) ==
                VK_SUCCESS && t[1] > t[0]) {
            const double ms = (double)(t[1] - t[0]) * dev.props.limits.timestampPeriod * 1e-6;
            s.ms_sum += ms, s.ms_min = std::min(s.ms_min, ms), s.ms_count++;
        }
        if (s.ms_count >= 300) {
            logf("vk: open NR: %.2f ms of GPU time a frame on average, %.2f at the least, over %u frames at %ux%u",
                 s.ms_sum / s.ms_count, s.ms_min, s.ms_count, s.width, s.height);
            s.ms_sum = 0, s.ms_min = 1e30, s.ms_count = 0;
        }
    }
    vkCmdResetQueryPool(cmd, s.queries, slot * 2, 2);
}

} // namespace

bool selected() {
    // The device is made for the network (its extensions and features, below) when it can be wanted this run: the
    // Backend row's Open, RX_VK_NR_BACKEND=open or RX_VK_DLSS_NR_BACKEND=open|3, or a model folder already in place
    // (so Auto can fall to it), or a GPU that is not NVIDIA's -- AMD's above all, where Open is the only backend there
    // is, and whose khr route asks for nothing the renderer does not already (only E4M3 matrices, where there are any).
    // Asked once, at device creation (dev.props is the GPU by then): the device cannot gain extensions afterwards, so a
    // model folder put in place later on an NVIDIA card wants a restart, and why() says so.
    static const bool open = [] {
        if (const char* e = getenv("RX_VK_NR_BACKEND"); e && !strcmp(e, "open")) return true;
        if (const char* e = getenv("RX_VK_DLSS_NR_BACKEND"); e && (!strcmp(e, "open") || !strcmp(e, "3"))) return true;
        if (opts.nr_backend == 3) return true;
        if (dev.physical && dev.props.vendorID != 0x10de) return true;   // (not NVIDIA: Open, the khr route)
        FILE* f = fopen((model_directory() + "/manifest.json").c_str(), "rb");
        if (f) fclose(f);
        return f != nullptr;
    }();
    return open;
}

bool ready() { return selected() && state().device_ok && state().phase.load() != kNoModel; }

void device_extensions(std::vector<const char*>& want, const std::vector<VkExtensionProperties>& have) {
    if (!selected()) return;
    State& s = state();
    auto add = [&](const char* name, bool& flag, const char* why) {
        flag = has(have, name);
        if (!flag) {
            logf("vk: open NR: this GPU has no %s (%s)", name, why);
            return;
        }
        for (const char* w : want)
            if (!strcmp(w, name)) return;
        want.push_back(name);
        logf("vk: open NR wants %s: enabled", name);
    };
    // (RX_VK_OPENNR_ROUTE=khr: only what the khr route can use, E4M3 matrices where there are any; otherwise what
    // upstream's NVIDIA routes need too, where the GPU has it -- an AMD GPU has none of the NV ones, and runs khr)
    if (!khr_forced()) {
        add(VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME, s.want_coop2,
            "upstream's GLSL kernels' workgroup-scope matrices; the khr route needs none");
    }
    add(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, s.want_fp8,
        "E4M3 matrices: upstream's routes need them, the khr route widens E4M3 to f16 without them");
    if (!khr_forced()) {
        add(VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME, s.want_cuda, "the PTX kernels: without it the GLSL kernels run");
        add(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME, s.want_sm, "the SM count the persistent kernels size by");
    }
}

void device_features(void**& tail) {
    if (!selected()) return;
    State& s = state();
    if (s.want_coop2) *tail = &s.coop2, tail = &s.coop2.pNext;
    if (s.want_fp8) *tail = &s.fp8, tail = &s.fp8.pNext;
    if (s.want_cuda) *tail = &s.cuda, tail = &s.cuda.pNext;
    if (s.want_sm) *tail = &s.sm, tail = &s.sm.pNext;
}

void device_created() {
    if (!selected()) return;
    State& s = state();
    // What upstream's vk::DeviceRequirements turns on, against what this device was made with. The renderer turns
    // on every core feature its GPU has (vk_main.cpp), so the core ones are the GPU's; the Vulkan 1.1 ones are asked
    // again because the renderer does not keep them. These every route needs; the matrices decide which route.
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f11;
    vkGetPhysicalDeviceFeatures2(dev.physical, &f2);
    std::string missing;
    auto need = [&](bool have, const char* name) {
        if (!have) missing += missing.empty() ? name : std::string(", ") + name;
    };
    need(dev.coop_matrix, "VK_KHR_cooperative_matrix");
    need(f11.storageBuffer16BitAccess && f11.uniformAndStorageBuffer16BitAccess, "16-bit storage");
    need(dev.vk12.storageBuffer8BitAccess && dev.vk12.uniformAndStorageBuffer8BitAccess && dev.vk12.shaderInt8,
         "8-bit storage");
    need(dev.vk12.shaderFloat16, "shaderFloat16");
    need(dev.vk12.vulkanMemoryModel && dev.vk12.vulkanMemoryModelDeviceScope, "the Vulkan memory model");
    need(dev.vk12.bufferDeviceAddress, "buffer device addresses");
    need(dev.vk13.subgroupSizeControl && dev.vk13.computeFullSubgroups, "subgroup size control");
    need(dev.vk13.maintenance4, "maintenance4");
    need(dev.features.shaderInt16 && dev.features.shaderInt64, "16- and 64-bit integers");

    // upstream's NVIDIA routes: VK_NV_cooperative_matrix2 with all seven of its features, and E4M3 matrices
    const bool fp8 = s.want_fp8 && s.fp8.shaderFloat8 && s.fp8.shaderFloat8CooperativeMatrix;
    const bool nv = s.want_coop2 && s.coop2.cooperativeMatrixWorkgroupScope && s.coop2.cooperativeMatrixFlexibleDimensions &&
                    s.coop2.cooperativeMatrixReductions && s.coop2.cooperativeMatrixConversions &&
                    s.coop2.cooperativeMatrixPerElementOperations && s.coop2.cooperativeMatrixTensorAddressing &&
                    s.coop2.cooperativeMatrixBlockLoads && fp8;
    const bool cuda = nv && s.want_cuda && s.cuda.cudaKernelLaunchFeatures;
    // the khr route: the first of its kernel variants whose matrix shapes the driver lists (gpu/vk/opennr/opennr_khr.h)
    opennr_khr::Device kd;
    kd.float8_matrices = fp8;
    {
        VkPhysicalDeviceSubgroupSizeControlProperties sc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &sc;
        vkGetPhysicalDeviceProperties2(dev.physical, &p2);
        kd.min_subgroup = sc.minSubgroupSize, kd.max_subgroup = sc.maxSubgroupSize;
        kd.subgroup_control_compute = dev.vk13.subgroupSizeControl && dev.vk13.computeFullSubgroups &&
                                      (sc.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT);
    }
    std::string shapes;
    if (dev.coop_matrix && vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
        uint32_t n = 0;
        vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(dev.physical, &n, nullptr);
        std::vector<VkCooperativeMatrixPropertiesKHR> props(n, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
        vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(dev.physical, &n, props.data());
        auto type = [](VkComponentTypeKHR c) {
            return c == VK_COMPONENT_TYPE_FLOAT16_KHR         ? "f16"
                   : c == VK_COMPONENT_TYPE_FLOAT32_KHR       ? "f32"
                   : c == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT   ? "e4m3"
                   : c == VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT   ? "e5m2"
                   : c == VK_COMPONENT_TYPE_BFLOAT16_KHR      ? "bf16"
                                                              : "int";
        };
        for (const auto& p : props) {
            kd.shapes.push_back({p.MSize, p.NSize, p.KSize, p.AType, p.BType, p.CType, p.ResultType, p.scope});
            if (p.scope != VK_SCOPE_SUBGROUP_KHR || !strcmp(type(p.AType), "int")) continue;
            char one[64];
            snprintf(one, sizeof one, "%s%ux%ux%u %s>%s", shapes.empty() ? "" : ", ", p.MSize, p.NSize, p.KSize,
                     type(p.AType), type(p.ResultType));
            shapes += one;
        }
    }
    std::string khr_why;
    const char* forced_variant = getenv("RX_VK_OPENNR_KHR_VARIANT");
    s.khr = dev.coop_matrix ? opennr_khr::choose(kd, forced_variant, &khr_why) : nullptr;
    if (!dev.coop_matrix) khr_why = "no VK_KHR_cooperative_matrix";

    // The route: upstream's fast one where the CUDA launch is, its GLSL one where only its matrices are, the khr one
    // anywhere else it can run; RX_VK_OPENNR_ROUTE over that, where the GPU allows it
    if (cuda) s.route = opennr_net::Route::Fast;
    else if (nv) s.route = opennr_net::Route::Glsl;
    else s.route = opennr_net::Route::Khr;
    if (const char* r = getenv("RX_VK_OPENNR_ROUTE")) {
        if (!strcmp(r, "khr")) s.route = opennr_net::Route::Khr;
        else if (!strcmp(r, "glsl") && nv) s.route = opennr_net::Route::Glsl;
        else if (!strcmp(r, "reference") && nv) s.route = opennr_net::Route::Reference;
        else if (!strcmp(r, "fast") && cuda) s.route = opennr_net::Route::Fast;
        else logf("vk: open NR: RX_VK_OPENNR_ROUTE=%s cannot run here (%s); %s", r,
                  nv ? "no VK_NV_cuda_kernel_launch" : "upstream's routes need VK_NV_cooperative_matrix2 and E4M3 matrices",
                  opennr_net::route_name(s.route));
    }
    if (s.route == opennr_net::Route::Khr && !s.khr)
        missing += std::string(missing.empty() ? "" : "; ") + "matrices it can use (upstream's routes need "
                   "VK_NV_cooperative_matrix2 and E4M3 matrices; the khr route found " + khr_why + ")";
    s.device_ok = missing.empty();
    logf("vk: open NR: cooperative matrices listed: %s", shapes.empty() ? "none" : shapes.c_str());
    if (!s.device_ok) {
        logf("vk: open NR is not available on this GPU: it lacks %s", missing.c_str());
        set_why(s, "Not available on this GPU: it lacks what the network needs (the log names it).");
        return;
    }
    if (s.route == opennr_net::Route::Khr) {
        logf("vk: open NR: this GPU can run the network: %s, variant %s (%s; %u-lane subgroups)%s%s",
             opennr_net::route_name(s.route), s.khr->name, s.khr->what, s.khr->subgroup,
             khr_why.empty() ? "" : "; not the others: ", khr_why.c_str());
    } else {
        s.khr = nullptr;
        logf("vk: open NR: this GPU can run the network (%s)", opennr_net::route_name(s.route));
    }
    if (dev.features.robustBufferAccess) logf("vk: open NR: robust buffer access is on, which upstream measured at ~15%%");
}

bool asked() {
    // Live, every frame, like the Streamline backend's: the row the player just changed reaches the next frame.
    if (const char* e = getenv("RX_VK_DLSS_NR")) return atoi(e) != 0;
    return nr_settings().on;
}

const char* why() {
    if (!selected())
        return "No model at the start: supply yours in nr-model (readme), then restart the game.";
    return state().why;
}

VkImage apply(VkCommandBuffer cmd, VkImage src, uint32_t sw, uint32_t sh, VkImageView* out_view, const NrFrame* frame) {
    State& s = state();
    if (!src || !sw || !sh || !s.device_ok) return VK_NULL_HANDLE;
    if (frame && frame->reset) s.history_valid = false;   // (a cut the chain saw: no history carried over it)
    const bool on = asked();
    const bool turned_on = on && !s.last_asked;
    s.last_asked = on;
    if (!on) return VK_NULL_HANDLE;
    int phase = s.phase.load();
    // (a model put in place after "no model" is seen the next time the row is turned on)
    if (phase == kNoModel && turned_on) s.phase = phase = kIdle;
    if (phase == kIdle) {
        s.model_dir = model_directory();
        logf("vk: open NR: checking the model directory %s", s.model_dir.c_str());
        set_why(s, "Checking the model directory...");
        s.check_done = false;
        s.phase = kChecking;
        if (s.checker.joinable()) s.checker.join();
        s.checker = std::thread([&s] {
            s.check = opennr_net::check_model(s.model_dir);
            s.check_done = true;
        });
        return VK_NULL_HANDLE;
    }
    if (phase == kChecking) {
        if (!s.check_done) return VK_NULL_HANDLE;
        s.checker.join();
        if (!s.check.ok) {
            logf("vk: open NR: no usable model: %s. Hookshot does not include the network's weights and has no way to "
                 "get them: the model directory is the player's to supply (RX_VK_OPENNR_MODEL, or nr-model in the game "
                 "folder; docs/neural-rendering.md)", s.check.why.c_str());
            // (the hint is one line, as long as the row's longest: the path and the reason are in the log)
            set_why(s, s.check.why.rfind("no manifest.json", 0) == 0
                           ? "No model: Hookshot has none and cannot get one. Supply yours in nr-model (readme)."
                           : "The model directory was refused: the log says why.");
            s.phase = kNoModel;
            return VK_NULL_HANDLE;
        }
        logf("vk: open NR: model directory checked: %u stage files (%.1f MiB, SHA-256 as the manifest says), %u "
             "tensors, the 71-block network", s.check.stages, (double)s.check.bytes / 1048576.0, s.check.tensors);
        s.phase = kChecked;
        phase = kChecked;
    }
    if (phase == kChecked) {
        if (!load(s, sw, sh)) {
            s.phase = kFailed;
            return VK_NULL_HANDLE;
        }
        set_why(s, "");
        s.phase = phase = kReady;
    }
    if (phase != kReady) return VK_NULL_HANDLE;
    if (opennr_net::chain_timed_out(s.net)) {
        // (upstream's watchdog let a chained wait go: that frame was wrong, and chaining is not trusted again. The
        // same graph with a barrier after every launch, as upstream's demo falls back)
        logf("vk: open NR: a chained wait timed out; the network now runs with barriers between launches");
        opennr_net::fall_back_to_barriers(s.net);
        s.width = s.height = 0;   // (the size check below builds it again)
    }
    if ((sw != s.width || sh != s.height) && !resize(s, sw, sh)) {
        s.phase = kFailed;
        return VK_NULL_HANDLE;
    }

    // ---- the motion, for the temporal path: the scene's camera-only motion (vk_camera_motion.h), when there is a
    // scene. A camera that did not move gets no pass at all (zero motion: the history at the same pixel), because the
    // motion module answers a repeated matrix with the target it last wrote, which here would be a moving frame's.
    static const bool temporal = !getenv("RX_VK_OPENNR_TEMPORAL") || atoi(getenv("RX_VK_OPENNR_TEMPORAL")) != 0;
    VkImageView motion = VK_NULL_HANDLE;
    float motion_size[2] = {1.0f, 1.0f};
    if (temporal) {
        SceneFrameInputs scene;
        if (scene_frame_inputs(&scene) && scene.have_view_projection && scene.depth_view) {
            const bool moved = !s.have_view_projection ||
                               memcmp(s.last_view_projection, scene.view_projection, sizeof s.last_view_projection) != 0;
            memcpy(s.last_view_projection, scene.view_projection, sizeof s.last_view_projection);
            s.have_view_projection = true;
            if (moved) {
                camera_motion::Matrix vp{};
                memcpy(vp.m, scene.view_projection, sizeof vp.m);
                motion = camera_motion::record(cmd, scene.depth_view, scene.depth_format, scene.render_width,
                                               scene.render_height, vp);
                motion_size[0] = (float)scene.render_width, motion_size[1] = (float)scene.render_height;
            }
        }
        if (motion && motion != s.bound_motion) {   // (the motion target is new: its size changed. Rare)
            std::lock_guard qk(dev.present_mu);
            device_wait_idle_locked();
            write_sets(s, motion);
        }
    }

    const uint32_t slot = (uint32_t)(s.frame % kTimingRing);
    read_timing(s, cmd, slot);
    if (s.queries) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, s.queries, slot * 2);
    if (s.fresh) {
        for (VkImage i : {s.in.image, s.out.image, s.history[0].image, s.history[1].image, s.no_motion.image})
            to_general(cmd, i);
        const VkClearColorValue zero{};
        vkCmdClearColorImage(cmd, s.no_motion.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &kOneLevel);
        barrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        s.fresh = false;
    }

    // the frame into our copy: whatever the window's format, the copy holds its code values (the game's is only read)
    barrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstSubresource = blit.srcSubresource;
    blit.srcOffsets[1] = {(int32_t)sw, (int32_t)sh, 1};
    blit.dstOffsets[1] = blit.srcOffsets[1];
    vkCmdBlitImage(cmd, src, VK_IMAGE_LAYOUT_GENERAL, s.in.image, VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_NEAREST);
    barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

    const NrSettings want = nr_settings();
    Push push{};
    push.full_width = s.built.full_width, push.full_height = s.built.full_height;
    push.valid_width = sw, push.valid_height = sh;
    // (single-frame: one seed, so a still frame stays still; temporal: a new one each frame, as upstream's demo, the
    // history holding the result together)
    push.seed = temporal ? (uint32_t)s.frame : 1234u;
    push.history_valid = temporal && s.history_valid ? 1u : 0u;
    push.motion_valid = motion ? 1u : 0u;
    push.auto_mask = want.auto_mask ? 1u : 0u;
    push.style = (float)want.style;
    push.local_tone = want.local_tone, push.local_structure = want.local_structure, push.skin = want.skin;
    push.motion_size[0] = motion_size[0], push.motion_size[1] = motion_size[1];
    push.blend_scale = opennr_net::blend_scale(s.net);
    push.intensity = want.intensity;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pre);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.layout, 0, 1, &s.sets[s.parity], 0, nullptr);
    vkCmdPushConstants(cmd, s.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, &push);
    vkCmdDispatch(cmd, (s.built.full_width + 7) / 8, (s.built.full_height + 7) / 8, 1);
    // (the network is foreign work -- dispatches and CUDA launches -- so whole-pipeline barriers on both sides)
    barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    ::mkwvk::rec::boundary(::mkwvk::rec::B_DIRECT);   // (The open network records into cmd itself: what this thread recorded through vk_work.h reaches cmd first)
    opennr_net::record(s.net, cmd);
    barrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
    // (vkCmdExecuteCommands leaves the primary's bound state undefined: everything again)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.post);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.layout, 0, 1, &s.sets[s.parity], 0, nullptr);
    vkCmdPushConstants(cmd, s.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, &push);
    vkCmdDispatch(cmd, (sw + 7) / 8, (sh + 7) / 8, 1);
    barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT);
    if (s.queries) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, s.queries, slot * 2 + 1);
    s.timed++;
    s.frame++;
    s.parity ^= 1u;
    s.history_valid = true;
    if (out_view) *out_view = s.out.view;
    return s.out.image;
}

void shutdown() {
    State& s = state();
    if (s.checker.joinable()) s.checker.join();
    if (!s.net && !s.pre) return;
    std::lock_guard qk(dev.present_mu);
    device_wait_idle_locked();
    std::lock_guard pq(dev.present_queue_mu);
    opennr_net::destroy(s.net);
    s.net = nullptr;
    for (Image* i : {&s.in, &s.out, &s.history[0], &s.history[1], &s.no_motion}) drop(*i);
    if (s.queries) vkDestroyQueryPool(dev.device, s.queries, nullptr);
    if (s.pool) vkDestroyDescriptorPool(dev.device, s.pool, nullptr);
    if (s.pre) vkDestroyPipeline(dev.device, s.pre, nullptr);
    if (s.post) vkDestroyPipeline(dev.device, s.post, nullptr);
    if (s.layout) vkDestroyPipelineLayout(dev.device, s.layout, nullptr);
    if (s.set_layout) vkDestroyDescriptorSetLayout(dev.device, s.set_layout, nullptr);
    if (s.linear) vkDestroySampler(dev.device, s.linear, nullptr);
    if (s.nearest) vkDestroySampler(dev.device, s.nearest, nullptr);
    s.queries = VK_NULL_HANDLE, s.pool = VK_NULL_HANDLE, s.pre = s.post = VK_NULL_HANDLE;
    s.layout = VK_NULL_HANDLE, s.set_layout = VK_NULL_HANDLE, s.linear = s.nearest = VK_NULL_HANDLE;
    s.width = s.height = 0;
    s.phase = kIdle;
}

} // namespace mkwvk::opennr
