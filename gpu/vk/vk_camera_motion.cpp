// (hookshot) See vk_camera_motion.h. One pipeline, one R16G16F target, one dispatch.
#include <cmath>
#include <cstring>

#include "vk_camera_motion.h"
#include "vk_texture.h"

#include "shaders/camera_motion_comp_spv.h"
#include "vk_work.h"   // (last: its vkCmd* go through the recorder, RX_VK_RECORDER; vk_work.h)

namespace mkwvk::camera_motion {
namespace {

struct Push {
    float inverse_view_projection[16];
    float previous_view_projection[16];
    int32_t render_size[2];
    float far_depth;
    float pad;
};
static_assert(sizeof(Push) == 16 * 4 * 2 + 16, "the shader's push constants");

struct State {
    bool tried = false;
    VkDescriptorSetLayout set = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkSampler point = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation memory = nullptr;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0;
    Matrix previous{};
    Matrix current{};   ///< the matrix the last record() was given, for the same-frame guard
    bool have_previous = false;
};

State g;

/// The inverse of a row-major 4x4 by Gauss-Jordan, in doubles. False when it is singular, which a
/// projection with a zero field of view or a degenerate viewport can be: the caller then writes no
/// motion rather than reprojecting through nonsense.
bool invert(const float* a, float* out) {
    double m[4][8]{};
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            const float v = a[r * 4 + c];
            if (!(v == v) || v > 1e30f || v < -1e30f) return false;   // (NaN or absurd)
            m[r][c] = v;
            m[r][c + 4] = r == c ? 1.0 : 0.0;
        }
    }
    for (int c = 0; c < 4; c++) {
        int pivot = c;
        for (int r = c + 1; r < 4; r++) {
            if (std::abs(m[r][c]) > std::abs(m[pivot][c])) pivot = r;
        }
        const double d = m[pivot][c];
        if (d > -1e-12 && d < 1e-12) return false;
        if (pivot != c) {
            for (int k = 0; k < 8; k++) {
                const double t = m[c][k];
                m[c][k] = m[pivot][k];
                m[pivot][k] = t;
            }
        }
        for (int k = 0; k < 8; k++) m[c][k] /= d;
        for (int r = 0; r < 4; r++) {
            if (r == c) continue;
            const double f = m[r][c];
            if (f == 0.0) continue;
            for (int k = 0; k < 8; k++) m[r][k] -= f * m[c][k];
        }
    }
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) out[r * 4 + c] = (float)m[r][c + 4];
    return true;
}

bool make_pipeline() {
    const VkDescriptorSetLayoutBinding b[3]{
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1000, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    li.bindingCount = 3, li.pBindings = b;
    if (vkCreateDescriptorSetLayout(dev.device, &li, nullptr, &g.set) != VK_SUCCESS) return false;
    const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1, pi.pSetLayouts = &g.set;
    pi.pushConstantRangeCount = 1, pi.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(dev.device, &pi, nullptr, &g.layout) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = sizeof CAMERA_MOTION_COMP_SPV, mi.pCode = CAMERA_MOTION_COMP_SPV;
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev.device, &mi, nullptr, &m) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
    ci.layout = g.layout;
    const VkResult made = vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, &ci, nullptr, &g.pipeline);
    vkDestroyShaderModule(dev.device, m, nullptr);
    if (made != VK_SUCCESS) {
        g.pipeline = VK_NULL_HANDLE;
        logf("vk: camera motion: its pipeline was not made (%d)", (int)made);
        return false;
    }
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    return vkCreateSampler(dev.device, &si, nullptr, &g.point) == VK_SUCCESS;
}

bool make_target(uint32_t width, uint32_t height) {
    if (g.image && g.width == width && g.height == height) return true;
    if (g.view) vkDestroyImageView(dev.device, g.view, nullptr), g.view = VK_NULL_HANDLE;
    if (g.image) destroy_image(g.image, g.memory), g.image = VK_NULL_HANDLE;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R16G16_SFLOAT;
    ci.extent = {width, height, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!create_image(ci, g.image, g.memory, false)) return false;
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = g.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = ci.format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(dev.device, &vi, nullptr, &g.view) != VK_SUCCESS) return false;
    g.width = width, g.height = height;
    // (GENERAL from here on, as the rest of this renderer's own images are)
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    b.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = g.image;
    b.subresourceRange = vi.subresourceRange;
    g.have_previous = false;   // (a new size has no history to reproject through)
    logf("vk: camera motion: %ux%u R16G16F", width, height);
    return true;
}

} // namespace

VkImageView record(VkCommandBuffer cmd, VkImageView depth, VkFormat depth_format, uint32_t width,
                   uint32_t height, const Matrix& view_projection) {
    if (!depth || width == 0 || height == 0) return VK_NULL_HANDLE;
    if (!dev.push_descriptor || !vkCmdPushDescriptorSetKHR) return VK_NULL_HANDLE;
    // (hookshot) Asked twice for the same frame -- the upscale seam wants motion, and so does frame
    // generation at present time -- this answers with what it already wrote. Recording it a second
    // time would reproject through the matrix the first recording had just stored as "previous",
    // which is this same frame's: exactly zero motion, silently, for whichever caller came second.
    if (g.view && g.have_previous && g.width == width && g.height == height &&
        memcmp(g.current.m, view_projection.m, sizeof g.current.m) == 0)
        return g.view;
    if (!g.tried) {
        g.tried = true;
        if (!make_pipeline()) return VK_NULL_HANDLE;
    }
    if (!g.pipeline || !make_target(width, height)) return VK_NULL_HANDLE;

    // GLSL loads a mat4 from sixteen floats column by column, and everything here is row major, so
    // both matrices go in transposed. Getting this wrong does not fail: it reprojects through the
    // transpose, which looks like plausible motion pointing the wrong way.
    const auto transpose_into = [](const float* src, float* dst) {
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++) dst[c * 4 + r] = src[r * 4 + c];
    };
    Push push{};
    float inverse_row[16];
    if (!invert(view_projection.m, inverse_row)) return VK_NULL_HANDLE;
    transpose_into(inverse_row, push.inverse_view_projection);
    // The first frame, and the frame after a reset, have nothing to reproject through: this frame's
    // matrix stands in, which makes the motion exactly zero rather than arbitrary.
    transpose_into(g.have_previous ? g.previous.m : view_projection.m, push.previous_view_projection);
    push.render_size[0] = (int32_t)width, push.render_size[1] = (int32_t)height;
    // (reversed depth: the far plane is 0. A format that is not reversed would want 1 here, and
    // nothing in this game has been seen using one)
    push.far_depth = 0.0f;
    (void)depth_format;

    const VkDescriptorImageInfo depth_info{VK_NULL_HANDLE, depth, VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo motion_info{VK_NULL_HANDLE, g.view, VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo sampler_info{g.point, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    VkWriteDescriptorSet w[3]{};
    for (int i = 0; i < 3; i++) w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, w[i].descriptorCount = 1;
    w[0].dstBinding = 0, w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, w[0].pImageInfo = &depth_info;
    w[1].dstBinding = 1, w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, w[1].pImageInfo = &motion_info;
    w[2].dstBinding = 1000, w[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, w[2].pImageInfo = &sampler_info;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipeline);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.layout, 0, 3, w);
    vkCmdPushConstants(cmd, g.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, &push);
    vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);

    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);

    g.previous = view_projection;
    g.current = view_projection;   // (what the guard above compares the next ask against)
    g.have_previous = true;
    return g.view;
}

VkImage motion_image() {
    return g.image;
}

VkFormat motion_format() {
    return g.image ? VK_FORMAT_R16G16_SFLOAT : VK_FORMAT_UNDEFINED;
}

void reset() {
    g.have_previous = false;
}

void shutdown() {
    if (g.view) vkDestroyImageView(dev.device, g.view, nullptr);
    if (g.image) destroy_image(g.image, g.memory);
    if (g.point) vkDestroySampler(dev.device, g.point, nullptr);
    if (g.pipeline) vkDestroyPipeline(dev.device, g.pipeline, nullptr);
    if (g.layout) vkDestroyPipelineLayout(dev.device, g.layout, nullptr);
    if (g.set) vkDestroyDescriptorSetLayout(dev.device, g.set, nullptr);
    g = State{};
}

} // namespace mkwvk::camera_motion
