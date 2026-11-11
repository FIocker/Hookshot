// (hookshot) The open neural rendering backend's network layer: OpenDLSS-NR's host code
// (third_party/opendlss-nr/src) behind a small interface, so the renderer's side (gpu/vk/vk_opennr.cpp) never
// includes it. The two cannot meet in one translation unit: upstream's code calls Vulkan through the volk shim
// (gpu/vk/opennr/volk.h), the renderer through vkx (vk_loader.h), and both spell the functions the same.
//
// The renderer's side only hands it a device and reads two buffers back, whatever kernels run behind it: upstream's
// (the Fast, Glsl and Reference routes, NVIDIA-only) or the khr route's (gpu/vk/opennr/khr, VK_KHR_cooperative_matrix
// alone: AMD and the rest), which run upstream's unfused graph on kernels of their own (docs/neural-rendering.md).
//
// Nothing here throws: upstream's code reports with exceptions, which are caught at this boundary and come back as
// text. Nothing here reads or writes the frame; the renderer's two passes do (gpu/vk/shaders/opennr_*.comp).
#pragma once

#include <cstdint>
#include <string>

#include <vulkan/vulkan.h>

namespace mkwvk::opennr_net {

/// A model directory checked the way upstream's nr::Model and nr::Graph check it, before any GPU work: the manifest
/// parses; every stage file is under model/, has the packed length the manifest says and the SHA-256 it says; every
/// tensor lies inside a stage that exists; and the network is the 71-block one of 310.8.0. CPU only, and seconds on
/// a real model (the hashing), so the renderer runs it off the display thread.
struct ModelCheck {
    bool ok = false;
    std::string why;        ///< what is wrong, for the log and the menu, when not ok
    uint64_t bytes = 0;     ///< the stage files' size
    uint32_t stages = 0, tensors = 0;
};
[[nodiscard]] ModelCheck check_model(const std::string& directory);

struct Device {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queue_family = 0, queue_index = 0;
    PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr;
};

/// Which kernels run (upstream's routes; every one gives the same bytes on a given GPU, which the harness checks)
enum class Route {
    Fast,        ///< upstream's default: PTX through VK_NV_cuda_kernel_launch, counter chaining
    Glsl,        ///< the fused GLSL kernels only (every DLSS5VK_PTX_* off): no CUDA launch needed
    Reference,   ///< the least fused GLSL route (DLSS5VK_UNFUSED=1): every intermediate materialized, up to 2560x1440
    Khr,         ///< upstream's unfused graph on the khr kernels (gpu/vk/opennr/khr): no NV extension at all
};

/// What a build for one size gives the renderer: the f32 [full rows][16] input it writes and the f32 [full rows][4]
/// head it reads, on the padded field (upstream's Geometry)
struct Built {
    uint32_t full_width = 0, full_height = 0;
    VkBuffer features = VK_NULL_HANDLE, head = VK_NULL_HANDLE;
    VkDeviceSize features_bytes = 0, head_bytes = 0;
    uint32_t dispatches = 0;
    bool chained = false;
};

class Network;

/// Loads the model (already checked: its hashes are not computed again) and the kernels from `kernel_dir`
/// (shaders/*.spv and ptx/*.ptx; for the khr route shaders-khr-<khr_variant>/*.spv, a name of opennr_khr.h kVariants).
/// Submits uploads on the device's queue and waits for them: the caller holds whatever lock that queue takes. Null and
/// `error` on failure.
[[nodiscard]] Network* create(const Device& device, const std::string& model_dir, const std::string& kernel_dir,
                              Route route, const char* khr_variant, std::string* error);
/// The graph for a frame of w x h, recorded once into a secondary command buffer that record() then executes every
/// frame (nothing is allocated per frame). The previous size's graph is released: the device must be idle. The first
/// build compiles every pipeline and JITs every PTX module, which is most of a second.
[[nodiscard]] bool build(Network* network, uint32_t width, uint32_t height, Built* out, std::string* error);
/// The network, into the frame's command buffer: its features buffer must be written and made visible to compute
/// before this; its head is written when this is done (the caller's barrier makes it visible).
void record(Network* network, VkCommandBuffer primary);
/// block70.layer0.blend_scale, the model's learned cap on the temporal blend (1 when the model has none)
[[nodiscard]] float blend_scale(const Network* network);
/// A counter-chained wait gave up in a finished frame (upstream's watchdog: that frame's bytes are wrong). The
/// caller then calls fall_back_to_barriers and builds again, as upstream's demo does (NrPass::fallBackToBarriers).
[[nodiscard]] bool chain_timed_out(Network* network);
/// No counter chaining from now on: a barrier after every launch. Takes effect at the next build.
void fall_back_to_barriers(Network* network);
[[nodiscard]] const char* route_name(Route route);
/// Releases everything. The device must be idle and every queue's lock held (upstream's context waits for the device).
void destroy(Network* network);

} // namespace mkwvk::opennr_net
