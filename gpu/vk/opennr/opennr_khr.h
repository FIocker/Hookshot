// (hookshot) The open backend's khr route: which of its kernel variants a GPU can run, from what its driver lists.
//
// The khr route is OpenDLSS-NR's network on VK_KHR_cooperative_matrix alone (gpu/vk/opennr/khr: no
// VK_NV_cooperative_matrix2, no PTX, no CUDA launch), for AMD's GPUs and anything else with KHR cooperative matrices.
// Its kernels come in variants, one per set of MMA shapes a driver may list (khr/khr_mma.glsl says what each define
// means); scripts/opennr_kernels.py KHR_VARIANTS is this same table and compiles each into its own kernel folder
// (shaders-khr-<name>), and tests/test_opennr_kernels.py checks the two tables agree.
//
// Header-only and free of Vulkan calls: the renderer (gpu/vk/vk_opennr.cpp) and the harness's probe
// (scripts/opennr_harness.py) both feed it what vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR returned.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace mkwvk::opennr_khr {

struct Variant {
    const char* name;
    bool e4;            ///< E4M3 cooperative matrices (VK_EXT_shader_float8's shaderFloat8CooperativeMatrix)
    uint32_t m, n, k;   ///< the FP8 GEMMs' MMA shape (f16 operands when !e4)
    uint32_t hk;        ///< the f16 GEMMs' MMA K (m x n x hk, f16 operands; with f16 accumulators only)
    bool acc32;         ///< f32 accumulators rounded to half after every step; else f16 accumulators
    uint32_t subgroup;  ///< the subgroup size its pipelines are made with (requiredSubgroupSize)
    const char* what;
};

/// In the order a GPU is offered them: the most faithful to the network's arithmetic first
inline constexpr Variant kVariants[] = {
    {"e4m3-k32-f16", true, 16, 16, 32, 16, false, 32, "NVIDIA's own MMA: the bytes of upstream's GLSL route"},
    {"e4m3-k16-f32", true, 16, 16, 16, 16, true, 32, "E4M3 into f32 (AMD RDNA 4)"},
    {"e4m3-k32-f32", true, 16, 16, 32, 16, true, 32, "E4M3 into f32"},
    {"f16-k16-f32", false, 16, 16, 16, 16, true, 32, "E4M3 widened to f16, into f32 (AMD RDNA 3)"},
    {"f16-k8-f32", false, 8, 8, 8, 8, true, 8, "E4M3 widened to f16, 8x8x8 into f32 at 8 lanes (llvmpipe)"},
};

/// One row of vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR
struct Shape {
    uint32_t m, n, k;
    VkComponentTypeKHR a, b, c, result;
    VkScopeKHR scope;
};

/// What decides it, besides the shapes
struct Device {
    std::vector<Shape> shapes;
    bool float8_matrices = false;            ///< VK_EXT_shader_float8 with shaderFloat8CooperativeMatrix, enabled
    uint32_t min_subgroup = 0, max_subgroup = 0;
    bool subgroup_control_compute = false;   ///< subgroupSizeControl + computeFullSubgroups, compute in requiredSubgroupSizeStages
};

inline bool lists(const Device& d, uint32_t m, uint32_t n, uint32_t k, VkComponentTypeKHR ab, VkComponentTypeKHR cr) {
    for (const Shape& s : d.shapes)
        if (s.m == m && s.n == n && s.k == k && s.a == ab && s.b == ab && s.c == cr && s.result == cr &&
            s.scope == VK_SCOPE_SUBGROUP_KHR)
            return true;
    return false;
}

/// Why this device cannot run variant v; empty when it can
inline std::string refusal(const Device& d, const Variant& v) {
    const VkComponentTypeKHR acc = v.acc32 ? VK_COMPONENT_TYPE_FLOAT32_KHR : VK_COMPONENT_TYPE_FLOAT16_KHR;
    const char* acc_name = v.acc32 ? "f32" : "f16";
    auto shape = [](uint32_t m, uint32_t n, uint32_t k, const char* ab, const char* cr) {
        return std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k) + " " + ab + " into " + cr;
    };
    if (v.e4 && !d.float8_matrices) return "no E4M3 cooperative matrices (VK_EXT_shader_float8)";
    if (v.e4 && !lists(d, v.m, v.n, v.k, VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT, acc))
        return "the driver lists no subgroup " + shape(v.m, v.n, v.k, "E4M3", acc_name);
    if (!v.e4 && !lists(d, v.m, v.n, v.k, VK_COMPONENT_TYPE_FLOAT16_KHR, acc))
        return "the driver lists no subgroup " + shape(v.m, v.n, v.k, "f16", acc_name);
    // (the f16 GEMMs use matrices only with f16 accumulators: otherwise they are integer arithmetic, khr/gemm_f16.comp)
    if (!v.acc32 && !lists(d, v.m, v.n, v.hk, VK_COMPONENT_TYPE_FLOAT16_KHR, acc))
        return "the driver lists no subgroup " + shape(v.m, v.n, v.hk, "f16", acc_name);
    if (!d.subgroup_control_compute || d.min_subgroup > v.subgroup || d.max_subgroup < v.subgroup)
        return "its compute pipelines cannot be made with " + std::to_string(v.subgroup) + "-lane subgroups";
    return {};
}

inline const Variant* find(const char* name) {
    for (const Variant& v : kVariants)
        if (name && !strcmp(v.name, name)) return &v;
    return nullptr;
}

/// The first variant the device can run (or `forced`, when it can run that one); null and `why` (one reason per
/// variant) when none
inline const Variant* choose(const Device& d, const char* forced, std::string* why) {
    std::string reasons;
    for (const Variant& v : kVariants) {
        if (forced && *forced && strcmp(forced, v.name)) continue;
        const std::string no = refusal(d, v);
        if (no.empty()) return &v;
        reasons += (reasons.empty() ? "" : "; ") + std::string(v.name) + ": " + no;
    }
    if (why) *why = forced && *forced && !find(forced) ? std::string("no khr variant is named ") + forced : reasons;
    return nullptr;
}

} // namespace mkwvk::opennr_khr
