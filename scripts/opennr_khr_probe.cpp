// (hookshot) The open NR harness's device probe (scripts/opennr_harness.py builds and runs it): for every Vulkan device,
// whether upstream's NVIDIA routes can run there (VK_NV_cooperative_matrix2 with all seven of its features and E4M3
// cooperative matrices) and which khr variants (gpu/vk/opennr/opennr_khr.h) it lists the matrix shapes for -- the same
// decision gpu/vk/vk_opennr.cpp makes for the renderer's device, from the same table.
//
// One line per device and one per variant, tab-separated, for the harness to read:
//   device <index> <name> <driver id> nv=<0|1> subgroups=<min>-<max>
//   variant <index> <name> ok|<why not>
#include "volk.h"   // (the harness's shim over libvulkan, gpu/vk/opennr/volk.h)

#include <cstdio>
#include <vector>

#include "opennr_khr.h"

using namespace mkwvk;

int main() {
    if (volkInitialize() != VK_SUCCESS) {
        printf("error\tno Vulkan loader\n");
        return 1;
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ii.pApplicationInfo = &app;
    VkInstance instance{};
    if (vkCreateInstance(&ii, nullptr, &instance) != VK_SUCCESS) {
        printf("error\tvkCreateInstance failed\n");
        return 1;
    }
    volkLoadInstance(instance);
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    for (uint32_t i = 0; i < count; i++) {
        VkPhysicalDevice p = devices[i];
        uint32_t n = 0;
        vkEnumerateDeviceExtensionProperties(p, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> ext(n);
        vkEnumerateDeviceExtensionProperties(p, nullptr, &n, ext.data());
        auto has = [&](const char* name) {
            for (const VkExtensionProperties& e : ext)
                if (!strcmp(e.extensionName, name)) return true;
            return false;
        };
        VkPhysicalDeviceCooperativeMatrix2FeaturesNV coop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV};
        VkPhysicalDeviceShaderFloat8FeaturesEXT fp8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &f13;
        void** tail = &f13.pNext;
        if (has(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) *tail = &coop, tail = &coop.pNext;
        if (has(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME)) *tail = &fp8, tail = &fp8.pNext;
        if (has(VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME)) *tail = &coop2, tail = &coop2.pNext;
        vkGetPhysicalDeviceFeatures2(p, &f2);
        VkPhysicalDeviceSubgroupSizeControlProperties sc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        driver.pNext = &sc;
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &driver;
        vkGetPhysicalDeviceProperties2(p, &p2);
        opennr_khr::Device d;
        d.float8_matrices = fp8.shaderFloat8 && fp8.shaderFloat8CooperativeMatrix;
        d.min_subgroup = sc.minSubgroupSize, d.max_subgroup = sc.maxSubgroupSize;
        d.subgroup_control_compute = f13.subgroupSizeControl && f13.computeFullSubgroups &&
                                     (sc.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT);
        if (coop.cooperativeMatrix && vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
            uint32_t shapes = 0;
            vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(p, &shapes, nullptr);
            std::vector<VkCooperativeMatrixPropertiesKHR> props(shapes, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
            vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(p, &shapes, props.data());
            for (const auto& s : props)
                d.shapes.push_back({s.MSize, s.NSize, s.KSize, s.AType, s.BType, s.CType, s.ResultType, s.scope});
        }
        const bool nv = coop2.cooperativeMatrixWorkgroupScope && coop2.cooperativeMatrixFlexibleDimensions &&
                        coop2.cooperativeMatrixReductions && coop2.cooperativeMatrixConversions &&
                        coop2.cooperativeMatrixPerElementOperations && coop2.cooperativeMatrixTensorAddressing &&
                        coop2.cooperativeMatrixBlockLoads && d.float8_matrices;
        printf("device\t%u\t%s\t%d\tnv=%d\tsubgroups=%u-%u\n", i, p2.properties.deviceName, (int)driver.driverID, nv ? 1 : 0,
               sc.minSubgroupSize, sc.maxSubgroupSize);
        for (const opennr_khr::Variant& v : opennr_khr::kVariants) {
            const std::string no = coop.cooperativeMatrix ? opennr_khr::refusal(d, v) : "no VK_KHR_cooperative_matrix";
            printf("variant\t%u\t%s\t%s\n", i, v.name, no.empty() ? "ok" : no.c_str());
        }
    }
    vkDestroyInstance(instance, nullptr);
    return 0;
}
