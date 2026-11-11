// (hookshot) The volk.h OpenDLSS-NR's host code includes (third_party/opendlss-nr/src/vk_context.h), as a shim.
//
// Upstream links volk; Hookshot has its own loader (gpu/vk/vk_loader.h) and does not. This declares the Vulkan
// entry points the vendored code calls, under the names it calls them by, in a namespace of their own (ovk) with a
// using-directive -- so they are not global C symbols named like libvulkan's own functions, which a shared object
// that exports `vkCreateBuffer` as a data symbol would be. The three volk calls the vendored code makes are here
// too. gpu/vk/opennr/volk_shim.cpp defines all of it.
//
// Only for the vendored sources and gpu/vk/opennr/*.cpp: everything else in gpu/vk uses vkx (vk_loader.h), and the
// two must not meet in one translation unit, where an unqualified vkCreateBuffer would be ambiguous.
#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#ifndef VK_ENABLE_BETA_EXTENSIONS
#define VK_ENABLE_BETA_EXTENSIONS   // (VK_NV_cuda_kernel_launch is declared in vulkan_beta.h)
#endif
#include <vulkan/vulkan.h>

#include <cstring>   // (upstream's vk_context.h and nr_model.h call memcpy with no include of their own: MSVC's headers bring it)

// global and instance level: resolved through vkGetInstanceProcAddr
#define OVK_GLOBAL(X) X(vkCreateInstance) X(vkEnumerateInstanceLayerProperties) X(vkEnumerateInstanceExtensionProperties)

#define OVK_INSTANCE(X)                                                                                                 \
    X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties)                                 \
    X(vkGetPhysicalDeviceProperties2) X(vkGetPhysicalDeviceFeatures2) X(vkGetPhysicalDeviceQueueFamilyProperties)       \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkEnumerateDeviceExtensionProperties) X(vkCreateDevice)                    \
    X(vkGetDeviceProcAddr) X(vkCreateDebugUtilsMessengerEXT) X(vkDestroyDebugUtilsMessengerEXT)                         \
    X(vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)

// device level: resolved through vkGetDeviceProcAddr once the device is known
#define OVK_DEVICE(X)                                                                                                   \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkDeviceWaitIdle) X(vkQueueSubmit) X(vkQueueWaitIdle)                      \
    X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory)            \
    X(vkBindBufferMemory) X(vkMapMemory) X(vkUnmapMemory) X(vkGetBufferDeviceAddress)                                   \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreateComputePipelines) X(vkDestroyPipeline)                   \
    X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) X(vkCreateDescriptorSetLayout)                                 \
    X(vkDestroyDescriptorSetLayout) X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) X(vkResetDescriptorPool)       \
    X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkCreateCommandPool) X(vkDestroyCommandPool)                \
    X(vkResetCommandPool) X(vkAllocateCommandBuffers) X(vkFreeCommandBuffers) X(vkBeginCommandBuffer)                   \
    X(vkEndCommandBuffer) X(vkCmdExecuteCommands) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets)                       \
    X(vkCmdPushConstants) X(vkCmdDispatch) X(vkCmdPipelineBarrier2) X(vkCmdCopyBuffer) X(vkCmdFillBuffer)               \
    X(vkCmdUpdateBuffer) X(vkCreateQueryPool) X(vkDestroyQueryPool) X(vkResetQueryPool) X(vkCmdResetQueryPool)          \
    X(vkCmdWriteTimestamp) X(vkGetQueryPoolResults) X(vkGetPipelineExecutablePropertiesKHR)                             \
    X(vkGetPipelineExecutableStatisticsKHR) X(vkGetPipelineExecutableInternalRepresentationsKHR)                        \
    X(vkCreateCudaModuleNV) X(vkCreateCudaFunctionNV) X(vkDestroyCudaFunctionNV) X(vkDestroyCudaModuleNV)               \
    X(vkCmdCudaLaunchKernelNV)

namespace ovk {
#define OVK_DECLARE(f) extern PFN_##f f;
OVK_GLOBAL(OVK_DECLARE)
OVK_INSTANCE(OVK_DECLARE)
OVK_DEVICE(OVK_DECLARE)
#undef OVK_DECLARE
extern PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;

/// Hookshot: resolve through the renderer's own loader instead of opening the Vulkan library a second time. Call it
/// before anything of OpenDLSS-NR's runs; the harness does not, and volkInitialize then opens the library itself.
void volkUseInstanceProcAddr(PFN_vkGetInstanceProcAddr get_instance_proc_addr);

// volk's three, with volk's meaning: the library and the global functions; the instance's; the device's
VkResult volkInitialize();
void volkLoadInstance(VkInstance instance);
void volkLoadDevice(VkDevice device);
} // namespace ovk

using namespace ovk;
