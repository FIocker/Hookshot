// (hookshot) The definitions behind gpu/vk/opennr/volk.h: OpenDLSS-NR's Vulkan entry points and volk's three calls.
#include "volk.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ovk {
#define OVK_DEFINE(f) PFN_##f f = nullptr;
OVK_GLOBAL(OVK_DEFINE)
OVK_INSTANCE(OVK_DEFINE)
OVK_DEVICE(OVK_DEFINE)
#undef OVK_DEFINE
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;

namespace {
PFN_vkGetInstanceProcAddr given = nullptr;
VkInstance loaded_instance = VK_NULL_HANDLE;

PFN_vkGetInstanceProcAddr open_library() {
#ifdef _WIN32
    HMODULE m = LoadLibraryA("vulkan-1.dll");
    return m ? (PFN_vkGetInstanceProcAddr)(void*)GetProcAddress(m, "vkGetInstanceProcAddr") : nullptr;
#else
    void* m = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!m) m = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    return m ? (PFN_vkGetInstanceProcAddr)dlsym(m, "vkGetInstanceProcAddr") : nullptr;
#endif
}
} // namespace

void volkUseInstanceProcAddr(PFN_vkGetInstanceProcAddr get_instance_proc_addr) {
    given = get_instance_proc_addr;
}

VkResult volkInitialize() {
    if (!vkGetInstanceProcAddr) vkGetInstanceProcAddr = given ? given : open_library();
    if (!vkGetInstanceProcAddr) return VK_ERROR_INITIALIZATION_FAILED;
#define OVK_LOAD_GLOBAL(f) f = (PFN_##f)vkGetInstanceProcAddr(VK_NULL_HANDLE, #f);
    OVK_GLOBAL(OVK_LOAD_GLOBAL)
#undef OVK_LOAD_GLOBAL
    return VK_SUCCESS;
}

void volkLoadInstance(VkInstance instance) {
    loaded_instance = instance;
#define OVK_LOAD_INSTANCE(f) f = (PFN_##f)vkGetInstanceProcAddr(instance, #f);
    OVK_INSTANCE(OVK_LOAD_INSTANCE)
    // (until volkLoadDevice: the device's functions through the instance, as volk does, so a caller that loads only
    // the instance still has them)
    OVK_DEVICE(OVK_LOAD_INSTANCE)
#undef OVK_LOAD_INSTANCE
}

void volkLoadDevice(VkDevice device) {
    if (!vkGetDeviceProcAddr && loaded_instance)
        vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)vkGetInstanceProcAddr(loaded_instance, "vkGetDeviceProcAddr");
    if (!vkGetDeviceProcAddr) return;
#define OVK_LOAD_DEVICE(f) f = (PFN_##f)vkGetDeviceProcAddr(device, #f);
    OVK_DEVICE(OVK_LOAD_DEVICE)
#undef OVK_LOAD_DEVICE
}
} // namespace ovk
