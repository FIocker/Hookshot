// (hookshot) The NVIDIA Streamline session; see gpu/vk/vk_streamline.h for why it exists beside the
// NGX one. Everything that touches Streamline is inside RX_VK_STREAMLINE and _WIN32; the stubs at
// the bottom are what a build without either compiles, so the renderer's call sites need no #ifdef.

#include "vk_streamline.h"

#if defined(RX_VK_STREAMLINE) && defined(_WIN32)

#include <windows.h>

#include <cctype>
#include <cstring>
#include <string>
#include <vector>

#include <softpub.h>
#include <wincrypt.h>
#include <wintrust.h>

#include "vk_streamline_api.h"

namespace mkwvk::streamline {
namespace {

// A custom engine names itself with its own project id rather than an application id from NVIDIA.
// A real UUID, because Streamline rejects anything else outright -- the same mistake cost a day on
// the NGX side (docs/upscaling.md: FAIL_InvalidParameter 0xbad00005).
constexpr const char* kProjectId = "6f2b1d84-97c5-4ea3-bd10-5c7a4e8f20b6";

struct State {
    HMODULE module{};
    bool started{};
    bool attached{};
    bool nr{};
    std::wstring plugin_dir;
    std::vector<std::string> instance_exts;
    std::vector<std::string> device_exts;
    // Pointers into instance_exts / device_exts, handed out by the extension getters. The strings
    // are never touched again once start() returns, so the pointers stay good for the process.
    std::vector<const char*> instance_names;
    std::vector<const char*> device_names;
    uint32_t extra_graphics{};
    uint32_t extra_compute{};
    std::string why;   ///< nr_why(): the short reason neural rendering is not available, for the menu

    Api api{};
};

State& state() {
    static State s;
    return s;
}

int log_level() {
    const char* const v = getenv("RX_VK_STREAMLINE_LOG");
    return v ? atoi(v) : 0;
}

void log_from_streamline(sl::LogType type, const char* message) {
    std::string text{message ? message : ""};
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
    if (type == sl::LogType::eError) logf("vk: streamline error: %s", text.c_str());
    else if (type == sl::LogType::eWarn) logf("vk: streamline warning: %s", text.c_str());
    else if (log_level() > 0) logf("vk: streamline: %s", text.c_str());
}

/// Where sl.interposer.dll may be: RX_VK_STREAMLINE_DIR, then a `streamline` folder beside the
/// executable, then the executable's own folder. A player who drops NVIDIA's DLLs next to the game
/// should not also have to set an environment variable.
std::vector<std::wstring> search_dirs() {
    std::vector<std::wstring> dirs;
    wchar_t env[1024];
    if (const DWORD n = GetEnvironmentVariableW(L"RX_VK_STREAMLINE_DIR", env, 1024); n && n < 1024)
        dirs.emplace_back(env);
    wchar_t exe[MAX_PATH];
    if (const DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH); n && n < MAX_PATH) {
        std::wstring dir{exe};
        if (const size_t cut = dir.find_last_of(L"\\/"); cut != std::wstring::npos) {
            dir.resize(cut);
            dirs.push_back(dir + L"\\streamline");
            dirs.push_back(dir);
        }
    }
    return dirs;
}

template <typename T>
bool resolve(HMODULE m, const char* name, T*& fn) {
    fn = reinterpret_cast<T*>(GetProcAddress(m, name));
    if (!fn) logf("vk: sl.interposer.dll has no %s: neural rendering is off", name);
    return fn != nullptr;
}

/// Whether the file carries a signature Windows trusts, and whether that signature is NVIDIA's.
///
/// Written out here rather than taken from Streamline's own sl_security.h on purpose: that header's
/// verifyEmbeddedSignature uses WINTRUST_SIGNATURE_SETTINGS, which the Windows SDK has and
/// llvm-mingw's headers do not, and Hookshot builds for Windows with either (third_party/a64-kit).
/// This uses only what both have. Trust alone is not enough -- it would accept anything signed by
/// anyone Windows trusts -- so the subject name is read too.
bool signed_by_nvidia(const std::wstring& path) {
    WINTRUST_FILE_INFO file{};
    file.cbStruct = sizeof file;
    file.pcwszFilePath = path.c_str();
    GUID action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    WINTRUST_DATA data{};
    data.cbStruct = sizeof data;
    data.dwUIChoice = WTD_UI_NONE;
    data.fdwRevocationChecks = WTD_REVOKE_NONE;   // (a player may well be offline)
    data.dwUnionChoice = WTD_CHOICE_FILE;
    data.pFile = &file;
    data.dwStateAction = WTD_STATEACTION_VERIFY;
    const LONG status = WinVerifyTrust(nullptr, &action, &data);
    data.dwStateAction = WTD_STATEACTION_CLOSE;
    WinVerifyTrust(nullptr, &action, &data);
    if (status != ERROR_SUCCESS) {
        logf("vk: sl.interposer.dll has no signature Windows trusts (status %#lx)",
             (unsigned long)status);
        return false;
    }
    // Whose signature it is. A trusted signature by somebody else is not a reason to load it.
    HCERTSTORE store{};
    HCRYPTMSG msg{};
    if (!CryptQueryObject(CERT_QUERY_OBJECT_FILE, path.c_str(),
                          CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED_EMBED,
                          CERT_QUERY_FORMAT_FLAG_BINARY, 0, nullptr, nullptr, nullptr, &store,
                          &msg, nullptr)) {
        logf("vk: sl.interposer.dll's signature could not be read (error %lu)", GetLastError());
        return false;
    }
    DWORD info_size = 0;
    bool nvidia = false;
    if (CryptMsgGetParam(msg, CMSG_SIGNER_INFO_PARAM, 0, nullptr, &info_size) && info_size) {
        std::vector<unsigned char> buffer(info_size);
        if (CryptMsgGetParam(msg, CMSG_SIGNER_INFO_PARAM, 0, buffer.data(), &info_size)) {
            const auto* signer = reinterpret_cast<CMSG_SIGNER_INFO*>(buffer.data());
            CERT_INFO want{};
            want.Issuer = signer->Issuer;
            want.SerialNumber = signer->SerialNumber;
            if (PCCERT_CONTEXT cert = CertFindCertificateInStore(
                    store, X509_ASN_ENCODING, 0, CERT_FIND_SUBJECT_CERT, &want, nullptr)) {
                wchar_t name[256] = {};
                CertGetNameStringW(cert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, nullptr, name, 256);
                nvidia = wcsstr(name, L"NVIDIA") != nullptr;
                if (!nvidia)
                    logf("vk: sl.interposer.dll is signed, but not by NVIDIA: not loading it");
                CertFreeCertificateContext(cert);
            }
        }
    }
    CryptMsgClose(msg);
    CertCloseStore(store, 0);
    return nvidia;
}

bool load(State& s, const std::wstring& path) {
    // We are about to load code from a path a player can write to, so a file that is not NVIDIA's
    // signed interposer is not loaded at all.
    if (!signed_by_nvidia(path)) return false;
    s.module = LoadLibraryW(path.c_str());
    if (!s.module) {
        logf("vk: could not load sl.interposer.dll (error %lu)", GetLastError());
        return false;
    }
    HMODULE m = s.module;
    Api& a = s.api;
    if (resolve(m, "slInit", a.init) && resolve(m, "slShutdown", a.shutdown) &&
        resolve(m, "slIsFeatureSupported", a.is_feature_supported) &&
        resolve(m, "slGetFeatureRequirements", a.get_feature_requirements) &&
        resolve(m, "slGetFeatureVersion", a.get_feature_version) &&
        resolve(m, "slGetFeatureFunction", a.get_feature_function) &&
        resolve(m, "slGetNewFrameToken", a.get_new_frame_token) &&
        resolve(m, "slSetConstants", a.set_constants) &&
        resolve(m, "slEvaluateFeature", a.evaluate_feature) &&
        resolve(m, "slFreeResources", a.free_resources) &&
        resolve(m, "slSetVulkanInfo", a.set_vulkan_info) &&
        resolve(m, "slSetTagForFrame", a.set_tag_for_frame))
        return true;
    FreeLibrary(s.module);
    s.module = nullptr;
    s.api = {};
    return false;
}

/// Whether Windows trusts the file's signature, whoever's it is: no signature at all is what a community-patched
/// nvngx_dlssnr.dll has (patching it invalidates NVIDIA's, and the patchers strip it).
bool trusted_signature(const std::wstring& path) {
    WINTRUST_FILE_INFO file{};
    file.cbStruct = sizeof file;
    file.pcwszFilePath = path.c_str();
    GUID action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    WINTRUST_DATA data{};
    data.cbStruct = sizeof data;
    data.dwUIChoice = WTD_UI_NONE;
    data.fdwRevocationChecks = WTD_REVOKE_NONE;
    data.dwUnionChoice = WTD_CHOICE_FILE;
    data.pFile = &file;
    data.dwStateAction = WTD_STATEACTION_VERIFY;
    const LONG status = WinVerifyTrust(nullptr, &action, &data);
    data.dwStateAction = WTD_STATEACTION_CLOSE;
    WinVerifyTrust(nullptr, &action, &data);
    return status == ERROR_SUCCESS;
}

/// NVIDIA's NGX signature override, as the player set it (read only: changing it is theirs to do):
/// HKLM\SOFTWARE\NVIDIA Corporation\Global\NGXCore EnableSignatureOverride. -1 when it is not there.
int signature_override() {
    DWORD value = 0, size = sizeof value;
    if (RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\NVIDIA Corporation\\Global\\NGXCore",
                     L"EnableSignatureOverride", RRF_RT_REG_DWORD, nullptr, &value, &size) != ERROR_SUCCESS)
        return -1;
    return (int)value;
}

/// The RTX generation from the device's name: 50 Blackwell, 40 Ada, 30 Ampere, 20 Turing; 0 for one it does not
/// recognize (a workstation card, a name it has not seen). Only for what the log and the menu say.
int rtx_series(VkPhysicalDevice gpu) {
    VkPhysicalDeviceProperties p{};
    vkGetPhysicalDeviceProperties(gpu, &p);
    const char* n = strstr(p.deviceName, "RTX ");
    if (!n || !isdigit((unsigned char)n[4]) || !isdigit((unsigned char)n[5])) return 0;
    const int series = (n[4] - '0') * 10;
    return series >= 20 && series <= 50 ? series : 0;
}

/// Why Streamline said no, in the log in full and for the menu in short. The usual case on an RTX 20, 30 or 40 is
/// NVIDIA's own model, which runs on RTX 50 only: NGX refuses it on older cards (FAIL_FeatureNotSupported, 0xbad00001,
/// and Streamline answers eErrorAdapterNotSupported or eErrorFeatureNotSupported). Community builds of
/// nvngx_dlssnr.dll recompile the model for Turing, Ampere and Ada (dlssnr-patcher; openNR's 310.9.16 compatibility
/// build): those carry no NVIDIA signature, so NGX loads them only with its signature override set.
void explain_refusal(State& s, VkPhysicalDevice gpu, sl::Result r) {
    const int series = rtx_series(gpu);
    const std::wstring dll = s.plugin_dir + L"\\nvngx_dlssnr.dll";
    const bool have = GetFileAttributesW(dll.c_str()) != INVALID_FILE_ATTRIBUTES;
    const bool signed_dll = have && trusted_signature(dll);
    const int override_value = signature_override();
    logf("vk: DLSS 5 neural rendering is not supported here (Streamline result %u). GPU: %s; nvngx_dlssnr.dll: %s; "
         "NGX signature override: %s",
         (unsigned)r, series ? (series == 50 ? "RTX 50" : series == 40 ? "RTX 40" : series == 30 ? "RTX 30" : "RTX 20")
                             : "not recognized as RTX 20 to 50",
         !have ? "not beside sl.interposer.dll" : signed_dll ? "NVIDIA's (signed)" : "unsigned (a patched build)",
         override_value < 0 ? "not set" : override_value ? "on" : "off");
    if (!have) {
        logf("vk:   put nvngx_dlssnr.dll beside sl.dlss_nr.dll (readme.md: DLSS 5 neural rendering)");
        s.why = "Not available: nvngx_dlssnr.dll is not beside the game's Streamline files.";
    } else if (series && series < 50 && signed_dll) {
        logf("vk:   NVIDIA's own nvngx_dlssnr.dll runs on RTX 50 only. On an RTX %d, a community build of it that "
             "adds Turing, Ampere and Ada is needed instead (readme.md says which, and how)", series);
        s.why = "Not available: NVIDIA's model is RTX 50 only. RTX 20-40: a patched nvngx_dlssnr.dll (readme).";
    } else if (!signed_dll && override_value <= 0) {
        logf("vk:   this nvngx_dlssnr.dll is unsigned, and NGX loads an unsigned one only with its signature "
             "override on: HKLM\\SOFTWARE\\NVIDIA Corporation\\Global\\NGXCore, DWORD EnableSignatureOverride = 1 "
             "(readme.md). Hookshot does not change it for you");
        s.why = "Not available: the patched nvngx_dlssnr.dll needs NGX's signature override (readme).";
    } else if (r == sl::Result::eErrorDriverOutOfDate) {
        s.why = "Not available: NVIDIA's driver is too old for DLSS 5 neural rendering.";
    } else {
        s.why = "Not available on this GPU or driver (the log says what Streamline answered).";
    }
}

void add_names(std::vector<std::string>& list, uint32_t count, const char** names) {
    for (uint32_t i = 0; i < count; i++) {
        bool have = false;
        for (const std::string& s : list) have = have || s == names[i];
        if (!have) list.emplace_back(names[i]);
    }
}

} // namespace

const Api& api() { return state().api; }

bool built_with_sdk() { return true; }

bool start() {
    State& s = state();
    if (s.started) return true;
    // RX_VK_DLSS_NR=0 is the only way to keep the interposer from being loaded at all. The menu's
    // own switch is deliberately not consulted here: the device must be created for the feature
    // before the options file is read, so waiting for the switch would mean a restart to honour it.
    if (const char* off = getenv("RX_VK_DLSS_NR"); off && atoi(off) == 0) return false;
    if (!s.module) {
        std::wstring found;
        for (const std::wstring& dir : search_dirs()) {
            const std::wstring candidate = dir + L"\\sl.interposer.dll";
            if (GetFileAttributesW(candidate.c_str()) != INVALID_FILE_ATTRIBUTES) {
                found = candidate;
                s.plugin_dir = dir;
                break;
            }
        }
        if (found.empty()) {
            logf("vk: no sl.interposer.dll beside the game or in RX_VK_STREAMLINE_DIR: DLSS 5 "
                 "neural rendering is unavailable (readme.md says which files to put there)");
            return false;
        }
        if (!load(s, found)) return false;
    }

    // Only neural rendering. Streamline's own DLSS and DLSS-G plugins are deliberately not asked
    // for: super resolution and frame generation go through our own NGX session (gpu/vk/vk_ngx.h),
    // and two NGX sessions over one device is the bug this avoids.
    sl::Feature features[] = {sl::kFeatureDLSS_NR};
    const wchar_t* plugin_paths[] = {s.plugin_dir.c_str()};
    sl::Preferences p{};
    p.logLevel = log_level() >= 2 ? sl::LogLevel::eVerbose : sl::LogLevel::eOff;
    p.logMessageCallback = log_from_streamline;
    p.pathsToPlugins = plugin_paths;
    p.numPathsToPlugins = 1;
    // eUseManualHooking: the renderer creates its own instance and device and hands them over with
    // slSetVulkanInfo, rather than letting the interposer wrap vkCreateDevice.
    // eUseFrameBasedResourceTagging: slSetTagForFrame, which is how the uplift buffers are tagged.
    // eDisableCLStateTracking: the feature binds its own pipelines and descriptors, and the replay
    // re-establishes its state after a present anyway.
    // eAllowOTA and eLoadDownloadedPlugins are deliberately NOT set: they let Streamline fetch and
    // load plugins over the network and bootstrap nvngx_update.exe, and a game has no business
    // starting an updater. DagorEngine's shipping integration sets neither either.
    p.flags = sl::PreferenceFlags::eDisableCLStateTracking |
              sl::PreferenceFlags::eUseManualHooking |
              sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    p.featuresToLoad = features;
    p.numFeaturesToLoad = 1;
    p.engine = sl::EngineType::eCustom;
    p.engineVersion = "1.0";
    p.projectId = kProjectId;
    p.renderAPI = sl::RenderAPI::eVulkan;
    if (const sl::Result r = s.api.init(p, sl::kSDKVersion); r != sl::Result::eOk) {
        logf("vk: NVIDIA Streamline did not start (result 0x%x): DLSS 5 neural rendering is off",
             (unsigned)r);
        return false;
    }

    sl::FeatureRequirements req{};
    if (const sl::Result r = s.api.get_feature_requirements(sl::kFeatureDLSS_NR, req);
        r != sl::Result::eOk) {
        logf("vk: Streamline found no sl.dlss_nr plugin (result 0x%x): put it next to "
             "sl.interposer.dll", (unsigned)r);
        s.api.shutdown();
        return false;
    }
    add_names(s.instance_exts, req.vkNumInstanceExtensions, req.vkInstanceExtensions);
    add_names(s.device_exts, req.vkNumDeviceExtensions, req.vkDeviceExtensions);
    s.extra_graphics = req.vkNumGraphicsQueuesRequired;
    s.extra_compute = req.vkNumComputeQueuesRequired;
    // The KHR and EXT buffer device address extensions cannot both be enabled on one device.
    if (std::find(s.device_exts.begin(), s.device_exts.end(),
                  std::string{"VK_KHR_buffer_device_address"}) != s.device_exts.end())
        s.device_exts.erase(std::remove(s.device_exts.begin(), s.device_exts.end(),
                                        std::string{"VK_EXT_buffer_device_address"}),
                            s.device_exts.end());
    for (const std::string& e : s.instance_exts) s.instance_names.push_back(e.c_str());
    for (const std::string& e : s.device_exts) s.device_names.push_back(e.c_str());
    s.nr = true;
    s.started = true;
    sl::FeatureVersion v{};
    if (s.api.get_feature_version(sl::kFeatureDLSS_NR, v) == sl::Result::eOk)
        logf("vk: NVIDIA Streamline %s started for DLSS 5 neural rendering (NGX %s), %u instance "
             "and %u device extensions, %u extra graphics and %u compute queues",
             v.versionSL.toStr().c_str(), v.versionNGX.toStr().c_str(),
             (unsigned)s.instance_exts.size(), (unsigned)s.device_exts.size(), s.extra_graphics,
             s.extra_compute);
    else
        logf("vk: NVIDIA Streamline started for DLSS 5 neural rendering");
    return true;
}

size_t instance_extensions(const char** out, size_t max) {
    State& s = state();
    if (!s.started) return 0;
    const size_t n = s.instance_names.size() < max ? s.instance_names.size() : max;
    for (size_t i = 0; i < n; i++) out[i] = s.instance_names[i];
    return n;
}

size_t device_extensions(const char** out, size_t max) {
    State& s = state();
    if (!s.started) return 0;
    const size_t n = s.device_names.size() < max ? s.device_names.size() : max;
    for (size_t i = 0; i < n; i++) out[i] = s.device_names[i];
    return n;
}

uint32_t extra_queues() {
    State& s = state();
    return s.started ? s.extra_graphics + s.extra_compute : 0;
}

bool attach_device(VkInstance instance, VkPhysicalDevice gpu, VkDevice device,
                   uint32_t graphics_family, uint32_t extra_queues_made) {
    State& s = state();
    if (!s.started || s.attached) return false;
    if (extra_queues_made < s.extra_graphics + s.extra_compute) {
        logf("vk: the graphics family has no room for the %u queues Streamline wants (only %u were "
             "made): DLSS 5 neural rendering is off",
             s.extra_graphics + s.extra_compute, extra_queues_made);
        s.nr = false;
        stop();
        return false;
    }
    // Streamline's queues follow the renderer's own in the graphics family, graphics first.
    sl::VulkanInfo info{};
    info.instance = instance;
    info.physicalDevice = gpu;
    info.device = device;
    info.graphicsQueueFamily = graphics_family;
    info.graphicsQueueIndex = 1;
    info.computeQueueFamily = graphics_family;
    info.computeQueueIndex = 1 + s.extra_graphics;
    if (const sl::Result r = s.api.set_vulkan_info(info); r != sl::Result::eOk) {
        logf("vk: Streamline did not take the Vulkan device (result 0x%x)", (unsigned)r);
        stop();
        return false;
    }
    sl::AdapterInfo adapter{};
    adapter.vkPhysicalDevice = gpu;
    if (const sl::Result r = s.api.is_feature_supported(sl::kFeatureDLSS_NR, adapter);
        r != sl::Result::eOk) {
        explain_refusal(s, gpu, r);
        s.nr = false;
        stop();
        return false;
    }
    s.attached = true;
    logf("vk: NVIDIA Streamline attached: DLSS 5 neural rendering is available");
    return true;
}

bool nr_available() {
    State& s = state();
    return s.started && s.attached && s.nr;
}

const char* nr_why() { return state().why.c_str(); }

void stop() {
    State& s = state();
    if (!s.started) return;
    const sl::Result r = s.api.shutdown();
    s.started = s.attached = false;
    logf("vk: NVIDIA Streamline stopped (result 0x%x)", (unsigned)r);
}

} // namespace mkwvk::streamline

#else // no Streamline headers, or not Windows

namespace mkwvk::streamline {

bool built_with_sdk() { return false; }
bool start() { return false; }
size_t instance_extensions(const char**, size_t) { return 0; }
size_t device_extensions(const char**, size_t) { return 0; }
uint32_t extra_queues() { return 0; }
bool attach_device(VkInstance, VkPhysicalDevice, VkDevice, uint32_t, uint32_t) { return false; }
bool nr_available() { return false; }
const char* nr_why() { return ""; }
void stop() {}

} // namespace mkwvk::streamline

#endif
