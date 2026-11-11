// (hookshot) See opennr_net.h. Upstream's classes, used as upstream's own demo uses them (demo/nr_pass.cpp in
// https://github.com/maanHimself/OpenDLSS-NR): one vk::Context adopting the renderer's device, the model, the kernels,
// and one graph per frame size recorded once into a secondary command buffer.
#include "volk.h"   // (first: upstream's headers include it, and it has to be the shim)

#include "opennr_net.h"

#include <cctype>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <memory>
#include <vector>

#include "json.h"
#include "opennr_khr.h"
#include "kernels.h"
#include "nr_graph.h"
#include "nr_model.h"
#include "numeric.h"
#include "reference.h"
#include "sha256.h"
#include "vk_context.h"

namespace mkwvk::opennr_net {
namespace {

bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize n = f.tellg();
    f.seekg(0);
    out.resize((size_t)n);
    return (bool)f.read(reinterpret_cast<char*>(out.data()), n);
}

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

} // namespace

ModelCheck check_model(const std::string& dir) {
    ModelCheck c;
    try {
        std::vector<uint8_t> text;
        if (!read_file(dir + "/manifest.json", text)) {
            c.why = "no manifest.json in " + dir;
            return c;
        }
        const json::Value m = json::parse(std::string(text.begin(), text.end()));
        if (!m.has("totals") || !m["totals"].has("blockCount")) {
            c.why = "manifest.json has no totals.blockCount";
            return c;
        }
        // (nr::Graph's own refusal, made before anything is loaded: this graph is the 71-block network of 310.8.0)
        const int64_t blocks = m["totals"]["blockCount"].integer();
        if (blocks != nr::Graph::kBlockCount) {
            c.why = "the model has " + std::to_string(blocks) + " blocks; this network is the " +
                    std::to_string(nr::Graph::kBlockCount) + "-block one (DLSS-NR 310.8.0)";
            return c;
        }
        // (nr::Model's: each stage file under model/, of the packed length, with the SHA-256 the manifest gives.
        // upstream compares its upper-case hex exactly; a lower-case manifest is the same digest and is taken)
        std::vector<std::pair<std::string, size_t>> stages;
        for (const json::Value& s : m["stages"].array) {
            std::vector<uint8_t> bytes;
            const std::string path = dir + "/model/" + s["file"].str();
            if (!read_file(path, bytes)) {
                c.why = "cannot read the stage file " + path;
                return c;
            }
            if (bytes.size() != (size_t)s["packedByteLength"].integer()) {
                c.why = "stage " + s["id"].str() + " is " + std::to_string(bytes.size()) + " bytes, the manifest says " +
                        std::to_string(s["packedByteLength"].integer());
                return c;
            }
            std::string want = s["sha256"].str();
            for (char& ch : want) ch = (char)toupper((unsigned char)ch);
            if (sha256Hex(bytes.data(), bytes.size()) != want) {
                c.why = "stage " + s["id"].str() + " does not have the SHA-256 the manifest gives (a corrupt or "
                        "mismatched model directory)";
                return c;
            }
            stages.emplace_back(s["id"].str(), bytes.size());
            c.bytes += bytes.size();
        }
        for (const json::Value& t : m["tensors"].array) {
            const std::pair<std::string, size_t>* stage = nullptr;
            for (const auto& s : stages)
                if (s.first == t["stage"].str()) stage = &s;
            if (!stage) {
                c.why = "tensor " + t["name"].str() + " is in a stage the manifest does not list";
                return c;
            }
            if ((uint64_t)t["stageOffset"].integer() + (uint64_t)t["byteLength"].integer() > stage->second) {
                c.why = "tensor " + t["name"].str() + " runs past the end of its stage";
                return c;
            }
            c.tensors++;
        }
        c.stages = (uint32_t)stages.size();
        if (!c.stages || !c.tensors) {
            c.why = "the manifest lists no stages or no tensors";
            return c;
        }
        c.ok = true;
    } catch (const std::exception& e) {
        c.why = std::string("manifest.json: ") + e.what();
    }
    return c;
}

class Network {
public:
    Route route = Route::Fast;
    std::unique_ptr<vk::Context> context;
    std::unique_ptr<nr::Model> model;
    std::unique_ptr<nr::Kernels> kernels;
    std::unique_ptr<nr::Graph> graph;
    nr::Activation* features = nullptr;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer secondary = VK_NULL_HANDLE;
    uint32_t pool_slot = 0;
    float blend = 1.0f;
};

void fall_back_to_barriers(Network* n) {
    nr::Kernels::setChainEnabled(false);
    if (n && n->kernels) n->kernels->resetChainTimeouts();
}

const char* route_name(Route route) {
    return route == Route::Fast        ? "PTX kernels through VK_NV_cuda_kernel_launch, counter chaining"
           : route == Route::Glsl      ? "the fused GLSL kernels, no PTX"
           : route == Route::Khr       ? "the khr kernels (VK_KHR_cooperative_matrix only), unfused"
                                       : "the GLSL reference kernels, unfused";
}

Network* create(const Device& d, const std::string& model_dir, const std::string& kernel_dir, Route route,
                const char* khr_variant, std::string* error) {
    const opennr_khr::Variant* khr = route == Route::Khr ? opennr_khr::find(khr_variant) : nullptr;
    if (route == Route::Khr && !khr) {
        if (error) *error = std::string("no khr kernel variant named ") + (khr_variant ? khr_variant : "(none)");
        return nullptr;
    }
    // Upstream's route switches are environment variables its kernels read once, at first use, so they are set
    // before any of it runs. The GLSL routes also point the PTX folder at an empty one: a PTX kernel reached anyway
    // would then fail to load, loudly, instead of calling a CUDA launch this device may not have.
    if (route != Route::Fast) {
        for (const char* name : {"DLSS5VK_PTX_GEMM", "DLSS5VK_PTX_GEMMT", "DLSS5VK_PTX_GEMMV", "DLSS5VK_PTX_BLOCK32",
                                 "DLSS5VK_PTX_FFN", "DLSS5VK_PTX_QKV", "DLSS5VK_PTX_ATTN"})
            set_env(name, "0");
        set_env("DLSS5VK_NO_PTX_MLP", "1");
        set_env("DLSS5VK_PTX_DIR", (kernel_dir + "/no-ptx").c_str());
        nr::Kernels::setChainEnabled(false);
    }
    // (the khr kernels are made for their variant's subgroup size: the patched copy of upstream's vk_context.cpp makes
    // its pipelines with it, scripts/opennr_kernels.py SOURCE_PATCHES; every other route keeps upstream's 32)
    set_env("HOOKSHOT_OPENNR_SUBGROUP_SIZE", khr ? std::to_string(khr->subgroup).c_str() : "0");
    auto n = std::make_unique<Network>();
    n->route = route;
    try {
        ovk::volkUseInstanceProcAddr(d.get_instance_proc_addr);
        n->context = std::make_unique<vk::Context>(d.instance, d.physical, d.device, d.queue_family, d.queue_index);
        n->model = std::make_unique<nr::Model>(*n->context, model_dir, false);   // (hashed by check_model already)
        n->kernels = std::make_unique<nr::Kernels>(
            *n->context, kernel_dir + (khr ? std::string("/shaders-khr-") + khr->name : std::string("/shaders")));
        n->kernels->setSiluTable(ref::siluTable());
        try {
            const nr::Tensor& b = n->model->tensor(70, 0, "blend_scale");
            if (b.byteLength >= 2) n->blend = num::f16ToF32((uint16_t)(b.bytes[0] | (b.bytes[1] << 8)));
        } catch (const std::exception&) {
            n->blend = 1.0f;   // (upstream's demo: a model without it blends uncapped)
        }
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.queueFamilyIndex = d.queue_family;
        VK_CHECK(vkCreateCommandPool(d.device, &pi, nullptr, &n->pool));
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = n->pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
        ai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(d.device, &ai, &n->secondary));
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        destroy(n.release());
        return nullptr;
    }
    return n.release();
}

bool build(Network* n, uint32_t width, uint32_t height, Built* out, std::string* error) {
    try {
        n->graph.reset();
        n->features = nullptr;
        const nr::Geometry g = nr::Geometry::fromValid(width, height);
        nr::Graph::Options options;
        options.fusedBlocks = n->route != Route::Reference && n->route != Route::Khr;   // (khr: its kernels are the unfused ones)
        n->graph = std::make_unique<nr::Graph>(*n->context, *n->model, *n->kernels, g, options);
        n->features = n->graph->allocate("input features", g.fullWidth * g.fullHeight, 16, nr::Format::F32);
        // A warm-up record and run (upstream's demo does the same): the first record allocates every activation,
        // builds every weight layout, compiles every pipeline and JITs every PTX module, so the first real frame does
        // none of it. Its descriptor sets go in one of the context's two pools; the frames' recording in the other.
        n->context->fillZero(n->features->buffer);
        n->context->resetDescriptorPool(n->pool_slot);
        VkCommandBuffer warm = n->context->beginCommands();
        n->graph->record(warm, *n->features);
        n->context->endAndSubmit(warm, true);
        if (n->kernels->chainTimeouts().waits) throw std::runtime_error("a chained wait timed out in the warm-up frame");
        n->pool_slot ^= 1;
        n->context->resetDescriptorPool(n->pool_slot);
        VK_CHECK(vkResetCommandPool(n->context->device(), n->pool, 0));
        VkCommandBufferInheritanceInfo inherit{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;   // (up to three frames in flight execute it)
        bi.pInheritanceInfo = &inherit;
        VK_CHECK(vkBeginCommandBuffer(n->secondary, &bi));
        n->graph->record(n->secondary, *n->features);
        VK_CHECK(vkEndCommandBuffer(n->secondary));
        out->full_width = g.fullWidth, out->full_height = g.fullHeight;
        out->features = n->features->buffer.buffer;
        out->features_bytes = n->features->validBytes();
        out->head = n->graph->head().buffer.buffer;
        out->head_bytes = n->graph->head().validBytes();
        out->dispatches = n->kernels->dispatchCount();
        out->chained = n->graph->chained();
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        n->graph.reset();
        n->features = nullptr;
        return false;
    }
    return true;
}

void record(Network* n, VkCommandBuffer primary) {
    if (n && n->graph) vkCmdExecuteCommands(primary, 1, &n->secondary);
}

float blend_scale(const Network* n) {
    return n ? n->blend : 1.0f;
}

bool chain_timed_out(Network* n) {
    if (!n || !n->kernels) return false;
    try {
        return n->kernels->chainTimeouts().waits != 0;
    } catch (const std::exception&) {
        return false;
    }
}

void destroy(Network* n) {
    if (!n) return;
    try {
        n->graph.reset();
        if (n->pool && n->context) vkDestroyCommandPool(n->context->device(), n->pool, nullptr);
        n->kernels.reset();
        n->model.reset();
        n->context.reset();
    } catch (const std::exception&) {
    }
    delete n;
}

} // namespace mkwvk::opennr_net
