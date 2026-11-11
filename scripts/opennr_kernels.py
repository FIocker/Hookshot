"""The open neural rendering backend's kernels: OpenDLSS-NR's SPIR-V and PTX, and the frame's own two passes.

Three kinds of file, made three ways, for reasons that are not the same:

* OpenDLSS-NR's thirteen GLSL kernels (third_party/opendlss-nr/shaders) are compiled with the flags of upstream's
  scripts/build_shaders.ps1 and checked in as one header, gpu/vk/shaders/opennr_kernels_spv.h -- like every other
  shader in gpu/vk/shaders, so a clone builds without glslang;
* the renderer's own preprocess and composite passes (gpu/vk/shaders/opennr_pre.comp, opennr_post.comp) the same
  way, one glslang header each, the way the rest of gpu/vk/shaders is made;
* the PTX kernels are generated at build time by nx2/build.py from upstream's Python generators
  (third_party/opendlss-nr/scripts/ptx), with exactly the calls build_shaders.ps1 makes. They are text, the
  generators need only Python, which every build has, and checking in 5.5 MB of generated text would only be a
  second copy of the generators to keep in step.

    python -I -B scripts/opennr_kernels.py              # write the SPIR-V headers
    python -I -B scripts/opennr_kernels.py --check      # compare instead of writing (tests/test_opennr_kernels.py)
    python -I -B scripts/opennr_kernels.py --materialize DIR
        # DIR/shaders/*.spv from the CHECKED-IN header and DIR/ptx/*.ptx freshly generated: the kernel folder
        # OpenDLSS-NR's host code reads, for the correctness harness (scripts/opennr_harness.py). The SPIR-V comes
        # from the header on purpose: the harness proves what ships, not a fresh compile of it.
    python -I -B scripts/opennr_kernels.py --prepare DIR
        # upstream's host sources copied to DIR with SOURCE_PATCHES applied (what nx2/build.py and the harness compile)

The khr route (gpu/vk/opennr/khr, for GPUs without VK_NV_cooperative_matrix2: AMD's, llvmpipe) is a fourth kind, made
like the first: checked in as gpu/vk/shaders/opennr_khr_spv.h. Its kernel folders hold, per variant (KHR_VARIANTS):
the four matrix kernels written for VK_KHR_cooperative_matrix (gpu/vk/opennr/khr/*.comp, compiled with the variant's
defines); upstream's ops, gemm_reduce, window_normalize and global_normalize compiled from build-time copies with their
hardware E4M3 conversions spelled in integers and their half fma, inverse square root and f32-to-half conversions
made exact (KHR_SOURCE_PATCHES, gpu/vk/opennr/khr/khr_e4m3.glsl and khr_numerics.glsl); upstream's preprocess as it is
(the same module as the other routes'); and a module that does nothing for the four fused kernels, which the khr
route's unfused graph never dispatches.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "opendlss-nr"
SHADERS = UPSTREAM / "shaders"
GENERATORS = UPSTREAM / "scripts" / "ptx"
OUT = ROOT / "gpu" / "vk" / "shaders"
KERNEL_HEADER = OUT / "opennr_kernels_spv.h"
# (the renderer's own passes around the network: name -> the glslang variable name of its header)
FRAME_SHADERS = {"opennr_pre": "OPENNR_PRE_COMP_SPV", "opennr_post": "OPENNR_POST_COMP_SPV"}

# The thirteen modules nr::Kernels loads (third_party/opendlss-nr/src/kernels.cpp, its constructor), in its order.
KERNELS = ["gemm_fp8", "gemm_f16", "ops", "window_normalize", "window_attend", "global_normalize", "global_attend",
           "preprocess", "fused_block32", "qkv_attention", "gemm_mlp", "global_attention", "gemm_reduce"]
# upstream's build_shaders.ps1: `glslang -V --target-env vulkan1.3 -I<shaders>`
KERNEL_FLAGS = ["-V", "--target-env", "vulkan1.3"]

# ---- the khr route (gpu/vk/opennr/khr/khr_mma.glsl says what each define means)
KHR = ROOT / "gpu" / "vk" / "opennr" / "khr"
KHR_HEADER = OUT / "opennr_khr_spv.h"
# name, defines, the MMA shapes a GPU must list for it (A/B, C/R component types; M x N x K) -- in the order a GPU is
# offered them, the most faithful first. gpu/vk/opennr/opennr_khr.h kVariants is the same table (the test compares).
KHR_VARIANTS = [
    ("e4m3-k32-f16", {"KHR_E4": 1, "KHR_M": 16, "KHR_N": 16, "KHR_K": 32, "KHR_HK": 16, "KHR_ACC32": 0, "KHR_SG": 32},
     "E4M3 16x16x32 into f16, f16 16x16x16 into f16: NVIDIA's own MMA, the bytes of upstream's GLSL route"),
    ("e4m3-k16-f32", {"KHR_E4": 1, "KHR_M": 16, "KHR_N": 16, "KHR_K": 16, "KHR_HK": 16, "KHR_ACC32": 1, "KHR_SG": 32},
     "E4M3 16x16x16 into f32 (AMD RDNA 4's WMMA): rounded to half after every k32 step"),
    ("e4m3-k32-f32", {"KHR_E4": 1, "KHR_M": 16, "KHR_N": 16, "KHR_K": 32, "KHR_HK": 16, "KHR_ACC32": 1, "KHR_SG": 32},
     "E4M3 16x16x32 into f32: rounded to half after every k32 step"),
    ("f16-k16-f32", {"KHR_E4": 0, "KHR_M": 16, "KHR_N": 16, "KHR_K": 16, "KHR_HK": 16, "KHR_ACC32": 1, "KHR_SG": 32},
     "E4M3 widened to f16, f16 16x16x16 into f32 (AMD RDNA 3's WMMA): rounded to half after every k32 step"),
    ("f16-k8-f32", {"KHR_E4": 0, "KHR_M": 8, "KHR_N": 8, "KHR_K": 8, "KHR_HK": 8, "KHR_ACC32": 1, "KHR_SG": 8},
     "E4M3 widened to f16, f16 8x8x8 into f32 at 8 lanes (llvmpipe): rounded to half after every k32 step"),
]
# the kernels the khr route writes itself, per variant; those it compiles from upstream with KHR_SOURCE_PATCHES, once;
# those it takes from upstream as they are (the same modules as opennr_kernels_spv.h's); those it never dispatches
KHR_OWN = ["gemm_fp8", "gemm_f16", "window_attend", "global_attend"]
KHR_PATCHED = ["ops", "gemm_reduce", "window_normalize", "global_normalize"]
KHR_UPSTREAM = ["preprocess"]
KHR_UNUSED = ["fused_block32", "qkv_attention", "gemm_mlp", "global_attention"]
# (upstream's kernel -> exact replacements in a build-time copy, each applied exactly once, as SOURCE_PATCHES)
_E4M3_IN_INTEGERS = [
    ("#extension GL_EXT_float_e4m3 : require\n", "",
     "no E4M3 type is left in the file once its conversions are spelled in integers (below)"),
    ('#define DLSS_E4M3_HW 1\n#include "common.glsl"\n',
     '#include "common.glsl"\n#include "khr_e4m3.glsl"   // (hookshot: upstream\'s conversions in integers)\n',
     "e4m3Hw / e4m3HwPair / e4m3HwToF32 without VK_EXT_shader_float8: khr_e4m3.glsl, equal by upstream's own statement"),
]
_NUMERICS = ('#include "common.glsl"\n', '#include "common.glsl"\n#include "khr_numerics.glsl"   // (hookshot)\n',
             "the exact half fma, inverse square root and conversions of khr_numerics.glsl")
KHR_SOURCE_PATCHES = {
    "ops": _E4M3_IN_INTEGERS,
    "gemm_reduce": _E4M3_IN_INTEGERS,
    "window_normalize": [
        _NUMERICS,
        ("    r[c] = fma(q[c], q[c], highSquare);\n", "    r[c] = fmaHalf(q[c], q[c], highSquare);\n",
         "the half fma rounded once on any GPU (llvmpipe's rounds twice)"),
        ("    r[c] = fma(k[c], k[c], highSquare);\n", "    r[c] = fmaHalf(k[c], k[c], highSquare);\n", "the same, for k"),
        ("  float16_t qNorm = float16_t(inversesqrt(float(qSum)));\n  float16_t kNorm = float16_t(inversesqrt(float(kSum)));\n",
         "  float16_t qNorm = inverseSqrtHalf(qSum);\n  float16_t kNorm = inverseSqrtHalf(kSum);\n",
         "RN16 of the exact 1 / sqrt: what upstream's reference computes, for every half; inversesqrt may be 2 ulp off"),
        ("float16_t(uintBitsToFloat(aux32[pc.scaleWordOffset + head]))",
         "float16_t(roundF16(uintBitsToFloat(aux32[pc.scaleWordOffset + head])))",
         "the learned f32 scale rounded to half on the bits, not by a conversion whose rounding mode is the driver's"),
    ],
    "global_normalize": [
        _NUMERICS,
        ("    r[c] = float16_t(sum);\n", "    r[c] = float16_t(roundF16(sum));\n",
         "the f32 square sum rounded to half on the bits (two occurrences: q and k)"),
        ("  float16_t qNorm = float16_t(inversesqrt(float(qSum)));\n  float16_t kNorm = float16_t(inversesqrt(float(kSum)));\n",
         "  float16_t qNorm = inverseSqrtHalf(qSum);\n  float16_t kNorm = inverseSqrtHalf(kSum);\n",
         "RN16 of the exact 1 / sqrt, as for the window blocks"),
        ("float16_t(uintBitsToFloat(aux32[pc.scaleWordOffset + head]))",
         "float16_t(roundF16(uintBitsToFloat(aux32[pc.scaleWordOffset + head])))", "the learned scale, as above"),
    ],
}


# Upstream's host code is built with MSVC only. What clang (and GCC) refuse in it, as exact replacements applied to a
# copy in the build's own folder (prepare_sources): third_party/ stays byte for byte upstream's, and each change
# here says why. A replacement that no longer matches exactly once fails the build instead of being skipped.
SOURCE_PATCHES = [
    ("kernels.h", "  struct Chain {\n", "  struct Chain {\n    Chain() {}   // (hookshot portability, scripts/opennr_kernels.py)\n",
     "Chain's default member initializers are used by a default argument (`const Chain& chain = Chain()`) inside "
     "the enclosing class Kernels, before Kernels is complete. MSVC accepts it; clang and GCC refuse it ('default "
     "member initializer needed within definition of enclosing class', CWG 1397). A user-provided constructor is "
     "the smallest change that compiles everywhere; nothing aggregate-initializes a Chain."),
    ("vk_context.cpp", "  subgroupInfo.requiredSubgroupSize = requiredSubgroupSize;\n",
     "  if (requiredSubgroupSize)   // (hookshot: the khr route's subgroup size, scripts/opennr_kernels.py)\n"
     "    if (const char* lanes = getenv(\"HOOKSHOT_OPENNR_SUBGROUP_SIZE\"); lanes && atoi(lanes) > 0)\n"
     "      requiredSubgroupSize = (uint32_t)atoi(lanes);\n"
     "  subgroupInfo.requiredSubgroupSize = requiredSubgroupSize;\n",
     "Upstream makes every subgroup pipeline with requiredSubgroupSize 32, which its NVIDIA-only kernels assume. The "
     "khr route's kernels are made for a variant's subgroup size (32, or 8 on llvmpipe, whose only size it is: 32 there "
     "is invalid usage); gpu/vk/opennr/opennr_net.cpp sets the variable for that route only, so every other route "
     "creates exactly upstream's pipelines."),
]
# The vendored host files the renderer compiles (not upstream's dlss5vk tool, main.cpp and verify.cpp, which only
# the harness links)
RENDERER_SOURCES = ["kernels.cpp", "nr_graph.cpp", "nr_model.cpp", "reference.cpp", "vk_context.cpp"]


def prepare_sources(out_dir):
    """Copy third_party/opendlss-nr/src to out_dir with SOURCE_PATCHES applied; returns out_dir. Unchanged files are
    not rewritten, so an object built from them is kept."""
    src, out_dir = UPSTREAM / "src", Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = {p.name: p.read_text(encoding="utf-8") for p in sorted(src.iterdir()) if p.is_file()}
    for name, old, new, why in SOURCE_PATCHES:
        if texts.get(name, "").count(old) != 1:
            raise RuntimeError("opennr portability patch for %s no longer matches upstream exactly once (%s)" % (name, why))
        texts[name] = texts[name].replace(old, new)
    for name, text in texts.items():
        target = out_dir / name
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8")
    return out_dir


def ptx_jobs():
    """(generator, arguments, output file) for every PTX variant upstream's build_shaders.ps1 generates, in its order.

    A variant the graph asks for and this list lacks is a hard error at record time for some kernel families and a
    fall-back to another route for others (upstream's docs/execution.md), so this list is the upstream one verbatim."""
    jobs = []
    for k in (64, 128, 256):
        jobs.append(("mlp_e4m3.py", [str(k)], "mlp_e4m3_K%d.ptx" % k, ["3", "1", "1"]))
    for c in (64, 128, 256, 512):
        jobs.append(("qkv_e4m3.py", [str(c)], "qkv_e4m3_K%d.ptx" % c, ["80"]))
    for k in (32, 64, 128, 256, 512, 1024):   # residual-seeded E4 (5), residual dual (13), plain E4 (4), f16 (8), SiLU (6)
        for f in (5, 13, 4, 8, 6):
            jobs.append(("gemm2_e4m3.py", [str(k), str(f)], "gemm2_e4m3_K%d_f%d.ptx" % (k, f), []))
    for c, r, extra in ((64, 4, 0), (128, 4, 0), (256, 3, 80), (256, 4, 64)):
        jobs.append(("ffn_e4m3.py", [str(c), str(r)], "ffn_e4m3_C%d_R%d.ptx" % (c, r), [str(extra)]))
        jobs.append(("ffn_e4m3.py", [str(c), str(r)], "ffn_e4m3_C%d_R%d_proj.ptx" % (c, r), [str(extra), "1"]))
    for k, f, p, s in ((4096, 1, 32, 4), (1024, 0, 16, 2), (1024, 1, 8, 4), (1024, 0, 8, 4)):
        jobs.append(("gemmt_e4m3.py", [str(k), str(f), str(p), str(s)], "gemmt_e4m3_K%d_f%d_p%d_s%d.ptx" % (k, f, p, s), []))
    for s, f in ((4, 4), (2, 8), (4, 8)):
        jobs.append(("reduce_e4m3.py", [str(s), str(f)], "reduce_e4m3_s%d_f%d.ptx" % (s, f), []))
    for k, f, s in ((1024, 6, 1), (4096, 5, 4), (1024, 8, 2), (1024, 5, 4)):
        jobs.append(("gemmv_e4m3.py", [str(k), str(f), str(s)], "gemmv_e4m3_K%d_f%d_s%d.ptx" % (k, f, s), []))
        jobs.append(("gemmv_e4m3.py", [str(k), str(f), str(s)], "gemmv_e4m3_K%d_f%d_s%d_m96.ptx" % (k, f, s),
                     ["4", "6", "0", "0", "0", "96"]))
    for padded in (64, 128, 192, 256):
        jobs.append(("global_attention_e4m3.py", [str(padded)], "global_attention_e4m3_p%d.ptx" % padded, []))
    jobs.append(("global_attention_stream_e4m3.py", ["normalize"], "global_normalize_e4m3.ptx", []))
    jobs.append(("global_attention_stream_e4m3.py", ["attention"], "global_attention_stream_e4m3.ptx", ["4"]))
    for k, f in ((512, 4), (512, 5), (512, 13), (256, 5), (4096, 5), (1024, 8), (1024, 5)):
        jobs.append(("gemmv_e4m3.py", [str(k), str(f), "1"], "gemmv_e4m3_K%d_f%d_s1_m96.ptx" % (k, f),
                     ["4", "6", "0", "0", "0", "96"]))
    for f in (2, 66, 74, 130, 48):
        jobs.append(("block32_e4m3.py", [str(f)], "block32_e4m3_f%d.ptx" % f, []))
    return jobs


def generate_ptx(out_dir, python=None):
    """Run every generator into out_dir. Upstream's generators import their siblings (ptxgen, swin), so each runs as a
    script from its own folder: not with -I, which would leave that folder off sys.path. They write nothing else."""
    out_dir = Path(out_dir).resolve()   # (absolute: each generator runs from its own folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    python = python or sys.executable
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    names = []
    for script, head, name, tail in ptx_jobs():
        target = out_dir / name
        r = subprocess.run([python, "-B", str(GENERATORS / script), *head, str(target), *tail], cwd=str(GENERATORS),
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if r.returncode or not target.is_file():
            raise RuntimeError("PTX generator %s %s failed:\n%s" % (script, " ".join(head + tail),
                                                                   r.stdout.decode("utf-8", "replace")[-2000:]))
        names.append(name)
    return names


def write_ptx_header(ptx_dir, header):
    """The generated PTX as one C++ header the renderer embeds (gpu/vk/vk_opennr.cpp writes it back out as files for
    OpenDLSS-NR's host code, which reads its kernels from a folder). Raw string literals: PTX has no ')opennr"'."""
    ptx_dir, header = Path(ptx_dir), Path(header)
    names = sorted(p.name for p in ptx_dir.glob("*.ptx"))
    lines = ["// Generated by nx2/build.py (scripts/opennr_kernels.py write_ptx_header) from",
             "// third_party/opendlss-nr/scripts/ptx: do not edit.", "#pragma once", "",
             "namespace mkwvk::opennr_ptx {", "struct File { const char* name; const char* text; };",
             "inline const File kFiles[] = {"]
    for n in names:
        text = (ptx_dir / n).read_text(encoding="utf-8")
        if ')opennr"' in text:
            raise RuntimeError(n + " holds the raw string delimiter")
        lines.append('    {"%s", R"opennr(%s)opennr"},' % (n, text))
    lines += ["};", "inline constexpr unsigned kCount = %d;" % len(names), "} // namespace mkwvk::opennr_ptx", ""]
    text = "\n".join(lines)
    if not header.is_file() or header.read_text(encoding="utf-8") != text:   # (unchanged: the object is kept)
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text(text, encoding="utf-8")
    return len(names)


def compile_kernel(glslang, name, spv):
    subprocess.run([glslang, *KERNEL_FLAGS, "-I" + str(SHADERS), str(SHADERS / (name + ".comp")), "-o", str(spv)],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def khr_folder(variant):
    """the kernel folder of a khr variant, beside `shaders` (what gpu/vk/opennr/opennr_net.cpp reads)"""
    return "shaders-khr-" + variant


def compile_khr(glslang, tmp):
    """[(entry name, words)] of every khr module: 'variant/name' for KHR_OWN, 'common/name' for the rest"""
    tmp = Path(tmp)
    def run(src, spv, defines=()):
        r = subprocess.run([glslang, *KERNEL_FLAGS, *["-D%s=%s" % kv for kv in defines], "-I" + str(KHR),
                            "-I" + str(SHADERS), str(src), "-o", str(spv)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError("glslang failed on %s %s:\n%s" % (src.name, " ".join("%s=%s" % kv for kv in defines),
                                                                 r.stdout.decode("utf-8", "replace")[-3000:]))
        return words_of(spv)
    modules = []
    for variant, defines, _ in KHR_VARIANTS:
        for name in KHR_OWN:
            modules.append(("%s/%s" % (variant, name), run(KHR / (name + ".comp"), tmp / ("%s-%s.spv" % (variant, name)),
                                                           sorted(defines.items()))))
    for name in KHR_PATCHED:
        text = (SHADERS / (name + ".comp")).read_text(encoding="utf-8")
        for old, new, why in KHR_SOURCE_PATCHES[name]:
            count = 2 if name == "global_normalize" and old == "    r[c] = float16_t(sum);\n" else 1
            if text.count(old) != count:
                raise RuntimeError("khr patch for %s.comp no longer matches upstream exactly (%s)" % (name, why))
            text = text.replace(old, new)
        copy = tmp / (name + ".khr.comp")
        copy.write_text(text, encoding="utf-8")
        modules.append(("common/" + name, run(copy, tmp / (name + ".khr.spv"))))
    for name in KHR_UNUSED:
        modules.append(("common/" + name, run(KHR / "unused.comp", tmp / (name + ".unused.spv"))))
    return modules


def khr_header(modules):
    out = ["// Generated by scripts/opennr_kernels.py from gpu/vk/opennr/khr and third_party/opendlss-nr/shaders: do not edit.",
           "// The khr route's SPIR-V (glslang %s): 'variant/name' for its own matrix kernels, 'common/name' for" %
           " ".join(KERNEL_FLAGS), "// the rest. gpu/vk/vk_opennr.cpp writes them out as the shaders-khr-<variant> folders.",
           "#pragma once", "", "#include <cstdint>", "", "namespace mkwvk::opennr_khr_spv {"]
    for entry, words in modules:
        ident = entry.replace("/", "__").replace("-", "_")
        out.append("inline constexpr uint32_t k_%s[] = {" % ident)
        for i in range(0, len(words), 8):
            out.append("    " + ",".join("0x%08x" % w for w in words[i:i + 8]) + ",")
        out.append("};")
    out += ["struct Module { const char* entry; const uint32_t* words; unsigned count; };",
            "inline constexpr Module kModules[] = {"]
    for entry, _ in modules:
        ident = entry.replace("/", "__").replace("-", "_")
        out.append('    {"%s", k_%s, (unsigned)(sizeof k_%s / sizeof k_%s[0])},' % (entry, ident, ident, ident))
    out += ["};", "} // namespace mkwvk::opennr_khr_spv", ""]
    return "\n".join(out)


def khr_header_words(text):
    """entry -> SPIR-V words of the checked-in khr header"""
    idents = dict(re.findall(r'\{"([\w/-]+)", k_(\w+),', text))
    words = {m.group(1): [int(w, 16) for w in re.findall(r"0x([0-9a-f]{8})", m.group(2))]
             for m in re.finditer(r"inline constexpr uint32_t k_(\w+)\[\] = \{(.*?)\};", text, re.S)}
    return {entry: words[ident] for entry, ident in idents.items()}


def khr_folder_modules(variant, khr_words, upstream_words):
    """name -> words of every module of a khr variant's folder: the thirteen nr::Kernels loads"""
    out = {}
    for name in KERNELS:
        if name in KHR_OWN:
            out[name] = khr_words["%s/%s" % (variant, name)]
        elif name in KHR_UPSTREAM:
            out[name] = upstream_words[name]
        else:
            out[name] = khr_words["common/" + name]
    return out


def words_of(spv):
    data = Path(spv).read_bytes()
    if len(data) % 4 or data[:4] != bytes((0x03, 0x02, 0x23, 0x07)):
        raise RuntimeError("%s is not SPIR-V" % spv)
    return [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]


def kernel_header(modules):
    out = ["// Generated by scripts/opennr_kernels.py from third_party/opendlss-nr/shaders: do not edit.",
           "// glslang %s -I<shaders>, as upstream's scripts/build_shaders.ps1. The SPIR-V words of each module;" %
           " ".join(KERNEL_FLAGS), "// gpu/vk/vk_opennr.cpp writes them out as the .spv files nr::Kernels loads.",
           "#pragma once", "", "#include <cstdint>", "", "namespace mkwvk::opennr_spv {"]
    for name, words in modules:
        out.append("inline constexpr uint32_t k_%s[] = {" % name)
        for i in range(0, len(words), 8):
            out.append("    " + ",".join("0x%08x" % w for w in words[i:i + 8]) + ",")
        out.append("};")
    out += ["struct Module { const char* name; const uint32_t* words; unsigned count; };",
            "inline constexpr Module kModules[] = {"]
    out += ['    {"%s", k_%s, (unsigned)(sizeof k_%s / sizeof k_%s[0])},' % (n, n, n, n) for n, _ in modules]
    out += ["};", "} // namespace mkwvk::opennr_spv", ""]
    return "\n".join(out)


def header_words(text):
    """name -> SPIR-V words of the checked-in kernel header (the parse --materialize and --check share)"""
    modules = {}
    for m in re.finditer(r"inline constexpr uint32_t k_(\w+)\[\] = \{(.*?)\};", text, re.S):
        modules[m.group(1)] = [int(w, 16) for w in re.findall(r"0x([0-9a-f]{8})", m.group(2))]
    return modules


def strip_generator(text):
    """A glslang header without glslang's own version comment (`// 1116.2.0`), as scripts/fsr3_shaders.py does: a
    runner with another glslang release would otherwise report a version bump as a source mismatch."""
    return [line for line in text.splitlines()
            if not line.strip().startswith("// ") or not line.strip()[3:].replace(".", "").isdigit()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glslang", default=None, help="glslangValidator; found on PATH by default")
    p.add_argument("--spirv-val", default=None, help="spirv-val, to validate each module")
    p.add_argument("--check", action="store_true", help="compare with the checked-in headers")
    p.add_argument("--materialize", metavar="DIR", help="write DIR/shaders/*.spv (from the header) and DIR/ptx/*.ptx")
    p.add_argument("--prepare", metavar="DIR", help="copy upstream's host sources to DIR with the portability patches")
    a = p.parse_args()

    if a.prepare:
        print("prepared %s" % prepare_sources(a.prepare))
        return 0
    if a.materialize:
        modules = header_words(KERNEL_HEADER.read_text(encoding="utf-8"))
        if sorted(modules) != sorted(KERNELS):
            print("opennr_kernels_spv.h holds %s, not the thirteen kernels" % sorted(modules), file=sys.stderr)
            return 1
        shaders = Path(a.materialize) / "shaders"
        shaders.mkdir(parents=True, exist_ok=True)
        for name, words in modules.items():
            (shaders / (name + ".spv")).write_bytes(b"".join(w.to_bytes(4, "little") for w in words))
        khr = khr_header_words(KHR_HEADER.read_text(encoding="utf-8"))
        for variant, _, _ in KHR_VARIANTS:
            folder = Path(a.materialize) / khr_folder(variant)
            folder.mkdir(parents=True, exist_ok=True)
            for name, words in khr_folder_modules(variant, khr, modules).items():
                (folder / (name + ".spv")).write_bytes(b"".join(w.to_bytes(4, "little") for w in words))
        n = len(generate_ptx(Path(a.materialize) / "ptx"))
        print("materialized %d SPIR-V modules, %d khr variants and %d PTX kernels in %s" %
              (len(modules), len(KHR_VARIANTS), n, a.materialize))
        return 0

    glslang = a.glslang or shutil.which("glslangValidator") or shutil.which("glslang")
    if not glslang:
        print("glslangValidator not found; pass --glslang", file=sys.stderr)
        return 2
    bad = []
    with tempfile.TemporaryDirectory(prefix="opennr-kernels-") as tmp:
        modules = []
        for name in KERNELS:
            spv = Path(tmp) / (name + ".spv")
            compile_kernel(glslang, name, spv)
            if a.spirv_val:
                subprocess.run([a.spirv_val, "--target-env", "vulkan1.3", str(spv)], check=True)
            modules.append((name, words_of(spv)))
        made = kernel_header(modules)
        khr_modules = compile_khr(glslang, tmp)
        if a.spirv_val:
            for entry, words in khr_modules:
                spv = Path(tmp) / "val.spv"
                spv.write_bytes(b"".join(w.to_bytes(4, "little") for w in words))
                r = subprocess.run([a.spirv_val, "--target-env", "vulkan1.3", str(spv)], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
                if r.returncode:
                    raise RuntimeError("spirv-val refused khr %s:\n%s" % (entry, r.stdout.decode("utf-8", "replace")))
        khr_made = khr_header(khr_modules)
        if a.check:
            have = khr_header_words(KHR_HEADER.read_text(encoding="utf-8")) if KHR_HEADER.is_file() else {}
            for entry, words in khr_modules:
                if have.get(entry) != words:
                    bad.append("khr " + entry)
                    print("khr %s: the checked-in SPIR-V differs from a compile of its source" % entry, file=sys.stderr)
            if sorted(have) != sorted(e for e, _ in khr_modules):
                bad.append("(the khr module list)")
        else:
            KHR_HEADER.write_text(khr_made, encoding="utf-8")
            print("wrote %s (%d modules, %d bytes)" % (KHR_HEADER.relative_to(ROOT), len(khr_modules), len(khr_made)))
        if a.check:
            have = header_words(KERNEL_HEADER.read_text(encoding="utf-8")) if KERNEL_HEADER.is_file() else {}
            for name, words in modules:
                if have.get(name) != words:
                    bad.append(name)
                    print("%s: the checked-in SPIR-V differs from a compile of its source (%s words checked in, %d made)"
                          % (name, len(have[name]) if name in have else "no", len(words)), file=sys.stderr)
            if sorted(have) != sorted(KERNELS):
                bad.append("(the module list)")
        else:
            KERNEL_HEADER.write_text(made, encoding="utf-8")
            print("wrote %s (%d modules, %d bytes)" % (KERNEL_HEADER.relative_to(ROOT), len(modules), len(made)))
        for name, variable in FRAME_SHADERS.items():
            header = OUT / (name + "_comp_spv.h")
            out = Path(tmp) / header.name
            src = OUT / (name + ".comp")
            subprocess.run([glslang, "-V", "--quiet", "-S", "comp", "--target-env", "vulkan1.3", "--variable-name",
                            variable, "-o", str(out), str(src)], check=True)
            if a.spirv_val:
                spv = Path(tmp) / (name + ".spv")
                subprocess.run([glslang, "-V", "--quiet", "-S", "comp", "--target-env", "vulkan1.3", "-o", str(spv),
                                str(src)], check=True)
                subprocess.run([a.spirv_val, "--target-env", "vulkan1.3", str(spv)], check=True)
            if a.check:
                have = header.read_text(encoding="utf-8") if header.is_file() else ""
                if strip_generator(have) != strip_generator(out.read_text(encoding="utf-8")):
                    bad.append(name)
                    print("%s: missing or differs from a compile of %s" % (header.name, src.name), file=sys.stderr)
            else:
                shutil.copyfile(out, header)
                print("wrote %s" % header.relative_to(ROOT))
    if bad:
        print("%d header(s) do not match their source: %s" % (len(bad), ", ".join(bad)), file=sys.stderr)
        return 1
    print("opennr kernels: %d kernels, %d khr variants and %d frame passes %s" %
          (len(KERNELS), len(KHR_VARIANTS), len(FRAME_SHADERS), "match" if a.check else "written"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
