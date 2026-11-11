# DLSS 5 neural rendering

## What it is, and the mistake that cost a week

DLSS 5 neural rendering (DLSS-NR; NVIDIA's own name for it is *3D-Guided Neural Rendering*) was
refused in this project three times on the grounds that it needs ray tracing. That was wrong, and
the record is kept here because the reasoning was plausible and still wrong.

The refusal conflated DLSS-NR with **DLSS Ray Reconstruction** (`sl.dlss_d`, NGX's `DLSSD`). Ray
reconstruction denoises a ray-traced image and needs a G-buffer -- albedo, normals, roughness --
which a recompilation of a 1998 rasterized game does not have and never will. DLSS-NR is a different
feature with a different plugin and different inputs. Streamline's own public headers settle it:

```
sl_core_types.h:210  kBufferTypeUpliftInputColor  = 70
sl_core_types.h:213  kBufferTypeUpliftOutputColor = 71   // may alias the input (in-place uplift)
sl_core_types.h:215  kBufferTypeUpliftControlMask = 72   // optional, 4-channel
sl_core_types.h:253  kFeatureDLSS_NR              = 1004
```

Colour in, colour out, plus depth and motion vectors. It is an image-enhancement pass, not a
denoiser, and it is why the feature can be put over rasterized content at all.

## Where it runs here

In the present, over the frame the game finished, before the menu and the FPS counter are drawn on
top (`gpu/vk/vk_replay.cpp`, `present_image`). It is not an upscaler and is not part of the
upscaling path: input and output are both at display resolution, and `Upscaling` / `Frame
generation` are untouched by it. It runs on every presented frame, so with frame generation at 4x it
runs four times per game frame -- each shown frame gets enhanced, which is the point, but it is four
times the cost.

The model is handed a flat depth and no motion at all, with an identity camera and no jitter
(`scene_constants`). That is deliberate, not laziness: by this point in the frame the picture is
already upscaled and already has the HUD on it, so the scene's real depth buffer does not describe
the image being handed over. Giving it the real camera would have it reproject detail that is not
where it thinks it is. The renderer *does* have real scene depth and camera motion (`vk_camera_motion.cpp`),
and feeding them is the obvious next experiment -- it would mean moving the pass before the HUD.

Dagor's shipping integration also runs NR on a **tonemapped, display-referred SDR image** and
inverse-tonemaps afterwards. Here the window image already is that, which is a second reason this is
the right place in the frame.

## The options struct

NVIDIA ships no `sl_dlss_nr.h`. The struct that configures the model is NDA-only, so
`gpu/vk/vk_dlssnr_options.h` declares it, and that file is the riskiest thing in this feature: it is
passed to code we did not compile, so a field at the wrong offset is not a crash but the model
reading one setting as another. Ten `static_assert`s on the offsets are there to make a future
Streamline that moves a field a build error.

Two sources are genuinely independent of each other and agree exactly on field order, count and
types: **DagorEngine** (BSD-3-Clause), which compiles against NVIDIA's real header, and **openNR**
(MIT), which recovered the layout from a prerelease Streamline 2.13 `sl.dlss_nr.dll`.

Three things this project previously had wrong, all inherited from openNR's defaults and all
corrected here:

| | was | is | why |
|---|---|---|---|
| `preset` | 1 | **0** | 0 is the default model; 1-7 pin presets A-G. A default of 1 silently pins preset A. |
| `skinStructureStrength` | -1 ("leave it to the model") | **1.0** | No source states a negative sentinel. openNR defaults -1 with no comment; Dagor's real header and two others default 1.0, and the documented range is 0-1. The "-1" meaning was read into openNR's default and then repeated -- including by me, in Elysium. |
| `useAutoMask` | `sl::Boolean` | **`uint32_t`** | `sl::Boolean` is `enum Boolean : char`: one byte, leaving three bytes of *uninitialized padding* inside a struct handed to NVIDIA's code. Same offsets either way. |

`globalToneStrength` and `performanceMode` are in the ABI and **ignored by Streamline 2.14's
plugin**. They are sent correctly anyway and the menu rows say so, because a row that looks broken
is worse than a row that explains itself.

### Not corroborated

The struct **GUID** `{29DFDFE0-...}` and `kStructVersion3` rest on openNR alone -- Dagor includes the
real header and so never restates them. If `slDLSSNRSetOptions` answers `eErrorInvalidParameter`,
that GUID is the first suspect, and `vk_dlssnr.cpp` says so in the log rather than leaving the next
person to rediscover it.

## Why a second NVIDIA path

Super resolution and frame generation go through NGX directly (`gpu/vk/vk_ngx.h`), which is measured
and works. DLSS-NR cannot: the public NGX SDK does not name the feature, and the only entry point
that configures it, `slDLSSNRSetOptions`, lives inside `sl.dlss_nr` and comes out through
`slGetFeatureFunction`. So Streamline sits beside the NGX session rather than replacing it, and it is
asked to load **only** `kFeatureDLSS_NR` -- not Streamline's own DLSS or DLSS-G plugins -- so that two
NGX sessions never come up against the same device.

`eAllowOTA` and `eLoadDownloadedPlugins` are deliberately not set: they let Streamline fetch plugins
over the network and bootstrap `nvngx_update.exe`, and a game has no business starting an updater.

## NGX direct: the second backend

`gpu/vk/vk_dlssnr_ngx.{h,cpp}` creates the same model as NGX feature 18 (`NVSDK_NGX_Feature_Reserved18`), over the one NGX
session DLSS super resolution owns (`dlss_ngx_session`), and needs no Streamline at all. The Neural tab's **Backend** row
(`opts.nr_backend`, saved as `nr_backend`; `RX_VK_DLSS_NR_BACKEND=auto|streamline|ngx` over it) chooses: Auto is Streamline
where `streamline::nr_available()`, NGX direct otherwise. `nr::apply` is the present's one call (`vk_replay.cpp`
`present_image`), and every other row applies to whichever runs.

Why it exists: Streamline's own plugin may gate RTX 20-40 even with a patched `nvngx_dlssnr.dll` (the "not measured here"
below), and on Linux there is no Streamline at all while NGX is there.

**The parameter set is openNR's** ([clshortfuse/openNR](https://github.com/clshortfuse/openNR), MIT), from its native
benchmark (`benchmarks/native/d3d11/main.cpp`, `benchmarks/native/include/neural_rendering_direct.hpp`). The public NGX
SDK names the feature id and nothing else. Names and types were taken as facts; no code was copied:

| when | keys | type |
|---|---|---|
| create | `Width`/`Height`/`OutWidth`/`OutHeight`, `DLSSNR.Width`/`Height`, `DLSSNR.InputWidth`/`InputHeight`, `DLSSNR.OutputWidth`/`OutputHeight`, `DLSSNR.Output.Width`/`Output.Height` | uint |
| create | `DLSSNR.Hint.Render.Preset` (0 default, 1-7 = A-G) | uint |
| create | `PerfQualityValue` = sl::DLSSMode - 1 (the Model row: 0 perf, 1 balanced, 2 quality, 5 DLAA) | int |
| create | `DLSSNRComputeScalingRatioCallback`: NGX calls it with a parameter block and it answers `DLSSNR.ScalingRatio` = 1, i.e. no upscaling | void* |
| create | `DLSSNR.ScalingRatio` 1, `DLSSNR.Scale` 1 / `DLSSNR.Upscaling` 0 | float / int |
| evaluate | `DLSSNR.Color`, `Output`, `MVec`, `Depth` (`NVSDK_NGX_Resource_VK`, images in `GENERAL`), `UI`/`UIAlpha` null | void* |
| evaluate | `DLSSNR.<Color/Output/MVec/Depth>Subrect<BaseX/BaseY/Width/Height>` | uint |
| evaluate | `Jitter.Offset.X/Y`, `DLSSNR.JitterOffsetX/Y` 0, `DLSSNR.MVecScaleX/Y` 1 | float |
| evaluate | `DLSSNR.DepthInverted` 0, `Enabled` 1, `Reset` (first frame of a feature), `UseAutoMask`, `UICorrection` 0 | int |
| evaluate | `DLSSNR.Intensity`, `LocalToneStrength`, `LocalStructureStrength`, `GlobalToneStrength`, `SkinStructureStrength` | float |
| evaluate | `DLSSNR.Style` | uint |

All of them are set with NGX's C functions (`NVSDK_NGX_Parameter_SetUI/SetI/SetF/SetVoidPointer`), not the
`NVSDK_NGX_Parameter` virtuals: NGX's Windows object is MSVC-built and llvm-mingw's vtable does not line up with it
(`vk_dlss.cpp`). Preset and Model are creation parameters, so changing either remakes the feature; the rest go with every
evaluation. A refusal is held against those exact settings (any row change tries again), and six refusals turn it off
for the run. The device is created with whatever extensions NGX names for feature 18
(`NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements`, `vk_main.cpp`), asked whatever the switch says because the
device exists before a player can turn it on.

Its inputs are the Streamline path's: the window image copied into an `R8G8B8A8_UNORM` image of its own, a flat 0.5 `R32`
depth, zero `R16G16` motion, no jitter. openNR's benchmark feeds `RGBA16F`; whether 8-bit colour is accepted by feature
18 on Vulkan is not known yet.

**Availability.** It reads `DLSSNR.Available` and `DLSSNR.FeatureInitResult` from NGX's capability parameters (and
`DLSSNR.NeedsUpdatedDriver` / `MinDriverVersionMajor/Minor` by the naming every NGX feature's keys follow; a key NGX does
not answer is reported as such) and logs them. Where NGX does not answer for the feature at all, the first frame's
`CreateFeature` is the answer. When it is refused:

- **Linux**: NGX loads features as `libnvidia-ngx-<name>.so*` from the folders `vk_dlss.cpp ngx_init` gives it, and then
  the game folder itself (`dlss_ngx_runtime` lists them all), and NVIDIA ships no `libnvidia-ngx-dlssnr.so` -- SDK 310.9.16's Linux libraries are
  dlss, dlssd and dlssg only. The log and the Neural tab say exactly that, and that the Windows build under Proton is the
  way to the feature today. The day NVIDIA ships the library, putting it in the game folder's `ngx` folder should be all
  it takes. That is a prediction, not a measurement.
- **Windows**: the same reasoning as `vk_streamline.cpp explain_refusal` -- the GPU's generation, the `nvngx_dlssnr.dll`
  found in NGX's folders and whether Windows trusts its signature (only in a build that links wintrust, i.e. one with
  Streamline's headers), and NGX's signature override, read and never written.

### What is measured for NGX direct, and what is not

- Measured, 2026-11-09, Linux, RTX 5060 Ti (Blackwell), driver 615.71.09, NGX SDK 310.9.16, windowed on Xvfb (a
  `HOS_HEADLESS=1` run never reaches `present_image`, so it never reaches either backend):
  - NGX asks for `VK_KHR_buffer_device_address` for feature 18 (`vk: dlss-nr wants ...: enabled`).
  - With Backend Auto, the run chose NGX direct (no Streamline on Linux); with NGX direct, the same.
  - NGX's own log (`RX_VK_DLSS_LOG=1`) shows it looking for `libnvidia-ngx-dlssnr.so` in the game folder's `ngx`, then
    `~/.local/share/hookshot/ngx`, then the game folder itself, and finding none.
  - NGX answers `DLSSNR.Available = 0`, `DLSSNR.FeatureInitResult = 0xbad00004` (`FAIL_FeatureNotFound`): the core
    knows the feature and has no library for it. The Linux message is logged once; the game went on to gameplay
    (~2,185 draws a frame, ~36 fps) for 10,000+ presents.
  - Nothing on Linux has created or evaluated feature 18, because there is nothing to create it from.
- Not measured: anything on Windows. No run has created feature 18 through NGX on Vulkan, evaluated it, or seen a frame
  from it; whether NGX's Windows core answers `DLSSNR.Available` for a patched DLL on RTX 20-40; whether
  `PerfQualityValue` and `Hint.Render.Preset` are taken at creation as openNR's D3D11/D3D12 benchmark takes them; and
  whether openNR's D3D parameter set is the whole Vulkan set (its benchmark loads the snippet itself, not through NGX's
  core, so the core's own checks are not exercised by it).

## What a player needs

Windows x64 only. Streamline's interposer and plugins are DLLs; on Linux, run the Windows build under
Proton, which is where this was measured for Elysium. These go beside the game, or in a `streamline`
folder beside it, or wherever `RX_VK_STREAMLINE_DIR` points:

```
sl.interposer.dll     the loader; its signature is checked before it is loaded
sl.common.dll         the interposer needs it whichever feature loads on top
sl.pcl.dll            likewise
sl.dlss_nr.dll        the Streamline plugin
nvngx_dlssnr.dll      the model -- must sit in the same folder as sl.dlss_nr.dll
```

Two lessons from other people's measurements, both worth not rediscovering: the plugin and the model
**must be colocated** (in separate folders the support query returned `FeatureNotSupported`), and the
`sl.*` trio **must be version-matched** -- a 2.13 plugin with 2.14 common/interposer is not a
supported combination.

Floor: **Streamline 2.14.1**, the first public tag carrying `kFeatureDLSS_NR` at all (2.14.0
introduced the plugin; v2.12.0 and earlier have neither the feature nor the uplift tags). CI builds
against v2.14.16.

**GPU**: NVIDIA's shipped `nvngx_dlssnr.dll` is **Blackwell-only** (its kernels are SM120). On an RTX 20, 30 or 40
NGX refuses it with `0xbad00001` (`FAIL_FeatureNotSupported`), and a probe on an RTX 4070 SUPER got
`AdapterUnsupported` from Streamline. The arch gate and the kernels are both inside that one file, which is why the
community route works: [dlssnr-patcher](https://github.com/dev-camo/dlssnr-patcher) (GPL-2.0) recompiles its PTX for
`sm_75`/`sm_86`/`sm_89`, rewrites the FP8 instructions older cards lack and widens the architecture the file declares,
and [openNR](https://github.com/clshortfuse/openNR) (MIT) reconstructs a pinned 310.9.16 compatibility DLL, an FP16 one
reported at about 3.5 ms on an RTX 4070 against 6 ms for the FP8 port. Both work from the player's own copy and ship
nothing of NVIDIA's.

Hookshot needs nothing special for them -- it loads whatever `nvngx_dlssnr.dll` sits beside `sl.dlss_nr.dll`, and
signature-checks only `sl.interposer.dll` -- but **a patched file has no NVIDIA signature, and NGX loads an unsigned
feature only with its signature override** (`HKLM\SOFTWARE\NVIDIA Corporation\Global\NGXCore`, DWORD
`EnableSignatureOverride` = 1), the same switch DLSS wrappers have always needed. That is the player's to set:
`vk_streamline.cpp explain_refusal` reads it for the log and never writes it. When Streamline refuses the feature, that
function logs the GPU's generation, whether the `nvngx_dlssnr.dll` there is NVIDIA's (signed) or a patched one, and the
override's state, says which of them is missing, and gives the Neural tab a one-line reason.

Not measured here: the patched DLLs through *Streamline* on *Vulkan*. Every report found used NGX directly from D3D12
(RenoDX's add-on, OptiScaler). If Streamline's own plugin turns out to gate on the architecture as well, the log line will
say `AdapterUnsupported` with a signed-off DLL and the override on, and the answer is the NGX-direct path for feature 18
(above, the Backend row's NGX direct) -- which is also the one that would bring NR to Linux.

## The open backend (OpenDLSS-NR)

A third way to run the same network, beside Streamline and NGX direct: **[OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR)**
(MIT, Copyright 2026 maan), a Vulkan reimplementation of the DLSS-NR 310.8.0 network that its author proves bit-exact
against NVIDIA's at all 75 block boundaries. It is vendored unmodified in `third_party/opendlss-nr` (its
`PROVENANCE.md` says what and from which commit) and runs on the renderer's own device: no NVIDIA runtime, no DLL, no
Streamline, and so **natively on Linux**. `gpu/vk/vk_opennr.cpp` is the backend; `gpu/vk/opennr/` is the layer around
upstream's code (its Vulkan shim, the network behind a small interface); `scripts/opennr_kernels.py` makes its kernels.

Selected with `RX_VK_NR_BACKEND=open` for now, and switched on like the others by the Neural rendering row or
`RX_VK_DLSS_NR=1`. When the Backend row lands it becomes that row's fourth value (the one `MERGE:` line in
`vk_opennr.cpp`). Other switches: `RX_VK_OPENNR_MODEL=<dir>`, `RX_VK_OPENNR_ROUTE=fast|glsl|reference`,
`RX_VK_OPENNR_TEMPORAL=0`, and upstream's own `DLSS5VK_*` switches, which its kernels read as they always do.

### What the player must supply, and what Hookshot does not do

**The weights.** The network reads a model directory -- `manifest.json` and its eleven stage files under `model/`, in
the layout of upstream's `docs/weights.md` -- and nothing public produces one. **Hookshot includes no weights and
provides no way to get them**: nothing here extracts, converts or downloads model data, and nothing reads any NVIDIA
binary. The player puts their own directory in `nr-model` beside the game, or names it with `RX_VK_OPENNR_MODEL`. It is
checked the way upstream's loader checks it -- every stage file's length and SHA-256 against the manifest, every
tensor inside its stage, the 71-block network and no other -- on a thread of its own, since hashing 141 MiB takes
a while; until it is done, and without one, frames go to the window untouched. Without a model the log says
`open NR: no usable model: ...` with the path it looked in, and the row's hint says the player has to supply one.
Whatever licence covers the data a player uses is theirs to respect; upstream says the same.

**The GPU**, today: an NVIDIA card whose driver has `VK_KHR_cooperative_matrix`, `VK_NV_cooperative_matrix2` (all
seven of its features) and `VK_EXT_shader_float8`. With `VK_NV_cuda_kernel_launch` as well the fast route runs
(upstream's PTX kernels, counter-chained); without it the GLSL kernels run alone, which give the same bytes. The
extensions are asked for only when this backend is selected (`vk_main.cpp`, three hooks), and what is missing is named
in the log.

### The frame

Same place and contract as the Streamline backend: over the finished frame at display size, before the menu, into the
frame's own command buffer (`present_image`, through `dlssnr::apply`; and, with no window, in the headless present
path, so a headless run exercises it). The window's frame is blitted into an RGBA8 copy, and two passes of ours
surround the network (`gpu/vk/shaders/opennr_pre.comp`, `opennr_post.comp`, following upstream's `docs/frame.md`):

- **in**: the sixteen lanes -- Gaussian noise hashed from the padded pixel, the frame centred
  (`f16((f16(c) - 0.5) * 0.125)`), the previous output reprojected (or the frame again without a history), the five
  conditioning scalars. The frame is already a display image, so it is its own proxy: no paper white, no shoulder.
  Outside the frame the field is a mirrored copy, as upstream pads it;
- **the network**: recorded once per frame size into a secondary command buffer and executed every frame. Nothing is
  allocated per frame; a new size rebuilds the graph with the device idle;
- **out**: `clamp(proxy + rgb / 4)`, the temporal blend by the head's own logit times the model's
  `blend_scale`, the next history truncated to half, then the style operator and the intensity blend.

**Temporal or single-frame: temporal**, with the scene's camera-only motion (`vk_camera_motion.h`) where there is a
scene; `RX_VK_OPENNR_TEMPORAL=0` is single-frame (lanes 7-9 a copy of the frame, a fixed noise seed, no blend). The
motion is the 3D scene's at its render size, so the HUD gets the scene's motion under it -- wrong for the HUD, and left
to the network's per-pixel blend logit to reject, which is what upstream's demo leaves disocclusion to as well. A camera
that did not move gets zero motion rather than a motion pass, because `camera_motion::record` answers a repeated matrix
with the target it last wrote (a moving frame's). Made frames of frame generation go through it like any other. How
well any of this looks is unmeasured (below).

**The Neural tab's rows**, against what the network actually takes:

| row | what it becomes here |
| --- | --- |
| Neural rendering | on / off, live |
| Style | lane 10 (`style / 128`) **and** upstream's post-blend style operator (Natural, Cinematic), as its demo does |
| Local tone | lane 11, and the style operator's strength |
| Local structure | lane 12 (or lanes 12-14 with Auto mask) |
| Skin structure | lane 13, with Auto mask only (as in NVIDIA's runtime: without it the lane is -1) |
| Auto mask | which form the structure / skin / mask triple of lanes 12-14 takes |
| Intensity | the blend back toward the frame after the network; not a network input |
| Global tone | **no counterpart**: the network has five conditioning scalars and this is not one (Streamline 2.14 ignores it too) |
| Preset | **no counterpart**: there is one network, the model directory's; NVIDIA's presets pick among networks |
| Model (Performance ... DLAA) | **no counterpart**: the network has no quality modes and always runs at the frame's size |

### What was measured, and what was not

There are no real weights on any machine this was built on, so **image quality and real-frame timing are not
measured**. What was measured is that the integration computes exactly what OpenDLSS-NR computes, with
`scripts/opennr_harness.py` (and `tests/test_opennr_harness.py`, its quick form; it says SKIP without the GPU). It
builds upstream's own `dlss5vk` tool from the vendored sources with Hookshot's clang, the same shim and portability
patch, and the kernels the renderer ships, then runs it on a **synthetic model directory** (deterministic E4M3
values in exactly the manifest layout, scaled so the activations stay alive through all 71 blocks, which the harness
checks) and a synthetic frame. Measured on an RTX 5060 Ti (Blackwell), driver 615.71.09, Linux:

- **Every route bit-exact with every other**, by upstream's own `parity` checker, on all 75 block boundaries and the
  head, at 384x256, 1280x720 and 2048x1152 (the last the game's default output size): the GLSL reference route
  (unfused, with every PTX family off and the PTX folder empty, so it provably launched none) as the references;
  upstream's `DLSS5VK_UNFUSED=1` alone; the default fast route (PTX + counter chaining); the PTX route with barriers;
  the fused GLSL route. Each also repeatable over three submissions, and the barrier and chained schedules identical.
- **Except when the activations get large.** With the synthetic decoder's weights 6-12% stronger (the harness's
  `OPENNR_SYNTH_GAIN_DECODER` at 0.85 or 0.9 instead of 0.8, decoder activations at an rms of 60-190 rather than
  under 10) the fused routes stopped agreeing with the unfused reference at 1280x720 and 2048x1152 -- from block 58
  for the fused GLSL kernels alone at 0.85, from blocks 40-45 for every fused route at 0.9 -- while they still agreed
  with each other at 384x256. Not root-caused. An f16 overflow inside a fused kernel, handled differently from the
  unfused kernels' publications, fits (the fused GLSL head carried NaNs at 0.9), but that is a guess. Whether real
  weights ever drive the network there is unknown; it is the first thing to look at if a real model's routes disagree.
- **Block 0 against upstream's CPU reference (`src/reference.cpp`) does NOT match as shipped**: every tensor-core
  kernel differs by an ulp in a quarter to a half of its values (the window normalize, which uses no tensor cores,
  matches). Upstream's reference models Ada's tensor cores -- an FP8 m16n8k32 step as two fixed-point groups of 16
  products with 13 fractional bits, an FP16 m16n8k16 step as two groups of 8 with 24. **This Blackwell card does each
  step as one group**: with the reference changed to one group of 32 FP8 products and one of 16 FP16 products (the
  harness builds both; the group width and the fraction bits are the only change), every kernel of block 0 matches
  exactly -- all 122,880 rows at 384x256 and 262,144 rows at 1280x720. The FP16 group needed at least 25 fractional
  bits; the FP8 group 19 passed 20,000 rows and failed 3 values in 8.4 million at 720p, 20 and 24 passed, so the
  harness uses 24. More bits could not be told apart on this data. Upstream's CPU reference covers block 0 only; at
  2048x1152 the CPU check was not run here (upstream's `verify` ran out of VRAM beside another game on the GPU).
- What that means: on Blackwell, OpenDLSS-NR is not bit-exact with upstream's captures, which were made on Ada; the
  same `mma.sync` instruction rounds differently on the two. Whether it is bit-exact with NVIDIA's own runtime on
  Blackwell (whose kernels run that instruction too) cannot be checked without captures from it.
- **In the game** (headless, the synthetic model, 2560x1440 frames, field 2560x1472): after the model check (a couple
  of seconds, off the display thread) it loads in about a second, then runs end to end -- 4,000-5,400 presents a run into
  gameplay, no device loss, no chained-wait timeout. The log's GPU time for the whole pass (copy, input, network,
  output) was 24-43 ms a frame on average and 20-35 ms at the least per 300 frames, over two runs; those stamps bracket the pass in a
  command buffer that overlaps the game's own work, and another game was running on the same GPU at the time, so
  they are an upper bound, not a cost. The harness measured the network alone at 2.5-5.6 ms at 384x256, 6-13 ms at
  1280x720 and 23-31 ms at 2048x1152 (minimum of three, across runs) (fast route; also with that other game running). Upstream measured 12.6 ms at
  2560x1440 on an RTX 4070 SUPER. With the synthetic weights the picture is saturated noise, as it should be; with
  Intensity at 0 the same pipeline gives back the game's frame unchanged, which checks the copy and the passes.
- Not measured: image quality, the temporal path's behaviour on real content, real-weight timing, Windows, and
  robust buffer access's cost (the renderer turns it on by default, device-wide; upstream measured ~15% for it).

VRAM: the model's raw tensors and their re-laid-out matrices, the activations of the field, and a 256 MiB host-visible
staging buffer that upstream's context keeps.

### An AMD route

The kernel layer is separate (`gpu/vk/opennr/opennr_net.h` is the seam: a device in, a features buffer and a head
buffer out), so an AMD port replaces what is behind it and keeps the frame passes, the model check and the menu
wiring. What it needs:

- **`VK_KHR_cooperative_matrix` instead of `VK_NV_cooperative_matrix2`**: upstream's GLSL GEMMs use coopmat2's
  workgroup scope, flexible dimensions, reductions and tensor addressing; they have to be rewritten for subgroup-scope
  KHR matrices of the sizes the driver lists. **No PTX** and no `VK_NV_cuda_kernel_launch`: the GLSL route is the
  whole of it, so counter chaining (PTX-only, and relying on NVIDIA's launch order) goes too.
- **FP8 on RDNA 4** (`VK_EXT_shader_float8` with E4M3 cooperative matrices), and **an FP16 path on RDNA 3**, which has
  no FP8: E4M3 weights and activations widened to half, with the E4M3 publications kept as explicit roundings so the
  network's arithmetic is the same.
- **Subgroup size 32**: upstream's fragment layouts assume it and AMD defaults to 64 (the renderer already pins 32
  where it can, `vk_main.cpp`).
- **Its own numerics model**: as Blackwell already differs from Ada, AMD's matrix units will round their own way.
  The harness's references come from the GLSL reference route and its CPU check takes an MMA model, so the port is
  checked the same way -- against the GLSL route on NVIDIA for the boundaries, and against a CPU model of its own
  arithmetic for block 0 -- and will at best be close to NVIDIA's bytes, not equal to them.

## On AMD

What exists for neural rendering on an AMD card, as of November 2026, and what of it can be used here:

- **[DLSS-NR on AMD](https://github.com/danielblnc/DLSS-NR-on-AMD)** (NeuralHub) reimplements the model in HIP and hooks
  D3D12 games. Its licence forbids redistribution, bundling and reverse engineering, and it is D3D12 only -- Vulkan is
  "planned". Hookshot is Vulkan, so it cannot hook it, and may not ship it.
- **[OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR)** (MIT) is a Vulkan reimplementation of the 310.8.0
  network, bit-exact block by block. Its kernels need `VK_NV_cooperative_matrix2` and `VK_NV_cuda_kernel_launch` (the
  fast path is PTX), so it is NVIDIA-only as it stands, and it reads the weights from a model directory that "nothing in
  this repository produces". It is now Hookshot's third backend (below, "The open backend"); what an AMD port of it
  needs is in that section. The model directory stays the player's to supply on AMD as on NVIDIA.
- **AMD's own machine-learning upscaler is now here**: FSR 4, on the Upscaler row (`docs/upscaling.md`).

## What is still unproven

Being plain about this, because the rest of this file reads confident:

- **Nobody has been found shipping or measuring DLSS-NR on Vulkan.** Dagor's NR path is written to be
  API-agnostic and does handle Vulkan resources, but every measurement found was D3D. This is the
  least-trodden configuration of the feature.
- Elysium reached start-up and feature support but **never drove `slEvaluateFeature` over real
  images**, so nothing in this project has yet seen the model produce a frame.
- Whether a flat depth and zero motion are *acceptable* rather than merely *accepted*. The four tags
  are declared required and are all supplied; whether the model wants physically meaningful depth is
  untested. The ReShade addon's present-time hook strongly suggests dummies are tolerated.
- In-place aliasing of tags 70 and 71. NVIDIA's header says it may alias; no integration found does
  it, and this one uses two images.
- The upper bound of `style`. Three values are documented (Default, Natural, Cinematic); the row
  offers exactly those three.
