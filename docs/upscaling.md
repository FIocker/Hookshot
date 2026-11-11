# Upscaling and frame generation

Most of this file is the record of how the upscalers got here: the plan it started as, and the corrections the runs made
to it. What the renderer does today is the section right below; where an older section disagrees with it, the older
section is history and says so.

## How it is wired now

AMD FSR 3.1 is upstream's implementation (NotRexed/Hookshot#1386, `pr/fsr3-upscale-framegen`), taken over this fork's
own earlier one, with the fork's AMD FSR 4 and DLSS frame generation put on the same inputs.

**One place makes every input.** `gpu/vk/vk_fsr3_inputs.cpp` `prepare()`, at the gameplay upscale's dispatch (the one
NVIDIA DLSS takes, found the same way: `fsr3_stand_in` in `vk_draw.cpp`, before `dlss_stand_in`), makes one `Frame` at
the render size with one pass (`shaders/fsr3_prep.comp`):

- **Colour:** the network's input as the network samples it (its square-root encoding), clamped to 0..1, NaN set to 0.
  The upscaler's picture is squared back into the network's output by `dlss_out.comp`, the pass DLSS's picture goes
  through, for FSR 3.1 and FSR 4 alike.
- **Depth:** the scene's depth is 0 at the near plane (NVN's -1..1 clip depth, not reversed). FSR's passes are compiled
  for reversed depth, so the copy is `1 - d`, and the factors that turn it into a view distance come from the projection
  the scene is drawn with (`fsr3_math.h`, checked by `tests/test_fsr3_math.cpp`): near 0.1, far 1000.5, a 45 degree
  field of view, read from the camera's uniform buffer as `dlss_note_scene_draw` reads it.
- **Jitter:** #841's, the one jitter in the game (`nx2/oot_jitter.py`, installed in every standalone lift now, and
  `host/rx_backend.c hos_dlss_jitter_clip`), on while DLSS, FSR 3.1 or FSR 4 draws, read back from the projection.
- **Motion:** the camera's, from the depth, with `dlss_prep.comp`'s math (this frame's jittered clip space to the frame
  before's without its jitter), in render pixels. What moves by itself has the motion of what is behind it until #1008's
  actor motion is shared.

**Who takes them.** FSR 3.1's upscaler (`vk_fsr3.cpp`) and its frame interpolation (`vk_fsr3fi.cpp`); in this fork also
AMD FSR 4 through AMD's runtime (`vk_ffx.cpp`: the same images, with `FFX_UPSCALE_ENABLE_DEPTH_INVERTED` set because the
copy is reversed and `DEPTH_INFINITE` only for an infinite projection, which this game's is not), and NVIDIA DLSS frame
generation (`vk_framegen.cpp dlssg_frames`: `DepthInverted` now true of the depth it is given, the planes and the field
of view from the same projection, and the clip-to-previous-clip pair the motion was made with).

**Frame generation** (`vk_framegen.cpp`, the method row in `framegen_methods.h`): at 2x to 8x Auto tries DLSS frame
generation, then FSR 4's, then FSR 3.1's, then the renderer's own blend; a method chosen outright is tried alone. FSR 3.1's
interpolation has no phase (AMD's `ffx_frameinterpolation.h` fixes t at 0.5), so it makes 2x sets only
(`framegen_methods::makes_set`): this fork used to repeat its one midpoint for every made frame of a 3x or 4x set, which
shows the same picture at two different times and judders. Each present's inputs are taken once (`fsr3_inputs::take`), so
no present interpolates with another frame's depth and motion.

**What the port fixed** in the fork's version:

- the camera hook (`nx2/oot_camera.py`, `host/hos_oot_camera.c`) published sead's projection object, near 1.0 and far
  20000, not the projection the scene is drawn with, and FSR was given reversed-infinite factors for a depth that is
  neither: gone, the camera comes from the projection;
- FSR's picture was blitted into the network's output without undoing the network's square-root encoding;
- the FSR files did not include `vk_work.h`, so `full_barrier` could leave out a barrier after their work;
- the camera-motion image never had its first layout transition;
- on a resize, images frames in flight could still use were destroyed (they are freed a few presents later now).

**What went with it:** the fork's viewport jitter (`vk_draw.cpp` `SceneJitter`, `RX_VK_JITTER`), the camera hook, the
seam and contract headers (`upscale_seam.h`, `upscale_contract.h`, `upscale_format.h`) and their tests, the per-network
log's contract roles, `RX_VK_FSR_STAGE`, `RX_VK_VP_HUNT` and `DescWriter::Source`. `vk_camera_motion.cpp` and
`scene_frame_inputs()` stay for the open neural rendering backend (`vk_opennr.cpp`), which nothing here changes.

## What exists today

| | Where | What it is |
|---|---|---|
| `resolution` | `gpu/vk/vk_menu.h`, `nx2/oot_patches.py` | The 3D scale: Auto (the game's dynamic resolution), or 100/75/67/50% of full |
| `upscale` | `gpu/vk/vk_menu.h`, `framegen_scale` | A spatial filter on the finished window image: off, Catmull-Rom, Lanczos |
| Frame generation | `gpu/vk/vk_framegen.cpp` | Image-space interpolation: quarter-size luma, 16x16 block matching within +-32 px, a 3x3 median, then a forward/backward blend per made frame |
| The game's own upscaler | `taau_stand_in`, `gpu/vk/vk_draw.cpp` | OoT evaluates a neural TAAU network (program `fd029a317a9a4b94`). Where the device has no hardware f16 matrix multiply (or with `RX_VK_NO_F16`), a compute stand-in replaces it, because the translated network loses the blue channel and casts the scene yellow; elsewhere the network itself runs. `RX_VK_TAAU_KEYS` overrides either way |
| The per-network log | `gpu/vk/vk_draw.cpp` | Once per network key: its dispatch size and its sampled textures |

Two consequences follow from that table, and they decide the whole design.

**The game already upscales temporally.** It renders the 3D at a dynamic internal resolution and
resolves it with its own network. A vendor upscaler does not need a new low-resolution path invented
for it; it needs to take that network's place. Note that on a card with hardware f16 matrix multiply
the network runs as itself, so a replacement cannot be built inside the stand-in alone: the dispatch
has to be recognized whether or not the stand-in is the thing currently handling it.

**The finished window image is the wrong place.** `upscale` and the frame interpolator both work on
the composited frame, after the HUD. A temporal upscaler there has no depth, no motion and a HUD
baked into its history. That is why the existing interpolator block-matches: it has nothing better to
work from.

## What the baseline actually is

Measured before any of this changes anything, because it decides whether an upscaler is worth
having here at all. Linux, RTX 5060 Ti, driver 615.71.09, headless under Xvfb, a warmed shader
cache, the opening and Kokiri Forest:

| | |
|---|---|
| Frame rate | **57-60 fps**, held over 33,600 presents in one run and 9,000 in another |
| A frame | 1106 draws (0 skipped), 99 dispatches, 148 passes |
| Scene | 1536x864 internal, resolved by the game's own network into a 2048x1152 scene |
| Present | 2560x1440 |

**60 fps is the floor here, not the target.** The game's own limiter lives at `mod + 0xfc` and the
engine's adaptive controller writes any rate from 40 up when variable refresh is on (`+0x231`), so
`HOS_FPS_TARGET=<fps>` can ask for more; the real-time frame step at `main+0x6e6cc4` already keeps
game speed honest at whatever rate comes out. What an upscaler buys against that is the two things
a floor does not give you: a higher internal resolution at the same rate, and the headroom to hold
a higher rate at all. The scene is 1536x864 for a 2560x1440 present — there is a lot of resolution
being left behind, and that is the gap FSR or DLSS closes.

So the argument for this work is image quality and headroom at a given cost. It is not "getting to
60", and this document should not be read as saying the job is done at 60.

## Correction: the game's network is not a temporal upscaler

**A run settles this, and it contradicts the section above.** With the per-network log reading the
contract, the gameplay network reports:

```
vk: network fd029a317a9a4b94: dispatch 32x36x1, 1 textures 1 images: 0:1536x864; no upscaler inputs
```

**One sampled texture, one written image, and nothing else.** No motion, no depth. Every one of the
eleven networks in the game is the same shape: at most two sampled textures, never a motion or depth
target among them.

So OoT's upscaler is a *single-image* neural upscaler, not a motion-vector temporal one, and the
premise this document started from — that a vendor upscaler takes that dispatch's place — is wrong.
FSR 3.1 and DLSS Super Resolution are temporal: they need motion and depth, and this dispatch is
never handed either. `temporal_possible()` therefore refuses at it, correctly, and will keep
refusing however the backend is wired.

What the run does confirm is that the opportunity is real: the input is **1536x864** against a
2048x1152 scene, so the game already renders the 3D at 75% and upscales. There is a genuine upscale
to improve on.

What it means is that the inputs have to be **assembled** rather than taken:

- **Depth** exists — the census found D32 depth attached to the scene passes — but it has to be
  carried to the resolve rather than read off this dispatch.
- **Motion** does not exist anywhere in the frame as a target. It has to be derived, and the only
  cheap source is the camera, which gives camera-only motion: correct for a still world, ghosting
  on anything that moves by itself.

That is precisely what #577 does, at the final resolve rather than at this dispatch, and this
finding is the argument for its integration point over the one below. The contract, the census and
the FSR passes stand; where they are plugged in has to change.

## The integration point (superseded by the correction above)

`taau_stand_in()` in `gpu/vk/vk_draw.cpp` already recognizes the network's dispatch by program key
and substitutes its own resolve. A vendor upscaler replaces the same dispatch, under the same
recognition, and falls back to the existing stand-in whenever it cannot run. That keeps one code path
for "the network did not run as itself" instead of two.

The replay rules in [design notes, section 4](design.md#4-the-lessons-from-mkw-as-rules-here) apply
unchanged. In particular the inputs are read from the submitted dispatch's own descriptors during
ordered replay, never from live guest memory.

## The camera, read out of the game rather than guessed

> **Superseded.** The hook this section describes is gone. It published sead's projection object (near 1.0, far 20000),
> which is not the projection the scene is drawn with: that one, read from the camera's uniform buffer, has the near
> plane at 0.1 and the far plane at 1000.5 (`p22 -1.0002, p23 -0.20002`). See "How it is wired now" above.

FSR's `device_to_view_depth` and `tan_half_fov` want the near plane and the vertical field of view,
and camera-derived motion wants the view-projection. Both are available: sead's perspective
projection object carries them, and its matrix update is already a patch site (`nx2/oot_wide.py`,
main+`0x3485cc0`, which is the whole function `f_834a5cc0`).

Reading that function's lifted C gives the layout, each offset from what the code does with it
rather than from a guess:

| offset | what it is | how the code shows it |
|---|---|---|
| `+0x98`, `+0x9C` | the near and far planes | loaded as a pair (`ldp s4, s3, [x0, #0x98]`), then **swapped by the byte at `+0x0A`** — a reversed-depth flag — before their difference is taken |
| `+0x0A` | which way round the depth runs | selects between the two above |
| `+0xAC` | `tan(fovy / 2)` | `1.0f / it` becomes the matrix's m11, which is `1 / tan(fovy/2)`. So the field of view is `2 * atan(it)` |
| `+0xB0` | the aspect ratio | divided into m00, and the ultrawide patch stores its own ratio back here |
| `+0xB4`, `+0xB8` | the projection's offsets | each doubled into m02 and m12 |

`x0` is the projection object throughout that function and `x1` is the 4x4 it writes, so one
injected block there can publish the near plane, the far plane and the field of view to host globals
for the renderer to read, the way `hos_wide_ratio` already travels the other way. Note that the path
which actually runs is one of the **inlined** copies, not the out-of-line function, and each holds
the projection in a different register; patching only the out-of-line one published nothing.

**A run confirms the layout and the values.** The object is built on the stack, so a published
pointer read a frame later is all zeros; snapshotting its first 48 floats at publish time gives:

```
+0x80: 0 0 0 0 1 0 1 20000
+0xa0: 0.7854 0.38268 0.92388 0.41421 0.5625 0 0 0
```

- `+0x98` = **1.0**, the near plane, and `+0x9c` = **20000**, the far plane
- `+0xac` = **0.41421** = tan(22.5 deg), so the vertical field of view is **45 degrees** — and
  `+0xa0` holds that same angle in radians, 0.7854
- `+0xb0` = **0.5625** = 1080/1920, so it is height over width, not width over height
- `+0xa4` and `+0xa8` are sin and cos of the half angle

That 45 degrees is also the needle for the rest of the matrix: a projection built from it has
m11 = 1/tan(22.5 deg) = **2.41421**, so the view projection the draws use can be found by looking
for that value in the uniform buffer a scene draw reads, rather than by guessing at another object's
offsets. `RX_VK_VP_HUNT=1` does that, and found it at once:

```
vp hunt: stage 5 slot 0, m11 at +68, matrix from +48:
    1.358   0       0      0
    0       2.414   0      0
    0       0      -1     -0.2
    0       0      -1      0
```

m11 is 2.414 as predicted, 2.414 / 1.358 = 1.777 = 16/9, and the last row makes w = -z. So the
scene draws read their projection from **stage 5, slot 0, byte 48** of a uniform buffer, and it
agrees with the camera the lifted code publishes.

It is the projection **alone**, though: the upper-left 3x3 is diagonal, with no rotation or
translation in it. A temporal upscaler reprojects with view times projection, so the view is still
to be found -- and it should be in the same buffer, since a draw needs both.

That removes the approximation the reference implementation settles for — it documents using "a
typical near plane and vertical field of view" because DLSS hands it no camera — and it is the same
data camera-derived motion needs.

## The input contract

> **History.** `upscale_contract.h`, `upscale_format.h` and their test are gone: the inputs are made in one place now
> (`vk_fsr3_inputs.cpp`), not classified out of the dispatch's bindings.

A temporal upscaler needs the network's own inputs, and it must only run when they are all present and
of the expected shape:

- low-resolution color, HDR, matching the output's format
- motion, two channels, normalized toward the previous frame, in render pixels
- depth, reversed, at the render extent
- a fresh jitter sample for this frame, in render pixels

and it must reset its history on a resolution change, on first use, and after a gap longer than about
250 ms (a pause, a menu, a load). A camera cut is a separate reset that this game does not hand us
directly.

`gpu/vk/upscale_contract.h` decides which sampled binding is which, and `gpu/vk/upscale_format.h`
decides what each Vulkan format counts as. Both are driven by `tests/upscale_contract_test.cpp`
without a game, a GPU or a dispatch. Motion and depth fix the render extent, because they are the two
a network cannot sample at another size; the colour at that extent is the input. A colour at a larger
extent is reported as full-size and nothing more: the renderer records what a dispatch's descriptors
sample *or store* in one list, so the history and the output cannot be told apart there, and the seam
in row 2 is where that is resolved.

Where any of that does not hold, the native dispatch or the stand-in stays. An upscaler that runs on
inputs it has guessed produces ghosting that looks like a renderer bug and is reported as one.

> The same contract, derived for a different Switch 2 title in a sibling GPL-3.0 project, is written up
> in Eglantine's `docs/native-dlss-replacement.md`. The formats there (B10G11R11 color, R16G16 motion,
> D32 depth) are that game's, not necessarily this one's: the shapes have to be read off OoT's own
> dispatch before anything is wired to them.

## The seam

> **History.** `upscale_seam.h` is gone with the contract; FSR takes the dispatch the way DLSS does (`fsr3_stand_in`).

`gpu/vk/upscale_seam.h` was where a backend was chosen. The renderer resolves a recognized dispatch once into an
`upscale::Request` — what the sampled bindings provide, the image the dispatch writes, and which binding the renderer
is going to hand over as colour — and `choose()` says which backend takes it. A backend never re-derives the dispatch
for itself.

`temporal_possible()` is the gate a temporal backend has to pass, and every clause in it is a wrong picture rather
than a missing feature: the inputs must be complete, the output must be resolved, writable and no smaller than the
render extent, and the colour the renderer hands over must be the colour the contract identified.

The bindings the contract reads are in **descriptor order**, resolved the way `build_descriptors` resolves them, so
index `i` is `dw.imgs[i]` — the descriptor a backend is handed. That matters because the obvious shortcut is wrong:
`draw_sampled` records a sampled texture only where the GPU has written it and records every storage image as well,
so binding 0 of that list is not `dw.imgs[0]`. A backend wired to that correspondence would read whichever texture
happened to be bound first and ghost, and the ghosting would be reported as a renderer bug rather than as a wrong
binding. The stand-in's own `dw.imgs[0]` is the colour only when the contract agrees that slot 0 is the colour.

The stand-in needs none of this. It is a spatial resolve, and it runs wherever `RX_VK_TAAU_KEYS` asks for it exactly
as it did before the seam existed, including where the contract is not met. That property is what makes introducing
the seam a no-op, and `tests/upscale_seam_test.cpp` pins it.

## Order of work

Each row is one pull request, and each one is useful on its own.

| # | Change | Why it is first |
|---|---|---|
| 1 | Name each network binding's role in the existing per-network log, and say whether the inputs are complete | Nothing can be built until the contract above is read off the real dispatch rather than assumed, and the log that reports a network's textures was already there |
| 2 | A vendor-neutral seam at `taau_stand_in`, with the stand-in as its only backend | No behaviour change, and it is the review that decides the shape |
| 3 | FSR 3.1 temporal upscaling as the first backend | MIT sources, no external runtime, works on every vendor, so it is the backend that can be the default |
| 4 | DLSS Super Resolution: Streamline on Windows, NGX on Linux | Needs a user-supplied runtime, so it comes after a working fallback exists |
| 4b | FSR 4 through AMD's runtime, behind the same seam | The same API as row 3, so it is a backend rather than a rewrite; AMD hardware only |
| 5 | Presets, quality modes and DLAA in the Insert menu and `hookshot_options.ini` | Settings for backends that exist, rather than settings-first |
| 6 | Frame generation from real motion and depth | Replaces the block matcher's guess where the inputs are available |

## What each vendor runtime allows

**FSR.** AMD's FidelityFX sources are MIT and can be vendored under `third_party/` with a `PROVENANCE.md`,
unmodified. There are two integration routes, and they are not interchangeable:

- *AMD's FidelityFX API*, loading AMD's own Vulkan runtime library at run time. This is what #577 does, and it has
  already vendored the five MIT API headers under `third_party/fidelityfx/` with provenance. The runtime libraries
  are AMD's and are not committed. FSR 4's newer models arrive this way, on Windows.
- *The FidelityFX SDK sources*, where the upscaler's passes are recorded from AMD's MIT shader sources and no runtime
  library is needed at all. This is the route that works on Linux as well as Windows.

This tree takes the **SDK-source route**, under `third_party/fidelityfx-sdk/` with its provenance note: AMD's GPU
headers and the eight upscaler passes from SDK v1.1.4, unmodified, MIT. No runtime library is needed and none is
included, and the same code path serves Windows and Linux.

The eleven variants (the eight passes, plus wave-free pyramids and a sharpening accumulate) are compiled by
`scripts/fsr3_shaders.py` into `gpu/vk/shaders/fsr3_*_comp_spv.h` and **checked in**, like every other shader here, so
a clone still builds with no glslang. `tests/test_fsr3_shaders.py` recompiles them all and compares, so the headers
and their sources cannot drift apart, and validates each module. That costs about 1.3 MB of generated hex in the
repository; generating them at build time instead would remove that at the price of a glslang dependency for every
build, and is the maintainer's call rather than something to decide in a patch.

Note that #577 vendors AMD's *API* headers under `third_party/fidelityfx/` for the other route. The two folders are
different things — API headers against a runtime library, versus GPU sources compiled here — but if both land, one of
them should go.

**DLSS.** The Streamline headers are MIT and can be vendored the same way. **The runtime libraries are
not**, must not be committed, and must not be bundled in a release. The user supplies them, or the
build fetches them only after the user has accepted NVIDIA's license. Streamline has no Linux build,
so Linux uses NGX directly — `libnvidia-ngx.so` ships with the NVIDIA driver, and the model library
sits beside the binary. The same two backends therefore need two loaders behind one interface.

Neither backend may be a build requirement: a tree with no SDK present builds, and a machine with no
runtime present runs, with the stand-in.

## Denoising upscalers, and how to find out

NVIDIA calls it Ray Reconstruction, AMD calls it Ray Regeneration (RDNA 4 and newer only). Both are the same kind of
thing and neither takes colour, motion and depth: they replace a ray-traced renderer's denoiser, and they want guide
buffers — albedo, normals, roughness, specular hit distance. Nothing in `gpu/` names such a target, and this game is
rasterized, so the working assumption has been that they cannot be given what they need here.

That assumption is now testable rather than repeated. A deferred renderer writes its guides as several colour
attachments of one extent in one pass, so `RX_VK_MRT_CENSUS=1` logs each distinct render target set once — how many
colour attachments the game binds together, their formats and the extent — from `targets()`, which resolves once per
`NOP_SET_RT` rather than per draw. Off by default and free while off.

Two outcomes, both useful:

- **Never more than one colour attachment.** There is no G-buffer, there are no guides, and a denoising upscaler is
  settled as not applicable to this title. Say so in the docs and stop proposing it.
- **Three or four at one extent.** Guides may exist, and identifying which attachment is which becomes the same kind
  of work the input contract above already is.

**The log has now been read, and the naive reading of it is the opposite of what was expected.** A
240 s run of the opening (`RX_VK_MRT_CENSUS=1`, 136 distinct target sets, 491 presents) found that
the game *does* bind multi-attachment passes:

| sets | colour attachments | what they are |
|---|---|---|
| 103 | 1 | the main scene at 2048x1152 B10G11R11, the 2560x1440 RGBA8 sRGB present, and the UI |
| 18 | 0 | depth-only passes |
| 7 | **6** | 1x1 through 64x64, every attachment B10G11R11: six faces of a cubemap down a mip chain, so an environment or irradiance probe |
| 6 | 2 | 170x96 and 384x216 R16F + R32F with D32; 341x192 and 768x432 RGBA16F + RGBA16F with D32; 480x270 two B10G11R11; 960x540 B10G11R11 + R8 |
| 2 | 3 | 341x192 and 768x432, R32F + *undefined* + R16F with D32 (slot 1 unbound) |

So the count alone does not settle it — the formats and extents do. A deferred G-buffer is three or
four attachments **at the main render resolution** carrying albedo (an 8-bit or sRGB colour),
packed normals (RGB10A2 or RG16) and roughness/metallic. Nothing here is that. The main scene is a
*single* HDR target; every multi-attachment pass is either a cubemap face chain or a small
post-process at a quarter resolution or less, in single-channel float or HDR pairs.

**Conclusion: there are no albedo, normal or roughness guides to give a denoising upscaler, so Ray
Reconstruction and Ray Regeneration remain inapplicable to this title** — but on the evidence above
rather than on the absence of a target named "normal" in `gpu/`, which was never a sound argument.

What *is* deliverable, and is what "the newest upscaling" means in practice on each vendor:

| | |
|---|---|
| NVIDIA | DLSS Super Resolution with the transformer presets, and DLAA at native resolution. A neural network upscaler, through NGX on Linux and Streamline on Windows |
| AMD | FSR 4's machine-learning upscaler, reached through the FSR 3.1 API that row 4 below targets. Radeon RX 9000/7000/6000; AMD's own API falls back to FSR 3.1.6 elsewhere |
| Everyone | FSR 3.1 compiled from the SDK sources in this tree, which needs no runtime and no particular vendor |

## Correction: FSR was not what cost the frame rate

For several rounds of work the record here said that engaging FSR 3.1 dropped the game from about
60 fps to 3. It did not. The cost was a bug in this project's own `capture_view_projection`, and
every hypothesis chased before that -- the pass count, the shared SPD counter, a system-memory
fallback, repeated evaluation -- was chasing a cost that was not there.

What settled it was a controlled pair rather than another hypothesis. Both arms ran headless, with
the pad driven by a frame-clocked `HOS_INPUT_SCRIPT` so that a slow arm and a fast one reach the
same place in the game, and both reached the same scene (same draw, dispatch and pass counts):

| | frame rate | note |
|---|---|---|
| `RX_VK_FSR=0` | 2.7, 2.9 fps | |
| `RX_VK_FSR=1` | 2.7, 3.0 fps | FSR running, `1536x864 -> 2048x1152`, its own log line present |

Identical. Two GPU measurements already agreed and had been read wrongly: the nine passes together
measured **0.03 ms a frame, 0.7% of GPU time, 102 us a span**, which is about a millisecond of GPU
work per evaluation. A backend costing a millisecond cannot take 300 off a frame.

The 300 ms was `capture_view_projection`, which walks five stages and sixteen slots reading a
uniform buffer from each to find the scene's view projection. It is meant to run once per render
target set, and it set its `done_for` guard **only when it found a matrix**. Most target sets --
shadow, UI, post -- hold no projection, so for those the guard never armed and the whole scan ran
again on every draw of them. At 1800 draws a frame that came to roughly 30,000 reads of guest
memory a frame; each read that lands in a captured page copies that whole 4 KB page out of the
capture, which is about 67 MB of memcpy a frame. It ran with FSR off exactly as with FSR on, which
is why both arms were slow and why the earlier comparison -- a heavy scene with FSR against a title
screen without it -- looked like FSR's doing.

Two things worth keeping from this:

- **A guard that is armed only on the path that succeeds is not a guard.** The failing path is the
  common one, and it is the one that repeats.
- **Compare two arms in the same place in the game, or do not compare.** The frame-clocked input
  script exists for that: without it, "before" and "after" are two different scenes and the
  difference between them is whatever you want it to be.

## FSR 3 frame interpolation: what is verified and what is not

`gpu/vk/vk_fsr3fi.*` records AMD's seventeen passes -- seven of optical flow, ten of interpolation --
in the order the SDK's own dispatch schedules them, behind `RX_VK_FSR_FG=<1-3>`. It takes the same
assembled inputs DLSS frame generation takes, so `framegen_record` asks DLSS first, this second, and
keeps the renderer's own colour interpolation for the frames neither takes.

Verified against the sources, not from memory: every binding number comes from the pass file's own
`FFX_..._BIND_*` defines, every descriptor type from the declarations in the callbacks headers, the
image formats from the GLSL's format qualifiers, and each of the four constant blocks carries a
`static_assert` on its size. The dispatch sizes and the SPD setups are transcribed from AMD's host
reference.

**What is inferred rather than verified.** The picture cannot be checked from here -- no capture, no
validation layers, no eye on the output -- so this list is the honest part, in the order worth
looking at:

1. ~~**`opticalFlowScale = 1 / displaySize`.**~~ **Checked.** No vendored source assigns it -- the
   FSR3 wrapper that would is not in the SDK subset here -- so it was derived rather than read. It
   holds: `GetOpticalFlowSize()` in `ffx_frameinterpolation_callbacks_glsl.h` is
   `(1.0 / opticalFlowScale) / opticalFlowBlockSize`, and with the staged `1 / displaySize` and a
   block size of 8 that is `displaySize / 8` -- exactly the extent `prepare()` allocates the flow
   images at, which a run confirms (`2048x1152 ... flow 256x144`). It is the only value that makes
   the shader's idea of the flow extent equal the image it is given. Where the display size is not
   a multiple of 8 the shader's integer divide floors while we allocate the ceiling, so it addresses
   inside the image rather than past it, which is the safe direction.
2. **A count above one repeated a frame.** FSR 3.1 interpolates the midpoint and only the midpoint --
   no phase in its constants, and its own header writes `t = 0.5` outright -- so a 3x set copied the
   one made frame rather than making two, and showed one picture at two times. It makes 2x sets only
   now (`framegen_methods::makes_set`), and `RX_VK_FSR_FG` is 0 (never) or 1 (over the row).
3. **A reset frame reports failure** although the chain is recorded, so the caller's own
   interpolation takes it rather than a blend against a black previous source. The SDK returns OK
   there; this is a deliberate difference.
4. **sRGB transfer and `{0, 1}` luminance**, right for the R8G8B8A8 window and untested against
   anything else. AMD's own sources are inconsistent here: the optical flow's transfer-function-0
   path treats the colour as linear while the interpolation applies an sRGB decode. Left as AMD has
   it.
5. Optical-flow confidence is a cleared image where the SDK binds null and reads zero through it;
   the flow vector-field dispatch rounds the last partial block up where the SDK truncates in float;
   the constants staged for the passes before the flow's own pyramid are a sane value rather than
   the SDK's leftovers. Each is commented where it is.

## The jitter, and why the renderer supplies it

> **Superseded.** The viewport jitter below is gone. #841's jitter (`nx2/oot_jitter.py`) offsets the projection where the
> game writes it, for every view of the picture's ratio, and costs nothing per draw; DLSS and both FSRs use it.

Neither upscaler had one, and there is none to extract. `capture_view_projection` refuses any
projection whose m02 is not exactly zero, and it finds one -- so the matrix the game hands its own
network carries no sub-pixel offset. Its network is single-image and has no use for one. That
settles the question without a run.

So the renderer supplies it, as a sub-pixel **viewport** offset rather than a change to the
projection. The projection lives in the game's own memory where the GPU reads it, so offsetting it
would mean binding a patched copy to every draw that reads it, which is the shape of per-draw work
that has already cost this renderer 300 ms a frame once. A viewport offset moves rasterization by
the same amount, costs two additions, and applies to colour, depth and motion together because they
are one pass.

It is applied only to the scene's own passes -- the ones drawing into a depth target of the extent
the upscaler is given -- never the HUD, the menus or the movie layer, which have nothing to resolve
them and would simply shimmer. The sequence is Halton with AMD's 8-times-ratio-squared phase count,
and the offset is handed to both backends so they can undo it. `RX_VK_JITTER=0` turns it off with an
upscaler still on, which is the comparison worth having; the sign convention is the thing to check
first if the image looks wrong, because a wrong sign there is soft or shimmering rather than broken.

## The settings a player actually sees

> **History.** The rows are the Upscaler (the game's own, DLSS, DLAA, FSR 3.1, FSR 4: `upscaler_rows.h`), Frame
> generation and the Frame generation method (`framegen_methods.h`) now; the Window scaling filter is a row of its own.

None of this has a switch of its own in the menu. It is the two rows that were already there, which
is the point: a player asking "what makes the picture bigger" should find one answer, not a filter
row and a second row that can disagree with it.

**Upscaling** carries five values. `Off`, `Catmull-Rom` and `Lanczos` are filters over the finished
picture, exactly as before. `FSR 3.1` and `DLSS` instead resolve the 3D scene itself, in the place
of the game's own upscaler. `window_filter()` and `scene_upscaler()` in `gpu/vk/vk_menu.h` say which
half a caller means, so the filter's shader is never handed a mode it does not have.

**Frame generation** stays what it was: how many frames to make between the game's own, 2x to 8x. It
does not ask which library makes them. Both backends are asked whenever it is on, and
`framegen_record` tries DLSS, then FSR 3, then its own colour blend -- so the setting means the same
thing on every machine, and a card that cannot do one of them simply gets the next.

**A build with no NGX SDK does not offer DLSS at all.** `dlss::built_with_sdk()` is a compile-time
fact, and `menu_init` clamps the Upscaling row to stop at FSR 3.1 when it is false. Every released
build is such a build: CI has no NVIDIA SDK and none may be committed, so a release that listed DLSS
would be listing something the binary cannot do. That was shipped once, in v2.1.0, and this is the
fix.

The `RX_VK_*` switches still win wherever they are set, so a test run is never at the mercy of a
saved ini.

## Neural rendering on AMD: the answer is the game's own network, not FSR 4

The obvious reading of "neural rendering on AMD" is FSR 4, and FSR 4 is the wrong place to start.
Its network does not ship as source: it lives in AMD's own runtime, so a backend for it would be the
same shape as the DLSS one -- a build-time gate pointing at a vendor SDK, off by default, untestable
on a machine without the hardware. Worth having eventually; not the thing that makes neural
upscaling work on AMD.

The thing that does is already in this tree. **Ocarina ships its own neural upscaler**, and
`gpu/vk/vk_coopmat.cpp` already runs it natively: the game's TAAU and movie networks are
`HMMA.16816.F16` programs, and that module lowers them to `OpCooperativeMatrixMulAddKHR` through
**`VK_KHR_cooperative_matrix`** -- the vendor-neutral Khronos extension, not NVIDIA's `NV` one. AMD
exposes it on RDNA 3 and newer. So the neural upscaler is not an NVIDIA feature here; it is a
feature of whatever GPU can multiply matrices.

What is not neutral is how the fragments get into the matrix. The module probes the device at start,
measuring which matrix positions each lane's elements are, and accepts the device only when a lane
holds **the same positions** the console's fragments do (PTX `mma.m16n8k16`'s layout), with at most a
different order inside the lane. Anything else falls back to the shuffle expansion in
`lower_to_maxwell.h`, which is correct but a fraction of the speed. AMD's WMMA distributes elements
differently, so the expected outcome on an AMD card today is: **the network runs, and runs slowly**.

The fix is not to teach the emitter AMD's layout. It is to stop depending on any layout. A
cooperative matrix's element distribution is deliberately opaque -- that is what
`OpCooperativeMatrixLoadKHR` is for -- so the general lowering is to round-trip the lane fragments
through shared memory in matrix order, load the matrix, multiply, store it back, and read each
lane's fragments out. That costs shared-memory traffic and works on every implementation, including
ones nobody has tested, instead of only on the one whose layout happens to match.

That is the recommendation, and it is deliberately not implemented here: there is no AMD GPU on this
machine, and the lesson of this document is what happens when something unverifiable gets written
anyway. The probe already reports what it measured, so the first real step is one log line from an
AMD card.

## FSR 3 and FSR 4 are two backends, not two versions

FSR 4 is not source-available: it ships as prebuilt libraries signed by AMD (`amd_fidelityfx_loader_vk.dll`,
`amd_fidelityfx_upscaler_vk.dll`, `amd_fidelityfx_framegeneration_vk.dll`, MIT-licensed in FSR SDK 2.4), so it cannot
come out of `third_party/fidelityfx-sdk/`. `gpu/vk/vk_ffx.h` is the second backend, behind the same seam:

- **FSR 3.1, from the SDK sources here** (`vk_fsr3.cpp`). No runtime, every vendor and both platforms.
- **FSR 4, through AMD's runtime** (`vk_ffx.cpp`). Windows only, because the runtime is Windows DLLs. It is compiled
  against AMD's FSR API headers (`third_party/ffx-api/`, MIT) and loads the loader from beside the game at run time:
  nothing is linked, and a PC without the DLLs runs exactly as before. The Windows release puts the DLLs there
  (`windows-builder.yml` packages them as the `amd-fsr` part, `nx2/make_game.py place_prebuilt_runtime`).

The runtime chooses the provider (`ffxQuery` `GET_VERSIONS` lists them, newest first, and the log prints the list): FSR 4
on an RX 9000 and the cards SDK 2.4 added, its own FSR 3.1 elsewhere. That is AMD's choice and the log says which was
made, so "FSR 4" in the menu on an NVIDIA card is honestly "what AMD's runtime runs on this card".

Three things the integration depends on, each from AMD's own sources rather than guessed:

- **Resource states.** Every image here lives in `VK_IMAGE_LAYOUT_GENERAL`, and AMD's Vulkan backend maps API states to
  layouts (`backend/vk/ffx_vk.cpp getVKImageLayoutFromResourceState`): only `COMMON`, `GENERIC_READ` and
  `UNORDERED_ACCESS` are `GENERAL` there. `COMPUTE_READ` -- the obvious choice for an input -- is
  `SHADER_READ_ONLY_OPTIMAL`, and would have the runtime transition from a layout the image is not in. So inputs are
  `GENERIC_READ` and outputs `UNORDERED_ACCESS`.
- **The inputs are FSR 3.1's** (`vk_fsr3_inputs.h` `Frame`). Jitter and motion in render pixels with a scale of 1, the
  depth reversed (`DEPTH_INVERTED`) with the scene's finite far plane (`DEPTH_INFINITE` only for an infinite projection:
  it used to be set always, for a depth that was neither reversed nor infinite), the colour as FSR 3.1's passes take it
  with automatic exposure, and the picture squared back by the same pass. The two backends are given the same picture.
- **Frame generation is dispatched by hand.** AMD's frame generation is built around replacing the application's
  swapchain with AMD's own, which paces the presents. Hookshot presents and paces its own made frames
  (`vk_framegen.cpp`), so it configures the context with no swapchain and dispatches `PrepareV2` and the generation
  itself, into its own images on its own command buffer -- the manual path AMD's documentation describes and
  discourages. It sits after DLSS frame generation and before FSR 3.1's in the chain, so a refusal costs nothing.

**Not measured.** There is no AMD GPU and no Windows on the machine this was written on. Both paths compile for Windows
and their refusals are logged with the API's result code; the first run on an RX 9000 is the first real test.
`RX_VK_FFX_LOG=1` routes AMD's own messages to the log, `RX_VK_FFX_DEBUG=1` turns on its debug checking (2 also draws
frame generation's debug view), `RX_VK_FFX_FG=0` turns its frame generation off and `RX_VK_FFX_FG_MAX` caps the frames it
is asked for.

## Frame generation, honestly

The existing interpolator shows made frames about one frame behind the game, and skips interpolation
across hitches over 100 ms. Real motion and depth improve what it makes, but they do not change
either property. Neither removes the latency a frame-interpolating renderer has by construction, and
#755 should be read before any of it is wired to the Insert menu.

**Correction: DLSS frame generation is not Windows-only.** This section used to say it was a
Streamline feature and therefore out of reach on Linux. Streamline has no Linux build, but frame
generation is an NGX feature in its own right -- `NVSDK_NGX_Feature_FrameGeneration`, with
`NVSDK_NGX_Parameter_FrameGeneration_Available` to ask about it and a full set of evaluation
parameters in `nvsdk_ngx_params_dlssg.h` -- and the SDK ships `libnvidia-ngx-dlssg.so` for
`Linux_x86_64` beside the super-resolution and ray-reconstruction libraries. It is reached exactly
as super resolution is, through the shared session in `gpu/vk/vk_ngx.h`.

`gpu/vk/vk_dlssg.h` is that backend, and it has run: on this GPU NGX reports Frame Generation
available, the feature is created at 2048x1152 from 1536x864 inputs, and evaluations are accepted.
Getting there cost two refusals, both worth writing down because each names a real requirement:

- **`FAIL_PlatformError` from `CreateFeature`, with the feature reporting itself available.** The
  device had been made without the extensions frame generation needs. The extension query was gated
  on `RX_VK_DLSS`, which a frame-generation run does not set, so a run that asked only for frame
  generation created a device for nothing. Each feature is asked separately now.
- **`FAIL_MissingInput` from `EvaluateFeature`.** NGX's own helper sets five matrices plus the
  camera's position and axes; this set two of them. They are in a struct NVIDIA calls
  `Opt_Eval_Params`, and they are not optional.

So the draw path now keeps the view and the projection as well as their product, and the present
path builds the rest: the projection's inverse, an identity lens clip, the camera's axes out of the
view's rows, its position from -R transpose times the translation, and the clip-to-previous-clip
pair from this frame's view projection and the last one. In doubles, because a float inverse of a
projection with a far plane in the thousands loses the precision the reprojection depends on. It makes the frames the existing interpolator would have made,
from the scene's own depth and the camera motion reprojected from it, and `MultiFrameCount` lets it
make more than one between two real frames -- which is the only way past a rate the game's own
simulation limits. It slots into `framegen_record` ahead of the colour interpolation: where it
refuses, or makes fewer frames than were asked for, the interpolation runs exactly as before and the
set is never shown half made. `RX_VK_DLSSG=<1-3>` asks for it and says how many frames to make;
without it nothing is loaded and nothing is logged.

> **Changed.** Its inputs are FSR 3.1's now (`vk_fsr3_inputs.h`), taken once a present: the reversed depth copy (so
> `DepthInverted` is true of what it is given; before, it was set for the scene's own depth, which is not reversed), the
> motion without the jitter, and the matrices and planes of the projection the scene is drawn with. The paragraph below
> is how it was.

The inputs it needs that colour interpolation does not -- the scene's depth, and the matrix its
draws used -- were kept by the draw path and read back through `scene_frame_inputs()`, because frame
generation is recorded at present time where neither is otherwise in reach. The motion is the same
camera-only reprojection the upscalers use, and asking for it twice in one frame is answered with
the one already written: recording it again would reproject through the matrix the first recording
had just stored as "previous", which is this same frame's, and the second caller would silently get
exactly zero motion.

## DLSS Super Resolution: what is in the tree, and what it is not

Row 4 of the order of work now has a backend: `gpu/vk/vk_dlss.h` and `gpu/vk/vk_dlss.cpp`, the same
shape as the FSR one (`supported()`, `prepare()`, `upscale()`, `output_image()`/`output_view()`,
`shutdown()`) so the seam can hold either.

It is now built against a real NGX SDK (`NVSDK_NGX_VERSION_API_MACRO` 0x15, the SDK whose Linux
feature libraries are `libnvidia-ngx-dlss.so.310.9.16` and its `dlssd`/`dlssg` companions), and
**it has run against the game**: Linux, RTX 5060 Ti, driver 615.71.09, the feature library beside
the binary, `RX_VK_DLSS=1`.

```
vk: dlss wants VK_NVX_binary_import: enabled
vk: dlss wants VK_NVX_image_view_handle: enabled
vk: dlss wants VK_KHR_buffer_device_address: enabled
vk: ngx initialized
vk: dlss: feature ready, 1536x864 -> 2048x1152
vk: dlss 1536x864 -> 2048x1152 with assembled depth and camera motion
```

Those three extensions are what NGX asked for on this driver, enabled at device creation because
they were asked for before it rather than written out from memory. The evaluation is the same
assembled inputs FSR gets: the dispatch's own colour, the scene's depth, and motion reprojected from
that depth.

What is still missing is the jitter. Neither backend is given one, because the game does not appear
to jitter its own projection -- its network is single-image and has no reason to. Without sub-pixel
diversity a temporal upscaler reconstructs from motion alone, which is softer than it should be.
Supplying a jitter means perturbing the projection the game writes and telling the upscaler what was
injected, and that is the next real piece of quality work rather than a flag to set.

Two things the backend needs that the upscaler half does not, both added with the seam wiring:

- **The device has to be created for it.** `dlss::device_extensions()` asks NGX itself
  (`NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements`) what Super Resolution needs on this
  driver, and `vk_main.cpp` enables what the GPU has. A list written out in our own source would be
  wrong on the next driver, and an extension missing at device creation is the one DLSS failure that
  cannot be recovered at evaluation time. Where the GPU lacks something NGX names, it is logged and
  left out: DLSS then reports itself unavailable, rather than `vkCreateDevice` failing and taking
  the renderer down with it.
- **NGX is handed resources, not views.** An `NVSDK_NGX_Resource_VK` is a view *and* its image,
  format and extent, so `DescWriter::Source` now records what each sampled binding is a view of,
  pushed in lockstep with the descriptor infos, and `camera_motion` publishes its target's image and
  format. Without those, a dispatch's colour cannot be given to DLSS at all.

Two switches, both off by default, both neutral while off:

| | |
|---|---|
| `RX_VK_DLSSG=<1-3>` | asks for DLSS frame generation and says how many frames to make between two of the game's. Without it nothing is loaded, nothing is looked at and nothing is logged; the renderer's own interpolation is unaffected either way |
| `RX_VK_DLSS=1` | asks for the backend at run time. Without it nothing is loaded, nothing is looked at and nothing is logged |
| `RX_VK_DLSS_SDK` | a **build-time** gate. Off, the NGX half of the backend is not compiled and `supported()` is false. On, the build needs an NGX SDK's `include/` on the include path |

### Ray Reconstruction, and what "neural rendering" can mean here

The SDK ships three Linux feature libraries and only two of them have a job in this game.

- `libnvidia-ngx-dlss.so` -- **Super Resolution**. Running (above). On this driver that is the
  transformer model, which is the neural network people mean by DLSS 4.
- `libnvidia-ngx-dlssg.so` -- **Frame Generation**. Written and wired; `gpu/vk/vk_dlssg.h`.
- `libnvidia-ngx-dlssd.so` -- **Ray Reconstruction**, which NVIDIA also calls the neural renderer.
  It replaces the denoiser in a ray-traced pipeline, and it is given a G-buffer -- albedo, normals,
  roughness, specular -- plus noisy ray-traced lighting. Ocarina of Time on NX2 is a rasterized game
  with no ray tracing and no such G-buffer. There is nothing here for it to denoise, and handing it
  invented inputs would be a worse picture dressed up as a feature. It is not wired, and the reason
  is the game, not the plumbing.

That paragraph is right about ray reconstruction and was wrong about everything it was then used to
conclude. **"Neural rendering" is also the name of a different NVIDIA feature, and that one does run
here**: DLSS 5 neural rendering (`kFeatureDLSS_NR`, NVIDIA's *3D-Guided Neural Rendering*) is an
uplift pass over a finished image -- `kBufferTypeUpliftInputColor` in, `kBufferTypeUpliftOutputColor`
out -- not a denoiser, and it needs no G-buffer. It is implemented and in the menu's Neural tab;
[neural-rendering.md](neural-rendering.md) has it, including the three times this project refused it
on the strength of the paragraph above.

### How NGX is actually reached on Linux

The first design resolved the `NVSDK_NGX_*` entry points by name from the NGX core the driver ships
(`libnvidia-ngx.so.1`), so that a gated build would not need the SDK's library at link time. That
does not work, and a run said so plainly:

```
vk: ngx loaded from libnvidia-ngx.so.1
vk: ngx: the NGX runtime has no NVSDK_NGX_VULKAN_Init_with_ProjectID
```

The driver's core exports none of the public API. Those nineteen Vulkan entry points live in the
SDK's own static library (`lib/Linux_x86_64/libnvsdk_ngx.a`), which calls into the core. So a build
with `RX_VK_DLSS_SDK=<path>` links that library -- `dlss_sdk_link()` in `nx2/build.py` -- and
`gpu/vk/vk_ngx.cpp` calls NGX directly. The licensing rule is unchanged and unbent: the SDK is the
builder's own copy under the path they name, nothing of NVIDIA's is in this repository, and a build
without that path links exactly what it linked before.

Two things cost a run each and are worth writing down:

- **The project id must be a real UUID.** NGX validates the shape and answers
  `NVSDK_NGX_Result_FAIL_InvalidParameter` (`0xbad00005`) otherwise. Ours spelled `hookshot-dlss`
  into the last two groups, which is not hex.
- **The feature libraries are the player's to supply.** `libnvidia-ngx-dlss.so.*` and its `dlssd`
  (ray reconstruction) and `dlssg` (frame generation) companions sit beside the binary, where NGX
  looks for them because the application data path points there. They are never ours to ship.

### The licensing decision, and why it is a gate rather than a hand-written ABI

The rule from CONTRIBUTING.md is not negotiable and is not negotiated: **no NVIDIA header and no
NVIDIA library is in this repository**, a tree with no SDK builds, and a machine with no DLSS
runtime runs. The question was how to reach NGX under that rule, and there were two answers.

The first — declare the minimal NGX ABI by hand and resolve the entry points with `dlopen` — is
what the FSR API route would look like, and it is the wrong answer here. NGX is not a handful of C
functions. Every call goes through `NVSDK_NGX_Parameter`, which is a **C++ class of pure-virtual
`Set`/`Get` overloads**: its vtable order is the ABI. `NVSDK_NGX_Resource_VK`,
`NVSDK_NGX_FeatureCommonInfo` and the DLSS create parameters are struct layouts, and the ~20
parameter names are strings. None of that can be derived from the exported function names, none of
it is checked by a compiler, and none of it can be tested without an NVIDIA GPU, the driver's NGX
core and the model library. A wrong vtable slot or a shifted field is memory corruption inside
NVIDIA's runtime on a user's machine — a worse failure than "DLSS is unavailable", and one that
would be reported as a driver crash.

So the second answer was taken: **the NGX half is gated at build time and is off by default.**
Behind the gate NVIDIA's own headers declare all of it and the compiler checks every line against
them; in front of the gate there is nothing of NVIDIA's at all. The five functions the gate covers
(`ngx_available`, `ngx_feature`, `ngx_evaluate`, `ngx_drop_feature`, `ngx_shutdown`) each refuse
when it is off, and everything else in the file — the output image, the extents, the layout
transitions, the barriers, and every clause that refuses an evaluation — is vendor-neutral and
compiles in every build.

Two things follow that are worth stating plainly:

- **This is not the FSR arrangement.** FSR 3.1 is compiled from MIT sources in this tree and is a
  backend that can be a default. DLSS is a backend that is off unless someone builds it on purpose
  with a licensed SDK they fetched themselves. That asymmetry is the licensing, not a judgment about
  the upscalers.
- **The gate is also the honest answer to the ABI question.** If a later change declares the ABI in
  this tree instead, the gated build stops needing NVIDIA's headers, and
  `tests/test_dlss_contract.py` fails on exactly that.

Even behind the gate, the entry points are **resolved by name at run time** from the NGX core the
driver ships (`libnvidia-ngx.so.1`, then `libnvidia-ngx.so`; `nvngx.dll` on Windows) through the
`LoadLibraryA`/`GetProcAddress` pattern the renderer already uses for `vulkan-1.dll`, which
`host/posix/win32_posix.c` makes `dlopen`/`dlsym` on Linux. So a gated build has no link dependency
on NVIDIA's library either, and on a machine with the SDK-built binary and no NVIDIA driver every
pointer is null and `supported()` is false instead of the process failing to start. The model
library (`libnvidia-ngx-dlss.so.*`) is NGX's to find, and it is told where to look: the game folder
goes in `NVSDK_NGX_FeatureCommonInfo::PathListInfo` and as the application data path.

One consequence of resolving rather than linking: the SDK's `NGX_VULKAN_CREATE_DLSS_EXT1` and
`NGX_VULKAN_EVALUATE_DLSS_EXT` helpers are **not** used, because they call
`NVSDK_NGX_VULKAN_CreateFeature` / `..._EvaluateFeature` by name and would hand the link back to
NVIDIA's library. What they do besides that is set the parameters below, and those come from the
SDK's own key macros here, so a wrong name is a compile error exactly as it would be through the
helper. What that trades away is the helper's completeness — a key DLSS wants and this code does not
set would be silently ignored — so the set is written out here to be reviewed against the SDK's
helper rather than taken on trust:

| Create | Evaluate |
|---|---|
| `Width`, `Height`, `OutWidth`, `OutHeight` | `Color`, `Depth`, `MotionVectors`, `Output` |
| `PerfQualityValue` | `Jitter.Offset.X`/`.Y`, `MV.Scale.X`/`.Y` |
| `DLSS.Feature.Create.Flags` (`IsHDR`, `MVLowRes`, `DepthInverted`) | `DLSS.Render.Subrect.Dimensions.Width`/`.Height` |
| `DLSS.Enable.Output.Subrects` (0) | `Reset`, `Sharpness` |
| `CreationNodeMask`, `VisibilityNodeMask` | `DLSS.Pre.Exposure`, `DLSS.Exposure.Scale` (both 1.0) |

The three create flags are statements about *this game's* inputs rather than preferences, and they
come from the input contract above: the colour is HDR, the motion is at the render extent, and the
depth is reversed. Each one set wrongly is a wrong picture.

### Windows and Streamline are deliberately not written

Streamline is the normal route to DLSS on Windows and the only route to DLSS frame generation, and
none of it is implemented. That is a decision, not an oversight: it needs a second runtime, a second
set of headers and a Windows machine to try it on. The shape is left for it — a third arm of the
gate implementing the same five functions, which is why they are named for the job rather than for
NGX — and nothing outside those five would change. Streamline has no Linux build, so the NGX arm
stays the Linux one whatever happens on Windows.

### What `dlss::Inputs` adds to `fsr3::Inputs`, and why

Field for field it is FSR's, plus the images and formats of the three inputs. FSR records its own
compute passes, so a `VkImageView` is the whole of an input to it; NGX is handed resources, and an
`NVSDK_NGX_Resource_VK` is a view *and* its image, its subresource range, its format and the extent
it is read at. There is nothing to default those to, so an evaluation refuses without them.

That has a consequence for wiring it up, and it is the main one: **the caller does not have them
all today.** `scene_depth` in `gpu/vk/vk_draw.cpp` carries its `VkImage` and format already, but the
colour arrives as `dw.imgs[i].imageView` with no image beside it, and
`camera_motion::record()` returns only a view. Both need an accessor before DLSS can be given a
dispatch. Those files were out of scope for the change that added this backend.

### Also not done, and needed before this can run

- **Nothing calls it.** `upscale_seam.h` has no `Backend::Dlss`, and `vk_draw.cpp` does not reach
  for `dlss::supported()`. The backend stands alone.
- **The device does not enable what NGX asks for.** NGX reports the instance and device extensions
  DLSS needs (`NVSDK_NGX_VULKAN_RequiredExtensions`), and `vk_main.cpp` enables none of them.
  Until it does, `Init` or `CreateFeature` will fail on a machine that is otherwise capable — which
  is a refusal and not a crash, but it is the first thing to check when DLSS reports unavailable on
  hardware that should have it.
- **Jitter.** As with FSR, the jitter the game's own network used is not extracted, so a run would
  pass zero sub-pixel offset. DLSS without jitter is not DLSS at its quality; a soft image from such
  a run is this, not the network behaving badly.
- **The library names are a guess.** `libnvidia-ngx.so.1` and `libnvidia-ngx.so` were not checked
  against an installed driver, because this machine has neither the runtime nor the SDK. If NGX
  loads nowhere, that list is the first place to look; the loader logs which name it opened.
- **The gated half has never been compiled against the real SDK.** It was syntax-checked against a
  hand-written stand-in outside the repository, which catches typos and arities against what the API
  is *believed* to be and proves nothing about the real one. Expect the first build with a real SDK
  to need corrections, in the `Init_with_ProjectID` argument list above all: whether that entry
  point takes the two `vkGet*ProcAddr` pointers varies by SDK version.

### What is checked without a GPU, a game or an SDK

`tests/test_dlss_contract.py` (with `tests/dlss_contract_test.cpp`) drives the real translation unit
and asserts the no-SDK contract: it compiles and links with no NVIDIA SDK, and with `RX_VK_DLSS=1`
and a device it reports DLSS unavailable *and says why*, without allocating an image, recording a
barrier or handing back an output. The log line is the point — a check that only asserted "false"
would pass against a `supported()` that had been deleted. It also fails if any NGX name in
`vk_dlss.cpp` escapes the build-time gate, if an NVIDIA SDK header or library appears in the tree,
or if a gated build stops needing NVIDIA's own headers. It needs `clang++` and the Vulkan headers,
and takes about a second.
