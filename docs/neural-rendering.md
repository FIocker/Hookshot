# DLSS 5 neural rendering

NVIDIA's DLSS 5 neural rendering model goes over the picture and rebuilds detail, tone and structure in it. Hookshot runs it
through NGX, the same runtime its DLSS uses, as a feature on DLSS's one NGX session (`gpu/vk/vk_dlssnr_ngx.cpp`). It is
off by default: the Insert menu's **Neural** tab turns it on.

## What you need

- An NVIDIA RTX card, and DLSS working (the Upscaler row offers DLSS).
- NVIDIA's model runtime, `nvngx_dlssnr.dll`, in the game folder's `ngx` folder, where `nvngx_dlss.dll` goes (or any of the
  folders NGX is told about: `RX_VK_DLSS_DIR`, then `%LOCALAPPDATA%\Hookshot\ngx`). Hookshot does not ship it or download
  it. The log says where it looked.
- NVIDIA's own `nvngx_dlssnr.dll` runs on RTX 50 cards. On older cards NGX refuses it; the log and the row say so.
- Linux: NVIDIA ships no `libnvidia-ngx-dlssnr.so` yet. NGX knows the feature, and the day the library ships, putting it in
  the game folder's `ngx` folder should be all this needs. Until then the Windows build under Proton has it.

## Where it runs (the Stage row)

- **After upscaler** (the default): on the temporal upscaler's picture (DLSS, DLAA, FSR 4 or FSR 3.1), before the game draws
  its HUD. The model never sees the hearts, the map or the text; over the finished frame it left a dark smudge behind them.
- **Before upscaler**: on the scene at its render size, which the upscaler then takes up to the output. The model's cost goes
  with its pixels, so this is much cheaper (on an RTX card at 4168x1976, DLSS Quality: 47 fps against 37), a little softer.
  Here the model gets the scene's own depth, motion and jitter, and the chain's resets at cuts, so a model that keeps
  history keeps the jitter the upscaler needs.
- Without a temporal upscaler, it runs at the present, over the finished frame (a flat depth, no motion), before anything of
  Hookshot's own (the menu, the FPS counter) is drawn on top. `RX_VK_DLSS_NR_STAGE=before`, `after` or `present` sets the
  stage for a run.

It is left out with HDR set to **Native**: the game's own HDR picture has light past 1, and the model's eight-bit picture
would cut every highlight at the paper white. Expanded HDR and SDR work.

## The rows

Neural rendering (on/off), Stage, Model (Performance, Balanced, Quality, DLAA), Style (Default, Natural, Cinematic),
Preset (Default, or DLSS presets A to G), Intensity, Local tone, Local structure, Global tone, Skin structure and Auto mask.
Every row reaches the next frame (`tests/test_nr_settings_live.py`). They are saved as `neural_rendering` and `nr_<name>`
(`nr_stage`, `nr_model`, ...) in `hookshot_options.ini`. Turning the row off lets go of the feature and its images.

## How it works

- **Parameters**: NGX knows the feature as `NVSDK_NGX_Feature_Reserved18`, and the public SDK names nothing else about it.
  The parameter names are the ones openNR (https://github.com/clshortfuse/openNR, MIT) drives it with; they are facts about
  the feature, written out in `vk_dlssnr_ngx.cpp`, and none of openNR's code is copied. Nothing of NVIDIA's is in the
  repository, and Hookshot reads, patches or changes none of NVIDIA's files or settings.
- **One NGX session**: `vk_dlss.cpp`'s `ngx_init` (`dlss_ngx_ready`), with DLSS's runtime folders and project. Its device
  extensions are asked of NGX with DLSS Frame Gen's code (`dlss_nr_device_extensions`), only on a device DLSS may run on.
- **Threads**: `nr::apply` has two callers, the temporal chain (the replay thread, the chain's lock held) and the present (the
  display thread); one at a time, under its own lock. Its NGX calls (create, evaluate, release) are made under DLSS Frame
  Gen's lock (`dlss_ngx_feature_mutex`), so the two features' calls never overlap. Its vkCmd* go through `vk_work.h`, with a
  `B_DIRECT` boundary before NGX records.
- **Known limits**: the feature and its images are made for one size, so a size change (Resolution Auto with Stage set to
  Before upscaler) makes them again with a device idle, a short hitch; six refusals turn it off for the run. The picture is
  handed over as RGBA8.

`RX_VK_DLSS_NR=0` or `1` overrides the row for a run; `RX_VK_DLSS=0` (DLSS off) leaves it off too.
