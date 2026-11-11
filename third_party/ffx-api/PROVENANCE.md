# AMD FSR API headers

The public headers of AMD's FSR API (the "FidelityFX API"), which `gpu/vk/vk_ffx.cpp` compiles against to reach AMD FSR 4
upscaling and frame generation. Unmodified upstream copies.

- Upstream: https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK
- Version: FSR SDK v2.4.0, commit `7539ae8b6568ad29ed78a753404351daf3b5cca7`
- License: MIT, in `LICENSE.txt`. The SDK's `docs/license.md` lists each of these files under its MIT section; keep the
  notice with any distribution.

| here | upstream |
|---|---|
| `api/include/ffx_api.h`, `ffx_api_types.h`, `vk/ffx_api_vk.h` | `Kits/FidelityFX/api/include/` |
| `upscalers/include/ffx_upscale.h` | `Kits/FidelityFX/upscalers/include/` |
| `framegeneration/include/ffx_framegeneration.h`, `ffx_framegeneration_api_types.h`, `vk/ffx_api_framegeneration_vk.h` | `Kits/FidelityFX/framegeneration/include/` |

The folder layout is upstream's, because the headers include each other by relative path.

Headers only. Nothing is linked: AMD's runtime, `amd_fidelityfx_loader_vk.dll` with `amd_fidelityfx_upscaler_vk.dll` and
`amd_fidelityfx_framegeneration_vk.dll` (signed by AMD, also MIT in SDK 2.4), is loaded at run time from beside the game.
That runtime is Windows only, so on Linux these headers are not compiled at all; see `docs/upscaling.md`.

Nothing in this folder is edited. A change belongs in a new folder with its own note, per `CONTRIBUTING.md`.
