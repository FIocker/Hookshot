// The Insert menu (vk_menu.cpp): options that apply while the game runs, kept in hookshot_options.ini beside the executable.
#pragma once

#include "vk_internal.h"

namespace mkwvk {

// (out of the box every feature is off: what the game does on its own. The ini beside the executable keeps a player's choices)
struct Options {
    int framegen = 0;     // made frames: 0 off, 1 Auto (to the display's refresh), or 2 to 8 times the game's frames
    int framegen_method = 0;   // (hookshot) which makes them at 2x-8x: gpu/vk/framegen_methods.h (0 Auto, 1 DLSS, 2 FSR 4, 3 FSR 3.1, 4 built-in)
    int pace = 0;         // frames shown on an even line (vk_framegen.cpp)
    int vsync = 0;        // (hookshot) V-Sync, the swapchain's present mode (vk_main.cpp create_swapchain, made again at once): 0 Auto
                          // (mailbox where the driver has it), 1 On (FIFO), 2 Off (immediate). RX_VK_FIFO / RX_VK_IMMEDIATE over it
    int hdr = 0;          // (hookshot) HDR10 out (vk_framegen.cpp framegen_hdr): 0 off, 1 the SDR picture expanded, 2 the game's own (native) HDR; when the display has it (the swapchain made again at once)
    int hdr_white = 200;  // HDR paper white, nits (the picture's white)
    int hdr_peak = 1000;  // HDR peak brightness, nits (its brightest highlights)
    int fps_cap = 0;      // (hookshot) Frame rate cap: 0 60 (the console's), 90, 120, 144, 165, 240, 6 Uncapped (host/nn_vi.c reads frame_rate_cap= at the start)
    int resolution = 0;   // the 3D's (Render resolution): 0 Auto (the game's dynamic resolution), else a share of the output,
                          // 1..6 = 100%, 75%, 67%, 50%, 58%, 33% (upscaler_rows.h; nx2/oot_patches.py)
    int output_res = 0;   // the game's whole frame, docked: 0 its 2048x1152, 1 2560x1440, 2 3840x2160, 3 1920x1080, 4 5120x1440, 5 5120x2880,
                          // 6 the player's own (vk_menu.cpp custom_output); saved as the size (output_res=3840x2160) and read once, at the start
                          // (host/nn_applet.c)
    int upscale = 0;      // to the window (Window scaling filter, saved as upscaling): 0 off (the plain copy), 1 Catmull-Rom, 2 Lanczos
    int reflex = 0;       // NVIDIA Reflex: 0 off, 1 on, 2 on + boost (vk_reflex.cpp; NVIDIA GPUs with VK_NV_low_latency2)
    int antilag = 0;      // AMD Anti-Lag 2: 0 off, 1 on (vk_antilag.cpp; AMD GPUs with VK_AMD_anti_lag)
    int dlss = 0;         // the Upscaler (its name from before FSR: options files and the tests' stand-ins keep it): 0 the game's
                          // own network, 1 NVIDIA DLSS, 2 NVIDIA DLAA (a build with NVIDIA's SDK only), 3 AMD FSR 3.1 (vk_fsr3.cpp),
                          // 4 AMD FSR 4, 5 its native AA (a build with FSR 4 only). upscaler_rows.h; vk_temporal.cpp. Saved as dlss
    int dlss_model = 0;   // DLSS's network, by its preset: 0 K (transformer), 1 M, 2 L (second-generation transformer), 3 NVIDIA's
                          // default (upscaler_rows.h model_preset; vk_dlss.cpp). RX_VK_DLSS_PRESET over it
    int sharpness = 0;    // 0 to 100
    int anisotropy = 4;   // (hookshot) Anisotropic filtering: 0 the game's own (1x or 2x), 1 2x, 2 4x, 3 8x, 4 16x (the
                          // device's limit at most; gpu/vk/vk_draw.cpp aniso_override, RX_VK_ANISO over it)
    int realtime = 1;     // game time follows the clock (host/nn_vi.c); 0: per frame (slower frames, slower game)
    int handheld = 0;     // the game's handheld profile: ~4 fps, softer; read once, at the start (host/nn_applet.c)
    int ultrawide = 0;    // the camera at the monitor's shape when it is wider than 16:9; read once, at the start (host/hos_main.c)
    int wide_menus = 1;   // ultrawide: the pause, item and map menus 0 at full width (stretched), 1 in the middle 16:9 with the sides
                          // blurred, 2 with black bars (nx2/oot_wide.py, vk_framegen.cpp)
    int show_fps = 0;
    int menu_shortcut = 0;   // 0 Insert or F1, 1 F1 only, 2 Insert, F1 or Esc (F1 always opens the menu)
    int menu_size = 2;   // (hookshot) the overlays' size: 1 Small, 2 Medium, 3 Large (gpu/vk/vk_ui.h unit; an older file's 0 Auto: Medium)
    int fullscreen = 0;   // borderless, at the monitor's size: F11 is the same setting
    int gpu_device = 0;   // graphics card index: 0 Auto, 1..N device 0..N-1 (gpu/vk/vk_main.cpp; RX_VK_DEVICE)
    int pad_background = 0;   // the controllers count while another window has the focus (host/nn_hid.c; HOS_INPUT_BACKGROUND)
    int mouse_native = 0; // the mouse turns the camera's own angles (host/mouse_native.c); 0: as the right stick
    int disable_camera_centering = 0; // 0: stock movement-follow turning; 1: keep the player's horizontal direction
    int mouse_sens = 100; // the mouse's sensitivity in percent: 0.1 degrees a count at 100 with native look (host/nn_hid.c)
    int mouse_invert = 0; // the mouse's up and down swapped
    int text_speed = 0;   // how fast dialogue prints: 0 the game's, 1 2x, 2 4x, 3 instant (host/nn_vi.c, nx2/oot_patches.py; HOS_TEXT_SPEED)
    int text_size = 0;    // dialogue and signs' text size: 0 the game's, 1 110%, 2 125%, 3 150%, 4 175% (host/oot_text_size.c,
                          // nx2/oot_patches.py; HOS_TEXT_SIZE)
    int invert_cam_x = 0; // right-stick camera horizontal inversion: 0 normal, 1 inverted
    int invert_cam_y = 0; // right-stick camera vertical inversion: 0 normal, 1 inverted
    int rumble = 100;     // the controllers' rumble in percent of the game's: 0 Off, 25 to 150 (host/pad_sdl.c)
    int stick_dead_zone = 0;  // a stick axis below it is 0: 0 XInput's 7849 (~24%), 1..5 5% to 30% (host/nn_hid.c)
    int pad_layout = 0;   // 0 Nintendo (default), 1 Xbox / PC, 2 Swap Face Buttons, 3 Custom
    int gyro = 1;         // motion controls from a pad's gyroscope (host/pad_sdl.c, host/hos_motion.h); on by default
    int gyro_sens_x = 100;   // the motion aim's horizontal sensitivity in percent
    int gyro_sens_y = 100;   // the motion aim's vertical sensitivity in percent
    int gyro_inv_x = 0;   // the motion aim's horizontal turned around
    int gyro_inv_y = 0;   // the motion aim's vertical turned around
    int gyro_deadzone = 1;   // the motion's deadzone: 0 tight, 1 standard (the console's default), 2 loose
    int prompts = 0;      // the button icons: 0 Auto (the controller), 1 Switch, 2 Xbox, 3 PS4, 4 PS5, 5 Steam Deck (host/prompts.c)
    int robust = 0;       // bounds-checked buffer access: 0 Auto (on), 1 Off, 2 On (gpu/vk/vk_main.cpp; RX_VK_ROBUST)
    int mesh_mode = 0;    // task and mesh programs: 0 Automatic, 1 Native, 2 Compute (gpu/vk/vk_main.cpp: mesh_shaders=;
                          // RX_VK_FORCE_MESH_FALLBACK and RX_VK_TASK_MESH_FALLBACK over it)
    int descriptor_reuse = 1;  // (on by default since options_version 4) a draw keeps its descriptors through other slots' registrations (gpu/vk/vk_draw.cpp; RX_VK_DESC_SLOTGEN)
    int sampler_cache = 0;     // sampler ids answer their parameters without the registrations' lock (gpu/vk/vk_draw.cpp; RX_VK_SAMPLER_CACHE)
    int preset = 0;       // quality preset: 0 Custom, 1 Minimum (4GB), 2 Low (6GB), 3 Balanced (8GB), 4 High (12GB), 5 Ultra (16GB+)
    int vram_mb = 0;      // dynamic texture cache budget limit in MB
    int mirror_mb = 0;    // staging mirror limit in MB
    // (hookshot) DLSS 5 neural rendering (gpu/vk/vk_dlssnr.h). Off out of the box, like every other
    // feature here. The model's own knobs are percentages because that is what a slider can say; the
    // backend divides by 100 for the float Streamline wants.
    int nr = 0;                 // 0 off, 1 on
    int nr_backend = 0;         // 0 Auto (Streamline where it is attached, else NGX direct), 1 Streamline, 2 NGX direct
                                //   (vk_dlssnr_ngx.h), 3 Open (OpenDLSS-NR on Vulkan, vk_opennr.h). Saved as nr_backend
    int nr_stage = 0;           // (super build) 0 after the upscaler (its picture, before the HUD), 1 before it (the render-size
                                //   scene, which the upscaler then takes: cheaper, the model working on fewer pixels; vk_dlss.cpp)
    int nr_model = 2;           // 0 Performance, 1 Balanced, 2 Quality, 3 DLAA
    int nr_style = 0;           // 0 Default, 1 Natural, 2 Cinematic
    int nr_preset = 0;          // 0 the plugin's default model, 1 to 7 pin presets A to G
    int nr_intensity = 100;     // how much of the uplift is mixed in, percent
    int nr_local_tone = 100;    // local tone strength, percent
    int nr_local_structure = 100;   // local structure strength, percent
    int nr_global_tone = 100;   // global tone strength, percent
    int nr_skin = 100;          // skin structure strength, percent (Auto mask decides where)
    int nr_auto_mask = 0;       // the model picks what to leave alone
    int budget = 0;       // the preset whose memory budgets (vram_mb, mirror_mb) hold: kept when another row turns the preset
                          // to Custom, so tuning Resolution after Low (6GB) keeps its cap. Saved as memory_budget; 0: none
    int audio_output = 0; // (hookshot) the speakers the game mixes for: 0 Auto (the default output's), 1 Stereo, 2 5.1; read once, at the start (host/nn_audio.c; HOS_AUDIO_CHANNELS)
    int discord = 1;   // (hookshot) the Discord status (host/discord_rpc.c): 1 on, shown while a Discord client runs on this PC
    int debug_tools = 0;   // (hookshot) the Debug tab's tools (#785: flight, free camera, teleport, inventory): 0 off; HOS_OOT_CHEATS=1 also turns them on
    int volume = 100;     // (hookshot) the game's sound in percent, 0 to 100 (host/nn_audio.c: HOS_OPT_VOLUME, 27)
    int mute_background = 0;   // (hookshot) no sound while another window has the focus (host/nn_audio.c: HOS_OPT_MUTE_BACKGROUND, 28)
    int mic = 1;          // (hookshot) the ocarina's microphone: 0 Off, 1 the system's default, 2.. a device as host->mic_name lists it (host/nn_audio_in.c)
};
extern Options opts;

// (hookshot) The Window scaling filter (opts.upscale) is a filter over the finished picture, 0-2; the
// Upscaler row (opts.dlss, gpu/vk/upscaler_rows.h) is what resolves the 3D scene itself. A value past the
// row (a hand-edited options file) is read as Off, never handed to the filter's shader as a mode it does
// not have.
inline int window_filter() {
    return opts.upscale >= 0 && opts.upscale <= 2 ? opts.upscale : 0;
}
/// AMD's scene upscaler the Upscaler row asks for: 0 none of AMD's (the game's own, DLSS or DLAA), 3
/// FSR 3.1 or 4 FSR 4 -- upscaler_rows.h's FSR3 and FSR4. Whether it runs is the backend's to say.
inline int scene_upscaler() {
    return opts.dlss == 3 || opts.dlss == 4 ? opts.dlss : 0;
}
/// (hookshot) DLSS 5 neural rendering's settings as the backend wants them, read live so that a
/// change in the menu reaches the next frame. The quality field is an sl::DLSSMode, which is what
/// DLSS-NR takes: 1 performance, 2 balanced, 3 quality, 6 DLAA.
struct NrSettings {
    bool on;
    unsigned quality;
    unsigned style;
    unsigned preset;
    float intensity, local_tone, local_structure, global_tone, skin;
    bool auto_mask;
    bool operator==(const NrSettings&) const = default;
};
inline NrSettings nr_settings() {
    const unsigned quality = opts.nr_model == 0 ? 1u : opts.nr_model == 1 ? 2u
                           : opts.nr_model == 3 ? 6u : 3u;
    return NrSettings{
        opts.nr != 0, quality, (unsigned)(opts.nr_style < 0 ? 0 : opts.nr_style),
        (unsigned)(opts.nr_preset < 0 ? 0 : opts.nr_preset),
        (float)opts.nr_intensity / 100.0f, (float)opts.nr_local_tone / 100.0f,
        (float)opts.nr_local_structure / 100.0f, (float)opts.nr_global_tone / 100.0f,
        (float)opts.nr_skin / 100.0f, opts.nr_auto_mask != 0,
    };
}

/// (vk_memory.cpp) true when the quality preset's memory budgets (vram_mb, mirror_mb) are to be used: the card has no more
/// video memory than the preset is for. A 4GB preset on an 8 GB card kept 256 MB of mirrors and 2.2 GB of textures: the
/// game's memory was read over PCIe and the textures made again every frame, 5 fps with the CPU idle (Discord, 11-08)
bool preset_caps_memory();

void menu_init();             // (the options from the file, once)
void menu_first_start_for_gpu(const VkPhysicalDeviceProperties& props, const VkPhysicalDeviceMemoryProperties& mem);   // (no options file: the preset for the card)
/// (hookshot) The V-Sync row as this run applies it: RX_VK_FIFO=1 is On and RX_VK_IMMEDIATE=1 Off, whatever the row says
/// (for the run only: the row keeps the player's choice). 0 Auto, 1 On, 2 Off
int menu_vsync();
void menu_set_gpu_count(int count);
void menu_tick();             // (each frame shown: the keys, while the game's window has the focus)
bool menu_is_open();
bool resolution_auto();       // (the game's dynamic resolution: Resolution Auto)
int resolution_scale();       // (the fixed path's scale in 1/10000: the Resolution row's, as a twelfth of the output)
void menu_option_set(int id, int value);   // (from the game's side: 7 the mouse's sensitivity, the - and = keys)
int menu_heart_protection();
int menu_player_flight();
int menu_free_camera();
int menu_link_speed();   // (Movement speed in percent, 100 while the tools are off)
int menu_take_cheat_action();
void menu_note_frame(bool game_frame, int shown = 1);   // (the frame rates the FPS counter shows)
/// The game's picture's size (the Output resolution, its crop) and the window's, at each present: the Window scaling filter
/// row says when it is not needed
void menu_note_present_sizes(uint32_t picture_w, uint32_t picture_h, uint32_t window_w, uint32_t window_h);
/// Over the window's image `index` (TRANSFER_DST_OPTIMAL before and after), w x h: the start-up screen, the menu and the FPS
/// counter, drawn with Dear ImGui (vk_imgui.h, vk_ui.h). Nothing recorded while none of them shows
void menu_record(VkCommandBuffer cmd, uint32_t index, uint32_t w, uint32_t h);
/// (headless shots, RX_VK_SHOT_MENU) The menu and the counter over another image: TRANSFER_DST_OPTIMAL before and after,
/// R8G8B8A8_UNORM with colour-attachment use, view a view of it
void menu_record_image(VkCommandBuffer cmd, VkImage image, VkImageView view, uint32_t w, uint32_t h);
/// (hookshot) The start-up screen while the shaders are prepared (vk_main.cpp, vk_progress.h), drawn by menu_record under the
/// counter and the menu. Null takes it away.
struct StartupView;
void startup_screen(const StartupView* view);
/// The start-up screen's fade to black as it ends: 0 none, 1 black (vk_main.cpp prepare_with_screen)
void startup_fade_to(float f);

} // namespace mkwvk
