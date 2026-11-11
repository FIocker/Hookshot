/* (hookshot) The Insert menu's Frame generation method row (opts.framegen_method, vk_menu.cpp) and the order
   gpu/vk/vk_framegen.cpp tries the methods in, as plain functions: no Vulkan, no menu. tests/test_framegen_methods.cpp
   checks them.

   Auto takes the first method that runs here, best first: NVIDIA DLSS frame generation (the game's depth and motion,
   NVIDIA's network), AMD FSR 4's (AMD's runtime, Windows), AMD FSR 3.1's (AMD's open shaders, any GPU), then the
   renderer's own blend, which always runs. A method chosen outright is tried alone; when it refuses a frame (no history
   yet, its runtime missing, a cut) that frame falls back to the built-in blend rather than showing nothing.

   FSR 3.1's interpolation has no phase input (ffx_frameinterpolation.h fixes t at 0.5): it makes the midpoint between two
   of the game's frames and nothing else, so it makes 2x sets only. At 3x and above a set of its frames would be the same
   midpoint picture at every made frame's time, which judders (the fork once did that), so those sets go on down the
   chain (makes_set). */
#pragma once

namespace mkwvk::framegen_methods {

enum { AUTO = 0, DLSS = 1, FSR4 = 2, FSR3 = 3, BUILTIN = 4, kMethods = 5 };

inline unsigned bit(int method) { return 1u << method; }
/// What can always be chosen: Auto and the built-in blend
inline unsigned always() { return bit(AUTO) | bit(BUILTIN); }

/// The row's next value from v (dir +1 / -1), wrapping over what runs here (available: bit() of each)
inline int step(int v, int dir, unsigned available) {
    available |= always();
    int at = v < 0 || v >= kMethods || !(available & bit(v)) ? AUTO : v;
    for (int i = 0; i < kMethods; i++) {
        at = (at + (dir > 0 ? 1 : kMethods - 1)) % kMethods;
        if (available & bit(at)) return at;
    }
    return AUTO;
}

/// The method shown and used: the one chosen when it runs here, else Auto (a saved choice is kept for a PC where it runs)
inline int effective(int chosen, unsigned available) {
    available |= always();
    return chosen >= 0 && chosen < kMethods && (available & bit(chosen)) ? chosen : AUTO;
}

/// Whether vk_framegen.cpp tries this vendor method for a frame, given the method in effect
inline bool tries(int effective_method, int method) { return effective_method == AUTO || effective_method == method; }

/// Whether a method can make a set of this multiplier (2 to 8): FSR 3.1 the midpoint only, so 2x only
inline bool makes_set(int method, int multiplier) { return method != FSR3 || multiplier == 2; }

/// The method vk_framegen.cpp tries first for a set of this multiplier, of the ones that run here (available): the built-in
/// blend when no vendor method would be tried
inline int first(int effective_method, unsigned available, int multiplier) {
    const int order[] = {DLSS, FSR4, FSR3};
    for (int method : order)
        if (tries(effective_method, method) && (available & bit(method)) && makes_set(method, multiplier)) return method;
    return BUILTIN;
}

} // namespace mkwvk::framegen_methods
