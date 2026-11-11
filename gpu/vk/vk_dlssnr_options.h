// (hookshot) DLSS 5 neural rendering's options struct.
//
// Streamline's public headers name the feature (kFeatureDLSS_NR = 1004) and its buffers
// (kBufferTypeUpliftInputColor / UpliftOutputColor / UpliftControlMask) but ship no sl_dlss_nr.h:
// the struct that configures the model is only declared in NVIDIA's NDA SDK. It is declared here
// with the layout openNR recovered from a prerelease Streamline 2.13 sl.dlss_nr.dll
// (https://github.com/clshortfuse/openNR, MIT, Copyright 2026 Carlos Lopez). The field names, order
// and defaults follow DagorEngine (BSD-3-Clause), which is the one integration that compiles against
// NVIDIA's real sl_dlss_nr.h -- so Dagor and openNR are genuinely independent of each other and they
// agree exactly on field order, count and types. Nothing of NVIDIA's is copied into this file.
//
// What is NOT independently corroborated, and what to suspect first if this struct is rejected: the
// GUID below and kStructVersion3 come from openNR alone (Dagor includes the real header and so never
// restates them). slDLSSNRSetOptions answering eErrorInvalidParameter means this GUID, before
// anything else -- vk_dlssnr.cpp says so in the log rather than leaving you to rediscover it.
//
// The static_asserts are the point of the file. This struct is passed to code we did not compile,
// so a field at the wrong offset is not a compile error or even a crash -- it is the model reading
// one setting as another, which looks like a bad-looking picture and sends you hunting in the wrong
// place. If a future Streamline moves a field, these fail at build time and say so.
#pragma once

#if defined(RX_VK_STREAMLINE) && defined(_WIN32)

#include <cstddef>
#include <cstdint>

#include <sl.h>

namespace mkwvk::dlssnr {

enum class DlssNrMode : uint32_t {
    eOff = 0,
    eOn = 1,
};

// {29DFDFE0-273A-4E72-B492-2DC823D5B1AD}
struct DlssNrOptions : sl::BaseStructure {
    static constexpr sl::StructType s_structType = sl::StructType(
        {0x29dfdfe0, 0x273a, 0x4e72, {0xb4, 0x92, 0x2d, 0xc8, 0x23, 0xd5, 0xb1, 0xad}});

    DlssNrOptions() : sl::BaseStructure(s_structType, sl::kStructVersion3) {}

    DlssNrMode mode = DlssNrMode::eOff;
    float intensity = 1.0f;
    float localToneStrength = 1.0f;
    float localStructureStrength = 1.0f;
    /// In the ABI and ignored by Streamline 2.14's plugin; kept because it is part of the
    /// layout and a later SDK is expected to act on it.
    float globalToneStrength = 1.0f;
    uint32_t style = 0;
    /// 0 is the default model; 1 to 7 pin presets A to G. openNR defaults this to 1,
    /// which silently pins preset A; Dagor's real header says 0, which is what this is.
    uint32_t preset = 0;
    /// Declared uint32_t on purpose. sl::Boolean is `enum Boolean : char`, one byte, which
    /// would leave three bytes of uninitialized padding at 0x3d-0x3f inside a struct we hand
    /// to code we did not compile. Dagor assigns a plain C++ bool to this field, which would
    /// not compile against an unscoped char enum, so the real header's type is not
    /// sl::Boolean either. Same offsets, same ABI, and the padding is guaranteed zero.
    uint32_t useAutoMask = 0;
    /// 0 to 1, and only meaningful where the model knows which pixels are a character: with
    /// useAutoMask on, or with a control mask whose skin channel is written.
    ///
    /// Not -1. openNR defaults this to -1 with no comment, and "negative leaves it to the model" was
    /// read into that and then repeated -- including by me, in Elysium. Dagor's real header defaults
    /// it to 1.0f, so does nashglyx/ssetest, and dlss5-anywhere documents the range as 0 to 1 with no
    /// sentinel. No source anywhere states that a negative value means anything.
    float skinStructureStrength = 1.0f;
    /// An sl::DLSSMode: 1 performance, 2 balanced, 3 quality, 6 DLAA. In the ABI and, like
    /// globalToneStrength, ignored by Streamline 2.14's plugin -- openNR's NGX-direct path does act
    /// on it, as PerfQualityValue = this - 1. Sent correctly regardless.
    uint32_t performanceMode = 3;
};

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winvalid-offsetof"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif
// sl::BaseStructure is next (8) + structType (16) + structVersion (8) = 0x20, so every field below
// is offset from there.
static_assert(sizeof(void*) == 8, "DLSS 5 neural rendering is 64-bit Windows only");
static_assert(sizeof(DlssNrOptions) == 0x48);
static_assert(offsetof(DlssNrOptions, mode) == 0x20);
static_assert(offsetof(DlssNrOptions, intensity) == 0x24);
static_assert(offsetof(DlssNrOptions, localToneStrength) == 0x28);
static_assert(offsetof(DlssNrOptions, localStructureStrength) == 0x2c);
static_assert(offsetof(DlssNrOptions, globalToneStrength) == 0x30);
static_assert(offsetof(DlssNrOptions, style) == 0x34);
static_assert(offsetof(DlssNrOptions, preset) == 0x38);
static_assert(offsetof(DlssNrOptions, useAutoMask) == 0x3c);
static_assert(sizeof(DlssNrOptions::useAutoMask) == 4, "no uninitialized padding at 0x3d");
static_assert(offsetof(DlssNrOptions, skinStructureStrength) == 0x40);
static_assert(offsetof(DlssNrOptions, performanceMode) == 0x44);
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

using PFun_slDLSSNRSetOptions = sl::Result(const sl::ViewportHandle& viewport,
                                           const DlssNrOptions& options);

} // namespace mkwvk::dlssnr

#endif
