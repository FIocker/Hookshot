// gpu/vk/fsr3_math.h: AMD FSR 3.1's camera from the scene's projection (tests/test_fsr3_math.py)
#include "../gpu/vk/fsr3_math.h"
#include "../gpu/vk/dlss_math.h"

#include <cmath>
#include <cstdio>

#define CHECK(name, condition) do { if (!(condition)) { std::fprintf(stderr, "FAIL %s at line %d\n", name, __LINE__); return 1; } } while (0)

namespace fm = mkwvk::fsr3_math;

static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b)); }

// a perspective projection, row major (clip = p * view position, view z negative in front), clip depth -1..1 (NVN's)
static void perspective(float p[16], double fovy, double aspect, double n, double f, bool reversed) {
    for (int i = 0; i < 16; i++) p[i] = 0.0f;
    const double c = 1.0 / std::tan(fovy / 2);
    p[0] = (float)(c / aspect), p[5] = (float)c, p[14] = -1.0f;
    // ndc z = -p22 + p23 / v (v = -z > 0): -1 at the near plane and 1 at the far one, or the other way round (reversed)
    const double zn = reversed ? 1.0 : -1.0, zf = -zn;
    // -p22 + p23 / n = zn, -p22 + p23 / f = zf (f infinite: -p22 = zf)
    double p23, p22;
    if (std::isinf(f)) p22 = -zf, p23 = (zn + p22) * n;
    else p23 = (zn - zf) / (1.0 / n - 1.0 / f), p22 = p23 / n - zn;
    p[10] = (float)p22, p[11] = (float)p23;
}

// the depth the renderer stores for a view distance: (ndc z + 1) / 2, as fsr3_prep.comp and dlss_prep.comp read it
static double stored(const float p[16], double v) { return ((-(double)p[10] + (double)p[11] / v) + 1.0) * 0.5; }

int main() {
    const double fov = 2.0 * std::atan(0.41421356), aspect = 16.0 / 9.0;
    for (int reversed = 0; reversed < 2; reversed++)
        for (double far_plane : {20000.0, (double)INFINITY}) {
            float p[16];
            perspective(p, fov, aspect, 1.0, far_plane, reversed != 0);
            fm::Camera c;
            CHECK("a perspective projection is a camera", fm::camera_from_projection(p, c));
            CHECK("the depth FSR is given is reversed: flipped where the game's is not", c.flip_depth == !reversed);
            CHECK("the near plane", near(c.near_plane, 1.0, 1e-3));
            CHECK("the far plane", std::isinf(far_plane) ? c.far_plane > 1e30f : near(c.far_plane, far_plane, 1e-2));
            CHECK("the field of view", near(c.fov_y, fov, 1e-5) && near(c.tan_half_fov_x, 0.41421356 * aspect, 1e-5));
            CHECK("the SDK's x and y factors", near(c.device_to_view_depth[2], 0.41421356 * aspect, 1e-5) &&
                                                   near(c.device_to_view_depth[3], 0.41421356, 1e-5));
            // FSR's own view distance (GetViewSpaceDepth) for the depth it is given gives the distance back
            for (double v : {1.0, 3.0, 50.0, 1000.0, 15000.0}) {
                if (v >= far_plane) continue;
                double d = stored(p, v);
                if (c.flip_depth) d = 1.0 - d;
                CHECK("FSR's view distance is the camera's", near(fm::view_distance(c, (float)d), v, 2e-2));
            }
            CHECK("the far end stays finite and positive (GetMaxDistanceInMeters)",
                  fm::view_distance(c, 0.0f) > 0.0f && std::isfinite(fm::view_distance(c, 0.0f)));
            CHECK("1 is the near plane", near(fm::view_distance(c, 1.0f), 1.0, 1e-3));
        }
    {   // the jitter on the projection (nx2/oot_jitter.py) moves neither the planes nor the field of view, once taken off
        float p[16], q[16];
        perspective(p, fov, aspect, 1.0, INFINITY, false);
        for (int i = 0; i < 16; i++) q[i] = p[i];
        q[2] += 0.0004f, q[6] -= 0.0007f;   // (P[0][2], P[1][2] as the host adds them)
        const float clip[2] = {-q[2], -q[6]};   // (the jitter as vk_fsr3_inputs.cpp reads it back: -P[0][2], -P[1][2])
        mkwvk::dlss_math::unjitter(q, clip);
        fm::Camera a, b;
        CHECK("jittered projection", fm::camera_from_projection(p, a) && fm::camera_from_projection(q, b));
        CHECK("same camera without its jitter", near(a.near_plane, b.near_plane, 1e-6) && near(a.fov_y, b.fov_y, 1e-6) &&
                                                    std::fabs(q[2]) < 1e-9f && std::fabs(q[6]) < 1e-9f);
        float px[2];
        mkwvk::dlss_math::jitter_to_pixels(clip, 1536, 864, px);
        CHECK("its jitter in FSR's render pixels: x right, y down", near(px[0], -0.0004 * 768, 1e-4) && near(px[1], -0.0007 * 432, 1e-4));
    }
    {   // what is not a scene's projection
        float p[16];
        perspective(p, fov, aspect, 1.0, 20000.0, false);
        fm::Camera c;
        float o[16];
        for (int i = 0; i < 16; i++) o[i] = p[i];
        o[14] = 0.0f, o[15] = 1.0f;   // (orthographic: a shadow map's, the HUD's)
        CHECK("not orthographic", !fm::camera_from_projection(o, c));
        for (int i = 0; i < 16; i++) o[i] = p[i];
        o[11] = 0.0f;   // (no depth range)
        CHECK("not without a depth range", !fm::camera_from_projection(o, c));
        for (int i = 0; i < 16; i++) o[i] = p[i];
        o[5] = NAN;
        CHECK("not a NaN field of view", !fm::camera_from_projection(o, c));
    }
    std::printf("fsr3 math: all checks passed\n");
    return 0;
}
