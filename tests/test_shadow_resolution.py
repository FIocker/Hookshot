"""Shadow resolution (gpu/vk/shadow_res.h): which depth textures are taken for shadow maps, the side their images get, and the
game's rectangles on them. Compiles the header as it is and checks:
- the predicate takes OoT's maps (the characters' 768 and less, the sun's 1600 x 1600 x 3 cascades) and leaves out the scene's
  odd dynamic-resolution sizes, small targets, non-square, multi-level, multisampled, cube and sparse ones;
- the side is the factor's, at most 8192 and 1 GB, and never smaller than the game's;
- for every qualifying size and factor, the game's whole width lands on the image's whole width (rs = side / w, as create()
  sets it), through both edge() and image_size(): no texel row left out and none past the image;
- two rectangles that meet still meet after scaling, at every factor (1.5x included, whose edges are not whole numbers);
- vk_draw.cpp's scissor and clears and vk_xfer.cpp's regions use edge(), and the bias keeps the game's slope term.

python -I -B tests/test_shadow_resolution.py [--cxx g++]
"""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu" / "vk"

FIXTURE = r'''
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include "shadow_res.h"
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL:%d %s\n", __LINE__, #x); std::exit(1); } } while (0)
using namespace shadow_res;
int main() {
    // the predicate: OoT's maps in, everything else out
    for (unsigned w : {256u, 512u, 768u, 1024u, 1600u, 2048u, 4096u}) CHECK(qualifies(true, true, 1, 1, false, w, w));
    for (unsigned w : {128u, 224u, 810u, 839u, 1000u, 4128u, 8192u}) CHECK(!qualifies(true, true, 1, 1, false, w, w));
    CHECK(!qualifies(false, true, 1, 1, false, 1600, 1600));   // (colour)
    CHECK(!qualifies(true, false, 1, 1, false, 1600, 1600));   // (a cube map, a 3D texture)
    CHECK(!qualifies(true, true, 4, 1, false, 1600, 1600));    // (multisampled)
    CHECK(!qualifies(true, true, 1, 2, false, 1600, 1600));    // (with levels)
    CHECK(!qualifies(true, true, 1, 1, true, 1600, 1600));     // (levels left out, or sparse)
    CHECK(!qualifies(true, true, 1, 1, false, 1600, 1280));    // (not square: a scene's depth)
    // the side
    CHECK(side(1600, 3, 2, 1.0f, 8192) == 1600);
    CHECK(side(1600, 3, 2, 1.5f, 8192) == 2400 && side(1600, 3, 2, 2.0f, 8192) == 3200);
    CHECK(side(1600, 3, 2, 3.0f, 8192) == 4800 && side(1600, 3, 2, 4.0f, 8192) == 6400);
    CHECK(side(768, 1, 4, 1.5f, 8192) == 1152 && side(768, 1, 4, 4.0f, 8192) == 3072);
    CHECK(side(4096, 1, 4, 4.0f, 8192) == 8192);                              // (8192 a side at most)
    CHECK(side(4096, 1, 4, 4.0f, 6000) == 6000);                              // (or the device's limit)
    CHECK(side(4096, 8, 4, 4.0f, 8192) <= 8192 && (unsigned long long)side(4096, 8, 4, 4.0f, 8192) * side(4096, 8, 4, 4.0f, 8192) * 8 * 4 <= (1ull << 30));
    CHECK(side(4096, 64, 4, 2.0f, 8192) == 4096);                             // (1 GB at most: no larger at all)
    // every qualifying size, every factor: the game's whole extent is the image's whole extent
    const float factors[] = {1.5f, 2.0f, 3.0f, 4.0f};
    int sizes = 0;
    for (unsigned w = 256; w <= 4096; w += 32) {
        if (!qualifies(true, true, 1, 1, false, w, w)) continue;
        for (float f : factors) {
            for (unsigned layers : {1u, 3u}) {
                const unsigned s = side(w, layers, 2, f, 8192);
                CHECK(s >= w && s <= 8192);
                if (s == w) continue;
                const float rs = (float)s / (float)w;
                CHECK(image_size(w, rs) == s);
                CHECK(edge(0, rs) == 0 && edge(w, rs) == (long long)s);
                // rectangles that meet still meet, and each edge moves forward with the game's
                long long last = -1;
                for (long long v = 0; v <= (long long)w; v += 1) {
                    const long long e = edge(v, rs);
                    CHECK(e > last && e >= 0 && e <= (long long)s);
                    last = e;
                }
                for (long long a = 0; a + 2 <= (long long)w; a += 97) {
                    const long long b = a + 1 + a % 13, c = b + 1 + b % 7;
                    if (c > (long long)w) break;
                    const long long ea = edge(a, rs), eb = edge(b, rs), ec = edge(c, rs);
                    CHECK(ea < eb && eb < ec);   // [a, b) and [b, c) share eb: no gap and no overlap
                }
            }
        }
        sizes++;
    }
    CHECK(sizes > 100);
    // at factor 1 nothing moves
    for (long long v : {-5ll, 0ll, 7ll, 1600ll}) CHECK(edge(v, 1.0f) == v);
    CHECK(image_size(1600, 1.0f) == 1600);
    std::printf("PASS: shadow_res.h takes OoT's shadow maps, sizes them within 8192 and 1 GB, and scales the game's edges onto the whole image (%d sizes)\n", sizes);
}
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cxx", default="g++")
    a = p.parse_args()
    # the callers use the header's rounding, and the bias keeps the game's slope term on a larger map
    draw = (VK / "vk_draw.cpp").read_text(encoding="utf-8")
    scissor = draw[draw.index("VkRect2D scissor_rect("):]
    scissor = scissor[:scissor.index("\n}\n")]
    if scissor.count("shadow_res::edge(") != 4:
        raise SystemExit("vk_draw.cpp scissor_rect: its four edges are not scaled with shadow_res::edge")
    if "clear_rect(" in draw:
        raise SystemExit("vk_draw.cpp: the clears' rectangle is scissor_rect's (clear_rect is gone)")
    if len(re.findall(r"scissor_rect\(s, [^)]*at\.scale\)", draw)) != 3:
        raise SystemExit("vk_draw.cpp: the draw's scissor and both clears use scissor_rect with the targets' scale")
    if not re.search(r"if \(at\.scale != 1\.0f\) d\.bias\[2\] \*= at\.scale;", draw):
        raise SystemExit("vk_draw.cpp: the slope-scaled bias is not scaled with the map")
    xfer = (VK / "vk_xfer.cpp").read_text(encoding="utf-8")
    if xfer.count("shadow_res::edge(") != 8:
        raise SystemExit("vk_xfer.cpp: a copy's and a region clear's edges are not scaled with shadow_res::edge")
    tex = (VK / "vk_texture.cpp").read_text(encoding="utf-8")
    if "shadow_res::qualifies(" not in tex or "shadow_res::side(" not in tex:
        raise SystemExit("vk_texture.cpp create: the shadow maps are not picked and sized by shadow_res.h")
    if "ci.extent = {image_width(t), image_height(t), t->depth};" not in tex:
        raise SystemExit("vk_texture.cpp compact_step: a compacted image is not made at its image's size")
    with tempfile.TemporaryDirectory(prefix="shadow-res-") as tmp:
        cpp, exe = Path(tmp) / "fixture.cpp", Path(tmp) / "fixture.exe"
        cpp.write_text(FIXTURE, encoding="utf-8")
        subprocess.run([a.cxx, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", "-I" + str(VK), str(cpp), "-o", str(exe)],
                       check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
