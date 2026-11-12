"""The texture budget counts an ASTC texture the GPU samples itself by its blocks, not as 16 bytes a texel.

create (gpu/vk/vk_texture.cpp) counted every image as texels x layers x bytes a texel, and for an ASTC image the GPU
samples natively (Intel's GPUs: `vk: ASTC sampled natively`) the bytes a texel were the block's 16: 16 (4x4 blocks) to
144 (12x12) times the image. A UHD 770 counted 17608 MB of textures against an integrated GPU's 2560 MB budget while every
allocation of the process was 4766 MB. texture_level_units is extracted unmodified and run on the cases below: blocks for
native ASTC, and for every other image (unit 1 x 1) the texels exactly as before. No GPU or game is involved.

python -I -B tests/test_native_astc_texture_count.py [--cxx g++]
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]

FIXTURE = r"""
#define _CRT_SECURE_NO_WARNINGS   // (the Windows CRT marks scanf deprecated: an error under -Werror)
#include <algorithm>
#include <cstdint>
#include <cstdio>

/* ACTUAL_HELPER */

int main() {
    unsigned w, h, d, bw, bh;
    while (std::scanf("%u %u %u %u %u", &w, &h, &d, &bw, &bh) == 5)
        std::printf("%llu\n", (unsigned long long)texture_level_units(w, h, d, bw, bh));
    return 0;
}
"""

# (ASTC block sizes: NVN 0x79.. as gpu/vk/vk_formats.cpp has them)
ASTC = [(4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (8, 5), (8, 6), (8, 8), (10, 5), (10, 6), (10, 8), (10, 10), (12, 10), (12, 12)]


def block(text, marker):
    start = text.index(marker)
    end = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def expected(w, h, d, bw, bh):
    w, h, d = max(1, w), max(1, h), max(1, d)
    return -(-w // bw) * -(-h // bh) * d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="g++")
    a = p.parse_args()
    source = (ROOT / "gpu/vk/vk_texture.cpp").read_text(encoding="utf-8")
    helper = block(source, "uint64_t texture_level_units(")
    create = block(source, "Texture* create(const Key& key,")
    assert "const bool astc_blocks = !astc && is_astc(x.format);" in create, "create: native ASTC is not told apart"
    assert "texture_level_units(x.width >> l, x.height >> l, x.depth >> l, astc_blocks ? (uint32_t)bw : 1u, astc_blocks ? (uint32_t)bh : 1u)" in create, \
        "create: the level chain is not counted through texture_level_units"

    cases = []
    # (every other image: unit 1 x 1, the texels as before, levels past 1 x 1 included)
    for w, h, d in [(1365, 1365, 1), (2560, 1440, 1), (32, 32, 32), (0, 7, 0), (1, 1, 1), (6144, 6144, 1)]:
        for level in range(14):
            cases.append((w >> level, h >> level, d >> level, 1, 1))
    # (native ASTC: whole blocks, a part-block counting as one)
    for bw, bh in ASTC:
        for w, h in [(1365, 1365), (13, 13), (6144, 6144), (1536, 3072), (171, 171), (1, 1), (0, 0)]:
            cases.append((w, h, 1, bw, bh))
    with tempfile.TemporaryDirectory(prefix="hookshot-native-astc-count-") as temp:
        src, exe = Path(temp) / "fixture.cpp", Path(temp) / "fixture.exe"
        src.write_text(FIXTURE.replace("/* ACTUAL_HELPER */", helper), encoding="utf-8")
        subprocess.run([a.cxx, "-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror", str(src), "-o", str(exe)], check=True)
        out = subprocess.run([str(exe)], input="".join("%d %d %d %d %d\n" % c for c in cases), capture_output=True, text=True,
                             check=True).stdout.split()
    assert len(out) == len(cases), (len(out), len(cases))
    fails = [(c, int(o), expected(*c)) for c, o in zip(cases, out) if int(o) != expected(*c)]
    for c, got, want in fails:
        print("FAIL %s: %d, expected %d" % (c, got, want))
    assert not fails
    for (w, h, d, bw, bh), o in zip(cases, out):
        if bw == bh == 1:
            assert int(o) == max(1, w) * max(1, h) * max(1, d)   # (unchanged for everything but native ASTC)

    # The array the UHD 770's log named: 1365 x 1365, 90 layers, 11 levels, ASTC 8 x 8. Counted as texels at 16 bytes,
    # 3410 MB; its blocks are 56 MB
    texels = sum(expected(1365 >> l, 1365 >> l, 1, 1, 1) for l in range(11))
    blocks = sum(expected(1365 >> l, 1365 >> l, 1, 8, 8) for l in range(11))
    assert texels * 90 * 16 == 3575649600 and blocks * 90 * 16 == 56365920, (texels, blocks)
    print("native ASTC texture count: %d cases pass (the 1365 x 1365 x 90 ASTC 8x8 array: %d MB counted before, %d MB of blocks)"
          % (len(cases), texels * 90 * 16 >> 20, blocks * 90 * 16 >> 20))


if __name__ == "__main__":
    main()
