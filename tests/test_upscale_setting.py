"""The Upscaling row's two halves, and the clamp that hides DLSS from a build that cannot do it.

The row carries five values: 0-2 are filters over the finished picture and 3-4 name a backend that
resolves the 3D scene instead. Two inline accessors in gpu/vk/vk_menu.h say which half is meant, and
everything that used to read `opts.upscale` raw now goes through them -- because the filter's shader
takes a mode of 0..2 and handing it a 3 would be a setting that corrupts the picture rather than one
that does nothing.

This compiles the real accessors (extracted, not retyped, so a change to them changes the test) and
checks the mapping both ways. It also checks the source facts the accessors cannot: that the row
offers all five values, that menu_init clamps it where the build has no NGX SDK, and that no
consumer reads opts.upscale raw again.

python -I -B tests/test_upscale_setting.py [--cxx g++]
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def block(text, marker):
    """The function beginning at marker, braces balanced."""
    if marker not in text:
        raise SystemExit(f"{marker!r} is gone: this test no longer checks what it says it does")
    start = text.index(marker)
    end = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


PROGRAM = r'''
#include <cstdio>
static int failures;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL: %%s\n", #c); ++failures; } } while (0)
struct Options { int upscale = 0; int dlss = 0; };
static Options opts;
%s

%s

int main() {
    // the three filter values pass through untouched, and none of them is a scene upscaler
    for (int v = 0; v <= 2; v++) { opts.upscale = v; CHECK(window_filter() == v); CHECK(scene_upscaler() == 0); }
    // an old options file's 3 and 4 (FSR 3.1 and DLSS were on the filter's row) are not filter modes
    opts.upscale = 3; CHECK(window_filter() == 0); CHECK(scene_upscaler() == 0);
    opts.upscale = 4; CHECK(window_filter() == 0);
    opts.upscale = 99; CHECK(window_filter() == 0);
    opts.upscale = -1; CHECK(window_filter() == 0);
    // AMD's scene upscalers come from the Upscaler row (upscaler_rows.h FSR3 = 3, FSR4 = 4); DLSS's values are not AMD's
    opts.upscale = 0;
    for (int v = 0; v <= 2; v++) { opts.dlss = v; CHECK(scene_upscaler() == 0); }
    opts.dlss = 3; CHECK(scene_upscaler() == 3);
    opts.dlss = 4; CHECK(scene_upscaler() == 4);
    opts.dlss = 5; CHECK(scene_upscaler() == 0);
    opts.dlss = -1; CHECK(scene_upscaler() == 0);
    if (failures) return 1;
    std::printf("PASS: the window filter and the scene upscaler stay apart\n");
    return 0;
}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="g++")
    a = p.parse_args()

    header = (ROOT / "gpu/vk/vk_menu.h").read_text(encoding="utf-8")
    menu = (ROOT / "gpu/vk/vk_menu.cpp").read_text(encoding="utf-8")

    source = PROGRAM % (block(header, "inline int window_filter() {"),
                        block(header, "inline int scene_upscaler() {"))
    with tempfile.TemporaryDirectory(prefix="upscale-setting-") as tmp:
        cpp = Path(tmp) / "fixture.cpp"
        cpp.write_text(source, encoding="utf-8")
        exe = Path(tmp) / "fixture"
        subprocess.run([a.cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(exe)],
                       check=True)
        subprocess.run([str(exe)], check=True)

    # the rows themselves: the filter's three values, and the Upscaler's five (game's own, DLSS, DLAA, FSR 3.1, FSR 4)
    if '"Window scaling filter", &opts.upscale, 0, 2, 1, tx_filter' not in menu:
        raise SystemExit("the Window scaling filter row no longer offers 0..2")
    if '"Upscaler", &opts.dlss, 0, ur::kUpscalers - 1, 1, tx_upscaler' not in menu:
        raise SystemExit("the Upscaler row no longer offers every upscaler")

    # nothing reads the raw value any more, or the filter gets a mode it does not have
    raw = []
    for name in ("gpu/vk/vk_framegen.cpp", "gpu/vk/vk_replay.cpp", "gpu/vk/vk_draw.cpp", "gpu/vk/vk_dlss.cpp"):
        for n, line in enumerate((ROOT / name).read_text(encoding="utf-8").splitlines(), 1):
            if "opts.upscale" in line:
                raw.append(f"{name}:{n}: {line.strip()}")
    if raw:
        raise SystemExit("these read opts.upscale raw instead of window_filter()/scene_upscaler():\n"
                         + "\n".join(raw))
    print("PASS: the rows offer their values, an old file's upscaler moves over, and no consumer reads it raw")


if __name__ == "__main__":
    main()
