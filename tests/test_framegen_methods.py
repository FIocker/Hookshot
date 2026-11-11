"""The Insert menu's Frame generation method row (gpu/vk/framegen_methods.h): its stepping, what a choice means, and that
the menu and vk_framegen.cpp use those rules rather than their own. No game or GPU required.

python -I -B tests/test_framegen_methods.py [--cxx g++]
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="g++")
    a = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="framegen-methods-") as tmp:
        exe = Path(tmp) / "t"
        subprocess.run([a.cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(ROOT / "tests" / "test_framegen_methods.cpp"),
                        "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
    menu = (ROOT / "gpu" / "vk" / "vk_menu.cpp").read_text(encoding="utf-8")
    fg = (ROOT / "gpu" / "vk" / "vk_framegen.cpp").read_text(encoding="utf-8")
    for needle, where, what in (
            ('"Frame generation method", &opts.framegen_method', menu, "the row"),
            ("framegen_methods::step(", menu, "set_row stepping through framegen_methods.h"),
            ("fgm::effective(", menu, "the row showing the method in effect"),
            ("fgm::effective(chosen, available)", fg, "vk_framegen.cpp choosing through framegen_methods.h"),
            ("fgm::tries(method, which)", fg, "each vendor method behind the choice"),
            ("fgm::makes_set(which, m)", fg, "each vendor method only for the sets it can make (FSR 3.1: 2x)"),
            ("tried(fgm::DLSS)", fg, "DLSS frame generation in the chain"),
            ("tried(fgm::FSR4)", fg, "FSR 4 frame generation in the chain"),
            ("tried(fgm::FSR3)", fg, "FSR 3.1 frame generation in the chain"),
            ("fgm::first(", fg, "the scene's inputs made when a vendor method may take them"),
            ("fgm::first(", menu, "the Frame generation row's hint naming the method that makes the frames")):
        if needle not in where:
            raise SystemExit("missing: %s (%s)" % (what, needle))
    # (the choice is read each frame, so the menu's change reaches the next frame: not cached in a static)
    if "static const int chosen" in fg or "static int chosen" in fg:
        raise SystemExit("vk_framegen.cpp caches the method: a change in the menu would wait for a restart")
    print("PASS: the menu and frame generation take the method from framegen_methods.h, read live")


if __name__ == "__main__":
    main()
