"""Open neural rendering's correctness harness, the quick form: OpenDLSS-NR on this GPU on a synthetic model and frame,
every route against the GLSL reference route bit for bit and block 0 against the CPU reference
(scripts/opennr_harness.py says how, and why synthetic: there are no real weights to test with, anywhere).

Without a GPU that has the network's extensions (VK_KHR_cooperative_matrix, VK_NV_cooperative_matrix2,
VK_EXT_shader_float8), without Vulkan, or without a compiler or the Vulkan headers, it says SKIP and passes: CI has no
such GPU. On a PC that has one it takes about twenty seconds.

    python -I -B tests/test_opennr_harness.py --cxx clang++ --headers <Vulkan-Headers include folder>
"""
import argparse
import subprocess
import sys
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}", "--headers", "{vulkan_headers}"]
# (it builds upstream's tool with the system's C++ library and runs it on the GPU, which says SKIP without one)
RUN_ALL_SKIP = {"windows": "builds and runs OpenDLSS-NR's tool: checked on Linux"}

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cxx", required=True)
    p.add_argument("--headers", required=True)
    a = p.parse_args()
    r = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "scripts" / "opennr_harness.py"), "run", "--cxx", a.cxx,
                        "--headers", a.headers], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = r.stdout.decode("utf-8", "replace")
    print(out.rstrip())
    skipped = any(line.startswith("SKIP:") for line in out.splitlines())
    if r.returncode or not (skipped or "OPENNR HARNESS: PASS" in out):
        print("FAIL: the open neural rendering harness did not pass")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
