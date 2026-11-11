"""The open neural rendering backend's khr route (gpu/vk/opennr/khr: VK_KHR_cooperative_matrix alone, the route AMD's
GPUs run), quick form: scripts/opennr_harness.py run --khr on the synthetic model and frame at 384x256, every khr
variant on every device here that can run one, each on a device made with nothing but what that route needs, against
the references of upstream's GLSL route, and block 0 against the CPU reference.

What a PC here can have: an NVIDIA GPU (the references, and the khr variants its driver lists the shapes for) and
llvmpipe (Mesa's CPU driver: KHR matrices, no NV extension, no E4M3 -- the f16 variant at 8 lanes, a couple of
minutes). Without a GPU that runs upstream's routes the references come from a khr run; without any device that runs
either, without Vulkan, a compiler or the Vulkan headers, it says SKIP and passes.

    python -I -B tests/test_opennr_khr.py --cxx clang++ --headers <Vulkan-Headers include folder>
"""
import argparse
import subprocess
import sys
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "{cxx}", "--headers", "{vulkan_headers}"]
RUN_ALL_SKIP = {"windows": "builds and runs OpenDLSS-NR's tool: checked on Linux"}

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cxx", required=True)
    p.add_argument("--headers", required=True)
    a = p.parse_args()
    r = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "scripts" / "opennr_harness.py"), "run", "--cxx", a.cxx,
                        "--headers", a.headers, "--khr", "--skip-routes"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = r.stdout.decode("utf-8", "replace")
    print(out.rstrip())
    skipped = any(line.startswith("SKIP:") for line in out.splitlines())
    ran = "-- khr " in out
    if r.returncode or not (skipped or (ran and "OPENNR HARNESS: PASS" in out)):
        print("FAIL: the open neural rendering khr route did not pass")
        return 1
    if not ran and not skipped:
        print("SKIP: no device here runs a khr variant")
    return 0


if __name__ == "__main__":
    sys.exit(main())
