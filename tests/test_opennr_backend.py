"""Open neural rendering's wiring, in the source: what no frame-level check can see.

* the open backend reads the Neural tab live: its asked() and apply() go through nr_settings() every call and hold
  no static copy of it (the upscaling rows once shipped dead that way; tests/test_nr_settings_live.py is the same
  check for the Streamline backend);
* dlssnr::apply hands the frame to it when it is selected, in BOTH of its builds -- the Streamline one and the stubs
  every other build (Linux's) compiles. A dispatch in only one would leave the backend unreachable on the other
  platform and nothing would say so;
* create_device asks it for its extensions before vkCreateDevice and chains its features before the features query,
  and it is told when the device exists;
* opennr::selected() takes the Backend row's fourth value, Open, and RX_VK_NR_BACKEND (no MERGE marker left).

    python -I -B tests/test_opennr_backend.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu" / "vk"


def fail(text):
    print("FAIL: " + text)
    sys.exit(1)


def blocks(text, head):
    """Every function body starting at a line that starts with `head`, by brace matching: [(start, body)]"""
    out = []
    for m in re.finditer(r"^" + re.escape(head), text, re.M):
        open_at = text.index("{", m.start())
        depth, i = 0, open_at
        while True:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                break
            i += 1
        out.append((m.start(), text[open_at:i + 1]))
    return out


def main():
    opennr = (VK / "vk_opennr.cpp").read_text(encoding="utf-8")
    asked = blocks(opennr, "bool asked()")
    apply = blocks(opennr, "VkImage apply(VkCommandBuffer cmd")
    if len(asked) != 1 or len(apply) != 1:
        fail("vk_opennr.cpp: expected one asked() and one apply(), found %d and %d" % (len(asked), len(apply)))
    if "nr_settings()" not in asked[0][1] or re.search(r"\bstatic\b", asked[0][1]):
        fail("opennr::asked() must read nr_settings() every call, with no static")
    if "nr_settings()" not in apply[0][1]:
        fail("opennr::apply() does not read nr_settings(): the Neural tab's rows would never reach the network")
    if re.search(r"static[^;]*NrSettings", apply[0][1]):
        fail("opennr::apply() caches the settings in a static")

    # (the frame reaches the open backend through the Backend row's one dispatch point, nr::apply in vk_dlssnr_ngx.cpp:
    #  its fourth value, Open, and Auto's fall to it; vk_dlssnr.cpp's apply() is Streamline's alone)
    ngx = (VK / "vk_dlssnr_ngx.cpp").read_text(encoding="utf-8")
    nr_apply = blocks(ngx, "VkImage apply(VkCommandBuffer cmd, VkImage src, uint32_t sw, uint32_t sh, VkImageView* out_view, "
                           "const NrFrame* frame) {\n    // (two callers")
    if len(nr_apply) != 1:
        fail("vk_dlssnr_ngx.cpp: nr::apply not found")
    if "if (b == OPEN) return opennr::apply(cmd, src, sw, sh, out_view, frame);" not in nr_apply[0][1]:
        fail("nr::apply does not hand the frame to the open backend when it is the one running")
    if "opennr::ready()" not in ngx:
        fail("the Backend row's Auto never falls to the open backend (nr::running_backend)")
    main_cpp = (VK / "vk_main.cpp").read_text(encoding="utf-8")
    create = blocks(main_cpp, "bool create_device()")
    if len(create) != 1:
        fail("vk_main.cpp: create_device() not found")
    body = create[0][1]
    at = {name: body.find(name) for name in ("opennr::device_extensions(want, exts)", "opennr::device_features(tail)",
                                             "opennr::device_created()", "vkGetPhysicalDeviceFeatures2(dev.physical, &f)",
                                             "vkCreateDevice(dev.physical", "optional(VK_KHR_COOPERATIVE_MATRIX")}
    missing = [n for n, i in at.items() if i < 0]
    if missing:
        fail("create_device() lacks %s" % missing)
    if not (at["optional(VK_KHR_COOPERATIVE_MATRIX"] < at["opennr::device_extensions(want, exts)"] <
            at["vkCreateDevice(dev.physical"]):
        fail("opennr::device_extensions must come after the cooperative-matrix extension and before vkCreateDevice")
    if not at["opennr::device_features(tail)"] < at["vkGetPhysicalDeviceFeatures2(dev.physical, &f)"]:
        fail("opennr::device_features must chain its structs before the features query fills them")
    if not at["vkCreateDevice(dev.physical"] < at["opennr::device_created()"]:
        fail("opennr::device_created must come after the device is made")

    # (merged: the Backend row's fourth value, Open, is what makes the device for the network, read in selected())
    selected = blocks(opennr, "bool selected()")
    if len(selected) != 1 or "opts.nr_backend == 3" not in selected[0][1] or "RX_VK_NR_BACKEND" not in selected[0][1]:
        fail("opennr::selected() no longer takes the Backend row's Open (3) and RX_VK_NR_BACKEND")
    if "MERGE:" in opennr:
        fail("vk_opennr.cpp still has a MERGE marker: the Backend row is merged")
    print("open neural rendering: settings read live, dispatched from nr::apply (the Backend row), device hooks in "
          "order, selected by the Backend row's Open")
    return 0


if __name__ == "__main__":
    sys.exit(main())
