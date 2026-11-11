"""The view-projection scan runs once per render target set, however it ends.

capture_view_projection walks five stages and sixteen slots, reading a uniform buffer from each, to
find the matrix the scene was drawn with. It is meant to do that once per render target set. It used
to arm its guard only on the branch that *found* a matrix, so for every target set that holds none
-- shadow, UI, post, which is most of them -- the guard never armed and the whole scan ran again on
every draw of that set: about 30,000 reads of guest memory a frame, each read landing in a captured
page copying that whole 4 KB page. It cost roughly 300 ms a frame and was mistaken for the cost of
FSR (docs/upscaling.md, "Correction: FSR was not what cost the frame rate").

So: the assignment has to come before the loop, not inside it.

python -I -B tests/test_vp_scan_once.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARD = "done_for = targets_serial;"


def block(text, marker):
    if marker not in text:
        raise SystemExit(f"{marker!r} is not in vk_draw.cpp: this test no longer checks anything")
    start = text.index(marker)
    end = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def main():
    source = (ROOT / "gpu" / "vk" / "vk_draw.cpp").read_text(encoding="utf-8")
    fn = block(source, "static void capture_view_projection(")

    # the scan this is about: without these the function is something else and the check is empty
    if "ub_read(" not in fn:
        raise SystemExit("capture_view_projection no longer reads uniform buffers")
    if "for (int stage" not in fn:
        raise SystemExit("capture_view_projection no longer walks the stages")
    if "if (done_for == targets_serial) return;" not in fn:
        raise SystemExit("capture_view_projection no longer skips a target set it has looked at")

    assignments = [i for i in range(len(fn)) if fn.startswith(GUARD, i)]
    if not assignments:
        raise SystemExit("capture_view_projection never records the target set it looked at")
    loop = fn.index("for (int stage")
    late = [i for i in assignments if i > loop]
    if late:
        raise SystemExit(
            "capture_view_projection arms its guard inside the scan, so a target set with no "
            "projection in it is scanned again on every draw"
        )
    print(f"PASS: the view-projection scan arms its guard before the loop ({len(assignments)} assignment)")


if __name__ == "__main__":
    main()
