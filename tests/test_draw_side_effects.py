"""CPU-only draw readiness decision check: inline fake worker completion, no Vulkan/shader execution.

The real draw decision block is extracted intact. Publication and final resource writes are CPU witnesses,
not a claim about GPU output, timing, or a reproduced game failure.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = r"""
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define PROF_EXACT() ((void)0)
#define PROF_HOT(x) ((void)0)
#define SKIP_HERE() return
using VkPipeline = int;
struct Desc { bool is_written = false; };
struct Info {
    bool stores_global_memory = false;
    std::vector<Desc> storage_buffers_descriptors, image_buffer_descriptors, image_descriptors;
};
struct Stage { Info info; };
int scans, early_reads, translation_waits, pipeline_waits, draws, output, st_waits, st_pending_skips;
double st_wait_ms;
// Instrument only fixture iteration: reject inspecting ordinary stages before acquire-ready publication.
struct Stages {
    std::atomic<bool>* ready;
    std::vector<Stage> values;
    auto begin() const {
        if (!ready->load(std::memory_order_acquire)) ++early_reads;
        ++scans;
        return values.begin();
    }
    auto end() const { return values.end(); }
};
struct Program {
    std::atomic<bool> ready{false};
    bool failed = false, fallback = false, fallback_task = false;
    int layout = 1, fallback_topology = 0;
    Stages stages{&ready, {}};
};
struct PipeEntry { std::atomic<bool> ready{false}; VkPipeline pipeline = 1; };
struct Targets { bool persistent = true; };
struct NvnState {};
struct ParLock {};
struct DrawArgs { bool indirect = true; int groups[3]{1, 1, 1}; int first_task = 0; };
int fallback_task_groups(const Program&) { std::abort(); }
void draw(int, NvnState&, const DrawArgs&) { std::abort(); }
Program* current;
PipeEntry pipe_entry;
int effect;
Stage shader_stage(int kind) {
    Stage s;
    if (kind == 1) s.info.stores_global_memory = true;
    if (kind == 2) s.info.storage_buffers_descriptors.push_back({true});
    if (kind == 3) s.info.image_buffer_descriptors.push_back({true});
    if (kind == 4) s.info.image_descriptors.push_back({true});
    // Nonempty read-only descriptors must still qualify for the color-only control.
    if (kind == 0) s.info.storage_buffers_descriptors.push_back({false});
    return s;
}
void publish() {
    // Put the write in a later stage, not stages[0].
    current->stages.values = {Stage{}, shader_stage(effect)};
    current->ready.store(true, std::memory_order_release);
}
void jobs_wait(const std::atomic<bool>& ready) {
    if (&ready == &current->ready) { ++translation_waits; publish(); }
    else { ++pipeline_waits; pipe_entry.ready.store(true, std::memory_order_release); }
}
bool targets_persistent(const Targets& at) { return at.persistent; }
bool exposure_log_on() { return false; }
void exposure_note(NvnState&, const Program&) {}
void ensure_layout(Program&) {}
// These cases use non-fallback programs, whose topology stays unchanged.
int pipeline_topology(const Program&, int drawn) { return drawn; }
PipeEntry* graphics_pipeline(const Program&, NvnState&, const Targets&, int, const unsigned char*) { return &pipe_entry; }
@WRITES_MEMORY@
void decision(const Program* cp, const Targets& at) {
    NvnState s; DrawArgs d; int cmd = 0, topo = 0; unsigned char types[8]{};
@BLOCK@
    // Fake draw/consumer witness only: not actual GPU shader execution or resource binding.
    ++draws;
    if (effect) output = 42;
}
int failures;
void check(bool ok, const char* label) { if (!ok) { ++failures; std::printf("FAIL %s\n", label); } }
void scenario(int kind, bool translated, bool pipeline_hot, bool persistent) {
    scans = early_reads = translation_waits = pipeline_waits = draws = output = st_pending_skips = 0;
    Program p; current = &p; effect = kind;
    if (translated) publish();
    pipe_entry.ready.store(pipeline_hot, std::memory_order_release);
    const bool force = std::getenv("RX_VK_PIPE_SKIP") && std::atoi(std::getenv("RX_VK_PIPE_SKIP")) == 0;
    const bool legacy = std::getenv("RX_VK_PIPE_SKIP") && std::atoi(std::getenv("RX_VK_PIPE_SKIP")) == 2;
    const bool skip_unknown = legacy && !translated && persistent;
    const bool should_draw = !skip_unknown && (pipeline_hot || force || !persistent || kind != 0);
    decision(&p, {persistent});
    char label[120];
    std::snprintf(label, sizeof label, "kind%d translated%d hot%d persistent%d force%d", kind, translated, pipeline_hot, persistent, force);
    const int before = failures;
    check(draws == int(should_draw), label);
    if (kind && should_draw) check(output == 42, "producer write absent at fake consumer");
    check(!early_reads, "metadata inspected before publication");
    check(translation_waits == int(!translated && !skip_unknown), "translation wait mode");
    if (pipeline_hot || force || !persistent || skip_unknown) check(scans == 0, "unnecessary classification");
    check(pipeline_waits == int(!pipeline_hot && should_draw), "pending pipeline must publish before draw");
    std::printf("%s %s draws%d skips%d scans%d twait%d pwait%d\n", failures == before ? "PASS" : "RED", label,
                draws, st_pending_skips, scans, translation_waits, pipeline_waits);
}
int main() {
    for (int kind = 0; kind <= 4; ++kind)
        for (bool translated : {false, true})
            for (bool hot : {false, true}) scenario(kind, translated, hot, true);
    scenario(1, true, false, false);
    scenario(1, false, false, false);
    scenario(0, false, false, false);
    std::printf("%d failures; fake CPU decision check only\n", failures);
    return failures ? 1 : 0;
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--source", type=Path, default=ROOT)
    args = parser.parse_args()
    src = (args.source / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
    try:
        helper_start = src.index("static bool writes_memory(const Program& p)")
        helper_end = src.index("enum class FallbackRole", helper_start)
        start = src.index("    static const bool always_wait =", src.index("void draw(VkCommandBuffer"))
        end = src.index("    cur_program = &p;", start)
    except ValueError:
        raise SystemExit("draw readiness block changed; update this test's extraction anchors") from None
    fixture = FIXTURE.replace("@WRITES_MEMORY@", src[helper_start:helper_end]).replace("@BLOCK@", src[start:end])
    with tempfile.TemporaryDirectory(prefix="draw-side-effects-") as tmp:
        cpp, exe = Path(tmp) / "test.cpp", Path(tmp) / "test"
        cpp.write_text(fixture, encoding="utf-8")
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Wno-unused-variable",
                        str(cpp), "-o", str(exe)], check=True, timeout=60)
        failed = False
        for mode in (None, "0", "1", "2"):
            env = os.environ.copy()
            env.pop("RX_VK_PIPE_SKIP", None)
            if mode is not None:
                env["RX_VK_PIPE_SKIP"] = mode
            result = subprocess.run([str(exe)], env=env, capture_output=True, text=True, timeout=10)
            lines = result.stdout.splitlines()
            complete = "0 failures; fake CPU decision check only" in lines and sum(l.startswith("PASS kind") for l in lines) == 23
            ok = result.returncode == 0 and complete
            print(f"RX_VK_PIPE_SKIP={mode if mode is not None else 'unset'}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                failed = True
                print("fixture failed or did not finish all 23 cases\n" + result.stdout + result.stderr)
        if failed:
            raise SystemExit(1)
    print("PASS: unknown translations wait by default; mode 2 restores their old skip; translated writers wait; hot draws do not scan stages")


if __name__ == "__main__":
    main()
