"""A build that failed before the link keeps the objects it compiled; one whose link failed forgets them (issue 1197).

A compiler stopped twice for memory, or the build closed or killed mid-compile, threw away every finished object: the next try on
a small PC compiled for up to an hour again. Objects are cached only once their compiler succeeded, so only a failed link (a
cached runtime object from another version, issue 1197) needs them gone. The lift is forgotten either way.

python -I -B tests/test_failed_compile_keeps_objects.py
"""
import contextlib
import importlib.util
import io
import os
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_game_under_test", ROOT / "nx2" / "make_game.py")
make_game = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_game)


def work_folder(d):
    work = Path(d)
    for sub in ("objcache", "sa/build/obj", "sa/gen"):
        (work / sub).mkdir(parents=True)
    (work / "objcache" / ("0" * 40 + ".o")).write_bytes(b"object")
    (work / "sa" / "build" / "obj" / "code_000.o").write_bytes(b"object")
    (work / "sa" / "gen" / make_game.LIFT_KEY).write_text("key\nprint\n")
    return work


class FailedCompileTests(unittest.TestCase):
    def test_before_the_link_the_objects_stay(self):
        with tempfile.TemporaryDirectory() as d:
            work = work_folder(d)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                make_game.forget_failed_build(str(work), linked=False)
            self.assertFalse((work / "sa" / "gen" / make_game.LIFT_KEY).exists())
            self.assertTrue((work / "objcache" / ("0" * 40 + ".o")).exists())
            self.assertTrue((work / "sa" / "build" / "obj" / "code_000.o").exists())
            self.assertIn("compiles only the files not compiled yet", out.getvalue())

    def test_a_failed_link_forgets_them(self):
        with tempfile.TemporaryDirectory() as d:
            work = work_folder(d)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                make_game.forget_failed_build(str(work), linked=True)
            self.assertFalse((work / "sa" / "gen" / make_game.LIFT_KEY).exists())
            self.assertFalse((work / "objcache").exists())
            self.assertFalse((work / "sa" / "build" / "obj").exists())
            self.assertIn("nothing of this one is reused", out.getvalue())

    def test_the_link_is_told_by_build_pys_line(self):
        """make_game counts the link as reached on build.py's "compiled in Ns, 0 failed" alone: not with a file that failed"""
        src = (ROOT / "nx2" / "make_game.py").read_text(encoding="utf-8")
        build = (ROOT / "nx2" / "build.py").read_text(encoding="utf-8")
        self.assertIn('print("compiled in %.0fs, %d failed" % (time.time() - t0, still), flush=True)', build)
        pattern = re.search(r're\.match\(r"(compiled in [^"]+)", line\)', src).group(1)
        self.assertTrue(re.match(pattern, "compiled in %.0fs, %d failed" % (790.4, 0)))
        self.assertFalse(re.match(pattern, "compiled in %.0fs, %d failed" % (790.4, 1)))
        self.assertFalse(re.match(pattern, "compiled in %.0fs, %d failed" % (790.4, 10)))
        self.assertIn("forget_failed_build(work, bool(linking))", src)
        self.assertIn('run(PY + ["nx2/build.py", "--standalone", "--jobs", str(a.jobs)], env, compile_line)', src)


if __name__ == "__main__":
    unittest.main()
