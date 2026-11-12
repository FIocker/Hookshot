"""Linux Builder's Play starts the game in a systemd scope of its own when the launcher runs in a systemd service (#2529):
a desktop that starts each launcher as app-<name>@<id>.service stops the service, and the game with it, when the window
closes. The scope is only asked for from a .service, only when systemd-run is there, and only when a first systemd-run
works; otherwise Play starts the game as before."""
import ast
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "builder/linux"
SOURCE = BUILDER / "hookshot_builder.py"

SERVICE = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/app-hookshot@0123abcd.service\n"
SCOPE = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/app-gnome-hookshot-4242.scope\n"
TOOL = "/usr/bin/systemd-run"
PREFIX = [TOOL, "--user", "--scope", "--collect", "--quiet", "--"]


def functions(*names):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    found = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names}
    missing = set(names) - set(found)
    assert not missing, "hookshot_builder.py no longer has %s: update this test" % ", ".join(sorted(missing))
    namespace = dict(os=os, shutil=shutil, socket=socket, subprocess=subprocess)
    exec(compile(ast.Module(body=[found[n] for n in names], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def check_own_scope(directory):
    own_scope = functions("own_scope")["own_scope"]
    env = {"PATH": "/usr/local/bin:/usr/bin"}

    def cgroup(text):
        path = Path(directory) / "cgroup"
        path.write_text(text, encoding="utf-8")
        return str(path)

    calls = []

    def run_ok(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=0)

    def which(name, path=None):
        assert name == "systemd-run" and path == env["PATH"], (name, path)
        return TOOL

    # In a launcher's service, with a user manager: the scope, after one trial run of true in it, in the game's environment.
    assert own_scope(env, cgroup(SERVICE), which, run_ok) == PREFIX
    assert len(calls) == 1 and calls[0][0] == PREFIX + ["true"], calls
    assert calls[0][1]["env"] is env and calls[0][1]["timeout"] > 0
    assert all(calls[0][1][k] is subprocess.DEVNULL for k in ("stdin", "stdout", "stderr"))

    # A hybrid hierarchy has cgroup v1 lines before the unified one.
    assert own_scope(env, cgroup("12:pids:/user.slice\n1:name=systemd:/user.slice\n" + SERVICE), which, run_ok) == PREFIX

    def never(*args, **kwargs):
        raise AssertionError("own_scope ran systemd-run where nothing has to change")

    # Nothing to do: a scope (a terminal, or a desktop that starts launchers in scopes) ends only when the game does,
    # cgroup v1 alone, no /proc, the root.
    for text in (SCOPE, "1:name=systemd:/user.slice/app-hookshot@1.service\n", "0::/\n", ""):
        assert own_scope(env, cgroup(text), never, never) == [], text
    assert own_scope(env, str(Path(directory) / "missing"), never, never) == []

    # No systemd-run: as before.
    assert own_scope(env, cgroup(SERVICE), lambda name, path=None: None, never) == []

    # A user manager that cannot be reached: as before, so that Play still starts the game.
    def failing(code=None, error=None):
        def run(command, **options):
            if error:
                raise error
            return SimpleNamespace(returncode=code)
        return run

    for run in (failing(code=1), failing(error=OSError("exec")),
                failing(error=subprocess.TimeoutExpired(PREFIX, 10))):
        assert own_scope(env, cgroup(SERVICE), which, run) == []


def check_play(directory):
    folder = Path(directory) / "game"
    folder.mkdir()
    launched = []

    def popen(command, **options):
        assert options["start_new_session"] and options["cwd"] == str(folder)
        launched.append(command)

    namespace = functions("play")
    namespace.update(
        HERE=str(BUILDER), own_scope=lambda env: list(PREFIX),
        linux_update=SimpleNamespace(game_exe=lambda f: str(Path(f) / "hookshot")),
        make_game=SimpleNamespace(game_env=lambda: {"PATH": "/usr/bin"}),
        subprocess=SimpleNamespace(Popen=popen, DEVNULL=subprocess.DEVNULL))
    exe = os.path.abspath(str(folder / "hookshot"))

    # The diagnostics script, in the scope.
    namespace["play"](str(folder))
    command = launched[-1]
    assert command[:len(PREFIX)] == PREFIX and command[len(PREFIX):len(PREFIX) + 2] == ["/bin/sh", "-c"], command
    assert command[-2:] == ["hookshot-play", exe], command

    # The game alone (the script could not be read), in the scope too.
    namespace["HERE"] = str(Path(directory) / "no-builder")
    namespace["play"](str(folder))
    assert launched[-1] == PREFIX + [exe], launched[-1]

    # Nothing to do: the command is the one Play always ran.
    namespace.update(HERE=str(BUILDER), own_scope=lambda env: [])
    namespace["play"](str(folder))
    assert launched[-1][:2] == ["/bin/sh", "-c"], launched[-1]


def main():
    with tempfile.TemporaryDirectory(prefix="hookshot-play-scope-") as directory:
        check_own_scope(directory)
        check_play(directory)
    print("PASS: Play leaves the launcher's systemd service for a scope of its own, and only when it is in one and can")


if __name__ == "__main__":
    main()
