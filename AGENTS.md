# Notes for AI coding agents

**Start with [docs/workflow.md](docs/workflow.md):** what to work on first (severity x reach x confidence, reach from
the hardware people run), the hardware tiers a change must not break, and the steps from claim to release. The open work, scored and sorted,
is in [docs/worklist.md](docs/worklist.md): pick from its top.

Follow [CONTRIBUTING.md](CONTRIBUTING.md): its rules apply to agents as to everyone, above all that no game files or
anything made from them leave the PC. The [development guide](docs/development.md) has the layout and the commands; the
[design notes](docs/design.md) have the architecture and the rules the GPU replay keeps.

What CONTRIBUTING.md does not spell out, and agents trip on:

- **Tests read the source.** Many checks in `tests/` cut a function out of a source file by its text and compile it
  with stubs (17 read `gpu/vk/vk_menu.cpp` alone). Renaming or reshaping a function can break a check far from your
  change: run `python -I -B tests/run_all.py` and update the check in the same PR rather than weakening it.
- **The renderer's interface is versioned.** A change to `MkwVkHost` or `MkwVkRenderer` in `gpu/vk/vk_api.h` bumps
  `MKWVK_API_VERSION`.
- **Generated files are generated.** The SPIR-V headers in `gpu/vk/shaders/` (`*_spv.h`) are compiled from the shader
  beside them; regenerate them with glslang rather than editing them (the shaders' tests show the command).
- **Two platforms.** Windows builds with llvm-mingw, Linux with clang/gcc. Keep host code portable, and paths UTF-8
  (no `CP_ACP`; `tests/test_utf8_paths.py`).
- **No game in CI.** Checks must run without a dump. A run in the game is for you to do locally and report.

## Team updates

Agents share claims, findings, pull requests and blockers in the Matrix room `#agent-updates:magicbean.dev`, which people
read too; each post is copied to Discord #dev-agents under the poster's Discord name. The steps are in
[.claude/skills/team-updates/SKILL.md](.claude/skills/team-updates/SKILL.md) (Claude Code loads it as `/team-updates`;
other agents follow the same file). The flow:

1. **The person you work for asks you to enable auto updates.**
2. **You send them to sign in on git.magicbean.dev with Discord:** run `python -I scripts/team_chat.py login` and give
   them the link. The first sign-in creates their account. The session stays in
   their config folder, never in the repo.
3. **From then on you post to the room automatically:** a recurring check about once an hour (Claude Code:
   `/loop 60m /team-updates`) that posts only when something happened.

What to post, and what not:

- **Only what someone could act on:** a claim, a finding, a PR, a blocker, a finished task. No "still working", no
  "no updates". The script allows **one post an hour** and no repeats; `--force` only when the person asked for an
  update.
- **Read before claiming** (`team_chat.py read`), so two people don't fix the same thing.
- The rule about game files applies here too: no game data, keys or anything from `work/`, and no secrets or personal
  details.

Pull requests: **open the description with why the PR is needed**, in one or two plain sentences, and **include
screenshots when the change shows on screen** (CONTRIBUTING.md, *Commits and pull requests*).
