# The pipeline state list's sources

**These are not Nintendo files, and they hold no game code or game data.** Each is a list of Vulkan pipelines that
Hookshot's renderer made while someone played on their own PC, written by Hookshot itself. Keep it that way: only lists
written by Hookshot and this folder's tool go in here. The lists named `contrib/pmwork-personal-fritchii-recipe*` are converted
instead: made from the pipeline records another emulator ships for the game (hashes and draw state, no code), matched to
Hookshot's shader hashes; their notes say how. They hold the same records as any other list and nothing more.

- **What a record is:** one pipeline the game asked for. Each record holds:
  - the hashes of its shaders and of the program variant
  - the target formats
  - the vertex formats
  - the draw state it was built with

  There are no shaders in it, only their hashes. A player's build turns those hashes back into shaders from the
  player's own `hookshot_shaders.bin`. `nx2/pipeline_states.py` documents the record layout.
- **What it is for:** `nx2/make_game.py` merges every list here into the game folder's
  `hookshot_pipeline_states_shipped.bin`. The first start's shader preparation then builds these pipelines before the
  title screen, so the first field load doesn't wash out white while they compile.

## The layout

| File | Written by | Changed how |
|---|---|---|
| `tour.bin.xz`, `tour.md` | the pipeline tour (`scripts/pipeline_tour.py`) | replaced whole by each tour |
| `contrib/<name>.bin.xz`, `contrib/<name>.md` | one contributor each | added; only its owner adds to it |

Each list has a note beside it (`<name>.md`) that says where it comes from: who, GPU and driver, build, settings and
route, and how many pipelines it holds. `nx2/pipeline_states.py check` stops on a list without one.

**The merge** (`nx2/pipeline_states.py merge`, and `nx2/make_game.py` at every build) takes the tour's list first, then
the contributed ones in the order of their names, and keeps each pipeline once. Its output depends only on these files:
the same files always give the same bytes. The lists are written xz-compressed with fixed settings, and the merged list
beside the game is not compressed.

One file per contribution means two pull requests that each add a list never touch the same file. A contributed list
only holds the pipelines that no other contributed list holds when it is added. It does not leave out the tour's,
because the tour's list is replaced whole and a later tour may not find them. When a later tour or another list finds
a pipeline again, the merge keeps it once.

## Contributing a list

Players without a clone send their list to the [Submit your pipelines](https://github.com/NotRexed/Hookshot/issues/2034)
issue: **Save my pipeline list** (the game's menu, Debug tab, or the builders' launcher page; readme, *Send your
pipeline list*) zips the game folder's `hookshot_pipeline_states.bin` with a filled-in report
(`hookshot_pipeline_report.txt`: the version, OS, GPU and driver, settings and mods), and nothing else. A maintainer then
adds it as below, with the report's lines as `--gpu`, `--build`, `--settings` and `--route`. The zip is made by
`gpu/vk/pipeline_share.h` in the game and `nx2/pipeline_share.py` in the builders, after the same checks as
`nx2/pipeline_states.py --check`.

With a clone:

1. Build the game from current `main`. A list from an older build may hold pipelines the current build no longer asks
   for.
2. Move the game folder's `hookshot_pipeline_states.bin` aside if there is one. Play: every pipeline the game makes in
   play is added to that file. A run doesn't write the pipelines that the shipped list already built before the title,
   so your list only holds what the current one misses. Play what the current lists miss, such as areas, dungeons,
   cutscenes, child and adult, or night.
3. Add it, from the repository's root:

   ```sh
   python -I nx2/pipeline_states.py add <github user>-<what you played> <game folder>/hookshot_pipeline_states.bin \
       --who "<github user>" --gpu "<GPU>, driver <version>, <OS>" --build "main at <commit>" \
       --settings "<resolution, upscaler, frame generation, mods>" --route "<what you played>"
   python -I nx2/pipeline_states.py check
   ```

   `add` writes `contrib/<name>.bin.xz` with only the pipelines that are new, and its note `contrib/<name>.md`. Running
   it again with the same name adds to your own list. Use a new name for a different PC or different settings.
4. Open a pull request with the two files. It changes no other file.

**A list depends on the settings it was recorded with.** A record holds the target formats and draw state, so a list
recorded at 1440p with DLSS shares few pipelines with one recorded at 360p with no upscaler. Say the settings in the
note, and record at settings that many players use. Builds from now on write version 2 lists (`HKSTATE2`): they hold
the game's own NVN formats and mesh draws, which each GPU maps to its own formats and to its mesh shaders or the mesh
fallback, so a list recorded on one GPU and at one upscaler, HDR mode or output size prepares the pipelines of the
others. The lists here are still version 1; the merge converts them as soon as one source is version 2
(`nx2/pipeline_states.py --v2` converts a list on its own). When the record format changes, each list here is
converted to the new format (or recorded again) in place, and the layout stays the same.

## The tour's list

The pipeline tour plays a fixed route on its own and records the pipelines it makes. It then replaces `tour.bin.xz` and
`tour.md` with that list:

```sh
python -I nx2/pipeline_states.py tour <tour lists>... --gpu "..." --build "main at <commit>" --settings "..." --route "..."
```

Nothing else writes `tour.bin.xz`, so it is a function of the build and the tour's route. Contributed lists are never
affected by a new tour.

## Where the current lists come from

These were one file, `assets/pipeline_states.bin.xz`, merged from three sources in turn. It was split by source with
nothing left out: the three lists hold the same 11,527 pipelines of 6,672 programs.

- `contrib/hckrman101-first-play`: **1,261 pipelines**, the first list (#1918).
- `contrib/joey-magnus-linux-dlss-hd-pack`: **3,015 more**, from a second player's play on Linux (RTX 4070 Ti, DLSS,
  HD texture pack Lite) (#1935).
- `contrib/jarynclouatre-most-of-the-game`: **7,251 more**, from play on Windows (RTX 4070 Ti SUPER). There were three
  runs of a scripted route through five areas, and a player's sessions through most of the game's areas and dungeons,
  as child and as adult (#2029).
