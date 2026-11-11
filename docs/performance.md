# Where the frame goes

Measured on Linux, RTX 5060 Ti, driver 615.71.09, headless under Xvfb, Ocarina in Kokiri Forest with
a warmed shader cache, driven there by a frame-clocked `HOS_INPUT_SCRIPT` so that two arms measure
the same place in the game. A frame of ~1800-2800 draws is gameplay; ~5 draws a frame is the title
screen, and comparing one against the other is how this project once blamed FSR for a collapse it
had nothing to do with (`docs/upscaling.md`).

## The frame is CPU-bound in recording, not on the GPU

A representative gameplay frame, from `vk: a frame's replay time (ms)`:

| | ms |
|---|---|
| `submit` (the whole of `run_submit`) | **30.6-35.0** |
| of which `draw` | 7.9-11.0 |
| `programs` | 3.3-4.7 |
| `descriptors` | 2.2-3.1 |
| `commands` | 2.0-2.6 |
| everything else named | < 1 each |
| **GPU time** | **~12** |

The named sub-counters are subsets of `submit`, and they account for roughly two thirds of it; the
rest is inside `replay_handle` walking the game's command stream. At ~30 fps the frame is ~33 ms, so
the GPU is idle for over half of it. **Recording the commands is the limit, not drawing them.**

## What the CPU is actually doing

`perf` on this kernel cannot open events with `-g`, so there are no call graphs; `-e cpu-clock`
(the software event) and a flat profile work. 30 s of a gameplay scene:

| share | symbol |
|---|---|
| **~30%** | one libc routine, disassembled to `rep movsb` -- ERMS `memmove` |
| 3.5% | `spin_until_set` |
| 2.1% | `__sched_yield` |
| ~1% each | guest code (`f_*`), `renderer_main`, `obj_lookup`, `build_descriptors`, `ub_try_gap` |

**The memcpy is not attributed.** Without call graphs there is no honest way to say whether it is
guest-memory reads, mirror syncing, staging uploads or the game's own work, and guessing is how the
upscaling work wasted a day. One guess was checked and was wrong: `ub_read` copies a whole 4 KB page
on a read that lands in a captured page, which looks wasteful until you find `submitted_page_reuse`
-- the copy is an investment that pays off because uniform reads cluster on a page.

## Parallel replay exists, and today it makes things worse

`RX_VK_PARALLEL=<threads>` (default 0, off) walks the command stream once on the replay thread to
find segment boundaries, then records segments of ~160 draws across worker threads, submitted in
order. It was written for Mario Kart World, where one command buffer held nearly a frame's draws.

Measured against the serial path, same scene:

| arm | fps | `submit` |
|---|---|---|
| serial | 29.9 / 30.5 / 30.5 | 30.55 ms |
| `RX_VK_PARALLEL=4` | 23.2 / 23.2 / 24.4 | 38.02 ms |

It costs about a fifth of the frame rate. The reason is in its own log line, and it is not the dry
pass:

```
parallel replay, a command buffer: dry pass 0.34 ms, segments 11.22 ms (4.7 of them),
                                   lock taken 739 times, waited for 3.65 ms (all threads)
```

The dry pass is 0.34 ms. The workers spend **3.65 ms waiting on a lock they take 739 times** per
command buffer, across 4.7 segments. `ParLock` (gpu/vk/vk_internal.h) is **one global recursive
mutex** over everything the threads share -- textures and their uploads, programs, pipelines,
samplers, pools and memory stamps. Three of its seventeen sites in the draw path are inside
`build_descriptors`' per-binding loop, so the count rises with every texture every draw binds.

### Which sites take it

`ParLock` counts by file and line now (free when parallel replay is off, since nothing in it runs
then), and the same report names them by the file and line of the build it runs in. Over 240 command
buffers, per command buffer (named here by the function each site is in, since line numbers move):

| site | taken | waited |
|---|---|---|
| the texture touch's locked slow path (`vk_draw.cpp`, after `texture_touch_quick`) | 352 | 1.18 ms |
| **the pool lookup in `gpu_buffer_bind` (`vk_memory.cpp`)** | 250 | **2.02 ms** |
| four other `vk_draw.cpp` sites together | 140 | 0.57 ms |

Two sites are the whole of the waiting, and each has a shape worth knowing:

- **The texture touch** is the slow path: `texture_touch_quick` is lock-free and
  the lock is only taken when it misses. 352 misses a command buffer is the cost, so the question is
  why the quick path misses so often, not how to make the locked path cheaper.
- **`gpu_buffer_bind`'s pool lookup** waits the longest. It takes `ParLock` **before** `pools_mu`, and its own
  comment says why: always in that order, because a mirror sync records into the shared upload
  command buffer. So the lock is held across the whole pool lookup even when no sync follows. The
  obvious fix -- take it only when a sync will actually record -- is the one thing that must not be
  done casually, because the ordering rule exists to stop a deadlock against whatever takes these
  two the other way round.

Those are the lock's two questions: why `texture_touch_quick` misses, and whether the `pools_mu` path
can resolve a pool without holding the shared-recording lock. Both are measurable with the harness
that produced this table (`texture_touch_misses` and `memory_pool_lock_reasons` count the reasons).
But the next section shows that answering them would not make the parallel path win here.

### Tuning does not rescue it, and the lock is not the main cost

Swept against serial, same scene:

| arm | fps | `submit` |
|---|---|---|
| **serial** | **31.1 / 30.9 / 29.3** | **31.65 ms** |
| `RX_VK_PARALLEL=2` | 24.7 / 25.4 / 26.4 | 34.86 ms |
| `RX_VK_PARALLEL=4` | 23.2 / 23.2 / 24.4 | 38.02 ms |
| `RX_VK_PARALLEL=8` | 16.4 / 16.4 / 14.7 | 63.41 ms |
| `=4`, `_SEG=480` | 7.3 / 18.9 / 18.6 | 50.25 ms |

Serial wins at every setting, it gets monotonically worse with more threads, and larger segments are
worse still (that 7.3 is one sample of three, so it stutters as well as being slow).

**Two threads already lose by 5 fps**, and two threads can barely contend. So the main cost is not
the lock: it is the fixed work per command buffer -- the dry pass, the segment list, a recording and
command buffer per segment, and the ordered submission -- and fixing the two lock sites above could
not have recovered it. That is worth saying plainly, because the lock numbers (739 acquisitions,
3.65 ms) invite the opposite conclusion. The lock is a second cost on top, which grows with threads.

**Why it was written, and where it probably does pay.** The design note says Mario Kart World hands
over nearly a frame's draws in *one* command buffer, 6-10 ms of replay on the critical path. Ocarina
submits about six command buffers a frame, so each one is small and the per-buffer overhead is paid
six times for proportionally less work. The feature is likely right for the game it was written for
and wrong for this one -- which makes it worth measuring again in the Mario Kart World fork rather
than deleting.

The frame rate is still waiting on the serial path: ~15 ms of `submit` that no sub-counter names,
and the ~30% of CPU in `memmove` that needs call graphs to attribute. That is where the next gain is.

For Ocarina, `RX_VK_PARALLEL` should stay off, which it is. If it is ever made to pay here, it needs
both fewer, larger command buffers' worth of work per segment and a split of `ParLock` by what it
protects; the split is delicate, since the mutex is recursive and taken reentrantly, so it has to
preserve every nesting order that exists today.

## NVIDIA's shader disk cache (opt-in)

The start-up preparation builds every program's stage libraries once, tens of thousands of them, into the driver's own
shader disk cache. On Windows that cache is smaller than this by default, so the driver trims it and the next start
builds them again. `nvidia_shader_cache_gb=<GB>` in `hookshot_options.ini` (there is no menu row; at most 64) sets
`__GL_SHADER_DISK_CACHE_SIZE` to that and turns the driver's trimming off, before the driver loads and only where the
environment does not set either already. Measured on the NX2navi build: the same preparation took 148 s before, 10.7 s
with 10 GB. It is off unless asked for, as it is the player's disk.
