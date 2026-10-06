# Fleet host benchmarks (Phase 0, 03 §3)

- **Plan items:** 03 §3.1 (host benchmark and allowlist, ≥ 25K hanchan/h), §3.2 (collection modes),
  §3.3 (bottleneck), §3.6 (learner capacity); Phase 0 exit needs hanchan/h per box type (`00`).
- **Throughput only.** No strength claims and no registry rows. The benchmark seeds come from a
  benchmark-only key (`0x62656e6368`, "bench") outside the seed families, so no nonces are used.
- **Raw results:** `workdir/fleet/<date>-<provider>-<gpu>-<cpu>/` (`box/` = setup log, `info.txt`,
  benchmark JSON; `p0-smoke/`, `p0-tput/` = logs and tensorboard).

## Hosts

| Date | Host | GPU | CPU | vCPU (quota) | RAM | $/h | Eval mode, hanchan/h | $ per 1M hanchan | P0 actor, hanchan/h | ≥ 25K/h |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-06 | Vast machine 42443, Washington | RTX 4090 | Ryzen 9 7950X (Zen 4 desktop) | 32 (no quota) | 124 GB | 0.713 | **51,421** | 13.9 | — | yes |
| 2026-10-06 | Vast machine 137569, California | RTX 4090 | EPYC 9684X (Zen 4 server) | 32 (30.7) | 108 GB | 0.464 | **49,741** | **9.3** | **31,680** | yes |
| 2026-10-06 | Vast machine 10447, Pennsylvania | RTX 4090 | EPYC 7V13 (Zen 3 server) | 32 (no quota visible) | 1 TB | 0.463 | **39,956** | 11.6 | 22,274 (800-game batches), 28,511 (2,000) | yes |
| 2026-10-06 | Vast machine 48161, Arizona | RTX 4090 | EPYC 7742 (Zen 2 server) | 32 (30.7) | 503 GB | 0.487 | **30,895** | 15.8 | — | yes |
| 2026-10-06 | Vast machines 27649 (Ontario) and 28903 (Thailand) | RTX 4090 | Ryzen 9 5950X; EPYC 7713 (Zen 3) | — | — | 0.387; 0.413 | not measured: both still pulling the image after 25 min, destroyed by the driver | — | — | — |
| reference | Workstation | RTX 3060 12 GB | i5-12600KF | 16 | 15 GB | — | ~22,000 (A/A′ run timing, 2026-10-05) | — | — | — |

- **Eval mode:** `tools/fleet/bench_1v3.py`. One checkpoint on both sides, argmax, AMP, 500 sets
  (2,000 concurrent games) per arena call, the way the lab plays. Steady median over chunks 2+.
- **P0 actor:** the real `p0.toml` pipeline (trainee samples with `explore_rate = 1.0`, behaviour
  sidecars, 800-game batches), with server and trainer on the same box. Median batch play time.

## Workflow (what a box needs)

1. `vastai search offers 'gpu_name=RTX_4090 num_gpus=1 cpu_cores_effective>=32 cpu_ram>=64
   cuda_vers>=12.4 reliability>0.98 disk_space>=50 direct_port_count>=1 inet_down>=300' --storage 50 -o dph_total`
2. `vastai create instance <offer> --image vastai/pytorch:2.6.0-cuda-12.4.1-py312-22.04 --disk 50 --ssh --direct`
   plus the template's `-e` variables and `--onstart-cmd entrypoint.sh`, **without any `-p` port
   mappings**: they publish Jupyter, Syncthing and the portal on the host's public IP. SSH is
   mapped by `--ssh`; tensorboard goes through `ssh -L 6006:localhost:6006`.
3. From a clean commit: rsync the repo and the frozen inputs (commands in the header of
   `tools/fleet/setup_box.sh`), then run `tools/fleet/setup_box.sh` in tmux.
4. Copy the results back to `workdir/fleet/…`, then **destroy the instance at once**:
   `vastai destroy instance <id> -y`.

Steps 3–4 plus the benchmarks are automated in `tools/fleet/box_bench.sh` (one detached driver
per instance; it always destroys its box, with a 2-hour watchdog and a 25-minute limit on startup).

**Gotchas found on the first box:**
- **Offer IDs change** when hosts relist, and `vastai search offers 'id=<offer>'` matches nothing
  at all. Look an offer up by `machine_id`.
- **Torch 2.6 cu124 has no Blackwell kernels:** no 5090s until a torch ≥ 2.7 / cu128 upgrade
  (an infrastructure change, so T-smoke). `setup_box.sh` refuses such a GPU.
- **`python` is not on PATH** in a non-interactive ssh session. The image's venv is
  `/venv/main/bin/python`, and `setup_box.sh` finds it. The image already has rsync, git, tmux,
  uv, pip and the Python headers.
- **Line endings:** six docs files are CRLF in the workstation tree but LF in the commits, and
  the workstation's `core.autocrlf = input` hides that. `setup_box.sh` now sets the same option.
- **CPU quota:** `nproc` shows 32 but the cgroup quota is 30.7 CPUs.
- **Idle billing:** the first box sat idle for ~7 h after the work was done ($3.78 total, about
  $0.40 of it the trial itself).

## 2026-10-06: RTX 4090 + EPYC 9684X (Vast contract 54461788)

- **Box:** offer 51245734, machine 137569, California; image above; commit `c6a19b0` (clean);
  `bench_1v3.py` sha256 `e0f735c5…`; model pilot A (`15121432…`). GRP-v2 (`db28beac…`) checked.
- **Timeline (UTC):** created 09:26; running 09:29 (image pull ~3.5 min); `setup_box.sh`
  09:31–09:34; smoke 09:35–09:38; eval benchmark 09:38–09:55; P0 throughput 09:56–10:15;
  destroyed 17:39.
- **Setup:** `cargo test` passes; `pytest` 55 passed, 3 skipped (the tests that need Tenhou data,
  which never goes to a community box, D23).

### P0 smoke (`p0-smoke.toml`; the hand-off's M4 checks)

| Check | Result |
|---|---|
| ≥ 2 parameter versions | 4 drains, 12 submissions |
| Every drained log has its sidecar | 32/32 |
| First-minibatch ratio ≈ 1 (AMP noise ~1e-3) | 1.00006, 1.00000, 1.00005 |
| Excluded fraction (NaN behaviour probability) | 0 |
| Memory | ~6 GB |

### Eval mode (15 min)

- 7 chunks of 2,000 games, 142–148 s each: 48,647–50,742 hanchan/h; steady median **49,741**.
- 18.1 cores busy (of a 30.7-CPU quota); GPU utilization ~10%; 1.5 GB of GPU memory.

### P0 mode (`p0.toml` in a scratch directory, ~19 min, everything on one box)

- **Actor:** 12 batches of 800 games. Median play time 90.9 s → **31,680 hanchan/h**; with
  submit and parameter fetch, a 92.2 s cycle → 31,249/h.
- **Learner:** each 800-game drain is 249 steps × 512 ≈ 127.5K samples (~159 per hanchan),
  trained in ~34 s including featurization (`num_workers = 2`, `file_batch_size = 50`):
  **~3,750 samples/s**, busy ~37% of the time.
- Box: 16.7 cores busy on average (cgroup); GPU utilization ~10%; RAM ≤ 9 GB.
- PPO diagnostics: excluded 0; policy lag mean ~2 versions (max 3); first-minibatch ratio
  0.98–1.006, as expected with lag (the ≈ 1 check needs lag 0, as in the smoke run); clip
  fraction ~5%; entropy ~0.49.

### Reading

- **Allowlist:** this host type passes in both modes.
- **Bottleneck (03 §3.3):** neither CPU (18 of 31 cores) nor GPU (~10%) is saturated, so the limit
  is the serial step loop between libriichi and the engine, not raw compute. P0's 800-game batches
  run 36% slower than the lab's 2,000-game chunks; batch size is the first lever to test (it also
  raises policy lag). A profile (`py-spy`) would settle it. *(Settled below: the Rust arena loop
  on the main thread; 2,000-game batches give +28%.)*
- **Learner capacity (03 §3.6):** ~3,750 samples/s covers the P0 arm's 1 learner + 2 actors
  (~2,800 samples/s at this actor rate), but not 03's fleet scale of 6–8K samples/s.
- **Policy lag:** ~2 versions without `force_sequential`, against the fleet soak's p95 ≤ 2
  versions (03 §3.7): borderline.

## 2026-10-06: CPU classes, all on RTX 4090 (4 boxes in parallel, `box_bench.sh`)

- **Code:** commit `dd052e7` (clean); `bench_1v3.py` `e0f735c5…`; pilot A. Setup and tests
  pass on both boxes that started (55 passed, 3 skipped).
- **Ryzen 9 7950X** (contract 54521468): 7 chunks, 50,420–52,513 hanchan/h, steady median
  **51,421**; GPU ~11%. This box has no cgroup CPU quota, so the CPU-busy figure is missing.
- **EPYC 7742** (contract 54521465): 4 chunks, 29,912–31,190 hanchan/h, steady median **30,895**;
  17.1 cores busy (of a 30.7-CPU quota), GPU ~7%.
- **Ryzen 9 5950X and EPYC 7713** (contracts 54521454, 54521461): still `loading` (pulling the
  image) after 25 min; destroyed unmeasured. The py-spy profile and the 2,000-game P0 test were
  planned on the 5950X, so they did not run.
- Cost of the round: $0.48 (the two boxes that never started cost little or nothing).

### Reading

- **Per-core speed decides the actor rate.** At the same ~17–18 busy cores (both EPYCs) and a near-idle GPU,
  Zen 4 (server 2.5–3.7 GHz or desktop 4.5–5.7 GHz) gives ~50K/h and Zen 2 gives ~31K/h. That
  fits the serial step loop being the bottleneck: newer cores help, higher desktop clocks add
  only ~3% over Zen 4 server cores.
- **Value:** the Zen 4 EPYC 9684X costs $9.3 per million hanchan, about half the plan's $18 (`03`
  §9). The 7950X is as fast but costs more per hour. Zen 2 clears the 25K/h bar but costs ~70%
  more per hanchan than the 9684X.
- **Allowlist rule (provisional):** prefer Zen 4 hosts (EPYC 9xx4, Ryzen 7000/9000) and rank
  them by $/hanchan; Zen 3 (~40K/h, see below) is a fair second; accept Zen 2 only when it is
  much cheaper per hour.

## 2026-10-06: Zen 3 retry with the extras (EPYC 7V13, contract 54525421)

- **Box:** machine 10447, $0.463/h, 256 host threads with no cgroup quota visible, 1 TB RAM.
  Code `dd052e7` plus uncommitted docs (the tree was synced before the notes were committed;
  no code differs). Cost $0.45.
- **Contended host:** `pytest` took 7 min 53 s here against 22–77 s on the other boxes, and setup
  took 12 min, so other tenants were probably loading the machine. Treat this box's numbers as a
  lower bound for Zen 3.
- **Eval mode:** 5 chunks, 38,249–40,787 hanchan/h (rising), steady median **39,956**; GPU ~8%.

### py-spy profile (`py-spy record --native --rate 50`, one 2,000-game chunk)

py-spy samples the Python (main) thread only; 2,412 samples, and 8,062 more failed to unwind
natively, so the shares below are indicative. Profiling also halved throughput (21,034/h).

| Main-thread time | Share |
|---|---|
| Rust arena (libriichi), outside `react_batch` | **88.8%** (a quarter of it in `syscall`, i.e. waiting) |
| Engine `react_batch` (torch forward, numpy, Python) | 2.3% |
| Startup (`load_engine`) | 2.5% |
| Other | 6.3% |

**Reading:** the bottleneck is libriichi's arena step loop on the main thread (game logic, event
handling, allocation), not Python and not inference. That matches the low GPU use, the partial CPU
use, and the per-core-speed dependence across CPU generations. The lever is in Rust: parallelize
or shorten what the main thread does per step (03 §6 items 7–8).

### P0 throughput at 800- vs 2,000-game actor batches (same box, server + trainer + client)

| Batch | Batches timed | Actor, hanchan/h | Learner busy per drain | Policy lag (mean / max) | First-minibatch ratio |
|---|---|---|---|---|---|
| 800 | 3 | 22,274 | 54 s of a 131 s cycle | 0.4 / 1 (one reading) | 0.995–1.005 |
| 2,000 | 3 | **28,511** (+28%) | 158 s of a 256 s cycle | 2.2 / 4 | 1.000–1.003 |

- Larger batches raise the actor rate by about 28% but raise policy lag past the fleet soak's
  p95 ≤ 2 versions. The batch size is a tuning-allowance item for the P0 arm (05 §2.2), to be
  written down before the runs.
- The learner here trained ~2.4K samples/s (800-game drains), against 3.75K/s on the 9684X box:
  the contended host again.
- Only three batches per setting: the ±% is rough.

## Featurization rate (2026-10-06, workstation)

`tools/fleet/bench_featurize.py` (sha256 `c8620e57…`, commit `dd052e7`): 256 games sampled evenly
from `quick500k` (2020–2025), i5-12600KF (6 performance + 4 efficiency cores, 16 threads, WSL2).
Results: `workdir/fleet/20261006-workstation-featurize/featurize.json`.

**Loader only** (`GameplayLoader.load_gz_log_files`: parse + obs v4, all four seats, ~670
samples per game), one fresh process per rayon thread count:

| Threads | Samples/s | Per thread | Peak RSS |
|---|---|---|---|
| 1 | 825 | 825 | 1.4 GB |
| 2 | 1,580 | 790 | 1.5 GB |
| 4 | 2,791 | 698 | 1.5 GB |
| 8 | 4,398 | 550 | 2.3 GB |
| 16 | 5,743 | 359 | 3.8 GB |

**SL input pipeline** (`FileDatasetsIter` + `DataLoader` as `train_offline_phase1.py` builds them:
BC, fp16 obs, Q6/Q8 masks, `file_batch_size` 20, batch 2048, no GPU):

| Workers | Samples/s | Peak RSS (workers) |
|---|---|---|
| 1 | 3,927 | 3.9 GB |
| 2 | 4,241 | 3.7 GB |

The 2-worker rate matches the pilot students' training rate (4.2K/s, `pilot-qs-a.toml`).

### Reading (03 §4.2 teacher-hardware rule)

- The rule needs (vCPU × samples/s per vCPU) ≥ 80% of the GPU's training rate. Here the pipeline
  gives **~265 samples/s per thread**: the loader scales sublinearly past the 6 performance cores,
  and the per-sample Python loop in `populate_buffer` caps the pipeline at ~74% of the loader.
- Using `03` §4.2's estimates for b24c512 (~8K samples/s on a 4090, ~35–55K/s on an H100) [calc]:
  an H100 would need well over 100 threads, against the 16–26 vCPU H100 instances come with, so
  the H100 is out for the teachers. A 4090 or 5090 needs ~6.4K/s or more, i.e. a box with ≥ 48 vCPU
  *if* the pipeline keeps scaling with more workers. That needs a large-RAM box to test (this
  workstation can't run more than 2 workers), and the per-sample Python loop is the first thing to
  optimize if it doesn't.
- Per-thread rates here come from a hybrid desktop CPU; server cores will differ. Re-measure on the
  teacher box (secure tier only: Tenhou data, D23).
