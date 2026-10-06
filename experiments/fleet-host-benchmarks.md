# Fleet host benchmarks (Phase 0, 03 §3)

- **Plan items:** 03 §3.1 (host benchmark and allowlist, ≥ 25K hanchan/h), §3.2 (collection modes),
  §3.3 (bottleneck), §3.6 (learner capacity); Phase 0 exit needs hanchan/h per box type (`00`).
- **Throughput only.** No strength claims and no registry rows. The benchmark seeds come from a
  benchmark-only key (`0x62656e6368`, "bench") outside the seed families, so no nonces are used.
- **Raw results:** `workdir/fleet/<date>-<provider>-<gpu>-<cpu>/` (`box/` = setup log, `info.txt`,
  benchmark JSON; `p0-smoke/`, `p0-tput/` = logs and tensorboard).

## Hosts

| Date | Host | GPU | CPU | vCPU (quota) | RAM | $/h | Eval mode, hanchan/h | P0 actor, hanchan/h | ≥ 25K/h |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-06 | Vast machine 137569, California | RTX 4090 (driver 580.119, CUDA 13.0) | EPYC 9684X (Zen 4) | 32 (30.7) | 108 GB | 0.464 | **49,741** | **31,680** | yes |
| reference | Workstation | RTX 3060 12 GB | i5-12600KF | 16 | 15 GB | — | ~22,000 (A/A′ run timing, 2026-10-05) | — | — |

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
  raises policy lag). A profile (`py-spy`) would settle it.
- **Learner capacity (03 §3.6):** ~3,750 samples/s covers the P0 arm's 1 learner + 2 actors
  (~2,800 samples/s at this actor rate), but not 03's fleet scale of 6–8K samples/s.
- **Policy lag:** ~2 versions without `force_sequential`, against the fleet soak's p95 ≤ 2
  versions (03 §3.7): borderline.
