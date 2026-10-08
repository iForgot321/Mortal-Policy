# Fleet end-to-end test (Phase 0, 03 §2, §10; D48)

- **Plan items:** the 03 §10 fleet scripts (provision, private network, start client, health
  check, teardown) and a first look at the 03 §3.7 recovery rule ("no box loss without automatic
  recovery within 30 min").
- **Plumbing only.** P0 ran for about 17 minutes. No strength claims, no registry
  rows, no lab seeds (the P0 actors use random training keys).
- **Code:** `tools/fleet/fleet.py` at `42e10b3`; the replacement actor got `afbd138` (the
  SSH-stability fix below, no box-side change). Config `experiments/configs/p0.toml`, init a copy of
  pilot A.
- **Raw results:** `workdir/fleet/runs/e2e-1/`: `events.jsonl`, `health.jsonl`, `fleet.log`, the
  pulled learner run dir (`learner-run/`), and each box's logs (`boxes/<name>-<instance>/`).

## 2026-10-08: one learner + one actor (2 client processes), Vast RTX 4090 boxes

| Box | Instance | CPU | $/h |
|---|---|---|---|
| learner, attempt 1 | 54824491 (machine 151441) | Threadripper PRO 7995WX | 0.464; failed its first sync, destroyed after 3 min |
| learner | 54825073 (machine 151246) | EPYC 9684X | 0.464 |
| actor1 | 54824492 (machine 114244) | EPYC 9654 | 0.464; destroyed by hand to simulate a loss |
| actor1, replacement | 54827143 (machine 151444) | EPYC 9654 | 0.464 |

Total cost **$0.41** (plus bandwidth) for 26 minutes of wall-clock. All four boxes shared one
public IP (one datacenter NAT), and hairpin connections between them worked.

| Step | What happened | Time |
|---|---|---|
| `up --actors 1 --procs 2 --verify --hard-hours 1.6` | Rent → running → sync → `setup_box.sh` with tests → start learner → join → verify → start clients | 9 min 24 s, including the failed learner |
| Failed learner | sshd dropped the connection 4 s after the box first answered (the image's entrypoint restarting it, probably). `provision` destroyed the box, marked the machine bad and rented another. **Fixed** (`afbd138`): wait for two SSH answers 20 s apart, and retry syncs. | +3 min |
| `join --verify` | All 4 checks pass on the real learner: the key authenticates and runs no command; a forward to `:22` is refused; a forward to `:5000` works; `-R` is refused | 2 s |
| First P0 cycle | 2 × 800 games, submit, drain 1,600, train; first-minibatch ratio mean **0.99999** (p99 1.0027): the behaviour sidecars cross machines intact | 2 min |
| Kill client-1 (SIGKILL) | `box_supervise.sh` restarted it after 10 s; health showed `fail` (client-1 not running) during the gap, then `ok` | 10 s |
| Kill the tunnel's ssh | `box_tunnel.sh` reconnected with a new PID | 5 s |
| Kill `server.py` on the learner | `box_learner.sh` stopped the trainer's whole group (38 processes) and both restarted. The trainer reloaded the last checkpoint (the init: the crash came before `save_every`), the actors rode out the gap on their backoff, and a drain followed 8 s after the restart (ratio mean 1.000008) | 22 s to restart |
| Destroy actor1's instance (`vastai destroy`) | `watch --replace --grace 2 --interval 60`: `lost` at the next check; after the grace, revoke its key, confirm destroyed, rent, set up, join, start | **recovered 591 s after detection**, ~11 min after the loss |
| Replacement actor | First submission 90 s after start; next drain ratio mean 0.99994 | |
| `down` (with `watch` still running) | Pull, destroy both, sweep labels, stop the watchdog. Zero instances left on the account; `watch` exited at its next round | 11 s |

### Readings

- **The 03 §3.7 recovery rule looks attainable** with the defaults (`--grace 10`, `--interval 300`):
  ~10 min grace + up to 5 min to the next check + ~7 min to provision ≈ 22 min, against a
  30-minute budget. Image pulls of 25+ min (seen on 2026-10-06) would break it. `provision` then
  gives up on the host after `WAIT_RUNNING_S` and tries another, which costs another ~7 min.
- **P0 actor rate through the tunnel:** 800 games in ~86 s per process with 2 processes, so
  **≈ 66K hanchan/h per box** (EPYC 9654). One process on the 9684X gave 31.7K/h on 2026-10-06.
  Submitting a batch and fetching new parameters takes ~1 s in total, so the tunnel costs nothing
  visible at this scale. That is a short sample (3 batches per process), not a benchmark.
- **The learner kept up:** queue 0, never blocked (`blocked 0%`), GPU ~94% while training a drain.
  One actor box is far below the learner's ~3.75K samples/s ceiling (`fleet-host-benchmarks.md`).
- **Known gap:** a server restart resets parameter versions to 0, so that drain's policy-lag
  numbers are wrong (README). The learner-replacement path (pull → new learner → push checkpoint →
  repoint actors) was not exercised.
