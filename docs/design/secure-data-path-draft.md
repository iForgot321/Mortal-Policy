# Draft: the secure data path for SL training (D23), for the user's review

**Status: draft, 2026-10-28.** The user chose to build the SL substrate next (G0's option 1: teachers →
students → the actor-size curves → π_SL → T1). This is its first piece: how the Tenhou training set
reaches a rented GPU, what protects it there, and how it is wiped. It implements D23 as written, so
it needs no `08` entry; the one exception (D50's vCPU floor) is flagged below. Nothing is rented and
no account is created until the user says yes.

## Why the data has to leave the workstation

The teachers (b24c512, `04` §5) train on the whole train window: 1,805,376 games, 2.34B presentations
(1 epoch × 2 augmentation × 1.17B usable decisions). On the 3060, b24c512 trains at **1,938
samples/s** (measured 2026-10-28: AMP, `torch.compile`, batch 1024, synthetic batches; b8c192 gives
11.0K by the same method against 10.5K in No-SP A's real run, so the method tracks real runs when
the GPU binds). That is ≈ 14 days per teacher and ≈ 6 weeks for the three. A rented 4090 or H100
cuts it to days, and D23 allows only secure-tier providers for that.

## What moves, and what never does

| Item | Size | Moves? |
|---|---|---|
| `train_E` ∪ `train_O` logs (`mjai/…json.gz`, 2015-01 → 2025-06) | 1,805,376 files, 7.69 GiB | Yes: the training set |
| `heldout_game` logs | 48,388 files, 0.21 GiB | Yes: the teacher's validation loss (`04` §3) |
| `splits/masks.jsonl` (Q6/Q8 and disconnect masks) | 16 MB | Yes: the loader needs it |
| GRP-v2 checkpoint | small | Yes: AWR's advantages. A model, not logs |
| `raw/`, `db/`, `meta/`, `names/`, `heldout_time`, the golden set | — | **No.** Golden-set and held-out-by-time checks run on the workstation |
| Player weights | — | Only if D14a adopts an arm other than A0 (no filter, the default) |

**What comes back:** checkpoints, TensorBoard scalars and logs. Weights aren't logs, so they may go
anywhere afterwards, including the community-tier fleet, as pilot A already does. **Per-game or
per-state data made from the logs on the box** (featurized samples, the teacher-output cache for
distillation, any dump) stays on the secure tier or comes back to the workstation. It never goes to a
community host.

## The path, step by step

1. **Pack (workstation, once, ≈ 10 min).** `tools/secure/pack.py` reads the frozen split lists,
   checks `SPLITS.sha256`, and writes ≈ 64 tar shards of ≈ 125 MB. Each shard is encrypted with
   `gpg --symmetric` (AES-256, with integrity check) under a random 256-bit key kept only on the
   workstation (`~/.config/mortal-secure/bundle.key`, mode 600). It also writes a manifest: each
   shard's ciphertext sha256, the files inside it and each file's sha256. The manifest's hash is the
   **bundle id**; every SL run records it, so a teacher names the exact data it saw. The
   ciphertext is safe to store anywhere, but the key is not.
2. **Store between runs: the user's choice.**
   - **(A) A private bucket (recommended):** Cloudflare R2 or Backblaze B2, S3-compatible.
     - Upload the ciphertext once; each box pulls it at datacenter speed through presigned URLs
       valid ≤ 2 h, so no bucket credential ever reaches a box.
     - About 8 GB costs ≈ $0.05–0.15/month at list prices (not re-checked).
     - The same bucket can hold the encrypted `raw/` backup that `04a` §10 still lists as open
       (35 GB, ≈ $0.25–0.55/month).
     - Needs an account you create, with a bucket-scoped key on the workstation.
   - **(B) Direct from the workstation:** rsync over SSH to each box. Nothing new to set up, but every
     box waits on the home upload: 7.9 GiB per box, and on 2026-10-18 the link fell to ≈ 200 KB/s
     (≈ 11 h). AWR ∥ BC, then the second seed and the students, is ≥ 4 transfers.
3. **Rent: secure tier only.** `tools/secure/secure.py up` searches only `datacenter=True` (Vast
   Secure Cloud: vetted ISO 27001 / Tier 3–4 datacenters). It records the instance in the ledger and
   re-checks the live instance's secure-cloud flag before any data moves. It also requires the
   contract's end date to exceed the run's length plus 2 days (the eval box was lost at its contract end on 2026-10-17).
4. **Ship and unpack: plaintext only in RAM.**
   - The box fetches the ciphertext shards (A or B).
   - The key is streamed over SSH into `gpg --batch --passphrase-fd 0`, piped into `tar -x` under
     `/dev/shm/mj` (tmpfs). The key never touches the box's disk.
   - The box then checks every file against the manifest and rebuilds the loader's file index
     from it.
   - About 8 GiB of plaintext plus ≈ 4 GB per loader worker needs ≥ 64 GB RAM; the candidates have
     94–126 GB.
   - **If the box's `/dev/shm` is too small** (unknown on Vast containers; the measurement session
     checks), the fallback is plaintext on the secure box's own disk, deleted at the end and the
     instance destroyed. D23 permits that, but it's weaker, so it needs the user's word.
5. **Train, with pulls.** A watchdog deadline and a dollar cap, as `fleet.py` has, and checkpoint
   pulls every 2–4 h, so a lost box costs hours, not days.
6. **Wipe.** `rm -rf /dev/shm/mj` and any cache, pull the last checkpoint, then destroy the instance
   (Vast deletes its storage), as D23 asks after each run. A lost or stopped host loses the RAM copy
   with it, and the watchdog destroys the instance through the API either way.

**The ledger** (`workdir/secure/ledger.jsonl`, append-only): for each box that received data, the
provider, machine, secure flag as checked, bundle id, shipped-at, wiped-at and destroyed-at. That is
the audit trail for D23.

## Guards in the tooling

- The secure tool and the community fleet don't share a path to the data. `fleet.py` and
  `setup_box.sh` never read `~/mj-data` or the key. A test checks `tools/fleet/` for both,
  except `bench_featurize.py`, which only the secure tool runs.
- `secure.py ship` refuses an instance that isn't in the ledger with a verified secure flag, and
  `up` refuses any offer the query didn't mark `datacenter=True`. Both get tests.
- Every secure run has its own `workdir/secure/<run>/`. Nothing there is synced to the fleet.

## Providers and prices (live, 2026-10-28)

| Vast Secure Cloud offer | $/h | vCPU | RAM | Notes |
|---|---|---|---|---|
| 4090, EPYC 7B13 (Maryland) | 0.780 | 32 | 126 GB | 307-day contract, 1.6 Gbit/s down. **The default** |
| 4090, EPYC 7Y83 (Hungary) | 1.357 | 32 | 94 GB | 180 days; VM-capable |
| 4090, Threadripper PRO (Ireland) | 0.428 | 12.8 | 101 GB | Usable only if ≈ 13 vCPU feeds the GPU (the measurement says) |
| H100 SXM, Xeon 8568Y+ (Czechia) | 3.434 | 24 | 106 GB | 337 days |
| H100 NVL, EPYC 9554 (Czechia) | 3.821 | 28 | 189 GB | 122 days |

Supply is thin: two 4090s have ≥ 32 vCPU. Lambda, Verda and Runpod Secure (`01` §10) would need new
accounts and tooling, so they stay the fallback. Blackwell cards stay out: torch 2.6 has no kernels
for them (D50).

**D50's vCPU floor:** D50 has the teachers on "a 4090 with ≥ 48 vCPU", a figure set by v4's
featurization. On v5 the SL pipeline gives ≈ 19K samples/s with 2 workers on the workstation
(`experiments/sp-ablation.md`), and a 4090 needs ≤ 8K for b24c512. So 32 vCPU, perhaps 13, likely
feed it. No secure 4090 has ≥ 48 vCPU today. If the measurement confirms it, a dated `08` note lowers
the floor before any teacher rents.

## Time and cost (estimates; the measurement session replaces them)

| | 4090 ($0.78/h) | H100 SXM ($3.43/h) |
|---|---|---|
| b24c512 training rate | ≈ 4.8–7.8K samples/s (2.5–4× the 3060's 1.9K) | ≈ 2–3× a 4090 |
| One teacher (2.34B presentations) | ≈ 83–135 h, **≈ $65–105** | ≈ 28–68 h, ≈ $95–230 |
| Three teachers (AWR ∥ BC, then the winner's second seed) | ≈ $200–320, ≈ 2 weeks | faster, ≈ 2× the money |

The plan's own estimate (30–40 h per 1B presentations, `03` §4.2, so 70–94 h) overlaps the fast end of the 4090 range. The
4090 wins per dollar unless an H100 is > 4.4× faster. If the second ≥ 32-vCPU 4090 ($1.36) runs BC
beside AWR, add ≈ $50–80; running them one after the other instead adds ≈ 3.5–5.5 days. Storage costs
cents. All of it sits inside the SL substrate's ≈ $1.0K (`00`).

## Before the first rental (workstation work)

1. `tools/secure/pack.py` and the bundle with its manifest; `tools/secure/secure.py`
   (up / ship / run / pull / wipe / down / ledger), built on `fleet.py`'s offer and SSH helpers, with
   the guard tests.
2. **The teacher trainer (its own draft, next).** `04` §5's heads (placement, kyoku-Δ, opponent
   tenpai, waits and hand value) are new code: `mortal/` has the policy head and arm V's next-rank head
   only. It also needs EMA, a cosine schedule and checkpoint pulls, plus a smoke run on the workstation
   with the quick500k subset. `train_offline_phase1.py` already has AWR/BC, AMP and resume.
3. If (A): you create the bucket and a bucket-scoped key; I upload the ciphertext.

## The first rental: a measurement session (D17a), ≈ $3–7, asked for when the tools are ready

One secure 4090 (the $0.78 box) for ≈ 2–3 h, optionally the H100 SXM for 1 h:
- **The data path end to end:** fetch, decrypt to RAM, verify, wipe, destroy, with the ledger. It
  also checks `/dev/shm`'s size and the transfer time.
- **The v5 pipeline's samples/s on the real data** at 2, 4 and 8 workers, pinned to 13 and 32 vCPU.
- **b24c512's training samples/s** with the real pipeline (compiled, batch 2048). This decides D17a
  by `03` §4.2's rule (feed ≥ 80% of the GPU), the D50 note and the teachers' run time.

## Decisions for the user

1. **Storage:** (A) a private bucket (recommended: it frees the runs from the home link, and it can
   hold the `raw/` backup), or (B) direct from the workstation.
2. **Plaintext only in RAM** (recommended), with the disk fallback only on your word if `/dev/shm`
   is too small.
3. **Vast Secure Cloud** as the provider (recommended), the others as fallback.
4. Later, separately: the measurement session's rental (≈ $3–7), then the teacher runs.

## Not in this draft

- The teacher trainer's heads and losses (next draft).
- Where the teacher-output cache for distillation lives: up to ≈ 0.8 TB for every head in FP16
  (`docs/design/cost-forecast-g1.md`), small if policy only or top-k. It's derived data, so it stays
  on the secure tier or comes to the workstation (730 GB free). That is decided with the student
  runs.
- Publishing weights trained on the data: `04` §1 asks for a re-read of Tenhou's terms and an `08`
  entry first.
