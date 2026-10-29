# Draft: the secure data path for SL training (D23), for the user's review

**Status: draft, 2026-10-28.** The user chose to build the SL substrate next (G0's option 1: teachers →
students → the actor-size curves → π_SL → T1). This is its first piece: how the Tenhou training set
reaches a rented GPU, what protects it there, and how it is wiped. It implements D23 as written, so
it needs no `08` entry; the exceptions (D50's vCPU floor, a provider outside `04` §1's named list)
are flagged below. Nothing is rented and no account is created until the user says yes.

**Decided by the user (2026-10-28):** decrypted data may sit on the secure box's own disk; RAM-only
isn't required. **Asked:** check for alternatives to Vast before choosing the provider (see
"Providers").

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
3. **Rent: secure tier only.** `tools/secure/secure.py` takes a box only from the chosen secure-tier
   provider (see "Providers"), through its API where it has one, or as a box rented by hand in its
   console. Either way the tool records it in the ledger before any data moves. On Vast it searches
   only `datacenter=True` and re-checks the live instance's flag, and it requires the contract's end
   date to exceed the run's length plus 2 days (the eval box was lost at its contract end on
   2026-10-17).
4. **Ship and unpack.**
   - The box fetches the ciphertext shards (A or B).
   - The key is streamed over SSH into `gpg --batch --passphrase-fd 0`, piped into `tar -x` under the
     run's data directory on the box's disk (the user's call: disk is fine). The key itself is never
     written on the box.
   - The box then checks every file against the manifest and rebuilds the loader's file index
     from it.
   - Disk: ≈ 8 GiB of data plus ≈ 16 GiB of ciphertext while unpacking (deleted once verified).
     RAM: ≈ 4 GB per loader worker, so ≥ 32 GB; the candidates have 58–240 GB.
5. **Train, with pulls.** A watchdog deadline and a dollar cap, as `fleet.py` has, and checkpoint
   pulls every 2–4 h, so a lost box costs hours, not days.
6. **Wipe.** Pull the last checkpoint, delete the data directory and any cache, then destroy the
   instance and its volumes, as D23 asks after each run. If a box is lost mid-run, the watchdog
   destroys the instance through the provider's API, or the ledger flags it for destruction by hand.

**The ledger** (`workdir/secure/ledger.jsonl`, append-only): for each box that received data, the
provider, machine, secure flag as checked, bundle id, shipped-at, wiped-at and destroyed-at. That is
the audit trail for D23.

## Guards in the tooling

- The secure tool and the community fleet don't share a path to the data. `fleet.py` and
  `setup_box.sh` never read `~/mj-data` or the key. A test checks `tools/fleet/` for both,
  except `bench_featurize.py`, which only the secure tool runs.
- `secure.py ship` refuses a box that isn't in the ledger under an allowed provider. On Vast, `up`
  also refuses any offer the query didn't mark `datacenter=True`. Both get tests.
- Every secure run has its own `workdir/secure/<run>/`. Nothing there is synced to the fleet.

## Providers (checked 2026-10-28, at the user's request)

**The bar** (D23, `04` §1): a provider-operated datacenter, never a community or peer host. The GPU
must run the pinned torch 2.6 + CUDA 12.4, so Ampere, Ada or Hopper only: no Blackwell (D50). The box
needs ≥ 12 vCPU to feed a 4090-class GPU on v5 (an H100 class wants ≈ 24) and ≥ 32 GB RAM.

**Cost per teacher run** (2.34B presentations). Hours scale a 4090's ≈ 4.8–7.8K samples/s (2.5–4× the
3060's measured 1.9K) by each GPU's speed against a 4090. The speeds are **estimates** from spec ratios
and `01` §10's ResNet-50 AMP figures (A100 1.55×, H100 PCIe 2.08×), not measured on this network.

| Provider (security) | GPU | $/h | vCPU / RAM | Speed vs 4090 | Hours | $ per teacher |
|---|---|---|---|---|---|---|
| Vast Secure Cloud (vetted third-party datacenters) | 4090 | 0.78 | 32 / 126 GB | 1 | 83–135 | **65–105** |
| Hyperstack (SOC 2 Type 2; on `04`'s list) | RTX A6000 | 0.50 | 28 / 58 GB | 0.55–0.65 | 128–245 | 64–123 |
| Massed Compute (owns its infrastructure; SOC 2) | RTX 6000 Ada | 0.79 | 12 / 72 GB | 0.9–1.0 | 83–150 | 66–119 |
| Hyperstack | A100 80GB PCIe | 1.35 | 28 / 120 GB | 1.3–1.6 | 52–104 | **70–140** |
| Massed Compute | A100 80GB PCIe | 1.35 | 16 / 96 GB | 1.3–1.6 | 52–104 | 70–140 |
| Massed Compute | L40S | 0.97 | 12 / 72 GB | 0.95–1.05 | 79–142 | 77–138 |
| Runpod Secure Cloud (on `04`'s list) | L40S | 1.09 | 16 / 94 GB | 0.95–1.05 | 79–142 | 86–155 |
| Hyperstack | H100 PCIe | 2.50 | 28 / 180 GB | 1.8–2.2 | 38–75 | 95–188 |
| Hyperstack | H100 SXM | 3.20 | 24 / 240 GB | 2.3–3.0 | 28–59 | 89–188 |
| Lambda (SOC 2 Type II; on `04`'s list) | A100 40GB | 1.99 | 30 / 225 GB | 1.3–1.5 | 55–104 | 110–207 |
| Lambda | H100 PCIe | 3.29 | 26 / 225 GB | 1.8–2.2 | 38–75 | 124–247 |
| Verda, spot (SOC 2; EU) | A100 80GB SXM | 1.10 | 22 / 120 GB | 1.4–1.7 | 49–96 | 54–106 + interruptions |
| Verda, spot | H100 SXM | 2.35 | 30 / 120 GB | 2.3–3.0 | 28–59 | 65–138 + interruptions |

**Ruled out:**
- Runpod Secure's 4090 ($0.89) and Massed Compute's A6000 ($0.57) have 6 vCPU, too few to feed
  the GPU.
- Hetzner's GPU servers are all Blackwell now (GEX45/63/131), and so are every provider's B200 and
  RTX PRO 6000 (D50).
- The hyperscalers and the larger clouds cost ≥ 1.5× for the same GPU:
  - AWS g6e.4xlarge (L40S, 16 vCPU) $3.00/h on demand;
  - GCP a2-highgpu-1g (A100 40GB) $3.67/h;
  - Scaleway L40S €1.40, H100 PCIe €2.73;
  - Crusoe L40S ≈ $1.45–1.50, H100 $3.90;
  - DigitalOcean L40S $1.57, H100 $3.39;
  - Nebius L40S from $1.55, H100 $4.50.

**Reading:**
- **Per run, five options land within ≈ 10–35% of each other:** the Vast 4090, Hyperstack's A6000
  and A100, and Massed Compute's RTX 6000 Ada and A100. The speed estimates are uncertain by more than
  that, so a measurement picks among them, not this table.
- **The A100 80GB at $1.35 (Hyperstack or Massed Compute) costs about the same per run as the Vast
  4090 and finishes ≈ 1.3–1.6× sooner.** Three teachers ≈ $210–420 and ≈ 4.5–9 days of compute
  (AWR ∥ BC on two boxes, then the second seed), against ≈ $200–320 and ≈ 7–11 days on Vast 4090s.
- **H100s cost ≈ 1.5–2× per run** and finish in ≈ 1.5–3 days. They're worth it only if the calendar
  binds.
- **Verda's spot prices are the cheapest per run,** but a spot box can be taken back at any time. The
  checkpoint pulls make that survivable, at the cost of lost hours and restarts.
- **Security:** Hyperstack, Massed Compute, Lambda and Verda operate their own datacenters, with one
  company and one contract. Vast Secure Cloud is a marketplace of vetted datacenters. All of them
  pass D23's bar, but the first-party clouds are the cleaner reading of it.
  - Massed Compute and Verda aren't on `04` §1's named list (Runpod Secure, Lambda, Hyperstack,
    GCP, AWS). Using either adds a dated `08` note naming it.
- **Tooling:** Vast has `fleet.py`. Every other provider needs an account (identity checks, possibly a
  quota request for an A100 or H100) and a small adapter: Hyperstack, Lambda and Verda have REST
  APIs. With a handful of multi-day runs, renting by hand in the console works too; the ledger and
  the guards don't depend on it.
- **Availability:** single A100s and H100s at these prices sell out. It can only be seen from inside
  an account, so it's checked when the account exists.

**Recommendation: Hyperstack's A100 80GB as the main candidate, with the Vast 4090 as the fallback.**
- **Why:**
  - It's on `04`'s list and SOC 2 Type 2, a first-party cloud, billed per minute.
  - Its 28 vCPU and 120 GB feed the GPU with room to spare.
  - It costs about the same per run as the Vast 4090 and finishes sooner.
  - The same account offers the H100 PCIe ($2.50) if the calendar binds.
- **The measurement session settles it:** 1 h on each, then the cheaper measured cost per teacher wins.

### The Hyperstack account, as its API shows it (2026-10-29 06:10 UTC, `tools/secure/hyperstack.py`)

- **The account can't rent yet:** credit $0, `can_create_instance: false`. Credit has to be added
  first (the user).
- **Single-GPU VMs exist only in CANADA-1.** NORWAY-1 has an A4000 only, and US-1 has 8-GPU nodes
  only. Every single-GPU flavor has 28 vCPU.
- **Stock and prices now** (the price book, which matches the public page):

| Flavor | $/h | RAM | Disk (root + ephemeral) | Single-GPU stock | Est. $ per teacher |
|---|---|---|---|---|---|
| A100 80GB PCIe, on demand | 1.35 | 120 GB | 100 + 750 GB | **0** | 70–140 |
| A100 80GB PCIe, **spot** | 1.08 | 120 GB | 100 + 750 GB | 7 | 56–112, plus preemptions |
| H100 PCIe, on demand | 2.50 | 180 GB | 100 + 750 GB | **0** | 95–188 |
| H100 PCIe, spot | 2.00 | 180 GB | 100 + 750 GB | 4 | 76–150, plus preemptions |
| L40, on demand | 1.00 | 58 GB | 100 + 725 GB | 33 | 87–159 (speed 0.85–0.95× a 4090) |
| RTX A6000, on demand | 0.50 | 58 GB | 100 GB | 3 | 64–123 (slow: 5–10 days) |

- **Spot VMs** can be taken back at any time **without notice**, with no hibernation or snapshots
  (Hyperstack's knowledge base). A teacher on spot needs checkpoint pulls every ≈ 1 h and a restart
  path: a new VM, the data fetched again from the in-region bucket, resume from the last checkpoint.
  Each preemption costs ≈ 0.5–1.5 h. That's cheap if they come rarely; how rarely is unknown.
- **The OS image:** "Ubuntu Server 22.04 LTS R550 CUDA 12.4" matches the pinned torch 2.6 + CUDA 12.4.
- **Object storage:** S3-compatible, region CANADA-1 only (the same region as the GPUs), at
  $0.000019397 per GB-hour (≈ $0.014/GB-month: ≈ $0.11/month for the bundle, ≈ $0.60 with the `raw/`
  backup). Egress price and presigned-URL support are checked by trying them. Presigned URLs are
  standard S3, so they're likely supported.
- **Reading:** on-demand A100s are sold out today, so the choice is between L40 on demand (in stock,
  no interruptions), A100 spot (cheapest per run if preemptions are rare) and A6000 (cheapest per
  hour, slowest). The speed estimates can't separate them; the measurement session can. All three
  are in CANADA-1, and each costs ≈ $0.50–1.10 for an hour.

**D50's vCPU floor:** D50 has the teachers on "a 4090 with ≥ 48 vCPU", a figure set by v4's
featurization. On v5 the SL pipeline gives ≈ 19K samples/s with 2 workers on the workstation
(`experiments/sp-ablation.md`), and a 4090 needs ≤ 8K for b24c512. So 12–32 vCPU likely feed it, and
no secure-tier 4090 has ≥ 48 vCPU today. D50 also names only 4090s and H100s. If the measurement
picks an A100 or a 32-vCPU 4090, a dated `08` note records it before any teacher rents.

## Before the first rental (workstation work)

1. `tools/secure/pack.py` and the bundle with its manifest; `tools/secure/secure.py`
   (up / ship / run / pull / wipe / down / ledger), built on `fleet.py`'s SSH helpers, with an adapter
   for the chosen provider and the guard tests.
2. **The teacher trainer (its own draft, next).** `04` §5's heads (placement, kyoku-Δ, opponent
   tenpai, waits and hand value) are new code: `mortal/` has the policy head and arm V's next-rank head
   only. It also needs EMA, a cosine schedule and checkpoint pulls, plus a smoke run on the workstation
   with the quick500k subset. `train_offline_phase1.py` already has AWR/BC, AMP and resume.
3. If (A): you create the bucket and a bucket-scoped key; I upload the ciphertext.

## The first rental: a measurement session (D17a), ≈ $4 (ran 2026-10-29, the user's yes)

**Boxes:** three Hyperstack VMs in CANADA-1 at once, quoted from the account on 2026-10-29:

| Flavor | $/h (with the IP) |
|---|---|
| L40, on demand | 1.007 |
| A100 80GB, spot | 1.087 |
| RTX A6000, on demand | 0.507 |

Optionally a fourth, H100 PCIe spot at $2.007/h.

Each box runs `tools/secure/secure.py`:
1. `up RUN --flavor F --hours 2.5 --yes`: SSH opens to the workstation's IPv4 only, and a watchdog
   destroys the VM at 2.5 h.
2. `setup RUN`: `bootstrap_vm.sh`, then `setup_box.sh` with the full test suite.
3. `ship RUN`: the bundle from the bucket, checked and unpacked.
4. `exec RUN "python tools/secure/bench_teacher.py --data <data dir>"`: the GPU-only rate, 15 min of
   the real trainer (b24c512, batch 2048, 4 workers, augmentation) and the pipeline by worker count.
5. `pull RUN` and `down RUN`: wiped and destroyed, with the ledger.

**What it measures:** each GPU's real samples/s, so the hours and dollars per teacher; whether the
28 vCPU feed it (`03` §4.2's 80% rule: GPU utilization, pipeline against GPU rate); and the data
path itself (setup and ship times, wipe and destroy in the ledger).

**Cost:** about 1.3 h per box, so ≈ $1.3 (L40) + $1.4 (A100 spot) + $0.7 (A6000) ≈ **$3.4**. At most
≈ $6.5 if every box runs to the watchdog. The H100 adds ≈ $2.6, at most $5.

**Reading:** the lowest measured dollars per teacher wins, unless it's spot and the run can't absorb
preemptions. The result settles D17a, and the `08` note on D50's GPU and vCPU floor follows it.

## The measurement session's result (2026-10-29 07:24–08:13 UTC): the A6000

Three VMs in CANADA-1 on the user's yes, $1.74 in all: L40 0.82 h ($0.82), A100 spot 0.48 h ($0.52),
A6000 0.79 h ($0.40). The ledger shows each one rented, shipped, wiped and destroyed, and the
watchdogs stopped. **The data path worked end to end on all three:**
- setup, including the full test suite: ≈ 10 min;
- the 9.32 GiB bundle fetched, checked and unpacked: 848–885 s (mostly unpacking 1.85M small files);
- wipe ≈ 1 min, destroy ≈ 20 s.

| | L40 | A100 80GB (spot) | RTX A6000 |
|---|---|---|---|
| $/h | 1.007 | 1.087 | 0.507 |
| GPU only, b24c512 compiled, batch 1024 / 2048 (samples/s) | 7,215 / 6,614 | 7,291 / 7,589 | 6,217 / 6,233 |
| **The real trainer** (BC, batch 2048, 4 workers, augmentation), samples/s | **5,346** | not measured¹ | **4,995** |
| GPU utilization in it | 72% | — | 74% |
| SL pipeline alone, 2 / 4 / 8 workers (samples/s) | 6.3K / 12.4K / 16.1K | 9.6K / 14.0K / 20.3K | 7.3K / 16.5K / 16.7K |
| **One teacher** (2.34B presentations) | **122 h, $122** | ≈ 107 h, ≈ $116² | **130 h, $66** |

¹ The trainer part crashed at start-up on all three: it loads `workdir/phase1/baseline.pth`, which
only the workstation has (fixed in `8ba1d33`). The L40 and A6000 ran it again before going down; the
A100 had already gone. ² The A100's estimate scales its GPU-only rate by the other two's real/GPU
ratio (0.80).

**Reading:**
- **D17a:** the A6000 costs about half the L40 or the A100 per teacher, at nearly the same speed.
  This network is small for these GPUs, so the faster cards don't pull ahead. It is on demand (no
  preemptions), at 130 h ≈ 5.4 days per teacher; AWR ∥ BC then the second seed is ≈ 11 days,
  ≈ $200 for the three. H100s aren't needed.
- **`03` §4.2's rule** passes everywhere. 28 vCPU feed each GPU: the pipeline alone gives
  12–17K samples/s with 4 workers against GPU rates of 6–7.6K.
- **The trainer leaves ≈ 26% of the GPU idle.** Its loop syncs with the GPU every step (the
  `assert` on the label mask), so the GPU waits while the next batch is prepared. Removing per-step
  syncs in `train_teacher.py` could bring the A6000 near its GPU-only 6.2K: ≈ 105 h, ≈ $53 per
  teacher.
- **D50** puts every role on 4090s, and teachers on "a 4090 with ≥ 48 vCPU, or an H100". Teachers
  on Hyperstack A6000s (28 vCPU) need a dated `08` note first, for the user to approve.

## Built so far (2026-10-29)

- **The bucket:** `mortal-sl-0d00ec51` at `https://ca1.obj.nexgencloud.io` (CANADA-1), made by
  `tools/secure/bucket.py init`.
  - One region-scoped access key (id 358). Its secret lives only in `~/.config/hyperstack/s3.json`
    (mode 600) on the workstation.
  - `bucket.py check`: an anonymous GET is refused (403), and a presigned GET works.
  - Measured from the workstation: a 150 MB multipart upload at 5.45 MB/s, download at 32.7 MB/s.
  - boto3 runs from its own venv (`~/.venvs/s3`), as vastai does from `~/.venvs/vast`.
- **The bundle:** `tools/secure/pack.py`, bundle **`45b6ffa43e2ceadd`**.
  - 1,853,764 logs (train_E 901,752, train_O 903,624, heldout_game 48,388) in 64 shards, plus
    `meta.tar.gpg` (the three split lists, `masks.jsonl`, `SPLITS.sha256`).
  - 9.32 GiB of ciphertext (the tar headers add ≈ 1.4 GiB to the 7.9 GiB of logs), packed in 160 s.
  - Every shard was decrypted again and matched its plaintext hash. The manifest has no game ids.
  - Key id `630df087bbced396`; the key is `~/.config/mortal-secure/bundle.key` (mode 600). **It
    needs a backup outside the workstation:** without it the bucket holds noise.
  - Uploaded with `bucket.py push … bundle/45b6ffa43e2ceadd`, the manifest last: 66 objects in
    14.5 min (06:26–06:41 UTC, ≈ 11 MB/s), every size matching the manifest.
- **Built 2026-10-29:**
  - `tools/secure/secure.py`, the VM lifecycle with the ledger.
  - `unpack.py`, tested on two bundle objects from the bucket; the extracted files are identical to
    the workstation's.
  - `bootstrap_vm.sh`.
  - `bench_teacher.py`, smoke-tested on the 3060: 1,693 samples/s at batch 512 with the real
    pipeline, GPU 84% busy.
  - The test suite passes (166).
- **Not built yet:** the teacher trainer's heads (its own draft).

## Decisions for the user

1. ~~Storage~~ **Decided 2026-10-28/29: a private bucket on Hyperstack's own object storage**
   (CANADA-1, the GPUs' region; the user).
2. ~~Plaintext only in RAM~~ **Decided 2026-10-28:** the secure box's disk is fine.
3. ~~Provider~~ **Decided 2026-10-28: Hyperstack.** The user asked to explore it first with a CLI.
   Hyperstack has none (only alpha SDKs), so `tools/secure/hyperstack.py` is a read-only stdlib client
   (flavors, stock, prices, images, regions, VMs, buckets; every request a GET). The Vast 4090 stays
   the fallback.
4. Later, separately: the measurement session's rental (≈ $3–7), then the teacher runs.

## Not in this draft

- The teacher trainer's heads and losses (next draft).
- Where the teacher-output cache for distillation lives: up to ≈ 0.8 TB for every head in FP16
  (`docs/design/cost-forecast-g1.md`), small if policy only or top-k. It's derived data, so it stays
  on the secure tier or comes to the workstation (730 GB free). That is decided with the student
  runs.
- Publishing weights trained on the data: `04` §1 asks for a re-read of Tenhou's terms and an `08`
  entry first.
