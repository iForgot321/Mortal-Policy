# 03: Throughput and Fleet

**Phase 0 deliverable. Re-checked at every gate.**

Strength in this genre scales with games played and searched. BigCoach reportedly used hundreds of millions of hanchan (hearsay; `01` §6). Under the $10–15K cloud budget (D21), **hanchan per dollar** is a top-level project metric.

The consequence: **scale out on consumer GPUs**. A 4090 or 5090 delivers about 2.5× more CNN training throughput per dollar than an H100 or H200. For small, launch-bound networks an H100 is estimated to be only about 1.0–1.3× faster than a 4090 (`01` §10).

---

## 1. Targets

*Revised 2026-10-01 per D30 and D39. Cumulative games are a **forecast**. Releasing compute depends on measured learning (D30) and on the fleet soak (§3.7).*

| Milestone | Per-box target | Fleet | Forecast cumulative hanchan (generated) |
|---|---|---|---|
| G0 (week 9) | ≥ 25K hanchan/h on the selected actor | **Fleet soak passes** (§3.7) | ≈ 50–60M (engine arms) |
| End of T1 (week 12) | ≥ 30K/h | 3 actors | +≈ 45M |
| End of T3 (week 18), if released | ≥ 30K/h (stretch 50K) | 4 actors (T2–T3) | +≈ 120M (≈ 90M of them champion-lineage) |
| Project end | — | — | ≈ 270–290M |

**Three counters, always reported separately** (D39):
- **generated** games;
- **consumed** training samples, with sample age (policy versions behind);
- **champion-lineage** games, i.e. those that trained the current champion's ancestry.

**Reference:** Mortal reports up to 40K hanchan/h on an RTX 4090 + Ryzen 9 7950X (16 cores, 32 threads) at game batch 2000, with an unstated network size.

Many Vast hosts run older EPYC chips (e.g. a 7542 at 2.9 GHz), so **benchmark candidate hosts and keep an allowlist** (§3).

## 2. Fleet architecture

```
        private network (mandatory): SSH tunnels to the learner (D48)
┌────────────────┐   params    ┌─────────────────────────┐   replays   ┌──────────────┐
│ LEARNER        │ ──────────▶ │ ACTORS ×N (stateless)   │ ──────────▶ │ LEARNER      │
│ 1× 4090, ≥32   │             │ 1× 4090, ≥32 vCPU       │             │ (server.py   │
│ vCPU, on-demand│ ◀────────── │ client.py + libriichi   │             │  buffer +    │
│ server.py +    │  get_param  │ interruptible OK        │             │  drain)      │
│ train_online   │             └─────────────────────────┘             └──────────────┘
└───────┬────────┘
        │ checkpoints → object storage (every ~200 iterations)
┌───────▼────────┐        ┌──────────────────┐       ┌───────────────────────┐
│ EVAL BOX       │        │ BURST            │       │ DEPLOY BOX (Phase 4)  │
│ 1× 4090, ≥32   │        │ SL teachers, CPU │       │ 1× 4090, ≥16 vCPU,    │
│ vCPU, on-demand│        │ benchmarks,      │       │ on-demand, reliable;  │
│ GSPRT queue    │        │ Mahjax spike     │       │ riichi.dev bot        │
└────────────────┘        └──────────────────┘       └───────────────────────┘
```

**Existing support.** Mortal-Policy already implements this actor–learner split (`01` §1):
- `client.py` actors loop: `get_param` → `train_play` (1v3) → `submit_replay`.
- `server.py` buffers the replays (with `capacity` and `force_sequential`).
- The learner drains the buffer, then featurizes it with `GameplayLoader`.
- **New work:** provisioning scripts, a reconnect/backoff loop in `client.py`, monitoring, and the private network (`tools/fleet/fleet.py`).

**Security:** `server.py` is plain TCP with **no authentication**. It binds to loopback only; actors reach it through restricted SSH tunnels to the learner (D48). Never expose it publicly.

**Roles** (all on RTX 4090s through Phase 2: the pinned torch 2.6 has no Blackwell kernels; D50):

| Role | Spec | Pricing tier | Notes |
|---|---|---|---|
| **Learner** | 1× RTX 4090, ≥ 32 vCPU, ≥ 64 GB RAM | On-demand, verified/reliable host | Its CPU also featurizes incoming replays. Upgrade to an H100 SXM only by the §5 rule. |
| **Actor** | 1× 4090, **≥ 32 vCPU** | Interruptible where it saves ≥ 25%; otherwise on-demand | Stateless; reconnects after interruption. Runpod's fixed 6–9 vCPU per consumer GPU is **too few** for these. |
| **Eval** | 1× 4090, ≥ 32 vCPU | On-demand | Runs the GSPRT queue 24/7 (`02`) |
| **Burst** | 1× H100 SXM or a 4090 with ≥ 48 vCPU | Hourly | SL teachers (choose by featurization rate, §4.2); benchmarks |
| **Deploy** | 1× 4090, ≥ 16 vCPU | On-demand, reliable, with low RTT to riichi.dev | Phase 4 |

**Lanes (T2–T3):** actors are split into a **champion lane** and a **candidate lane**.
- The candidate trains on a **dedicated candidate-learner box**, budgeted in T2 and T3 (D39).
- The eval box is never borrowed, so evaluation capacity stays independent of training.
- See `05` §6.

## 3. Phase 0 benchmarks (days 1–5)

1. **Host benchmark.**
   - Rent 4–6 candidate Vast hosts (4090 and 5090, 32–48 vCPU, different CPU generations) for 2–3 h each.
   - Run 1v3 collection with the quick student.
   - Record hanchan/h, GPU utilization, CPU utilization, and the CPU model.
   - **Allowlist** hosts at ≥ 25K/h.
2. **Collection modes.** On the best host, measure:
   - 1v3 trainee-only (the Mortal-Policy default);
   - 4-seat self-play (RL-6);
   - evaluation with argmax on both sides.
3. **Bottleneck.** Is it libriichi on the CPU, inference on the GPU, or Python/IPC? Optimize only the named bottleneck (§6).
4. **Featurization rate for SL.** Measure the `GameplayLoader` samples/s per vCPU on the training data. This decides the burst GPU for the teachers (§4.2).
5. **CPU-only inference experiment.** *Deferred 2026-10-08 (D47): v4 actor boxes are already CPU-bound.*
   - Rent a GCP c3d-highcpu-180 spot instance (~$1.64/h) for a few hours.
   - Run libriichi with the student net on CPU (oneDNN, BF16 if available).
   - If a b6c128/b8c192 student reaches ≥ 20K hanchan/h on a 96–180-thread box, monthly CPU servers become a **cheaper actor tier**: OVH RISE-XL at 96 threads is $354/mo plus $354 setup, about $0.005/thread-h.
   - This is speculative; the benchmark decides.
6. **Learner capacity.**
   - Training samples/s on the learner's 4090 for the chosen student, including the KL-anchor forward pass and the critic.
   - Compare against the fleet's sample production. Trainee-only 1v3 produces roughly 150–200 samples per hanchan [rough]. At 3.6M hanchan/day that is about 6–8K samples/s; RL-6 multiplies it by about 2.5.
7. **Fleet soak (gate for T1; D39).** Run the leading engine end-to-end at T1 scale (learner + 3 actors + eval) for **≥ 48 h**, including:
   - real featurization;
   - the critic and KL-anchor forward passes;
   - the ordered-target extraction (`05` §3.3).

   **Pass criteria:**
   - replay-queue length bounded (no upward trend over the last 24 h);
   - policy lag p95 ≤ 2 parameter versions;
   - learner useful-update rate ≥ 90% of actor sample production, and actors blocked < 10% of wall-clock;
   - no box loss without automatic recovery within 30 min.

   Failing the soak moves the dates; it never shrinks test sizes (D39).

## 4. Benchmark matrix and specific decisions

### 4.1 Inference (actors and eval)

Candidate networks (obs v4, 1012×34) [calc]:

| Net | Params | MACs/inference | Role |
|---|---|---|---|
| b6c128 | 2.1M | 35M | Lower bound; candidate for CPU-only |
| b8c192 | 3.6M | 82M | Student candidate |
| b12c256 | 6.8M | 189M | Student candidate |
| b16c256 | 8.5M | 242M | Student candidate |
| b24c512 | 42.1M | 1,339M | Teacher |

**Modes:** eager FP32; AMP FP16/BF16; `torch.compile(mode="reduce-overhead")` with fixed batch buckets (256/512/1024/2048); Torch-TensorRT FP16.

**Batch:** 500 to 4000 concurrent games.

**Output:** hanchan/h for every net × mode × batch on the allowlisted host type. This table gives the *speed* side only. T-NI against the best single teacher qualifies the student candidates (`04` §6). The **actor size is chosen from RL learning curves** at two budgets (D31; `05` §2).

### 4.2 SL teacher hardware

- **Compute:** about 8–9 EFLOP for 1B samples at b24c512. That is roughly 30–40 h on a 4090, or 5–8 h on an H100 [calc, rough]. Either costs well under $100 per run.
- **The real constraint is CPU featurization.** H100 instances usually come with only 16–26 vCPU. Featurized samples are about 137 KB each at FP32, so they can't be cached for 1B samples; they are generated on the fly.
- **Rule:** if (instance vCPU × measured samples/s per vCPU) can't feed ≥ 80% of the GPU's training rate, use a **4090 with ≥ 48 vCPU** instead of an H100.
- Run the teachers on **secure-tier** providers only (D23).

## 5. Learner upgrade rule

Upgrade the learner from a 4090 to an H100 SXM (~$2.4–3.4/h; 2–3.5× faster for CNN training) **only if both** hold:
- actors are measurably blocked on server buffer overflow (`capacity` reached) for more than 20% of wall-clock;
- **and** the learner's GPU utilization is above 85%.

Cheaper alternatives, tried first:
- subsample harvested seats;
- use a smaller critic;
- move featurization to the actors (they submit features instead of mjai logs).

Fund the upgrade from the reserve, and record it in `08`.

## 6. Optimization checklist, in order of expected return

1. **Right-size the network.** Distill to the smallest non-inferior student (`04` §6). Inference cost varies about 40× across candidates.
2. **Fixed-shape batching plus CUDA graphs.** Small Conv1d nets are launch-bound. CUDA graphs report up to 5× on launch-bound sections (vendor claim).
3. **FP16/BF16 at inference.** TensorRT typically gives about 1.5× more (KataGo's experience), but not always.
4. **Fuse normalization at inference.** BatchNorm folds into the conv; GroupNorm (used in Mortal-Policy's online code) does not. Benchmark a BN student and adopt it only if RL stability passes T-NI.
5. **Cut the cost of opponent inference.** 3 of 4 seats run the champion in 1v3. Use the student as the opponent, and batch all seats together.
6. **Harvest more seats per game** (RL-6, `05`).
7. **Process layout.** Pin env workers to cores, use shared-memory observation transfer, and target ≥ 60% of wall-clock in collection.
8. **Python overhead.** Keep per-step logic in Rust; profile with `py-spy`. If Python takes more than 20%, move the hot loop.

## 7. Simulator decision: Mahjax

**Payoff is higher at fleet scale:**
- One TanyaoDojo report gives 21–58K env steps/s on a single Ada-generation GPU during real PPO training.
- At roughly 700–900 steps per hanchan [inf], that would be about 90–290K hanchan/h **per GPU**, 3–9× a libriichi actor box.
- If that holds with our network, it could replace most of the actor fleet, or triple the scale.

**Spike (2 days, Phase 0):**
1. **Rule parity.** Replay 10K libriichi-generated mjai logs through Mahjax. Compare the legal-action sets, agari scores and yaku, ryukyoku and final placements at every step. Pass on 100% agreement, or documented, acceptable rule differences.
2. **Throughput** on 1× 5090 and on 1× H100, first with a random policy, then with a JAX port of a b8c192-class network.
3. **Integration cost.** Porting the v4 observation encoder (1012 channels, including the single-player EV tables) to JAX, versus retraining the student on Mahjax's own observation.

**Adopt** (D12) only if parity passes, end-to-end throughput is at least 3× a libriichi box per dollar, and integration is ≤ **4 weeks**. The allowance was raised from 2 weeks under the larger budget. The work would run in parallel with Phase 1–2 on libriichi. Otherwise, re-survey quarterly.

## 8. Search throughput budget (preview of `06`)

- libriichi runs about 118 kyoku/s at batch 2000 [calc]. That is **throughput, not latency**.
- Single-request search latency is measured **request-to-response on the deploy box** (`06` §4). The expectation is about 1–3K truncated rollouts per 3 s.
- Expert iteration (S-4) is **unfunded by default** (D33b). If G1 approves its equal-cost test:
  - its budget is set in **GPU-hours**;
  - coverage is derived from the measured cost ratio R: search_share = fR / (1 − f + fR);
  - for example, R = 10 and 5% of decisions searched gives a 34.5% compute share.

## 9. Cost model

All prices as of 2026-10-01 (`01` §10).

| Item | Price | Planning use |
|---|---|---|
| 4090/5090 with ≥ 32 vCPU (Vast, on-demand) | $0.47–0.57/h; planning $0.55 | Every box role |
| Same, interruptible | Roughly 25–35% lower (bid-dependent) | Actors |
| H100 SXM | Median $3.40 on-demand; Vast median $2.40; Runpod Secure $2.69 | Burst only; learner upgrade from reserve |
| GCP c3d-highcpu-180 spot | $1.64/h (us-central1) | CPU-only inference benchmark |
| OVH RISE-XL (96T Turin) | $354/mo + $354 setup | CPU-actor tier, only if §3.5 passes |
| Object storage for checkpoints and logs | Small, under $50/mo | Always |

**Per million hanchan:**
- At 30K/h per box, 1M hanchan takes about 33 box-hours, roughly **$18** on-demand or about **$12–14** on interruptible.
- The ≈ 270–290M forecast is about **$3.5–5K** of actor time, inside the Phase 1–3 allocations in `00`. This is a forecast; tranches T2 and T3 are released by evidence (D30).

## 10. Deliverables and exit (Phase 0)

- Host allowlist with hanchan/h per host type.
- The inference matrix (§4.1).
- Featurization rate and the teacher-hardware decision (§4.2).
- CPU-only verdict: deferred (D47).
- Learner-capacity number.
- The fleet-soak report (§3.7), which is the gate for T1.
- Mahjax verdict (`08` D12).
- Fleet scripts: provision, VPN join, start client, health-check, teardown.
- A capacity and cost forecast through G1.
