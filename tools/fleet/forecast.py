"""Capacity and cost forecast through Gate G1 (03 §10; docs/design/cost-forecast-g1.md).

Every input is a named constant below, with where it came from. Re-run when prices or throughput
are re-measured (01 §10 says to re-verify prices before each phase):
    python tools/fleet/forecast.py > workdir/forecast.md
Prints, per scenario: Phase 1 and Phase 2 line items, totals against the 00 budget, generated
hanchan, and the learner-capacity check.
"""

# ---- measured inputs (Phase 0) ----
HANCHAN_PER_ACTOR_H = 31_700        # P0 actor, 1 process, 800-game batches, 4090 + EPYC 9684X (fleet-host-benchmarks)
MULTIPROC_GAIN = 1.5                # 2-3 actor processes per box: 1.48-1.62x in eval mode; P0 itself not measured
LEARNER_SAMPLES_S = 3_750           # P0 learner, v4, 4090 box, 2 workers, featurization-bound (fleet-host-benchmarks)
SAMPLES_PER_HANCHAN = 159           # trainee seat only (fleet-host-benchmarks)
SPENT_SO_FAR = 5.33                 # benchmark rentals to 2026-10-08 (STATUS)

# SL: train window 2015-01 -> 2025-06 = train_E + train_O (splits.md), less ~1.4% Q6/Q8 masks
TRAIN_DECISIONS = 1_187_754_829 * 0.986
PRESENTATIONS = TRAIN_DECISIONS * 2  # 04 §5 default: 1 epoch with m<->p augmentation
TEACHER_RUNS = 3                    # AWR, BC, second seed of the winner (04 §5)
STUDENT_CANDIDATES = 3              # b8c192, b12c256, b16c256 (04 §6)

# ---- schedule (00): Phase 1 = weeks 4-9, Phase 2 = weeks 10-18 ----
WEEK_H = 168
PHASE1_WEEKS = 6
TRANCHE_WEEKS = 3
ARM_SEEDS = 6                       # P0 x1, P1 x2, V x2, H x1 (05 §2.1)
ARM_HOURS = 6 * 24                  # 3 boxes x 6 days per arm-seed (05 §2.2)
ACTOR_SIZE_RUNS, ACTOR_SIZE_HOURS = 2, 4 * 24   # 2 students x 12 box-days on a 3-box layout (D31)
SOAK_HOURS = 48                     # learner + 3 actors (+ the eval box already running) (03 §3.7)
TUNING_BOX_DAYS = 4 * 1             # <= 1 box-day of sweeps per arm (05 §2.2)

PLAN = {'phase0': 600, 'phase1': 3_200, 'phase2': 5_200}   # 00 phase map (D42)

# ---- scenarios: $/box-hour by role, featurization rate for SL, and levers ----
SCENARIOS = {
    'plan': dict(
        note='00 as written: $0.55 for every box; teachers and students at a mid featurization rate',
        actor=0.55, learner=0.55, eval=0.55, secure=0.55, feat_rate=8_000,
        multiproc=False, students_share_stream=False, t1_actors=3),
    'measured': dict(
        note='measured 2026-10-06 medians: actors 4090 on-demand $0.50, learner/eval 5090 $0.67, '
             'secure-tier 5090 $0.90 (range 0.67-1.30)',
        actor=0.50, learner=0.67, eval=0.67, secure=0.90, feat_rate=8_000,
        multiproc=False, students_share_stream=False, t1_actors=3),
    'levers': dict(
        note='measured prices plus levers: interruptible actors ($0.40), 4090 learner/eval ($0.50), '
             '2-3 actor processes per box, T1 with 2 actor boxes, the 3 students distilled from one '
             'featurized stream, secure-tier at its low end ($0.67)',
        actor=0.40, learner=0.50, eval=0.50, secure=0.67, feat_rate=8_000,
        multiproc=True, students_share_stream=True, t1_actors=2),
    'pessimistic': dict(
        note='5090 everywhere ($0.67), secure-tier $1.30, SL featurization only 5K samples/s',
        actor=0.67, learner=0.67, eval=0.67, secure=1.30, feat_rate=5_000,
        multiproc=False, students_share_stream=False, t1_actors=3),
}


def forecast(s):
    sl_run_h = PRESENTATIONS / s['feat_rate'] / 3600     # SL runs are featurization-bound on v4
    student_runs = 1 if s['students_share_stream'] else STUDENT_CANDIDATES
    rate = HANCHAN_PER_ACTOR_H * (MULTIPROC_GAIN if s['multiproc'] else 1)
    box3 = s['learner'] + 2 * s['actor']                 # 1 learner + 2 actors

    p1 = [
        ('Eval box 24/7 (also L1 smoke deploy, oracle search pilot)', PHASE1_WEEKS * WEEK_H * s['eval']),
        (f'Teachers: {TEACHER_RUNS} runs x {sl_run_h:.0f} h, secure-tier', TEACHER_RUNS * sl_run_h * s['secure']),
        (f'Teacher-output cache: 1 pass x {sl_run_h:.0f} h, secure-tier', sl_run_h * s['secure']),
        (f'Students: {student_runs} featurization pass(es) x {sl_run_h:.0f} h, secure-tier', student_runs * sl_run_h * s['secure']),
        (f'Engine arms: {ARM_SEEDS} arm-seeds x 3 boxes x {ARM_HOURS} h', ARM_SEEDS * ARM_HOURS * box3),
        (f'Actor-size curves: {ACTOR_SIZE_RUNS} x 3 boxes x {ACTOR_SIZE_HOURS} h', ACTOR_SIZE_RUNS * ACTOR_SIZE_HOURS * box3),
        (f'Fleet soak: learner + 3 actors x {SOAK_HOURS} h', SOAK_HOURS * (s['learner'] + 3 * s['actor'])),
        (f'Arm tuning allowance: {TUNING_BOX_DAYS} box-days', TUNING_BOX_DAYS * 24 * s['actor']),
        ('Quick sweeps, quick CQL model: workstation', 0.0),
        ('Object storage, ~1.5 months', 75.0),
    ]
    tranche_h = TRANCHE_WEEKS * WEEK_H
    t1 = tranche_h * (s['eval'] + s['learner'] + s['t1_actors'] * s['actor'])
    t23 = tranche_h * (s['eval'] + 2 * s['learner'] + 4 * s['actor'])
    p2 = [
        (f"T1: eval + learner + {s['t1_actors']} actors x {tranche_h} h", t1),
        (f'T2: eval + learner + candidate learner + 4 actors x {tranche_h} h (released by evidence)', t23),
        (f'T3: same as T2 (released by evidence)', t23),
        ('Object storage, ~2 months', 100.0),
    ]
    learner_cap = LEARNER_SAMPLES_S * 3600 / SAMPLES_PER_HANCHAN
    # (generated, consumed): one learner consumes at most learner_cap hanchan/h of trainee samples
    def seg(actors, hours, learners=1):
        prod = actors * rate
        return prod * hours, min(prod, learners * learner_cap) * hours
    hanchan = {
        'engine arms': tuple(ARM_SEEDS * x for x in seg(2, ARM_HOURS)),
        'actor-size curves': tuple(ACTOR_SIZE_RUNS * x for x in seg(2, ACTOR_SIZE_HOURS)),
        'fleet soak': seg(3, SOAK_HOURS),
        'T1': seg(s['t1_actors'], tranche_h),
        'T2 + T3 (two lanes, a learner each)': tuple(2 * x for x in seg(4, tranche_h, learners=2)),
    }
    load = {
        'arm (2 actors : 1 learner)': 2 * rate / learner_cap,
        f"T1 ({s['t1_actors']} actors : 1 learner)": s['t1_actors'] * rate / learner_cap,
        'T2/T3 lane (2 actors : 1 learner)': 2 * rate / learner_cap,
    }
    return p1, p2, hanchan, learner_cap, load


def main():
    print('# Forecast through G1 (generated by tools/fleet/forecast.py)\n')
    print(f'Inputs: SL presentations {PRESENTATIONS / 1e9:.2f}B (1 epoch x2 augmentation of '
          f'{TRAIN_DECISIONS / 1e9:.2f}B usable decisions); actor {HANCHAN_PER_ACTOR_H:,} hanchan/h per box '
          f'(x{MULTIPROC_GAIN} with several processes); learner {LEARNER_SAMPLES_S:,} samples/s; '
          f'{SAMPLES_PER_HANCHAN} trainee samples per hanchan. Spent so far ${SPENT_SO_FAR:.2f}.\n')
    summary = []
    for name, s in SCENARIOS.items():
        p1, p2, hanchan, cap, load = forecast(s)
        t1_total, t2_total = sum(c for _, c in p1), sum(c for _, c in p2)
        p2_t1_only = p2[0][1] + 50.0
        print(f'## Scenario: {name}\n\n{s["note"]}.\n')
        print('| Phase 1 item | $ |\n|---|---|')
        for item, c in p1:
            print(f'| {item} | {c:,.0f} |')
        print(f'| **Phase 1 total** (plan ${PLAN["phase1"]:,}) | **{t1_total:,.0f}** |\n')
        print('| Phase 2 item | $ |\n|---|---|')
        for item, c in p2:
            print(f'| {item} | {c:,.0f} |')
        print(f'| **Phase 2 total, all tranches** (plan ${PLAN["phase2"]:,}) | **{t2_total:,.0f}** |\n')
        print('| Hanchan (M) | Generated | Consumed by the learner |\n|---|---|---|')
        for k, (g, c) in hanchan.items():
            print(f'| {k} | {g / 1e6:.0f} | {c / 1e6:.0f} |')
        gen, con = (sum(v[i] for v in hanchan.values()) for i in (0, 1))
        print(f'| **through G1** | **{gen / 1e6:.0f}** | **{con / 1e6:.0f}** |\n')
        print(f'Learner capacity ≈ {cap / 1e3:.0f}K hanchan/h of trainee samples. Actor production as a share of it:')
        for k, v in load.items():
            print(f'- {k}: {v:.0%}' + ('  **(learner-bound)**' if v > 1 else ''))
        print()
        summary.append((name, t1_total, t2_total, p2_t1_only, gen, con))

    plan_total = PLAN['phase0'] + PLAN['phase1'] + PLAN['phase2']
    print('## Summary through G1\n')
    print(f'| Scenario | Phase 1 | Phase 2 (T1-T3) | Phase 0-2 total (plan ${plan_total:,}) | If only T1 is released | Hanchan generated / consumed (M) |')
    print('|---|---|---|---|---|---|')
    for name, a, b, t1o, gen, con in summary:
        total = SPENT_SO_FAR + 10 + a + b   # + ~$10 of Phase 0 fleet-script tests
        print(f'| {name} | {a:,.0f} | {b:,.0f} | {total:,.0f} | {SPENT_SO_FAR + 10 + a + t1o:,.0f} | {gen / 1e6:.0f} / {con / 1e6:.0f} |')


if __name__ == '__main__':
    main()
