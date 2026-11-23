"""Which decisions the curves' late stages change (exploratory, outside D68's pre-registration;
`experiments/size-curves.md`, "Stage-to-stage drift"). On the students' states, per run:

1. **close calls or reversals:** for each decision whose top move changed in stage 3 (p50 -> p75)
   or stage 4 (p75 -> `full`), the earlier snapshot's probability gap between the old and the new
   move, against the same gap (top move minus runner-up) over all decisions;
2. **back or on:** of the decisions changed in both stages, how many returned to p50's move and
   how many went to a third one; and whether stage 4's changes moved toward the student's move
   or away from it.

    python experiments/curves_flips.py            (from the repo root; seconds on the GPU)

True FP32 (no TF32, no AMP), so a near-tie isn't flipped by numerics. Writes
workdir/lab/curves-diag/flips.json.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RUNS = {'curves-b12-s1': 'b12c256', 'curves-b16-s1': 'b16c256'}
STUDENTS = os.path.join(ROOT, 'workdir', 'secure', 'runs', 'students-s1', 'box', 'students-s1')
OUT = os.path.join(ROOT, 'workdir', 'lab', 'curves-diag', 'flips.json')
BINS = (0.05, 0.2, 0.5)  # probability gaps: < 5 points is a close call, > 50 a reversal


def bins(gaps):
    """Shares of `gaps` in [0, .05), [.05, .2), [.2, .5), [.5, 1]."""
    edges = (0.0, *BINS, np.inf)
    n = max(len(gaps), 1)
    return {f'{lo:g}-{hi:g}' if hi != np.inf else f'>={lo:g}': float(((gaps >= lo) & (gaps < hi)).sum() / n)
            for lo, hi in zip(edges[:-1], edges[1:])}


def top2_gap(p):
    s = np.sort(p, -1)
    return s[:, -1] - s[:, -2]


def stage(p_old, p_new, kinds):
    """One stage's changes: the rate, the old snapshot's gap between the two moves, by kind."""
    a0, a1 = p_old.argmax(-1), p_new.argmax(-1)
    changed = a0 != a1
    idx = np.nonzero(changed)[0]
    gap = p_old[idx, a0[idx]] - p_old[idx, a1[idx]]  # how much the old snapshot preferred its move
    gap_new = p_new[idx, a1[idx]] - p_new[idx, a0[idx]]  # how much the new one prefers its own
    return changed, {
        'changed': float(changed.mean()), 'n': int(changed.sum()),
        'old_gap_median': float(np.median(gap)) if len(idx) else None,
        'old_gap_bins': bins(gap),
        'new_gap_median': float(np.median(gap_new)) if len(idx) else None,
        'new_gap_bins': bins(gap_new),
        'by_kind': {k: int((kinds[idx] == k).sum()) for k in np.unique(kinds[idx])},
    }


def analyse(p, kinds):
    """`p`: {'student', 'p50', 'p75', 'full'} -> (N, A) probabilities over the legal mask;
    `kinds`: the decision kind of p50's top move, (N,)."""
    a = {k: v.argmax(-1) for k, v in p.items()}
    c3, s3 = stage(p['p50'], p['p75'], kinds)
    c4, s4 = stage(p['p75'], p['full'], kinds)
    both = c3 & c4
    back = both & (a['full'] == a['p50'])
    toward = c4 & (a['full'] == a['student'])  # stage 4 restored the student's move
    away = c4 & (a['p75'] == a['student'])  # stage 4 left it
    return {
        'decisions': int(len(kinds)),
        'all_decisions_top2_gap': {k: {'median': float(np.median(top2_gap(p[k]))), 'bins': bins(top2_gap(p[k]))}
                                   for k in ('p50', 'p75')},
        'stage3': s3, 'stage4': s4,
        'p50_to_full_changed': float((a['p50'] != a['full']).mean()),
        'both_stages': {'n': int(both.sum()), 'share_of_stage4': float(both.sum() / max(c4.sum(), 1)),
                        'back_to_p50': int(back.sum()), 'to_a_third_move': int((both & ~back).sum())},
        'stage4_vs_student': {'toward': int(toward.sum()), 'away': int(away.sum()),
                              'neither': int((c4 & ~toward & ~away).sum())},
    }


def probabilities(states, specs):
    """{label: (N, A) probabilities} on `states`' decisions, featurized as the first spec's
    observation version, in true FP32."""
    import torch
    import rl_drift
    from lab.arena import load_engine
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    version = load_engine(next(iter(specs.values())), enable_amp=False, name='x').version
    obs, masks, _ = rl_drift.load_states(states, version=version)
    out = {}
    for label, spec in specs.items():
        e = load_engine(spec, enable_amp=False, name='x')
        if e.version != version:
            raise ValueError(f'{spec} is obs v{e.version}, not v{version}')
        chunks = []
        with torch.inference_mode():
            for s in range(0, len(obs), 2048):
                o = obs[s:s + 2048].to(e.device, torch.float32)
                m = masks[s:s + 2048].to(e.device)
                chunks.append(torch.softmax(e.dqn.logits(e.brain(o), m).float(), -1).cpu())
        out[label] = torch.cat(chunks).double().numpy()
    return out


def main():
    import rl_drift
    res = {}
    for run, net in RUNS.items():
        snaps = os.path.join(ROOT, 'workdir', 'fleet', 'runs', run, 'snapshots')
        specs = {'student': os.path.join(STUDENTS, f'ema-{net}.pth'),
                 **{k: os.path.join(snaps, f'{k}.pth') for k in ('p50', 'p75', 'full')}}
        p = probabilities(os.path.join(ROOT, 'workdir', 'diag', f'credit-{net}'), specs)
        kinds = np.array([rl_drift.kind(x) for x in p['p50'].argmax(-1)])
        res[run] = analyse(p, kinds)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
