"""D31's curves: the pre-registered reading (`experiments/curves_reading.py`) and the launch
helper's quote logic (`experiments/curves_common.sh`), before any run exists."""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

import curves_reading as cr  # noqa: E402

C12, C6 = 12 * cr.PLANNING_USD_PER_BOX_DAY, 6 * cr.PLANNING_USD_PER_BOX_DAY


def fig(full12, full16, p12, p16):
    """Δu lists per size at 12 and 6 box-days, in Δu per dollar."""
    return {'full': {'b12c256': [x / C12 for x in full12], 'b16c256': [x / C12 for x in full16]},
            'p50': {'b12c256': [x / C6 for x in p12], 'b16c256': [x / C6 for x in p16]}}


TWO = {'b12c256': 2, 'b16c256': 2}


def test_the_larger_net_needs_a_lead_past_the_spread():
    # b16 ahead by 0.012 Δu with seeds 0.004 apart: S is the noise floor (0.0043), L well past 1.5 S
    # (and at 6 box-days by 0.010)
    d = cr.decide(fig([0.020, 0.024], [0.032, 0.036], [0.010, 0.012], [0.020, 0.022]), TWO)
    assert d['pick'] == 'b16c256' and not d['close'] and not d['third_seed_needed']
    x = d['readings']['full']
    assert x['L'] == pytest.approx(0.012 / C12) and x['S'] == pytest.approx(0.0043 / C12)


def test_the_cheaper_net_wins_ties_and_leads_that_dont_clear_the_spread():
    assert cr.decide(fig([0.03, 0.03], [0.03, 0.03], [0.01, 0.01], [0.01, 0.01]), TWO)['pick'] == 'b12c256'
    # a lead of 0.006 against a seed range of 0.010: no
    d = cr.decide(fig([0.020, 0.030], [0.031, 0.031], [0.01, 0.01], [0.01, 0.01]), TWO)
    assert d['pick'] == 'b12c256' and d['readings']['full']['S'] == pytest.approx(0.010 / C12)
    # b12 ahead: never close, the cheaper net
    d = cr.decide(fig([0.040, 0.040], [0.020, 0.020], [0.02, 0.02], [0.01, 0.01]), TWO)
    assert d['pick'] == 'b12c256' and not d['close']


def test_a_close_reading_asks_for_the_third_seed_only_with_two():
    # L = 0.005, S = the 0.0043 floor: 0.5 S < L < 1.5 S at 12 box-days
    f = fig([0.030, 0.030], [0.035, 0.035], [0.01, 0.01], [0.01, 0.01])
    d = cr.decide(f, TWO)
    assert d['close'] and d['third_seed_needed'] and d['pick'] == 'b16c256'  # the rule's pick, pending batch 3
    assert not cr.decide(f, {'b12c256': 3, 'b16c256': 3})['third_seed_needed']
    # close at 6 box-days only counts too ("on either reading")
    d = cr.decide(fig([0.03, 0.03], [0.05, 0.05], [0.010, 0.010], [0.0145, 0.0145]), TWO)
    assert d['readings']['p50']['close'] and not d['readings']['full']['close'] and d['third_seed_needed']


def test_a_crossover_is_reported_and_read_by_rule_1():
    d = cr.decide(fig([0.020, 0.020], [0.030, 0.030], [0.012, 0.012], [0.008, 0.008]), TWO)
    assert d['crossover'] and d['pick'] == 'b16c256'


def test_the_interval_uses_the_seeds_or_the_measurement_noise_whichever_is_larger():
    v = {'b12c256': [0.020 / C12, 0.024 / C12], 'b16c256': [0.032 / C12, 0.036 / C12]}
    se = {s: [0.0021 / C12] * 2 for s in v}
    got = cr.interval(v, se)
    seeds_var = (0.004 / C12) ** 2 / 2  # each size's sample variance of two values 0.004 apart
    assert got['se'] == pytest.approx(((seeds_var / 2) * 2) ** 0.5) and got['df'] == 2
    assert got['ci95'][0] < got['L'] < got['ci95'][1] and got['ci95'][0] < got['ci95_normal'][0]
    same = {s: [0.03 / C12, 0.03 / C12] for s in v}  # seeds agree: the measurement noise remains
    assert cr.interval(same, se)['se'] == pytest.approx(((0.0021 / C12) ** 2 / 2 * 2) ** 0.5)


def test_the_report_from_registry_rows(tmp_path, monkeypatch):
    reg = tmp_path / 'runs.jsonl'
    monkeypatch.setattr(cr, 'RUNS_JSONL', str(reg))
    monkeypatch.setattr(cr, 'FLEET', str(tmp_path / 'fleet'))
    monkeypatch.setattr(cr, 'OUT', str(tmp_path / 'reading.json'))
    monkeypatch.setattr(cr, 'billed', lambda run: 160.0)
    du = {('curves-b12-s1', 'full'): 0.020, ('curves-b12-s2', 'full'): 0.024,
          ('curves-b16-s1', 'full'): 0.032, ('curves-b16-s2', 'full'): 0.036}
    rows = []
    for run in cr.DEFAULT_RUNS:
        d = tmp_path / 'fleet' / run
        d.mkdir(parents=True)
        (d / 'snapshots.jsonl').write_text(''.join(json.dumps({'label': lab, 'box_hours': bh}) + '\n'
                                                   for lab, bh in (('p50', 144.1), ('full', 288.2))))
        for label in ('p50', 'full'):
            mean = du.get((run, label), 0.01)
            rows.append({'test': 'measure', 'verdict': 'measured', 'run_id': f'{run}-{label}',
                         'description': f'curves: {run} {label} at 3x student',
                         'result': {'sets': 40000, 'mean': mean, 'sd': 0.42, 'ci': [mean - 0.004, mean + 0.004]}})
    reg.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert cr.main([]) == 0
    rep = json.loads((tmp_path / 'reading.json').read_text())
    assert rep['decision']['pick'] == 'b16c256' and rep['missing'] == []
    assert rep['runs']['curves-b12-s1']['full']['box_days'] == pytest.approx(288.2 / 24, abs=1e-3)
    assert rep['runs']['curves-b12-s1']['heldout_bc_s1'] is None  # not measured: reported as missing data
    reg.write_text(''.join(json.dumps(r) + '\n' for r in rows[:-1]))
    assert cr.main([]) == 1  # a missing measure: no reading
    assert json.loads((tmp_path / 'reading.json').read_text())['missing'] == ['curves-b16-s2 full']


# ---------------------------------------------------------------- the launch helper

FAKE_FLEET = r'''#!/bin/bash
# a stand-in for fleet.py: `up` prints the quote in $QUOTE_FILE, and with --yes records the call
echo "$*" >> "$CALLS"
if [[ $1 == up ]]; then
    cat "$QUOTE_FILE"
    if [[ " $* " == *" --yes "* ]]; then mkdir -p "workdir/fleet/runs/$3"; echo '{"boxes": {"learner": {}}}' > "workdir/fleet/runs/$3/fleet.json"; fi
fi
'''


def run_rent(tmp_path, quote, **env):
    fake = tmp_path / 'fake_fleet.sh'
    fake.write_text(FAKE_FLEET)
    (tmp_path / 'quote.txt').write_text(quote)
    calls = tmp_path / 'calls.txt'
    calls.write_text('')
    script = f'''source {ROOT}/experiments/curves_common.sh
cd {tmp_path}
RUN=curves-test; HOURS=96; UP="--run curves-test --config c.toml"
rent; echo "rc=$?"'''
    e = {**os.environ, 'F': f'bash {fake}', 'QUOTE_FILE': str(tmp_path / 'quote.txt'), 'CALLS': str(calls),
         'QUOTE_SLEEP': '0', 'QUOTE_HOURS': '1', **env}
    e.pop('MAX_DOLLARS', None)
    p = subprocess.run(['bash', '-c', script], capture_output=True, text=True, env=e, timeout=60)
    return p.stdout + p.stderr, calls.read_text().splitlines()


def test_rent_takes_a_quote_within_the_limits_and_sets_the_stop(tmp_path):
    out, calls = run_rent(tmp_path, '3 allowlisted machines; the 3 best:\n≈ $2.40/h for the fleet\n')
    assert 'rc=0' in out
    assert calls[0] == 'up --run curves-test --config c.toml --max-dph 1.00'  # the quote, per-box limit set
    assert calls[1].endswith('--max-dph 1.00 --yes')
    assert (tmp_path / 'workdir/fleet/runs/curves-test/max-dollars').read_text().strip() == '288'  # 1.25 × 2.40 × 96


def test_rent_refuses_a_quote_above_the_fleet_limit_and_a_missing_one(tmp_path):
    out, calls = run_rent(tmp_path, '≈ $3.10/h for the fleet\n')
    assert 'rc=1' in out and 'not rented' in out and not any('--yes' in c for c in calls)
    assert len(calls) == 4  # re-quoted every QUOTE_SLEEP for QUOTE_HOURS (4 per hour)
    out, calls = run_rent(tmp_path, 'error: only 1 allowlisted machines within the price limit for 3 boxes\n')
    assert 'rc=1' in out and not any('--yes' in c for c in calls)
    out, calls = run_rent(tmp_path, '≈ $3.10/h for the fleet\n', MAX_FLEET_DPH='3.20', MAX_DPH='1.30')
    assert 'rc=0' in out and calls[-1].endswith('--max-dph 1.30 --yes')


def test_the_curves_scripts_parse():
    for name in ('curves_common.sh', 'curves_b0.sh', 'curves_launch.sh', 'curves_batch.sh', 'curves_measure_chain.sh',
                 'curves_chain.sh'):
        subprocess.run(['bash', '-n', os.path.join(ROOT, 'experiments', name)], check=True)
