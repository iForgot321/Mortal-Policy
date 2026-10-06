"""Experiment registry, ablation table and nonce ledger (02 §9).

All three are append-only JSON-lines files under lab/registry/ and are the
project's permanent record, so they live in git. Inconclusive and failed rows
stay: they are evidence too.
"""
import datetime
import hashlib
import json
import os
import subprocess

from . import LAB_DIR, REGISTRY_DIR, ROOT, load_seeds

RUNS = os.path.join(REGISTRY_DIR, 'runs.jsonl')
NONCES = os.path.join(REGISTRY_DIR, 'nonces.jsonl')
TABLE = os.path.join(REGISTRY_DIR, 'ablation.md')


def now():
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec='seconds')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def git_state():
    def run(*args):
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {'commit': run('rev-parse', 'HEAD'), 'dirty': bool(run('status', '--porcelain'))}


def lab_config_hash():
    """Hash of the lab's frozen configuration (test definitions, seed families)."""
    h = hashlib.sha256()
    for name in ('lab.toml', 'seeds.toml', 'population.toml'):
        with open(os.path.join(LAB_DIR, name), 'rb') as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def allocate_nonces(family, count, run_id, purpose, block=None, path=NONCES):
    """Reserve [start, start + count) in a family's nonce range.

    Family A may reuse a named fixed block (tracking runs). Families B and C
    always get a fresh range, so a confirmation range is never reused (D36).
    """
    families = load_seeds()
    fam = families[family]
    ledger = _read_jsonl(path)
    if block is not None:
        if family != 'A':
            raise ValueError('named reusable blocks are allowed in family A only')
        for row in ledger:
            if row['family'] == 'A' and row.get('block') == block:
                if row['count'] < count:
                    raise ValueError(f'block {block} has only {row["count"]} sets')
                _append_jsonl(path, {**row, 'run_id': run_id, 'purpose': purpose,
                                     'reused': True, 'time': now()})
                return row['start'], fam['key']
    used = [r['start'] + r['count'] for r in ledger if r['family'] == family and not r.get('reused')]
    start = max(used, default=fam['nonce_start'])
    if start + count > fam['nonce_end']:
        raise RuntimeError(f'family {family} nonce range exhausted')
    _append_jsonl(path, {'family': family, 'start': start, 'count': count, 'run_id': run_id,
                         'purpose': purpose, 'block': block, 'time': now()})
    return start, fam['key']


def new_run_id(kind):
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    return f'{stamp}-{kind.replace("/", "")}'


def record_run(row, path=RUNS):
    _append_jsonl(path, row)
    render_table()


def _fmt(x, nd=4):
    return '' if x is None else f'{x:+.{nd}f}' if isinstance(x, float) else str(x)


def render_table(runs_path=RUNS, out_path=TABLE):
    """Regenerate the markdown ablation table from the run registry."""
    header = ('| Run | Mode | Description | Test | Family (nonces) | Sets | Δu | 95% CI | LLR | '
              'Verdict | Δ avg rank | avg pt | Calibrated | Commit |\n'
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n')
    lines = []
    for r in _read_jsonl(runs_path):
        res = r.get('result', {})
        sec = r.get('secondary', {})
        ci = res.get('ci')
        avg_rank = sec.get('avg_rank')
        lines.append('| {} | {} | {} | {} | {} [{}, +{}) | {} | {} | {} | {} | {} | {} | {} | {} | {}{} |\n'.format(
            r['run_id'], r.get('mode', ''), r.get('description', '').replace('|', '/'), r['test'],
            r['family'], r['nonce_start'], r['nonce_count'], res.get('sets', ''),
            _fmt(res.get('mean')), '' if not ci else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]',
            _fmt(res.get('llr'), 2), r.get('verdict', ''),
            '' if avg_rank is None else f'{avg_rank - 2.5:+.4f}', _fmt(sec.get('avg_pt'), 2),
            'yes' if r.get('calibrated') else 'NO', r['git']['commit'][:8],
            '+dirty' if r['git']['dirty'] else ''))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('# Ablation table\n\nGenerated from `runs.jsonl` by `python -m lab table`; do not edit by hand. '
                'Δ avg rank is relative to 2.5. Rows with Calibrated = NO used uncalibrated '
                'boundaries and do not count toward any rung (02 §4.1).\n\n')
        f.write(header)
        f.writelines(lines)
