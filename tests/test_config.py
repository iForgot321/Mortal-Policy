"""The trainers pickle the config into every checkpoint (`mortal/config.py`)."""
import os
import pickle
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_a_config_with_inline_tables_pickles(tmp_path):
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text("[policy.schedules]\nt0 = 50\nlr = { v0 = 2e-5, p = 0.5 }\n"
                   "[x]\nrows = [{ a = 1 }, { a = 2 }]\n")
    code = ("import pickle, config; c = config.config; pickle.dumps(c); "
            "assert type(c['policy']['schedules']['lr']) is dict and type(c['x']['rows'][1]) is dict; "
            "print(c['policy']['schedules']['lr']['v0'])")
    out = subprocess.run([sys.executable, '-c', code], cwd=os.path.join(ROOT, 'mortal'), capture_output=True,
                         text=True, env={**os.environ, 'MORTAL_CFG': str(cfg)}, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == '2e-05'


def test_every_test_that_picks_a_config_prefers_the_box_copy():
    """The first test module to import `config` fixes it for the whole pytest process, and box setup
    runs the suite (tools/fleet/setup_box.sh). A default naming experiments/configs/ alone points at
    workstation paths that don't exist on a box, and fails some later test there (2026-11-11)."""
    import re
    bad = []
    for name in sorted(os.listdir(os.path.join(ROOT, 'tests'))):
        if not name.endswith('.py'):
            continue
        with open(os.path.join(ROOT, 'tests', name), encoding='utf-8') as f:
            for n, line in enumerate(f, 1):
                if re.search(r"setdefault\(\s*'MORTAL_CFG'", line) and 'box_cfg' not in line:
                    bad.append(f'{name}:{n}')
    assert not bad, f"prefer workdir/box/configs/<name>.toml when it exists, as tests/test_teacher.py does: {bad}"
