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
