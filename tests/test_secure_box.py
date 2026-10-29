"""tools/secure/secure.py: `up` without --yes only quotes (no POST), and an out-of-stock flavor is refused."""
import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'secure', Path(__file__).resolve().parents[1] / 'tools' / 'secure' / 'secure.py')
secure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(secure)


def test_up_without_yes_rents_nothing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(secure, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(secure, 'LEDGER', tmp_path / 'ledger.jsonl')
    monkeypatch.setattr(secure, 'api', lambda method, path, body=None: calls.append((method, path)) or {})
    monkeypatch.setattr(secure, 'flavor_price', lambda f: (1.0, {'gpu': 'L40', 'gpu_count': 1, 'cpu': 28,
                                                                 'ram': 58.0, 'stock_available': True}))
    monkeypatch.setattr(secure, 'public_ipv4', lambda: '192.0.2.1')
    with pytest.raises(SystemExit, match='--yes'):
        secure.main(['up', 'r1', '--hours', '2'])
    assert all(m == 'GET' for m, _ in calls)
    assert not (tmp_path / 'runs' / 'r1' / 'state.json').exists()
    assert not (tmp_path / 'ledger.jsonl').exists()


def test_out_of_stock_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(secure, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(secure, 'api', lambda *a, **k: pytest.fail('no API call expected'))
    monkeypatch.setattr(secure, 'flavor_price', lambda f: (1.35, {'gpu': 'A100', 'gpu_count': 1, 'cpu': 28,
                                                                  'ram': 120.0, 'stock_available': False}))
    monkeypatch.setattr(secure, 'public_ipv4', lambda: '192.0.2.1')
    with pytest.raises(SystemExit, match='out of stock'):
        secure.main(['up', 'r2', '--hours', '2', '--yes'])
