"""tools/secure/hyperstack.py: the key file's permissions, and that the client only ever GETs."""
import importlib.util
import os
import urllib.request
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'hyperstack', Path(__file__).resolve().parents[1] / 'tools' / 'secure' / 'hyperstack.py')
hs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hs)


def test_key_file_must_be_private(tmp_path, monkeypatch):
    monkeypatch.delenv('HYPERSTACK_API_KEY', raising=False)
    key = tmp_path / 'api_key'
    key.write_text('secret\n')
    monkeypatch.setattr(hs, 'KEY_FILE', key)
    os.chmod(key, 0o644)
    with pytest.raises(SystemExit, match='chmod 600'):
        hs.api_key()
    os.chmod(key, 0o600)
    assert hs.api_key() == 'secret'


def test_env_key_wins(monkeypatch, tmp_path):
    monkeypatch.setenv('HYPERSTACK_API_KEY', ' from-env ')
    monkeypatch.setattr(hs, 'KEY_FILE', tmp_path / 'missing')
    assert hs.api_key() == 'from-env'


def test_every_request_is_a_get(monkeypatch):
    seen = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, *a):
            return b'{"data": []}'

    def fake_urlopen(req, timeout):
        seen.append((req.get_method(), req.full_url, req.get_header('Api_key')))
        return Resp()

    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    for path, _ in hs.COMMANDS.values():
        hs.get(path, key='k')
    assert {m for m, _, _ in seen} == {'GET'}
    assert all(url.startswith(hs.API + '/') and key == 'k' for _, url, key in seen)
    with pytest.raises(ValueError):
        hs.get('core/flavors', key='k')
