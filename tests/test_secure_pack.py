"""tools/secure/pack.py: a shard decrypts back to the tar it hashed, and the plaintext side is deterministic."""
import importlib.util
import io
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'pack', Path(__file__).resolve().parents[1] / 'tools' / 'secure' / 'pack.py')
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)

pytestmark = pytest.mark.skipif(shutil.which('gpg') is None, reason='needs gpg')


@pytest.fixture
def keyed(tmp_path, monkeypatch):
    key = tmp_path / 'cfg' / 'bundle.key'
    monkeypatch.setattr(pack, 'KEY', key)
    monkeypatch.setattr(pack, 'GNUPGHOME', tmp_path / 'cfg' / 'gnupg')
    pack.cmd_key(None)
    yield key
    subprocess.run(['gpgconf', '--homedir', str(tmp_path / 'cfg' / 'gnupg'), '--kill', 'gpg-agent'],
                   capture_output=True)


def test_shard_round_trip(tmp_path, keyed):
    src = tmp_path / 'src'
    (src / '2015' / '01').mkdir(parents=True)
    files = {'2015/01/a.json.gz': b'alpha', '2015/01/b.json.gz': b'beta' * 1000}
    for rel, data in files.items():
        (src / rel).write_bytes(data)
    rels = [(f'mjai/{r}', r) for r in sorted(files)]
    out1, out2 = tmp_path / 'o1', tmp_path / 'o2'
    out1.mkdir()
    out2.mkdir()
    s1 = pack.write_shard('shard-000.tar.gpg', src, rels, out1)
    s2 = pack.write_shard('shard-000.tar.gpg', src, rels, out2)
    assert s1['files'] == 2
    assert s1['plain_sha256'] == s2['plain_sha256']  # same data, same plaintext tar
    assert s1['cipher_sha256'] != s2['cipher_sha256']  # fresh salt and IV every time
    assert b'beta' not in (out1 / 'shard-000.tar.gpg').read_bytes()

    plain = subprocess.run(pack.GPG + ['--passphrase-file', str(keyed), '--decrypt', str(out1 / 'shard-000.tar.gpg')],
                           capture_output=True, check=True, env=pack.gpg_env()).stdout
    with tarfile.open(fileobj=io.BytesIO(plain)) as tar:
        got = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    assert got == {f'mjai/{r}': d for r, d in files.items()}


def test_key_is_private_and_never_replaced(keyed):
    assert keyed.stat().st_mode & 0o077 == 0
    before = keyed.read_text()
    pack.cmd_key(None)
    assert keyed.read_text() == before
