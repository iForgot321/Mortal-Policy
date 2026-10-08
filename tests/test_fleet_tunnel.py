"""The D48 tunnel against a real sshd: the learner's restrictions (sshd Match block +
authorized_keys options, as fleet.py writes them), the actor-side checks fleet.py runs after a
join, and box_tunnel.sh carrying traffic and reconnecting.

Skipped unless FLEET_TEST_SSHD is an sshd command line (absolute path; it runs as this user on
127.0.0.1). The workstation has no sshd, so this uses Ubuntu 22.04's, the Vast image's version:
    apt-get download openssh-server libwrap0 && for d in *.deb; do dpkg-deb -x $d root; done
    FLEET_TEST_SSHD="env LD_LIBRARY_PATH=$PWD/root/usr/lib/x86_64-linux-gnu $PWD/root/usr/sbin/sshd" pytest tests/test_fleet_tunnel.py
"""
import getpass
import os
import shlex
import signal
import socket
import subprocess
import threading
import time

import pytest

from test_fleet import FLEET, fleet

SSHD = os.environ.get('FLEET_TEST_SSHD')
pytestmark = pytest.mark.skipif(not SSHD, reason='FLEET_TEST_SSHD is not set')


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def wait_for(cond, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.2)
    return False


def listening(port):
    with open('/proc/net/tcp') as f:
        return f'0100007F:{port:04X} 00000000:0000 0A' in f.read()


def echo_server(port):
    """server.py's stand-in on the learner: echoes one message per connection."""
    srv = socket.create_server(('127.0.0.1', port))

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            with conn:
                conn.sendall(conn.recv(1024))
    threading.Thread(target=serve, daemon=True).start()
    return srv


def roundtrip(port, payload=b'hello learner'):
    with socket.create_connection(('127.0.0.1', port), timeout=10) as c:
        c.sendall(payload)
        return c.recv(1024)


@pytest.fixture
def learner(tmp_path):
    """A user-mode sshd configured like a learner: the fleet Match block for this user, and an
    actor key installed with fleet.authorized_line."""
    user = getpass.getuser()
    server_port, ssh_port = free_port(), free_port()
    subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', tmp_path / 'host'], check=True)
    subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', tmp_path / 'actor'], check=True)
    comment = 'fleet-test-actor1-1'
    (tmp_path / 'authorized_keys').write_text(
        fleet.authorized_line((tmp_path / 'actor.pub').read_text(), server_port, comment) + '\n')
    (tmp_path / 'sshd_config').write_text(
        f'Port {ssh_port}\nListenAddress 127.0.0.1\nHostKey {tmp_path}/host\nPidFile {tmp_path}/sshd.pid\n'
        f'AuthorizedKeysFile {tmp_path}/authorized_keys\nStrictModes no\nUsePAM no\n'
        f'PasswordAuthentication no\nKbdInteractiveAuthentication no\n\n# fleet tunnels (D48)\n'
        + fleet.sshd_match_block(user, server_port))
    sshd = shlex.split(SSHD)
    eff = subprocess.run([*sshd, '-T', '-f', tmp_path / 'sshd_config', '-C', f'user={user},host=actor,addr=192.0.2.1'],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    for want in ('allowtcpforwarding local', f'permitopen 127.0.0.1:{server_port}', 'permittty no',
                 'forcecommand /bin/false'):  # what learner_sshd_script checks on the box
        assert want in eff
    proc = subprocess.Popen([*sshd, '-D', '-e', '-f', tmp_path / 'sshd_config'],
                            stdout=subprocess.DEVNULL, stderr=open(tmp_path / 'sshd.log', 'w'))
    assert wait_for(lambda: listening(ssh_port)), (tmp_path / 'sshd.log').read_text()
    blob = (tmp_path / 'host.pub').read_text().split()[1]
    (tmp_path / 'known_hosts').write_text(f'[127.0.0.1]:{ssh_port} ssh-ed25519 {blob}\n')
    box = {'ip': '127.0.0.1', 'ssh_port': ssh_port}
    ssh = (fleet.actor_ssh_command(box).replace(fleet.TUNNEL_KEY, str(tmp_path / 'actor'))
           .replace(fleet.TUNNEL_KNOWN_HOSTS, str(tmp_path / 'known_hosts'))
           .replace(f'{fleet.TUNNEL_USER}@', f'{user}@'))
    yield {'dir': tmp_path, 'user': user, 'server_port': server_port, 'ssh_port': ssh_port, 'ssh': ssh}
    proc.terminate()
    proc.wait(10)


def verify(learner):
    return subprocess.run(['bash', '-c', fleet.PRELUDE + fleet.tunnel_verify_script(learner['ssh'], learner['server_port'])],
                          capture_output=True, text=True, timeout=200)


def test_actor_key_can_only_reach_the_server_port(learner):
    p = verify(learner)  # server.py not up yet: the allowed forward gets "connect failed"
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.count('ok   ') == 4
    srv = echo_server(learner['server_port'])
    try:
        p = verify(learner)
        assert p.returncode == 0, p.stdout + p.stderr
    finally:
        srv.close()


def test_verify_catches_an_unrestricted_key(learner):
    """The checks must fail when the restrictions are missing (negative control)."""
    d = learner['dir']
    pub = (d / 'actor.pub').read_text().split()
    (d / 'authorized_keys').write_text(f'{pub[0]} {pub[1]} unrestricted\n')
    cfg = (d / 'sshd_config').read_text()
    (d / 'sshd_config').write_text(cfg.split('# fleet tunnels (D48)')[0])
    os.kill(int((d / 'sshd.pid').read_text()), signal.SIGHUP)
    time.sleep(1.5)
    p = verify(learner)
    assert p.returncode == 1
    assert p.stdout.count('FAIL') == 3, p.stdout  # a command runs, :22 forwards, -R is accepted


def test_wrong_host_key_is_refused(learner):
    d = learner['dir']
    subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', d / 'other'], check=True)
    (d / 'known_hosts').write_text(f'[127.0.0.1]:{learner["ssh_port"]} ssh-ed25519 '
                                   f'{(d / "other.pub").read_text().split()[1]}\n')
    p = subprocess.run(f'{learner["ssh"]} -W 127.0.0.1:{learner["server_port"]} </dev/null', shell=True,
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 255 and 'Host key verification failed' in p.stderr


def test_box_tunnel_carries_traffic_and_reconnects(learner):
    d = learner['dir']
    local = free_port()
    srv = echo_server(learner['server_port'])
    (d / 'tunnel.env').write_text(
        f'LEARNER_IP=127.0.0.1\nLEARNER_SSH_PORT={learner["ssh_port"]}\nTUNNEL_USER={learner["user"]}\n'
        f'LOCAL_PORT={local}\nREMOTE_PORT={learner["server_port"]}\nKEY={d}/actor\nKNOWN_HOSTS={d}/known_hosts\n')
    tunnel = subprocess.Popen(['bash', os.path.join(FLEET, 'box_tunnel.sh'), d / 'tunnel.env'],
                              stdout=open(d / 'tunnel.log', 'w'), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        assert wait_for(lambda: listening(local)), (d / 'tunnel.log').read_text()
        assert roundtrip(local) == b'hello learner'
        first = int((d / 'tunnel.pid').read_text())
        os.kill(first, signal.SIGTERM)  # a dropped connection
        assert wait_for(lambda: (d / 'tunnel.pid').read_text().strip() not in ('', str(first))
                        and listening(local), timeout=30), (d / 'tunnel.log').read_text()
        assert roundtrip(local, b'again') == b'again'
    finally:
        os.killpg(tunnel.pid, signal.SIGTERM)
        tunnel.wait(10)
        srv.close()
