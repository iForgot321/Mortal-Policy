import torch
import os
import secrets
import socket
import struct
import time
from typing import *
from io import BytesIO
from functools import partial
from tqdm.auto import tqdm as orig_tqdm
from config import config

tqdm = partial(orig_tqdm, unit='batch', dynamic_ncols=True, ascii=True)

def parameter_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def filtered_trimmed_lines(lines):
    return filter(lambda l: l, map(lambda l: l.strip(), lines))

def iter_grads(parameters, take=False):
    for p in parameters:
        if p.grad is not None:
            if take:
                # Set to zero instead of None to preserve the layout and make it
                # easier to assign back later
                yield p.grad.clone()
                p.grad.zero_()
            else:
                yield p.grad

# server.py has no authentication of its own, and actors reach it through SSH tunnels whose
# connections arrive from loopback like the trainer's (D48). So the trainer's two requests,
# submit_param and drain, carry a token that server.py writes at each start into a file only
# the learner's own user can read; an actor's requests (get_param, submit_replay) need none.
def server_token_path(cfg=None):
    """server.py's token file: beside its buffer dir, in the run's directory."""
    server = (cfg or config)['online']['server']
    return os.path.join(os.path.dirname(os.path.abspath(server['buffer_dir'])), 'server.token')

def new_server_token(token_path):
    """A fresh token, written owner-only and atomically (server.py, at start)."""
    token = secrets.token_hex(32)
    os.makedirs(os.path.dirname(token_path), exist_ok=True)
    tmp = f'{token_path}.{os.getpid()}.tmp'
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(token + '\n')
    os.replace(tmp, token_path)
    return token

def server_token():
    """The running server.py's token, read at every request: a restarted server has a new one."""
    with open(server_token_path()) as f:
        return f.read().strip()

class ServerRefused(Exception):
    pass

def drain():
    return drain_msg()['drain_dir']

def drain_msg():
    """The server's drain reply: `drain_dir`, `behaviour_drain_dir` (D46
    sidecars), `param_versions` (the version each drained log was played
    with) and the server's current `param_version`."""
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    while True:
        with socket.socket() as conn:
            conn.connect(remote)
            send_msg(conn, {'type': 'drain', 'token': server_token()})
            msg = recv_msg(conn)
        if msg.get('status') == 'denied':
            raise ServerRefused(f'server.py refused drain: the token in {server_token_path()} is not its own')
        if msg['count'] == 0:
            time.sleep(5)
            continue
        return msg

def submit_param(mortal, dqn, is_idle=False):
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    with socket.socket() as conn:
        conn.connect(remote)
        send_msg(conn, {
            'type': 'submit_param',
            'mortal': mortal.state_dict(),
            'dqn': dqn.state_dict(),
            'is_idle': is_idle,
            'token': server_token(),
        })
        # the reply says the parameters were taken, not dropped
        rsp = recv_msg(conn)
    if rsp.get('status') != 'ok':
        raise ServerRefused(f'server.py refused submit_param ({rsp.get("status")}): '
                            f'the token in {server_token_path()} is not its own')

def send_msg(conn: socket.socket, msg, packed=False):
    if packed:
        tx = msg
    else:
        buf = BytesIO()
        torch.save(msg, buf)
        tx = buf.getbuffer()
    conn.sendall(struct.pack('<Q', len(tx)))
    conn.sendall(tx)

def recv_msg(conn: socket.socket, map_location=torch.device('cpu')):
    rx = recv_binary(conn, 8)
    (size,) = struct.unpack('<Q', rx)
    rx = recv_binary(conn, size)
    # weights_only: messages hold only dicts, strs, numbers, bytes and tensors, and a
    # peer (an actor host, D48) must not be able to run code by sending a pickle
    return torch.load(BytesIO(rx), weights_only=True, map_location=map_location)

def recv_binary(conn: socket.socket, size):
    assert size > 0
    ret = bytearray(size)
    buf = memoryview(ret)

    while len(buf) > 0:
        n = conn.recv_into(buf)
        if n == 0:
            raise UnexpectedEOF()
        buf = buf[n:]
    return bytes(ret)

class UnexpectedEOF(Exception):
    def __init__(self):
        super().__init__('unexpected EOF')
