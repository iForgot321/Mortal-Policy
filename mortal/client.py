import prelude

import logging
import random
import socket
import torch
import numpy as np
import time
import gc
from os import path
from model import Brain, CategoricalPolicy
from player import TrainPlayer
from common import send_msg, recv_msg, UnexpectedEOF
from config import config

CONNECT_TIMEOUT = 10
IO_TIMEOUT = 300
MAX_BACKOFF = 60

def request(remote, msg, map_location=None, expect_reply=True):
    # One request/response with the server, retried with capped exponential
    # backoff and jitter on network errors, so a stateless actor survives
    # learner restarts and preemptions (03 §2).
    delay = 1.0
    while True:
        try:
            with socket.create_connection(remote, timeout=CONNECT_TIMEOUT) as conn:
                conn.settimeout(IO_TIMEOUT)
                send_msg(conn, msg)
                return recv_msg(conn, map_location=map_location) if expect_reply else None
        except (OSError, UnexpectedEOF) as e:
            logging.warning(f'server {remote} unreachable ({e!r}); retrying in ~{delay:.0f}s')
            time.sleep(delay * (0.5 + random.random()))
            delay = min(delay * 2, MAX_BACKOFF)

def main():
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    device = torch.device(config['control']['device'])
    version = config['control']['version']
    num_blocks = config['resnet']['num_blocks']
    conv_channels = config['resnet']['conv_channels']

    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels,Norm = "GN").to(device).eval()
    dqn = CategoricalPolicy().to(device).eval()
    if config['online']['enable_compile']:
        mortal.compile()
        dqn.compile()

    train_player = TrainPlayer()
    param_version = -1

    pts = np.array([90, 45, 0, -135])
    history_window = config['online']['history_window']
    history = []

    while True:
        while True:
            msg = {
                'type': 'get_param',
                'param_version': param_version,
            }
            rsp = request(remote, msg, map_location=device)
            if rsp['status'] == 'ok':
                param_version = rsp['param_version']
                break
            time.sleep(3)
        mortal.load_state_dict(rsp['mortal'])
        dqn.load_state_dict(rsp['dqn'])
        logging.info('param has been updated')

        rankings, file_list = train_player.train_play(mortal, dqn, device)
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = rankings @ pts / rankings.sum()

        history.append(np.array(rankings))
        if len(history) > history_window:
            del history[0]
        sum_rankings = np.sum(history, axis=0)
        ma_avg_rank = sum_rankings @ np.arange(1, 5) / sum_rankings.sum()
        ma_avg_pt = sum_rankings @ pts / sum_rankings.sum()

        logging.info(f'trainee rankings: {rankings} ({avg_rank:.6}, {avg_pt:.6}pt)')
        logging.info(f'last {len(history)} sessions: {sum_rankings} ({ma_avg_rank:.6}, {ma_avg_pt:.6}pt)')

        logs = {}
        for filename in file_list:
            with open(filename, 'rb') as f:
                logs[path.basename(filename)] = f.read()

        request(remote, {
            'type': 'submit_replay',
            'logs': logs,
            'param_version': param_version,
        }, expect_reply=False)
        logging.info('logs have been submitted')
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
