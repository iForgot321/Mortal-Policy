#!/usr/bin/env python3
"""Write a train_teacher.py final checkpoint's EMA weights as a plain checkpoint the lab loads
(`mortal` and `policy_net` at the top level, as lab/arena.py reads them), read-only.

    python tools/secure/export_ema.py workdir/teacher/teacher-bc-s1/final.pth workdir/teacher/teacher-bc-s1/ema.pth
"""
import hashlib
import os
import sys

import torch


def main(src, dst):
    st = torch.load(src, weights_only=False, map_location='cpu')
    ema = st['ema_weights']
    with open(src, 'rb') as f:
        src_sha = hashlib.sha256(f.read()).hexdigest()
    out = {'mortal': ema['mortal'], 'policy_net': ema['policy_net'], 'heads': ema['heads'],
           'config': st['config'], 'steps': st['steps'], 'presentations': st['presentations'],
           'weights': 'ema', 'source': {'path': src, 'sha256': src_sha}}
    if os.path.exists(dst):
        sys.exit(f'{dst} exists')
    torch.save(out, dst)
    os.chmod(dst, 0o444)
    with open(dst, 'rb') as f:
        print(f'{dst}: EMA weights of {src} (step {st["steps"]:,}), sha256 {hashlib.sha256(f.read()).hexdigest()}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
