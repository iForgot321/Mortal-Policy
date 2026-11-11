"""Pieces of the students' trainer (train_students.py; 04 §6, docs/design/distillation-draft.md, D66,
D67): the frozen teachers, the ensemble's targets, the distillation loss and the validation pass.
Kept apart so tests can use them. (Importing it needs MORTAL_CFG, as dataloader.py does.)

The ensemble π_ens is the mean of the teachers' masked softmax probabilities at T = 1; its heads'
targets are the teachers' mean outputs as probabilities (placement, tenpai, waits) and values
(kyoku-Δ). A distilled student's loss (04 §6, the draft):

    L = KL(π_ens ‖ π_s) + w_hard · CE(human action)
        + w_placement · MSE(placement) + w_kyoku_delta · MSE(kyoku-Δ) + w_aux · [MSE(tenpai) + MSE(waits)]

with the waits term weighted by the ensemble's tenpai probability of each opponent, as the teachers'
waits loss counted only tenpai opponents. D66's control trains on the labels instead
(`teacher.teacher_loss`, BC).
"""
import hashlib

import torch
from torch.utils.data import DataLoader

from checkpoint import load_checkpoint
from dataloader import FileDatasetsIter
from model import Brain, CategoricalPolicy, TeacherHeads
from teacher import teacher_loss

HEAD_KEYS = ('placement', 'kyoku_delta', 'tenpai', 'waits')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_frozen_teacher(path, device, sha256=None):
    """A teacher checkpoint with `mortal`, `policy_net` and `heads` at the top level (the
    `ema-from-state.pth` exports) as (brain, policy, heads), frozen and in eval mode. With `sha256`,
    the file must match it."""
    if sha256 is not None and (got := sha256_file(path)) != sha256:
        raise SystemExit(f'{path}: sha256 {got[:16]}… does not match the config\'s {sha256[:16]}…')
    st = load_checkpoint(path)
    if st.get('weights') != 'ema':
        raise SystemExit(f'{path} is not an EMA export (tools/secure/export_ema.py)')
    if st.get('heads') is None:
        raise SystemExit(f'{path} has no heads: the students distil them too')
    cfg = st['config']
    brain = Brain(version=cfg['control']['version'], conv_channels=cfg['resnet']['conv_channels'],
                  num_blocks=cfg['resnet']['num_blocks'], Norm=cfg['resnet'].get('norm', 'GN'))
    policy = CategoricalPolicy()
    heads = TeacherHeads(hidden=cfg['teacher']['hidden'], detach=cfg['teacher']['detach'])
    for m, k in ((brain, 'mortal'), (policy, 'policy_net'), (heads, 'heads')):
        m.load_state_dict(st[k])
        m.to(device).eval().requires_grad_(False)
    return brain, policy, heads


@torch.no_grad()
def ensemble_targets(teachers, obs, mask):
    """The ensemble's targets on a batch, fp32: `policy` (masked probabilities, 0 off the mask),
    `placement` (4 probabilities), `kyoku_delta` (u), `tenpai` (3 probabilities) and `waits`
    (3 × 34 probabilities). Runs under the caller's autocast."""
    acc = {}
    for brain, policy, heads in teachers:
        phi = brain(obs)
        out = heads(phi)
        parts = {
            'policy': policy.logits(phi, mask).float().softmax(-1),
            'placement': out['placement'].float().softmax(-1),
            'kyoku_delta': out['kyoku_delta'].float(),
            'tenpai': out['tenpai'].float().sigmoid(),
            'waits': out['waits'].float().sigmoid(),
        }
        for k, v in parts.items():
            acc[k] = acc[k] + v if k in acc else v
    return {k: v / len(teachers) for k, v in acc.items()}


def kl_to_ensemble(logits, mask, p_ens):
    """KL(π_ens ‖ π_s) per decision, exact over the legal mask. `logits` are the student's masked
    raw logits; illegal actions contribute exactly 0 and their -inf never reaches the arithmetic."""
    logq = logits.float().log_softmax(-1).masked_fill(~mask, 0.0)
    return (torch.special.xlogy(p_ens, p_ens) - p_ens * logq).sum(-1)


def distill_loss(logits, out, actions, mask, targets, cfg):
    """The distilled student's loss (the module's docstring) and its parts, detached. `logits`
    are the student's masked raw logits, `out` its TeacherHeads output, `targets` from
    `ensemble_targets`, `cfg` the [distill] weights."""
    kl = kl_to_ensemble(logits, mask, targets['policy']).mean()
    hard = -logits.float().log_softmax(-1).gather(1, actions[:, None]).squeeze(1).mean()
    placement = (out['placement'].float().softmax(-1) - targets['placement']).square().mean()
    kd = (out['kyoku_delta'].float() - targets['kyoku_delta']).square().mean()
    tenpai = (out['tenpai'].float().sigmoid() - targets['tenpai']).square().mean()
    w = targets['tenpai'][..., None].expand_as(targets['waits'])
    waits = (w * (out['waits'].float().sigmoid() - targets['waits']).square()).sum() / w.sum().clamp(min=1e-8)
    total = (kl + cfg['w_hard'] * hard + cfg['w_placement'] * placement + cfg['w_kyoku_delta'] * kd
             + cfg['w_aux'] * (tenpai + waits))
    parts = {'kl': kl, 'hard': hard, 'placement': placement, 'kyoku_delta': kd, 'tenpai': tenpai, 'waits': waits}
    return total, {k: v.detach() for k, v in parts.items()}


class Student:
    """One student: its networks, optimizer, scheduler, scaler and EMA, and which loss it trains on
    ('distill' or 'labels', D66's control)."""

    def __init__(self, name, kind, brain, policy, heads, optimizer, scheduler, scaler, ema, params):
        self.name, self.kind = name, kind
        self.brain, self.policy, self.heads = brain, policy, heads
        self.optimizer, self.scheduler, self.scaler, self.ema = optimizer, scheduler, scaler, ema
        self.params = params

    @property
    def modules(self):
        return {'mortal': self.brain, 'policy_net': self.policy, 'heads': self.heads}

    def forward(self, obs, mask):
        phi = self.brain(obs)
        return self.policy.logits(phi, mask).float(), self.heads(phi)


@torch.no_grad()
def validate_students(students, teachers, files, *, version, pts, masks, loss_cfg, device, batch_size,
                      enable_amp, file_batch_size=4):
    """Per student and per weights (raw, EMA): the human move's NLL and top-1, each head against the
    true labels (`teacher_loss`'s parts), the KL to π_ens and the argmax agreement with it, over
    every seat's decisions in `files`. Also the ensemble's own NLL and top-1, as the reference.
    Returns {'ensemble': {...}, name: {'raw': {...}, 'ema': {...}}}."""
    nets = [m for s in students for m in s.modules.values()]
    training = [m.training for m in nets]
    for m in nets:
        m.eval()
    vds = FileDatasetsIter(version=version, file_list=list(files), pts=pts, file_batch_size=file_batch_size,
                           masks=masks, compute_advantage=False, obs_fp16=True, teacher_labels=True)
    sums, n = {}, 0

    def add(key, k, v, count):
        d = sums.setdefault(key, {})
        d[k] = d.get(k, 0.0) + float(v) * count

    for obs, actions, mask, _, kd, rank, opp in DataLoader(vds, batch_size=batch_size, num_workers=0):
        obs, actions, mask = obs.to(device).float(), actions.to(device), mask.to(device)
        kd, rank, opp = kd.to(device), rank.to(device), opp.to(device)
        b = len(actions)
        with torch.autocast(device.type, enabled=enable_amp):
            tgt = ensemble_targets(teachers, obs, mask)
        p_ens = tgt['policy']
        add('ensemble', 'policy', -p_ens.gather(1, actions[:, None]).clamp(min=1e-12).log().mean(), b)
        add('ensemble', 'top1', (p_ens.argmax(-1) == actions).float().mean(), b)
        for tag in ('raw', 'ema'):
            if tag == 'ema':
                for s in students:
                    s.ema.swap()
            try:
                for s in students:
                    with torch.autocast(device.type, enabled=enable_amp):
                        logits, out = s.forward(obs, mask)
                    _, parts = teacher_loss(logits, actions, torch.ones_like(kd), out, kd, rank, opp, loss_cfg)
                    key = (s.name, tag)
                    for k, v in parts.items():
                        add(key, k, v, b)
                    add(key, 'top1', (logits.argmax(-1) == actions).float().mean(), b)
                    add(key, 'kl_ens', kl_to_ensemble(logits, mask, p_ens).mean(), b)
                    add(key, 'agree_ens', (logits.argmax(-1) == p_ens.argmax(-1)).float().mean(), b)
            finally:
                if tag == 'ema':
                    for s in students:
                        s.ema.swap()
        n += b
    for m, t in zip(nets, training):
        m.train(t)
    res = {'ensemble': {k: v / n for k, v in sums['ensemble'].items()}, 'samples': n}
    for s in students:
        res[s.name] = {tag: {k: v / n for k, v in sums[(s.name, tag)].items()} for tag in ('raw', 'ema')}
    return res
