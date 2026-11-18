"""K2's KL anchor (05 §5.1, D59): a penalty c_KL · KL(π_θ ‖ π_SL) that keeps the online policy
near the supervised one. The diagnosis (`experiments/rl-diagnosis.md`) found P0 and P1 at a
reverse KL of ≈ 0.4 nats/decision by p25, with nothing but PPO's per-update clip to hold them.

π_SL is a frozen copy of a policy checkpoint on the learner, by default the run's frozen
opponent (`baseline.train.state_file`: the initialization). The KL is exact over the legal mask
(`ppo.reverse_kl`) and averaged over decisions with more than one legal action, as
`experiments/rl_drift.py` measures it.

Steps here are the run clock (`schedules.py`): learner steps since the run's first launch, the
clock K2's LR, clip and entropy schedules use too.

Config (`[policy.kl_anchor]`), two coefficient rules:

    enabled = true
    state_file = '...'      # π_SL; default baseline.train.state_file
    mode = 'adaptive'       # or 'fixed'
    coef = 0.1              # c_KL at step 0
    # 'fixed': c(t) = coef · (1 + t / decay_steps) ** -coef_decay   (Ataraxos-style power law)
    coef_decay = 0.3
    decay_steps = 100_000
    # 'adaptive': every adapt_every steps, the mean KL since the last update is compared with
    # target(t) = target · (1 + t / decay_steps) ** target_growth; above target · tolerance the
    # coefficient is multiplied by factor, below target / tolerance divided by it (PPO's
    # adaptive-KL rule), within [coef_min, coef_max]. It is not divided before the KL first
    # reaches target / tolerance or `hold_windows` updates have passed: the run starts at KL 0,
    # and halving the coefficient while the policy hasn't moved yet would leave it at coef_min
    # just as the KL overshoots (the local smoke, `experiments/k2-retry.md`). The hold ends, so
    # a starting coefficient strong enough to keep the KL below the band still relaxes.
    target = 0.02           # nats/decision at the start
    target_growth = 0.0     # > 0 lets the target rise over the run: the anchor's decay
    adapt_every = 400
    hold_windows = 5
    tolerance = 1.5
    factor = 2.0
    coef_min = 0.01         # in the local smoke, 0.1 held the KL near 0.007: the 0.02 target
    coef_max = 10.0         # needs ≈ 0.03, and a floor far below that switches the anchor off

Re-anchoring in the run (D64, D68: `experiments/size-curves.md`). The fleet asks for it by writing
`reanchor.json` beside the trainer's state file (`request_path`; `fleet.py reanchor`):

    {"label": "p25", "file": "<a checkpoint on this machine>", "sha256": "<its sha256>", ...}

The trainer polls it at start and at every save (`poll`). A request for a checkpoint that isn't
the current anchor makes that checkpoint the anchor from then on, once its sha256 matches. The
adaptive coefficient carries over; the window's KL, measured against the old anchor, is dropped,
and the start-of-run hold applies again, since the KL starts again from ≈ 0. The new anchor's
weights are kept in the trainer's state (`state_dict`), so a restarted or replaced learner resumes
on it whatever files it has; the config's `state_file` stays the run's first anchor (stage 1) and
is still checked against the checkpoint's.
"""
import hashlib
import json
import os

import torch

from checkpoint import load_checkpoint
from model import Brain, CategoricalPolicy

MODES = ('fixed', 'adaptive')
REQUEST_NAME = 'reanchor.json'


def request_path(state_file):
    """Where the fleet asks the trainer that owns `state_file` to re-anchor."""
    return os.path.join(os.path.dirname(state_file), REQUEST_NAME)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def arch_of(state):
    """The network a policy checkpoint holds, as Brain's arguments."""
    cfg = state['config']
    return {'version': cfg['control'].get('version', 1), 'num_blocks': cfg['resnet']['num_blocks'],
            'conv_channels': cfg['resnet']['conv_channels'], 'Norm': cfg['resnet'].get('norm', 'GN')}


def build_policy(arch, mortal, policy_net, device):
    """(brain, policy) from their weights, frozen and in eval mode."""
    brain = Brain(**arch)
    brain.load_state_dict(mortal)
    policy = CategoricalPolicy()
    policy.load_state_dict(policy_net)
    for m in (brain, policy):
        m.to(device).eval().requires_grad_(False)
    return brain, policy


def load_policy(state_file, device):
    """A policy checkpoint's (brain, policy), frozen and in eval mode, and its arch."""
    state = load_checkpoint(state_file)
    arch = arch_of(state)
    return (*build_policy(arch, state['mortal'], state['policy_net'], device), arch)


def same_file(a, b):
    """Whether two checkpoint paths name the same repo file, compared from `workdir/` on; paths
    outside a workdir are compared whole."""
    def rel(p):
        i = p.find('workdir/')
        return p[i:] if i >= 0 else None
    ra, rb = rel(a), rel(b)
    if ra is None or rb is None:
        return a == b
    return ra == rb


class KLAnchor:
    def __init__(self, cfg, *, default_state_file, device, version):
        self.mode = cfg.get('mode', 'adaptive')
        if self.mode not in MODES:
            raise ValueError(f'kl_anchor.mode must be one of {MODES}, got {self.mode!r}')
        self.state_file = cfg.get('state_file') or default_state_file
        self.coef0 = float(cfg.get('coef', 0.1))
        self.coef_decay = float(cfg.get('coef_decay', 0.3))
        self.decay_steps = float(cfg.get('decay_steps', 100_000))
        self.target0 = float(cfg.get('target', 0.02))
        self.target_growth = float(cfg.get('target_growth', 0.0))
        self.adapt_every = int(cfg.get('adapt_every', 400))
        self.hold_windows = int(cfg.get('hold_windows', 5))
        self.tolerance = float(cfg.get('tolerance', 1.5))
        self.factor = float(cfg.get('factor', 2.0))
        self.coef_min = float(cfg.get('coef_min', 0.01))
        self.coef_max = float(cfg.get('coef_max', 10.0))
        if self.decay_steps <= 0 or self.tolerance <= 1 or self.factor <= 1 or not 0 < self.coef_min <= self.coef_max:
            raise ValueError('kl_anchor: decay_steps > 0, tolerance > 1, factor > 1, 0 < coef_min <= coef_max')
        self.device, self.version = device, version
        self.brain, self.policy, self.arch = load_policy(self.state_file, device)
        if self.brain.version != version:
            raise ValueError(f'kl_anchor: {self.state_file} is obs version {self.brain.version}, the run {version}')
        # the run's first anchor (stage 1), then one record per re-anchor
        self.sha256 = sha256_file(self.state_file)
        self.stages = []
        # requests that failed in this process, so a bad one isn't retried at every save
        self.failed = set()
        # the adaptive coefficient, and the KL seen since its last update
        self.adaptive_coef = min(max(self.coef0, self.coef_min), self.coef_max)
        self.kl_sum = 0.0
        self.kl_n = 0
        self.reached = False
        self.windows = 0

    @torch.no_grad()
    def logits(self, obs, masks):
        """π_SL's masked raw logits on a batch (no grad)."""
        return self.policy.logits(self.brain(obs), masks)

    def target(self, step):
        return self.target0 * (1 + step / self.decay_steps) ** self.target_growth

    def coef(self, step):
        if self.mode == 'fixed':
            return self.coef0 * (1 + step / self.decay_steps) ** -self.coef_decay
        return self.adaptive_coef

    def observe(self, kl, step):
        """Record a batch's mean KL at learner step `step` (counted from 1); in 'adaptive' mode,
        every adapt_every steps, steer the coefficient toward target(step)."""
        # a tensor stays on its device until an update needs it (no sync per step)
        self.kl_sum = self.kl_sum + (kl.detach() if torch.is_tensor(kl) else float(kl))
        self.kl_n += 1
        if self.mode != 'adaptive' or step % self.adapt_every != 0:
            return
        mean = float(self.kl_sum) / self.kl_n
        self.kl_sum, self.kl_n = 0.0, 0
        target = self.target(step)
        self.windows += 1
        self.reached = self.reached or mean >= target / self.tolerance
        if mean > target * self.tolerance:
            self.adaptive_coef = min(self.adaptive_coef * self.factor, self.coef_max)
        elif mean < target / self.tolerance and (self.reached or self.windows > self.hold_windows):
            self.adaptive_coef = max(self.adaptive_coef / self.factor, self.coef_min)

    @property
    def stage(self):
        return len(self.stages) + 1

    @property
    def anchor_sha256(self):
        """The current anchor's sha256: the last re-anchor's file, else the config's."""
        return self.stages[-1]['sha256'] if self.stages else self.sha256

    def reanchor(self, file, sha256, *, label=None, step=None, t=None):
        """Make the policy checkpoint at `file` the anchor from here on. Its sha256 must be
        `sha256`. Returns the stage's record."""
        got = sha256_file(file)
        if got != sha256:
            raise ValueError(f'kl_anchor: {file} has sha256 {got[:12]}, the request {sha256[:12]}')
        state = load_checkpoint(file, map_location='cpu')
        arch = arch_of(state)
        if arch['version'] != self.version:
            raise ValueError(f'kl_anchor: {file} is obs version {arch["version"]}, the run {self.version}')
        self.brain, self.policy = build_policy(arch, state['mortal'], state['policy_net'], self.device)
        self.arch = arch
        # the window so far measured the KL to the old anchor; the policy starts again at ≈ 0
        self.kl_sum, self.kl_n = 0.0, 0
        self.reached, self.windows = False, 0
        rec = {'stage': self.stage + 1, 'label': label, 'file': file, 'sha256': sha256,
               'steps': step, 't': t, 'coef': self.coef(t or 0)}
        self.stages.append(rec)
        return rec

    def poll(self, path, step, t):
        """Act on the fleet's request at `path`, if any: re-anchor when it names a checkpoint other
        than the current anchor. Returns a record to log ({'reanchor': label, ...} or
        {'reanchor_failed': label, 'error': ...}) when something happened, else None. A request
        for an earlier stage's anchor is ignored."""
        try:
            with open(path) as f:
                req = json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as e:
            return {'reanchor_failed': None, 'error': f'unreadable request {path}: {e!r}'}
        sha = req.get('sha256')
        done = {self.sha256, *(s['sha256'] for s in self.stages)}
        if sha in done or (sha, req.get('file')) in self.failed:
            return None
        try:
            rec = self.reanchor(req['file'], sha, label=req.get('label'), step=step, t=t)
        except (KeyError, OSError, ValueError, RuntimeError) as e:
            self.failed.add((sha, req.get('file')))
            return {'reanchor_failed': req.get('label'), 'sha256': sha, 'error': repr(e)[:300]}
        return {'reanchor': rec['label'], **rec}

    def state_dict(self):
        state = {'mode': self.mode, 'state_file': self.state_file, 'adaptive_coef': self.adaptive_coef,
                 'kl_sum': float(self.kl_sum), 'kl_n': self.kl_n, 'reached': self.reached,
                 'windows': self.windows, 'sha256': self.sha256, 'stages': list(self.stages)}
        if self.stages:
            # the current anchor travels with the trainer's state (the module docstring)
            state['arch'] = self.arch
            state['weights'] = {'mortal': self.brain.state_dict(), 'policy_net': self.policy.state_dict()}
        return state

    def load_state_dict(self, state):
        # compared from workdir/ on: a box writes /workspace/Mortal-Policy/..., the workstation
        # /home/.../Mortal-Policy/..., for the same file
        if not same_file(state["state_file"], self.state_file):
            raise ValueError(f"kl_anchor: the checkpoint anchors to {state['state_file']}, the config to {self.state_file}")
        if state.get('sha256', self.sha256) != self.sha256:
            raise ValueError(f'kl_anchor: {self.state_file} has changed since the checkpoint was saved '
                             f"(sha256 {self.sha256[:12]}, the checkpoint's {state['sha256'][:12]})")
        stages = state.get('stages') or []
        if stages:
            arch = state['arch']
            if arch['version'] != self.version:
                raise ValueError(f'kl_anchor: the checkpoint\'s anchor is obs version {arch["version"]}, the run {self.version}')
            w = state['weights']
            self.brain, self.policy = build_policy(arch, w['mortal'], w['policy_net'], self.device)
            self.arch = arch
        self.stages = list(stages)
        self.adaptive_coef = state['adaptive_coef']
        self.kl_sum = state['kl_sum']
        self.kl_n = state['kl_n']
        self.reached = state['reached']
        self.windows = state['windows']
