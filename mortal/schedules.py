"""K2's schedules (05 §5.1, D59): the policy LR, PPO's clip ε and the entropy weight, annealed
together on power laws (Ataraxos-style) over the run clock.

The run clock t counts learner steps since this run's first launch, not the checkpoint's global
`steps`: a run initialized from a B0 snapshot starts at ≈ 245K global steps, where a decay
counted from 0 would already be well along. The origin is saved in the checkpoint
(`schedule_origin`), so a restarted trainer keeps the same clock. The KL anchor's coefficient
and target (`kl_anchor.py`) run on the same clock.

Config (`[policy.schedules]`); a name left out keeps its constant from the config (`[policy]`'s
clip_ratio and entropy_weight, `[optim.scheduler]` for the LR):

    t0 = 100_000                                  # shared time scale, in learner steps
    lr = { v0 = 2e-5, p = 0.3, lo = 4e-6 }        # value(t) = clip(v0 · (1 + t/t0)^-p, lo, hi)
    clip_ratio = { v0 = 0.2, p = 0.2, lo = 0.1 }
    entropy_weight = { v0 = 1e-3, p = 0.5 }       # each may set its own t0
"""
NAMES = ('lr', 'clip_ratio', 'entropy_weight')


class PowerLaw:
    """value(t) = clip(v0 · (1 + t / t0) ** -p, lo, hi). p > 0 decays, p < 0 grows, p = 0 is
    constant."""

    def __init__(self, v0, p=0.0, t0=100_000, lo=None, hi=None):
        self.v0, self.p, self.t0 = float(v0), float(p), float(t0)
        self.lo = None if lo is None else float(lo)
        self.hi = None if hi is None else float(hi)
        if self.t0 <= 0:
            raise ValueError(f't0 must be > 0, got {t0}')
        if self.lo is not None and self.hi is not None and self.lo > self.hi:
            raise ValueError(f'lo {lo} > hi {hi}')

    def __call__(self, t):
        v = self.v0 * (1 + max(t, 0) / self.t0) ** -self.p
        if self.lo is not None:
            v = max(v, self.lo)
        if self.hi is not None:
            v = min(v, self.hi)
        return v

    def __repr__(self):
        return f'PowerLaw(v0={self.v0:g}, p={self.p:g}, t0={self.t0:g}, lo={self.lo}, hi={self.hi})'


def build(cfg, constants):
    """{name: PowerLaw} for the names `cfg` schedules; the rest get constant ones from
    `constants`, and 'lr' gets None when unscheduled (the existing LR scheduler keeps it)."""
    unknown = set(cfg) - set(NAMES) - {'t0'}
    if unknown:
        raise ValueError(f'unknown schedules {sorted(unknown)}; known: {NAMES}')
    t0 = cfg.get('t0', 100_000)
    out = {}
    for name in NAMES:
        if name in cfg:
            spec = dict(cfg[name])
            spec.setdefault('t0', t0)
            out[name] = PowerLaw(**spec)
        else:
            out[name] = None if name == 'lr' else PowerLaw(constants[name])
    return out


def origin(state, steps):
    """The run clock's origin: the saved one, or `steps` on a run's first launch."""
    return state.get('schedule_origin', steps) if state is not None else steps
