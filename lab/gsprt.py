"""Sequential and fixed-N tests on duplicate seed-set scores (02 §4).

The unit of analysis is the seed-set: the challenger's mean utility over the 4
seat rotations of one seed. Everything here is pure numpy so the live runner
and the calibration simulator (calibrate.py) apply the identical procedure.
"""
from dataclasses import dataclass

import numpy as np

CONTINUE, PASS, FAIL, INCONCLUSIVE = 0, 1, 2, 3
STATUS_NAMES = {CONTINUE: 'continue', PASS: 'pass', FAIL: 'fail', INCONCLUSIVE: 'inconclusive'}
# Sample variances at or below this count as zero: no verdict (D36).
ZERO_VAR = 1e-12


@dataclass(frozen=True)
class TestSpec:
    name: str
    mu0: float
    mu1: float
    cap: int
    lower: float
    upper: float


@dataclass(frozen=True)
class Schedule:
    min_sets: int
    look_every: int
    floor_frac: float
    sigma_set: float

    @property
    def var_floor(self):
        return self.floor_frac * self.sigma_set ** 2

    def looks(self, cap):
        """Set counts at which the test looks, ending exactly at the cap."""
        pts = list(range(self.min_sets, cap, self.look_every))
        return np.asarray(pts + [cap], dtype=np.int64)


def specs_from_config(cfg):
    return {
        name: TestSpec(name, t['mu0'], t['mu1'], int(t['cap']), t['lower'], t['upper'])
        for name, t in cfg['gsprt']['tests'].items()
    }


def schedule_from_config(cfg, sigma_set=None):
    g = cfg['gsprt']
    return Schedule(
        min_sets=int(g['min_sets']),
        look_every=int(g['look_every']),
        floor_frac=float(g['floor_frac']),
        sigma_set=float(sigma_set if sigma_set is not None else g['sigma_set']),
    )


def llr(n, mean, var, mu0, mu1):
    """Gaussian GSPRT log-likelihood ratio (Fishtest LLR_alt2 form)."""
    return n / (2 * var) * ((mean - mu0) ** 2 - (mean - mu1) ** 2)


def evaluate(n, s1, s2, spec, sched, lower=None, upper=None):
    """Status and LLR at one look, vectorized over tests.

    n is the (scalar) number of sets seen; s1 and s2 are arrays of the sums of
    set scores and of their squares. Returns (status, llr) arrays.
    """
    lower = spec.lower if lower is None else lower
    upper = spec.upper if upper is None else upper
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    mean = s1 / n
    var = np.maximum((s2 - s1 * mean) / (n - 1), 0.0)
    zero = var <= ZERO_VAR
    value = llr(n, mean, np.maximum(var, sched.var_floor), spec.mu0, spec.mu1)
    value = np.where(zero, 0.0, value)

    status = np.full(s1.shape, CONTINUE, dtype=np.int8)
    if n >= sched.min_sets:
        status[(value >= upper) & ~zero] = PASS
        status[(value <= lower) & ~zero] = FAIL
    if n >= spec.cap:
        status[status == CONTINUE] = INCONCLUSIVE
    return status, value


class SequentialTest:
    """One live GSPRT. Feed it set scores in order; it looks only at the
    scheduled set counts, whatever the chunking of the input."""

    def __init__(self, spec, sched):
        self.spec = spec
        self.sched = sched
        self.scores = []
        self.looks = sched.looks(spec.cap)
        self.look_idx = 0
        self.status = CONTINUE
        self.llr = 0.0
        self.n_decided = None

    def sets_until_next_look(self):
        if self.status != CONTINUE:
            return 0
        return int(self.looks[self.look_idx]) - len(self.scores)

    def add(self, set_scores):
        self.scores.extend(float(x) for x in set_scores)
        while self.status == CONTINUE and self.look_idx < len(self.looks) \
                and self.looks[self.look_idx] <= len(self.scores):
            n = int(self.looks[self.look_idx])
            x = np.asarray(self.scores[:n])
            status, value = evaluate(n, x.sum(), np.square(x).sum(), self.spec, self.sched)
            self.status, self.llr = int(status), float(value)
            if self.status != CONTINUE:
                self.n_decided = n
            self.look_idx += 1
        return self.status

    def summary(self):
        n = self.n_decided or len(self.scores)
        x = np.asarray(self.scores[:n])
        return {
            'test': self.spec.name,
            'status': STATUS_NAMES[self.status],
            'sets': n,
            'mean': float(x.mean()) if n else None,
            'sd': float(x.std(ddof=1)) if n > 1 else None,
            'llr': self.llr,
            'bounds': [self.spec.mu0, self.spec.mu1],
            'boundaries': [self.spec.lower, self.spec.upper],
            'sigma_set_assumed': self.sched.sigma_set,
            'sets_played': len(self.scores),
        }


def set_scores(seeds, seats, ranks, util):
    """Per-set challenger utility from per-game arena records.

    Records come 4 per seed, the challenger at seat i % 4 in game i. Returns
    (unique_seeds, set_score) in seed order.
    """
    seeds = np.asarray(seeds)
    seats = np.asarray(seats)
    ranks = np.asarray(ranks)
    if len(seeds) % 4:
        raise ValueError('game records are not whole sets')
    s = seeds.reshape(-1, 4)
    if not (s == s[:, :1]).all():
        raise ValueError('a set mixes seeds')
    if not (seats.reshape(-1, 4) == np.arange(4)).all():
        raise ValueError('challenger seats are not 0..3 within each set')
    # Sum in integer hundredths so a set holding each rank once scores exactly
    # 0 (in floats, 0.9 + 0.5 - 0.1 - 1.3 != 0); the deterministic A/A relies
    # on exact zeros.
    util = np.asarray(util, dtype=np.float64)
    centi = np.rint(util * 100)
    if np.abs(centi / 100 - util).max() < 1e-12:
        return s[:, 0], centi.astype(np.int64)[ranks].reshape(-1, 4).sum(axis=1) / 400
    return s[:, 0], util[ranks].reshape(-1, 4).mean(axis=1)


def _atoms(x):
    """Distinct values and counts, when the data is discrete enough."""
    values, counts = np.unique(np.asarray(x, dtype=np.float64), return_counts=True)
    return values, counts


def bootstrap_mean_ci(x, level=0.95, resamples=10_000, seed=0):
    """Percentile bootstrap CI of the mean over sets (block = seed-set).

    Raw set scores take at most 35 distinct values, so resampling is done
    exactly via multinomial counts over the atoms.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    rng = np.random.default_rng(seed)
    values, counts = _atoms(x)
    if len(values) <= 1024:
        draws = rng.multinomial(n, counts / n, size=resamples)
        means = draws @ values / n
    else:
        means = np.empty(resamples)
        for i in range(resamples):
            means[i] = x[rng.integers(0, n, n)].mean()
    alpha = (1 - level) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(lo), float(hi)


def confirm(x, kind, delta=0.0, level=0.95, resamples=10_000, seed=0):
    """Fixed-N T-confirm (02 §4, D36). Pass if the lower end of the two-sided
    bootstrap CI is > 0 (superiority) or > -delta (non-inferiority)."""
    if kind not in ('superiority', 'non-inferiority'):
        raise ValueError(kind)
    x = np.asarray(x, dtype=np.float64)
    lo, hi = bootstrap_mean_ci(x, level, resamples, seed)
    threshold = 0.0 if kind == 'superiority' else -delta
    return {
        'kind': kind,
        'delta': delta,
        'sets': len(x),
        'mean': float(x.mean()),
        'ci': [lo, hi],
        'level': level,
        'passed': bool(lo > threshold),
    }
