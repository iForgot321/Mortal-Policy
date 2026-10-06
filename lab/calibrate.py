"""Calibration of the sequential procedure by simulation (02 §4.1, D36).

Simulates the exact procedure (minimum sample, look schedule, variance floor,
cap) on streams resampled from an empirical set-score distribution shifted to
H0 and to H1, plus adversarial variants of it, and reports false-pass and
false-fail rates. A test type is accepted only if both are <= 5%.

Paths are simulated to the cap once, recording the LLR at every look, so any
boundary pair can be scored afterwards without re-simulating.
"""
from dataclasses import dataclass

import numpy as np

from .gsprt import ZERO_VAR, llr

MAX_ATOMS = 256


@dataclass
class Dist:
    """A discrete set-score distribution."""
    name: str
    values: np.ndarray
    probs: np.ndarray

    @property
    def mean(self):
        return float(self.values @ self.probs)

    @property
    def sd(self):
        return float(np.sqrt(np.square(self.values - self.mean) @ self.probs))

    def shifted(self, to_mean):
        return Dist(self.name, self.values - self.mean + to_mean, self.probs)


def empirical(x, name='empirical'):
    """Distribution of observed set scores, compressed to <= MAX_ATOMS atoms
    by quantile bins (bin means preserve the overall mean exactly)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    values, counts = np.unique(x, return_counts=True)
    if len(values) > MAX_ATOMS:
        bins = np.array_split(x, MAX_ATOMS)
        values = np.array([b.mean() for b in bins])
        counts = np.array([len(b) for b in bins])
    return Dist(name, values, counts / counts.sum())


def independent_ranks(util, rank_probs=(0.25, 0.25, 0.25, 0.25), name='independent-ranks'):
    """Set scores when the 4 rotations are independent games: the planning
    model of 02 §4 (no duplicate benefit). Exact 4-fold convolution."""
    util = np.asarray(util, dtype=np.float64)
    p = np.asarray(rank_probs, dtype=np.float64)
    # sums of 4 draws, keyed by the rank-count vector (exact, no float merging)
    dist = {(): 1.0}
    for _ in range(4):
        nxt = {}
        for key, pk in dist.items():
            for r in range(4):
                k = tuple(sorted(key + (r,)))
                nxt[k] = nxt.get(k, 0.0) + pk * p[r]
        dist = nxt
    values = np.array([util[list(k)].mean() for k in dist])
    probs = np.array(list(dist.values()))
    order = np.argsort(values)
    return Dist(name, values[order], probs[order])


def _restandardize(name, values, probs, mean, sd):
    m = values @ probs
    s = np.sqrt(np.square(values - m) @ probs)
    return Dist(name, (values - m) / s * sd + mean, probs)


def variants(base):
    """The base distribution plus adversarial variants with the same mean and
    SD (02 §4.1 step 3): rare large losses, skew either way, heavy tails."""
    m, s = base.mean, base.sd
    v, p = base.values, base.probs
    out = [base]

    # sparse: 1% of sets lose 6 SD; the rest shrink to keep mean and SD
    q, k = 0.01, 6.0
    loss = m - k * s
    m1 = (m - q * loss) / (1 - q)
    a = np.sqrt((s ** 2 - (1 - q) * (m1 - m) ** 2 - q * (loss - m) ** 2) / ((1 - q) * s ** 2))
    out.append(Dist('sparse', np.append((v - m) * a + m1, loss), np.append(p * (1 - q), q)))

    z = (v - m) / s
    for gamma, name in ((-0.6, 'skew-left'), (0.6, 'skew-right')):
        out.append(_restandardize(name, np.exp(gamma * z), p, m, s))

    # heavy: a 10% component at 3x the spread
    out.append(_restandardize('heavy', np.concatenate([z, 3 * z]), np.concatenate([0.9 * p, 0.1 * p]), m, s))
    return out


def simulate_paths(dist, spec, sched, n_tests=10_000, seed=0):
    """LLR at every look for n_tests independent streams drawn from dist.

    Returns (looks, llr_paths, zero_var) with shapes (L,), (n_tests, L),
    (n_tests, L).
    """
    rng = np.random.default_rng(seed)
    looks = sched.looks(spec.cap)
    v, p = dist.values, dist.probs
    s1 = np.zeros(n_tests)
    s2 = np.zeros(n_tests)
    paths = np.empty((n_tests, len(looks)))
    zero = np.empty((n_tests, len(looks)), dtype=bool)
    n_prev = 0
    for j, n in enumerate(looks):
        counts = rng.multinomial(int(n - n_prev), p, size=n_tests)
        s1 += counts @ v
        s2 += counts @ np.square(v)
        n_prev = n
        mean = s1 / n
        var = np.maximum((s2 - s1 * mean) / (n - 1), 0.0)
        zero[:, j] = var <= ZERO_VAR
        paths[:, j] = llr(n, mean, np.maximum(var, sched.var_floor), spec.mu0, spec.mu1)
    return looks, paths, zero


def score_paths(looks, paths, zero, lower, upper):
    """Outcome rates and sample sizes for one boundary pair."""
    live = ~zero
    hit_up = (paths >= upper) & live
    hit_lo = (paths <= lower) & live
    hit = hit_up | hit_lo
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, hit.argmax(axis=1), len(looks) - 1)
    rows = np.arange(len(paths))
    passed = any_hit & hit_up[rows, first]
    failed = any_hit & hit_lo[rows, first]
    n = looks[first]
    return {
        'pass': float(passed.mean()),
        'fail': float(failed.mean()),
        'inconclusive': float((~any_hit).mean()),
        'mean_sets': float(n.mean()),
        'p95_sets': float(np.quantile(n, 0.95)),
    }


def calibrate(base, spec, sched, n_tests=10_000, seed=0, boundary_grid=None, max_error=0.05):
    """Error rates of every variant of base under H0 and H1, for the nominal
    boundaries and (if they fail) a grid of wider ones.

    A false pass is a pass under H0; a false fail is anything but a pass under
    H1 (inconclusive counts as a fail, 02 §4). Returns a dict with per-variant
    rates at the nominal boundaries and the narrowest passing pair.
    """
    if boundary_grid is None:
        uppers = [spec.upper, 3.2, 3.5, 4.0, 4.5, 5.0]
        lowers = [spec.lower, -3.2, -3.5, -4.0, -4.5, -5.0]
        boundary_grid = [(lo, up) for up in uppers for lo in lowers]

    sims = []
    for i, dist in enumerate(variants(base)):
        h0 = simulate_paths(dist.shifted(spec.mu0), spec, sched, n_tests, seed + 2 * i)
        h1 = simulate_paths(dist.shifted(spec.mu1), spec, sched, n_tests, seed + 2 * i + 1)
        sims.append((dist.name, h0, h1))

    def rates(lower, upper):
        rows = []
        for name, h0, h1 in sims:
            r0 = score_paths(*h0, lower, upper)
            r1 = score_paths(*h1, lower, upper)
            rows.append({
                'variant': name,
                'false_pass': r0['pass'],
                'false_fail': 1 - r1['pass'],
                'inconclusive_h0': r0['inconclusive'],
                'inconclusive_h1': r1['inconclusive'],
                'mean_sets_h0': r0['mean_sets'],
                'mean_sets_h1': r1['mean_sets'],
            })
        return rows

    def ok(rows):
        return all(r['false_pass'] <= max_error and r['false_fail'] <= max_error for r in rows)

    nominal = rates(spec.lower, spec.upper)
    result = {
        'test': spec.name,
        'n_tests': n_tests,
        'base': {'name': base.name, 'mean': base.mean, 'sd': base.sd, 'atoms': len(base.values)},
        'sigma_set': sched.sigma_set,
        'nominal': {'lower': spec.lower, 'upper': spec.upper, 'rows': nominal, 'accepted': ok(nominal)},
        'selected': None,
    }
    if ok(nominal):
        result['selected'] = {'lower': spec.lower, 'upper': spec.upper}
        return result

    best = None
    for lower, upper in boundary_grid:
        rows = rates(lower, upper)
        if ok(rows):
            cost = max(max(r['mean_sets_h0'], r['mean_sets_h1']) for r in rows)
            if best is None or cost < best[0]:
                best = (cost, lower, upper, rows)
    if best is not None:
        result['selected'] = {'lower': best[1], 'upper': best[2], 'rows': best[3]}
    return result
