"""Run Mahjax's `lax.cond` / `lax.switch` as selects (03 §7 spike finding).

Under `vmap`, a `lax.cond` with a per-environment predicate becomes a select,
and JAX broadcasts every operand of the cond to the batch size first, closed-over
constants included. Mahjax's step has conds around code that reads its 22 MiB
yaku table (and the hand table), so each environment in a batch carried its own
copy: about 22 MiB of temporaries per environment, which caps a 12 GB GPU at
~200 environments. Evaluating both branches and selecting the result is what the
batched cond does anyway, without the broadcast (temporaries fall to ~0.1 MiB
per environment). JAX code is pure, so the results are identical.

Call `apply()` before tracing any Mahjax function.
"""
import jax
import jax.numpy as jnp
from jax._src.lax.lax import _convert_element_type


def _pick(which, xs):
    """xs[which] elementwise, with the dtype and weak type `lax.cond` gives."""
    xs = [jnp.asarray(x) for x in xs]
    dtype = jnp.result_type(*xs)
    weak = all(x.weak_type for x in xs)
    out = xs[0].astype(dtype)
    for i, x in enumerate(xs[1:], 1):
        out = jax.lax.select(jnp.broadcast_to(which == i, out.shape), x.astype(dtype), out)
    # private, but it is how lax.cond keeps a weak type
    return _convert_element_type(out, dtype, weak_type=weak)


def cond_select(pred, true_fn, false_fn, *operands):
    t, f = true_fn(*operands), false_fn(*operands)
    # lax.cond's operand order is (false, true) for index 0/1
    return jax.tree.map(lambda a, b: _pick(jnp.asarray(pred).astype(jnp.int32), [b, a]), t, f)


def switch_select(index, branches, *operands):
    outs = [branch(*operands) for branch in branches]
    index = jnp.clip(jnp.asarray(index).astype(jnp.int32), 0, len(branches) - 1)
    return jax.tree.map(lambda *xs: _pick(index, xs), *outs)


def apply():
    jax.lax.cond = cond_select
    jax.lax.switch = switch_select
