"""Loading checkpoints as data, never as code.

A pickle can run code when it is loaded. Checkpoints are written on rented hosts (the fleet's
learners on community boxes, the secure tier's VMs) and loaded on the workstation, which holds the
Tenhou data, the API keys and the riichi.dev token. So every checkpoint goes through
`load_checkpoint`: torch's weights-only unpickler (tensors, plain containers, strings and numbers),
plus numpy scalars, which the v1-era trainer checkpoints carry in `best_perf`. The messages between
`server.py` and its peers are loaded the same way (`common.recv_msg`, D48).

tests/test_checkpoint.py keeps unrestricted loads out of the code.
"""
import numpy as np
import torch

try:
    from numpy._core.multiarray import scalar as _np_scalar
except ImportError:  # numpy < 2
    from numpy.core.multiarray import scalar as _np_scalar

# A numpy scalar pickles as scalar(dtype, bytes). Neither call runs code: an object dtype gets back
# the object the restricted unpickler already built.
_NUMPY_SCALARS = [_np_scalar, np.dtype] + [type(np.dtype(t)) for t in ('float64', 'float32', 'int64', 'int32', 'bool')]


def load_checkpoint(path, map_location='cpu'):
    """`torch.load(path)` restricted to data. Raises `pickle.UnpicklingError` on anything else."""
    with torch.serialization.safe_globals(_NUMPY_SCALARS):
        return torch.load(path, weights_only=True, map_location=map_location)
