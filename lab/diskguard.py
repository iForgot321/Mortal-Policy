"""Free-space guard for lab runs.

Game logs are nearly all the lab writes: about 0.17 MB per set with four neural seats, so 3.3 GB
for a 20K-set measure and 17-18 GB for a 100K-set run. On the workstation the Linux disk is a file
on the Windows system drive. If that drive fills, the file can't grow and Linux gets write errors,
which puts the registry and the checkpoints at risk (2026-11-10: C: was down to 2.2 GB).

- **At the start**, before any nonce is reserved, a run refuses to begin unless every watched
  filesystem would keep `LAB_MIN_FREE_GB` (default 10) free after the run's estimated logs.
- **Before each chunk**, a run waits while a watched filesystem is under half that reserve, and
  goes on when space is back. Nothing is lost by the wait.

Watched: the filesystem of workdir/lab, and under WSL the Windows drive that holds the disk file
(`LAB_HOST_DRIVE`, default /mnt/c; set it empty to watch none). `LAB_MIN_FREE_GB=0` turns the
guard off. Space freed inside Linux doesn't count on the host drive until the disk file is
compacted, so the host figure is the conservative one.
"""
import os
import platform
import shutil
import sys
import time

from . import RUNS_DIR

GB = 1 << 30
# measured: 3.3 GB / 20K sets and 17.5 GB / 100K sets with four neural seats (less against the
# rule-based floor)
LOG_MB_PER_SET = 0.18
DEFAULT_MIN_FREE_GB = 10.0
DEFAULT_HOST_DRIVE = '/mnt/c'


def min_free_gb():
    return float(os.environ.get('LAB_MIN_FREE_GB', DEFAULT_MIN_FREE_GB))


def _is_wsl():
    return 'microsoft' in platform.uname().release.lower()


def watched():
    """Paths whose filesystems the guard watches."""
    path = RUNS_DIR
    while not os.path.isdir(path):
        path = os.path.dirname(path)
    paths = [path]
    host = os.environ.get('LAB_HOST_DRIVE', DEFAULT_HOST_DRIVE if _is_wsl() else '')
    if host and os.path.isdir(host):
        paths.append(host)
    return paths


def free_gb(path):
    return shutil.disk_usage(path).free / GB


def log_gb(sets, logs=True):
    """Estimated size of a run's game logs."""
    return sets * LOG_MB_PER_SET / 1024 if logs else 0.0


def short(need_gb):
    """[(path, free GB)] for the watched filesystems with less than need_gb free."""
    out = []
    for path in watched():
        free = free_gb(path)
        if free < need_gb:
            out.append((path, free))
    return out


def check_start(sets, logs=True):
    """Exit unless every watched filesystem has room for `sets` sets of logs plus the reserve."""
    reserve = min_free_gb()
    if reserve <= 0:
        return
    need = log_gb(sets, logs) + reserve
    low = short(need)
    if low:
        where = '; '.join(f'{path} has {free:.1f} GB free' for path, free in low)
        sys.exit(f'disk guard: not started. {sets:,} sets need about {log_gb(sets, logs):.1f} GB of logs '
                 f'plus a {reserve:g} GB reserve ({need:.1f} GB), and {where}. Free space, or set '
                 'LAB_MIN_FREE_GB (0 turns the guard off). See lab/diskguard.py.')


def wait_for_space(poll=60, say_every=600):
    """Block while a watched filesystem is under half the reserve; returns the seconds waited."""
    floor = min_free_gb() / 2
    if floor <= 0:
        return 0
    waited = 0
    while low := short(floor):
        if waited % say_every == 0:
            where = '; '.join(f'{path} has {free:.1f} GB free' for path, free in low)
            print(f'disk guard: waiting, under {floor:g} GB free ({where}); waited {waited // 60} min',
                  flush=True)
        time.sleep(poll)
        waited += poll
    if waited:
        print(f'disk guard: space is back after {waited // 60} min; continuing', flush=True)
    return waited
