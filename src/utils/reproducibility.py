"""
Reproducibility utilities: deterministic seeding of all RNG sources.

Usage — call set_all_seeds(seed) at the very start of every run,
before any model creation, data loading, or random operations:

    from src.utils.reproducibility import set_all_seeds, get_worker_init_fn
    set_all_seeds(seed=0)

Sources of randomness that are seeded here:
  - Python built-in  random
  - NumPy            np.random
  - PyTorch CPU      torch.manual_seed
  - PyTorch CUDA     torch.cuda.manual_seed_all
  - PYTHONHASHSEED   environment variable

Sources that CANNOT be fully controlled:
  - Some CUDA operations (e.g., atomics) are inherently non-deterministic.
    Setting torch.backends.cudnn.deterministic=True mitigates this but can
    reduce throughput. We do NOT enable it by default; set force_deterministic=True
    only when strict reproducibility is more important than speed.
  - DataLoader multi-process shuffling is handled via worker_init_fn below.
"""

from __future__ import annotations
import os
import random
from typing import Optional
import numpy as np
import torch


def set_all_seeds(seed: int, force_deterministic: bool = False) -> None:
    """
    Seed every RNG source that is relevant to this experiment.

    Args:
        seed:               Integer seed for this run (0–4 for the five experiment seeds).
        force_deterministic: If True, also set torch.backends.cudnn.deterministic=True
                            and benchmark=False. This may slow down training.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if force_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True can help performance but may cause minor run-to-run
        # variation from cuDNN algorithm selection. Acceptable for this study;
        # document as a known source of minor non-determinism.
        torch.backends.cudnn.benchmark = True


def _seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_worker_init_fn(base_seed: int):
    """
    Returns a DataLoader worker_init_fn that seeds each worker's Python and
    NumPy RNGs independently, derived from the base experiment seed.

    Implemented with functools.partial of a module-level function so it is
    picklable under the 'spawn' / 'forkserver' start methods (macOS, Python
    3.14+ on Linux); a nested closure fails there.

    Usage:
        DataLoader(..., worker_init_fn=get_worker_init_fn(seed))
    """
    import functools
    return functools.partial(_seed_worker, base_seed=base_seed)


def make_run_id(prefix: str, seed: Optional[int] = None) -> str:
    """
    Unique, sortable run identifier: <prefix>[_s<seed>]_<UTC timestamp>_<8 hex>.

    The random suffix comes from os.urandom, so it is unique even when the
    experiment RNGs have been seeded identically.
    """
    import datetime
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seed_part = f"_s{seed:02d}" if seed is not None else ""
    return f"{prefix}{seed_part}_{stamp}_{os.urandom(4).hex()}"
