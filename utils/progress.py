"""Progress bar helpers for the training pipeline.

Wraps tqdm so that ML modules stay free of UI code.
Falls back to silent no-ops if tqdm is not installed.
"""

from __future__ import annotations

try:
    from tqdm import tqdm as _tqdm
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class _Noop:
    """Silent stand-in used when tqdm is not installed."""

    def __init__(self, iterable=None, **_):
        self._it = iter(iterable) if iterable is not None else iter([])

    def __iter__(self):        return self._it
    def __enter__(self):       return self
    def __exit__(self, *_):    pass
    def update(self, n=1):     pass
    def close(self):           pass
    def set_postfix(self, **_): pass


def fold_bar(splits: list, desc: str = "CV folds"):
    """Iterable progress bar over fold (train_idx, test_idx) splits."""
    return _tqdm(splits, desc=desc, unit="fold") if _AVAILABLE else _Noop(splits)


def img_bar(total: int, desc: str, *, leave: bool = False):
    """Context-manager / manual-update bar for image-level processing."""
    return _tqdm(total=total, desc=desc, unit="img", leave=leave) if _AVAILABLE else _Noop()


def epoch_bar(n_epochs: int, desc: str):
    """Iterable progress bar over fine-tuning epochs."""
    return (
        _tqdm(range(n_epochs), desc=desc, unit="ep", leave=False)
        if _AVAILABLE else _Noop(range(n_epochs))
    )
