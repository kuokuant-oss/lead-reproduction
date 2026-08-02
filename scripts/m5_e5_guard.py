"""Runtime guard: E5 is pure re-scoring, so any fit path is a hard failure.

E5 reuses E4's 24 persisted states. Nothing may be fitted, refitted, or have its
context resampled. Rather than rely on the runner simply not calling those
methods, this installs a guard that makes calling them raise -- so a future edit
that reintroduces a fit fails loudly at the first attempt instead of quietly
producing numbers that are not a replication.

`arm()` must be called before any TabPFN model is loaded. `assert_armed()` lets
callers prove the guard is active before scoring.
"""

from __future__ import annotations

from typing import Any


class FitAttemptedError(RuntimeError):
    """Raised when E5 code tries to fit. Never caught, never downgraded."""


_ARMED = False
_BLOCKED = (
    "fit",
    "fit_from_preprocessed",
    "_fit",
    "partial_fit",
)


def _blocker(name: str, owner: str):
    def _raise(*_args: Any, **_kwargs: Any):
        raise FitAttemptedError(
            f"HARD FAILURE: E5 is pure re-scoring and {owner}.{name}() was called. "
            "E5 must reuse the 24 E4 persisted states via "
            "load_fitted_tabpfn_model; fitting, refitting, context resampling, "
            "model-seed changes and scaler re-selection are all prohibited."
        )

    return _raise


def arm() -> list[str]:
    """Replace every fit entry point with a raising stub. Returns what it blocked."""
    global _ARMED
    blocked: list[str] = []

    from tabpfn import TabPFNClassifier, TabPFNRegressor

    for cls in (TabPFNClassifier, TabPFNRegressor):
        for name in _BLOCKED:
            if hasattr(cls, name):
                setattr(cls, name, _blocker(name, cls.__name__))
                blocked.append(f"{cls.__name__}.{name}")

    # The tree comparator is fixed too: it is reloaded, never refit.
    try:
        from sklearn.base import BaseEstimator  # noqa: F401
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            HistGradientBoostingClassifier,
            RandomForestClassifier,
        )

        for cls in (
            RandomForestClassifier,
            ExtraTreesClassifier,
            HistGradientBoostingClassifier,
        ):
            for name in ("fit", "partial_fit"):
                if hasattr(cls, name):
                    setattr(cls, name, _blocker(name, cls.__name__))
                    blocked.append(f"{cls.__name__}.{name}")
    except ImportError:  # pragma: no cover - sklearn is a hard dependency
        pass

    _ARMED = True
    return sorted(blocked)


def assert_armed() -> None:
    if not _ARMED:
        raise FitAttemptedError(
            "HARD FAILURE: the E5 no-fit guard was not armed before scoring"
        )
