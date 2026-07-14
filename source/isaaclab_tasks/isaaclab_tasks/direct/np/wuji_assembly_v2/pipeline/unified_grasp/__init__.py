"""Grasp pipelines with legacy v80 exports loaded only on demand."""

from __future__ import annotations

__all__ = [
    "FAILURE_CATEGORIES",
    "PROGRESS_COLUMNS",
    "V80_PARTS",
]


def __getattr__(name: str):
    if name in __all__:
        from . import v80_reports

        return getattr(v80_reports, name)
    raise AttributeError(name)
