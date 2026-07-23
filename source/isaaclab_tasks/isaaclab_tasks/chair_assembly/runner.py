"""Top-level runner facade for packaged chair assembly backends."""

from __future__ import annotations

from .backend import AssemblyBackend
from .catalog import get_task_catalog, validate_task_catalog
from .models import AssemblyReport


class ChairAssemblyRunner:
    def __init__(self, backend: AssemblyBackend):
        self.backend = backend

    def run(self, stages: tuple[int, ...] | None = None) -> AssemblyReport:
        validate_task_catalog()
        return self.backend.run(get_task_catalog(stages))

