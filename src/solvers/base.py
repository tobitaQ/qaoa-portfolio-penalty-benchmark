# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Layer 3: Solver abstraction interface.

All solvers implement solve(problem) → SolverResult.
The backend (local, cloud, real hardware) is selected at construction time.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from src.qubo.portfolio import QUBOProblem


@dataclass
class SolverResult:
    """Unified result object for all solvers."""

    bitstring: np.ndarray           # Binary solution vector x ∈ {0,1}^N
    energy: float                   # QUBO energy xᵀQx + offset
    energy_no_offset: float         # xᵀQx (for solver comparison)
    selected_assets: list[int]      # Indices where x_i = 1
    runtime_seconds: float
    backend: str                    # e.g. "braket_local", "classical_sa"
    metadata: dict = field(default_factory=dict)

    @property
    def num_selected(self) -> int:
        return int(np.sum(self.bitstring))

    def feasible(self, num_select: int) -> bool:
        """True if exactly num_select assets were chosen."""
        return self.num_selected == num_select


class AbstractSolver(ABC):
    """
    Layer 3: Device abstraction.  Backend changes; interface stays constant.
    """

    def __init__(self, backend: str) -> None:
        self.backend = backend

    @abstractmethod
    def solve(self, problem: QUBOProblem, seed: int = 42) -> SolverResult:
        """Solve the QUBO problem and return a SolverResult."""

    def _make_result(
        self,
        problem: QUBOProblem,
        bitstring: np.ndarray,
        runtime: float,
        extra: dict | None = None,
    ) -> SolverResult:
        x = np.asarray(bitstring, dtype=float)
        energy_no_offset = float(x @ problem.Q @ x)
        return SolverResult(
            bitstring=x,
            energy=energy_no_offset + problem.offset,
            energy_no_offset=energy_no_offset,
            selected_assets=[i for i, v in enumerate(x) if v > 0.5],
            runtime_seconds=runtime,
            backend=self.backend,
            metadata=extra or {},
        )

    @staticmethod
    def _timer() -> float:
        return time.perf_counter()
