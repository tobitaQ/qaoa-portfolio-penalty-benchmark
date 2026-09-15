# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""Pytest configuration.

Pins the OpenMP thread count before any test imports a simulator, for the same
reason the experiment scripts do: lightning.qubit reduces over the statevector
in parallel and the reduction order follows thread scheduling, so the same run
repeated in the same process differs by ~1e-13 at the default thread count.
lightning.gpu is unaffected. Without this, the determinism regression test in
tests/test_braket_solver.py skips itself rather than failing, which is the wrong
default for a property the paper claims.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
