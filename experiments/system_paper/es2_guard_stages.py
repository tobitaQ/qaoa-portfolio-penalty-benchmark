# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
E-S2: at which stage does a misconfigured cloud run stop, and what would it
have cost?

The system paper's principle P2 says that billing has to be refused by the
*configuration*, before anything is submitted, because the two controls the
platform offers work differently: AWS Budgets notifies after the fact, and
Amazon Braket's per-QPU spending limit (November 2025) rejects a task whose
list-price estimate exceeds the remaining cap -- but only for on-demand QPU
tasks, never for simulators, and it caps dollars rather than recognising a
wrong run. This harness measures that claim instead of asserting it.

Design
------
Rows are misconfigurations that a user of this code base can actually write,
columns are guard configurations, and each cell records the stage at which
the run stops, the tasks that would have reached the service and the
list-price bill.

Misconfigurations (``MISCONFIGURATIONS``):

  qpu_optimizer          ``--backend braket_ionq``: the 50-step ADAM loop on
                         hardware. A QPU has no adjoint gradient, so PennyLane
                         evaluates one parameter-shift circuit per gate
                         parameter and the framework submits 1 + 2·p·[C(N,2)+2N]
                         tasks per optimizer step (177 at N = 8, p = 2).
  trainable_sampling_sv1 the final 1,000-shot sampling call on SV1 with the
                         optimizer's trainable tensors: 1 + 2·p·[C(N,2)+2N]
                         tasks for one bitstring (the 177 of the 2026-08-09
                         pilot; ``paper01_sv1_task_ledger.csv`` run 1).
  trainable_sampling_qpu the same call on IonQ Forte-1 with 100 shots.
  qpu_sampling_1000_shots ``--sample-backend braket_ionq`` with the default
                         ``--shots 1000``: one correctly built task at
                         $0.30 + 1000 × $0.08. Not a structural error; listed
                         so the table shows what the structural guards do
                         *not* see.
  untagged_cloud_run     ``--backend braket_sv1`` without ``--results-tag``:
                         no money at stake, but the run's output path is the
                         committed baseline behind Table I.

Guard configurations (``CONFIGURATIONS``):

  none        every guard of this code base disabled (the counterfactual)
  cap         no structural guard; a Braket spending limit of ``--cap`` USD on
              the QPU (simulators are outside its scope, as the API states)
  structural  the guards as committed: QPU refused in the optimizer slot by the
              constructor and by the CLI, ``frozen_angles()`` before the
              sampling QNode, ``--results-tag`` required for cloud backends and
              the canonical files never overwritten by an untagged run
  both        structural guards plus the spending limit

Stages: ``cli`` (the driver refuses before constructing a solver),
``constructor`` (``BraketSolver.__init__`` refuses before any AWS object
exists), ``pre-submit`` (the misconfiguration is removed before the call
reaches the device: the sampling call becomes one task), ``service`` (the
spending limit rejects a task at ``CreateQuantumTask``) and ``submitted``
(nothing stops the run; every task is executed and billed).

Where the numbers come from
---------------------------
Nothing here talks to AWS. Task counts are *measured*, not assumed, on two
stand-ins that share the framework's execution path, and tied to the one bill
that exists:

* ``NoDerivativeDevice`` -- ``default.qubit`` declaring, like a QPU device,
  that it offers no derivative method. PennyLane's ``best`` diff method then
  resolves to parameter-shift exactly as it does for ``braket.aws.qubit`` on
  hardware, and ``qml.Tracker`` counts the tapes the solver's own QNode sends.
* ``braket.local.qubit`` -- the Braket plugin's own local device, which also
  lacks adjoint support (``--cross-check-local``; slow, N = 8 by default).
* the parameter-shift transform's tape count (``param_shift_tape_count``),
  which is what the plugin executed for the *sampling* call on SV1 -- the
  local devices execute that call once whatever the angles' flag, so its count
  rests on the transform and on the ledger: 1 + 176 = 177 tasks at N = 8.

The guards' stages are measured too: the structural cells run the committed
code with the device factory replaced by the stand-in, and the ``none`` cells
run the same code with the guards monkey-patched out, so a guard that stopped
firing would change the CSV.

Prices are Braket list prices of 2026-08 (``PRICES``); the spending-limit
model accepts tasks while the cumulative estimate stays within the cap and
rejects the first task that would exceed it, which is the documented behaviour
("Braket rejects the task immediately with a validation error").

Run:
    python -m experiments.system_paper.es2_guard_stages            # ~1.8 h, N=20 is 94 min
    python -m experiments.system_paper.es2_guard_stages --sizes 8 --no-cross-check-local   # ~1 min

The time goes into building the parameter-shift tapes in Python (a 20-asset
ansatz has 460 gate parameters per layer pair), not into simulating them; a
Lightning-based stand-in is no faster.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

import experiments.paper01_qubo_baseline.run_experiment as run_experiment
import src.solvers.braket_solver as solver_mod
from experiments.paper01_qubo_baseline.repro_parameter_shift_fallback import (
    param_shift_tape_count,
)
from experiments.paper01_qubo_baseline.run_experiment import write_csv
from src.qubo.portfolio import PortfolioQUBO
from src.solvers.braket_solver import BraketSolver

RESULTS_DIR = Path(__file__).parent / "results"
STAGES_CSV = RESULTS_DIR / "paper02_es2_guard_stages.csv"
COUNTS_CSV = RESULTS_DIR / "paper02_es2_task_counts.csv"

#: Braket list prices as billed to this project in 2026-08 (Sec. VI-B of the
#: numerical paper). SV1 bills execution time with a 3-second minimum per
#: task; a sampling task ran 0.11 s, so every one bills the minimum.
PRICES = {
    "ionq_task_usd": 0.30,
    "ionq_shot_usd": 0.08,
    "sv1_usd_per_minute": 0.075,
    "sv1_minimum_seconds": 3.0,
}
#: The project's Budgets figure, reused as the per-QPU spending limit.
DEFAULT_CAP_USD = 50.0
P_LAYERS = 2
OPTIMIZER_STEPS = 50
SIZES = (8, 12, 16, 20)
LEDGER_SAMPLING_TASKS_N8 = 177  # paper01_sv1_task_ledger.csv, run 1

CONFIGURATIONS = ("none", "cap", "structural", "both")


class NoDerivativeDevice:
    """Factory for a statevector device that, like a QPU, has no gradient method.

    Built lazily so importing this module does not import PennyLane.
    """

    @staticmethod
    def make(qml, wires: int):
        class _NoDerivativeDevice(qml.devices.DefaultQubit):
            def supports_derivatives(self, execution_config=None, circuit=None):
                return False

        return _NoDerivativeDevice(wires=wires)


def tapes_per_gradient(n: int, p: int = P_LAYERS) -> int:
    """Shifted tapes of one parameter-shift gradient: 2·p·[C(N,2) + 2N]."""
    return 2 * p * (math.comb(n, 2) + 2 * n)


def ionq_task_usd(shots: int) -> float:
    return PRICES["ionq_task_usd"] + shots * PRICES["ionq_shot_usd"]


def sv1_task_usd(seconds: float) -> float:
    return max(seconds, PRICES["sv1_minimum_seconds"]) / 60.0 * PRICES["sv1_usd_per_minute"]


def toy_problem(n: int, seed: int = 0):
    """A dense N-asset QUBO; the counts depend on N and p only, not on the data."""
    rng = np.random.default_rng(seed)
    return PortfolioQUBO().formulate(
        rng.uniform(0.05, 0.30, n), np.eye(n) * 0.04, num_select=max(1, n // 5))


def _tape_count(tracker_totals: dict) -> int:
    """Tapes the device executed. ``default.qubit`` reports them as
    ``simulations`` (its ``executions`` counts per-measurement work); the
    Braket local device reports ``executions``."""
    if "simulations" in tracker_totals:
        return int(tracker_totals["simulations"])
    return int(tracker_totals.get("executions", 0))


# ----------------------------------------------------------------------
# Measured task counts
# ----------------------------------------------------------------------

def measure_loop_tapes(n: int, steps: int, device: str = "mock",
                       shots: int = 100) -> dict:
    """Run the committed solver on a no-derivative device and count its tapes.

    Args:
        n: Problem size.
        steps: ADAM steps to run (the per-step count is what is reported).
        device: ``"mock"`` for ``NoDerivativeDevice`` or ``"braket_local"``
            for the plugin's own local simulator.
        shots: Shots of the final sampling call.

    Returns:
        Dict with the total tape count, the per-step count and its formula.
    """
    import pennylane as qml

    if device == "mock":
        dev = NoDerivativeDevice.make(qml, n)
    elif device == "braket_local":
        dev = qml.device("braket.local.qubit", wires=n)
    else:
        raise ValueError(device)
    solver = BraketSolver(backend="braket_local", p_layers=P_LAYERS,
                          n_optimizer_steps=steps, n_shots=shots)
    solver._make_device = lambda n_, qml_, backend=None: dev  # type: ignore[method-assign]
    t0 = time.perf_counter()
    with qml.Tracker(dev) as tracker:
        solver.solve(toy_problem(n), seed=42)
    elapsed = time.perf_counter() - t0
    total = _tape_count(tracker.totals)
    # One forward + the shifted tapes per step, then one sampling tape.
    expected_per_step = 1 + tapes_per_gradient(n)
    return {
        "n": n, "device": device, "steps": steps, "tapes_total": total,
        "sampling_tapes": 1, "tapes_per_step": (total - 1) / steps,
        "expected_per_step": expected_per_step,
        "matches_formula": (total - 1) == steps * expected_per_step,
        "elapsed_s": round(elapsed, 1),
    }


def measure_sampling_call(n: int) -> dict:
    """What the committed sampling path hands the device.

    Records whether the angles reaching ``_sample_best`` were trainable, that
    they were frozen before the QNode, and that the device saw one tape.
    """
    import pennylane as qml

    seen: dict = {}
    original = solver_mod.frozen_angles

    def spy(params):
        seen.setdefault("trainable_on_arrival", bool(qml.math.requires_grad(params)))
        out = original(params)
        seen["trainable_after"] = bool(qml.math.requires_grad(out))
        return out

    dev = qml.device("default.qubit", wires=n)
    solver = BraketSolver(backend="braket_local", p_layers=P_LAYERS,
                          n_optimizer_steps=1, n_shots=100)
    solver._make_device = lambda n_, qml_, backend=None: dev  # type: ignore[method-assign]
    with _patched(solver_mod, "frozen_angles", spy):
        with qml.Tracker(dev) as tracker:
            solver.solve(toy_problem(n), seed=42)
    # default.qubit differentiates with backprop, so the optimizer costs one
    # simulation per step and the sampling call is the other one.
    return {"n": n, "sampling_tapes_committed": _tape_count(tracker.totals) - 1,
            "frozen_before_qnode": seen.get("trainable_after") is False,
            "tapes_if_trainable": 1 + tapes_per_gradient(n)}


# ----------------------------------------------------------------------
# Guard toggling
# ----------------------------------------------------------------------

@contextlib.contextmanager
def _patched(obj, name: str, value) -> Iterator[None]:
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def guards(enabled: bool, qml, mock_dev, keep_frozen: bool = False) -> Iterator[None]:
    """Run the committed code with its structural guards on or off.

    With ``enabled=False`` the QPU set is emptied (constructor and CLI stop
    refusing hardware in the optimizer slot), the cloud backends are declared
    local (the CLI stops asking for ``--results-tag``), the canonical-file
    refusal returns None, and ``frozen_angles`` returns its input untouched.
    In both settings the device factory returns ``mock_dev`` for every cloud
    backend, so no AWS session is ever created. ``keep_frozen`` leaves
    ``frozen_angles`` in place while the other guards are off, for the row
    whose misconfiguration is not the trainable flag.
    """
    original_make = BraketSolver._make_device

    def make(self, n, qml_, backend=None):
        backend = backend or self.backend
        if backend in ("braket_sv1", "braket_ionq"):
            return mock_dev
        return original_make(self, n, qml_, backend)

    def explode(*args, **kwargs):
        raise AssertionError("an AWS session was requested")

    stack = contextlib.ExitStack()
    with stack:
        stack.enter_context(_patched(BraketSolver, "_make_device", make))
        stack.enter_context(_patched(BraketSolver, "_aws_session", staticmethod(explode)))
        if not enabled:
            stack.enter_context(_patched(solver_mod, "QPU_BACKENDS", frozenset()))
            stack.enter_context(_patched(run_experiment, "QPU_BACKENDS", frozenset()))
            stack.enter_context(_patched(
                run_experiment, "LOCAL_BACKENDS",
                run_experiment.LOCAL_BACKENDS | {"braket_sv1", "braket_ionq"}))
            stack.enter_context(_patched(
                run_experiment, "canonical_overwrite_refusal", lambda *a, **k: None))
            if not keep_frozen:
                stack.enter_context(_patched(solver_mod, "frozen_angles", lambda p: p))
        yield


# ----------------------------------------------------------------------
# The cells
# ----------------------------------------------------------------------

@dataclass
class Cell:
    misconfiguration: str
    n: int
    configuration: str
    stage: str
    tasks_would_submit: int
    tasks_billed: int
    usd_billed: float
    usable_result: str          # yes / no / unknown
    baseline_overwritten: str   # yes / no / n-a
    count_source: str
    note: str = ""


def _cap_outcome(tasks: int, usd_per_task: float, cap: float | None,
                 on_qpu: bool) -> tuple[str, int, float]:
    """Apply the spending-limit model: (stage, tasks billed, USD billed)."""
    if cap is None or not on_qpu:
        return "submitted", tasks, tasks * usd_per_task
    accepted = min(tasks, int(math.floor(cap / usd_per_task + 1e-9)))
    if accepted < tasks:
        return "service", accepted, accepted * usd_per_task
    return "submitted", tasks, tasks * usd_per_task


def cell_qpu_optimizer(n: int, configuration: str, cap: float | None,
                       per_step: int, qml, mock_dev) -> Cell:
    structural = configuration in ("structural", "both")
    shots = 100
    with guards(structural, qml, mock_dev):
        if structural:
            # The CLI is the surface a user touches; it refuses before
            # constructing anything, and the constructor refuses independently.
            code = run_experiment.main([
                "--backend", "braket_ionq", "--s3-bucket", "b", "--results-tag", "es2",
                "--quantum-n", str(n), "--classical-n", str(n)])
            try:
                BraketSolver(backend="braket_ionq", s3_bucket="b", n_shots=shots)
                constructor_refused = False
            except ValueError:
                constructor_refused = True
            assert code == 2 and constructor_refused
            return Cell("qpu_optimizer", n, configuration, "cli", 0, 0, 0.0, "no", "n-a",
                        "measured: CLI exit code 2; constructor ValueError",
                        "refused at two independent points before any device or session")
        # Unguarded: the constructor accepts the QPU (checked here, without
        # running the loop); every optimizer step then submits ``per_step``
        # tasks and the sampling call, with trainable angles, its own.
        solver = BraketSolver(backend="braket_ionq", s3_bucket="b", n_shots=shots)
        assert solver.backend == "braket_ionq"
    tasks = OPTIMIZER_STEPS * per_step + 1 + tapes_per_gradient(n)
    stage, billed, usd = _cap_outcome(tasks, ionq_task_usd(shots), cap, on_qpu=True)
    usable = "yes" if billed == tasks else "no"
    return Cell("qpu_optimizer", n, configuration, stage, tasks, billed, round(usd, 2),
                usable, "n-a", "tracker: per-step tapes on a no-derivative device",
                f"{per_step} tasks/step x {OPTIMIZER_STEPS} steps + sampling; "
                f"constructor accepted the QPU with the guard removed")


def cell_trainable_sampling(n: int, configuration: str, cap: float | None,
                            target: str, sampling: dict) -> Cell:
    structural = configuration in ("structural", "both")
    name = f"trainable_sampling_{target}"
    on_qpu = target == "qpu"
    shots = 100 if on_qpu else 1000
    per_task = ionq_task_usd(shots) if on_qpu else sv1_task_usd(0.11)
    if structural:
        tasks = sampling["sampling_tapes_committed"]
        stage, billed, usd = _cap_outcome(tasks, per_task, cap, on_qpu)
        stage = "pre-submit" if stage == "submitted" else stage
        return Cell(name, n, configuration, stage, tasks, billed, round(usd, 2),
                    "yes" if billed == tasks else "no", "n-a",
                    "tracker: sampling tapes of the committed path",
                    "angles frozen before the QNode" if sampling["frozen_before_qnode"]
                    else "WARNING: angles were not frozen")
    tasks = sampling["tapes_if_trainable"]
    stage, billed, usd = _cap_outcome(tasks, per_task, cap, on_qpu)
    if stage == "service":
        usable = "unknown"   # the forward sample is one tape of the batch
    else:
        usable = "yes"
    src = "transform: 1 + param-shift tapes" + (
        "; equals SV1 ledger run 1 (177)" if n == 8 and tasks == LEDGER_SAMPLING_TASKS_N8 else "")
    return Cell(name, n, configuration, stage, tasks, billed, round(usd, 2), usable,
                "n-a", src, "local devices execute this call once; the metered device did not")


def cell_qpu_sampling_default_shots(n: int, configuration: str, cap: float | None,
                                    qml, mock_dev) -> Cell:
    structural = configuration in ("structural", "both")
    shots = 1000
    with guards(structural, qml, mock_dev, keep_frozen=True):
        solver = BraketSolver(backend="lightning_cpu", sample_backend="braket_ionq",
                              s3_bucket="b", p_layers=P_LAYERS, n_optimizer_steps=1,
                              n_shots=shots)
        # The structural guards accept this configuration; count what reaches
        # the (mock) hardware device.
        with qml.Tracker(mock_dev) as tracker:
            with _patched(solver_mod, "QPU_BACKENDS", frozenset()):  # read locally, not from S3
                solver.solve(toy_problem(n), seed=42)
    tasks = _tape_count(tracker.totals)
    stage, billed, usd = _cap_outcome(tasks, ionq_task_usd(shots), cap, on_qpu=True)
    return Cell("qpu_sampling_1000_shots", n, configuration, stage, tasks, billed,
                round(usd, 2), "yes" if billed == tasks else "no", "n-a",
                "tracker: tapes reaching the mock hardware device",
                "not a structural error: the guards pass it (angles frozen in every "
                "configuration here); only the cap sees the shot count")


def cell_untagged_cloud_run(configuration: str, qml, tmp_dir: Path) -> Cell:
    """An untagged SV1 run against a directory holding the baseline files."""
    structural = configuration in ("structural", "both")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    baseline = tmp_dir / "paper01_results.csv"
    conv = tmp_dir / "paper01_convergence.csv"
    baseline.write_text("n_assets,solver\n8,committed-baseline\n", encoding="utf-8")
    conv.write_text("n_assets,step\n8,0\n", encoding="utf-8")
    before = hashlib.sha256(baseline.read_bytes()).hexdigest()
    mock_dev = NoDerivativeDevice.make(qml, 8)
    with _patched(run_experiment, "RESULTS_DIR", tmp_dir):
        with guards(structural, qml, mock_dev):
            with qml.Tracker(mock_dev) as tracker:
                code = run_experiment.main([
                    "--backend", "braket_sv1", "--s3-bucket", "b",
                    "--quantum-n", "8", "--classical-n", "8", "--steps", "1"])
    after = hashlib.sha256(baseline.read_bytes()).hexdigest()
    overwritten = "no" if after == before else "yes"
    tasks = _tape_count(tracker.totals)
    if structural:
        assert code == 2 and overwritten == "no" and tasks == 0
        return Cell("untagged_cloud_run", 8, configuration, "cli", 0, 0, 0.0, "no",
                    overwritten, "measured: exit code, file digest, mock-device tapes",
                    "refused before any solver was built")
    assert code == 0
    # SV1 has adjoint gradients, so the real service would see 1 task per step
    # + 1 sampling task; the no-derivative mock counts more. Report the SV1
    # figure for the bill and the mock's count in the note.
    sv1_tasks = 1 + 1
    usd = sv1_task_usd(0.7) + sv1_task_usd(0.11)
    return Cell("untagged_cloud_run", 8, configuration, "submitted", sv1_tasks, sv1_tasks,
                round(usd, 3), "yes", overwritten,
                "measured: exit code and file digest; bill from the SV1 ledger's per-task times",
                f"the run itself is cheap; the loss is the committed baseline "
                f"(mock device executed {tasks} tapes for the 1-step run)")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def run(sizes=SIZES, cap: float = DEFAULT_CAP_USD, cross_check_local: bool = True,
        steps_for_count: int = 1, log: Callable[[str], None] = print) -> tuple[list[dict], list[Cell]]:
    import pennylane as qml

    counts: list[dict] = []
    per_step: dict[int, int] = {}
    sampling: dict[int, dict] = {}
    for n in sizes:
        log(f"[counts] N={n}: solver loop on the no-derivative device ...")
        c = measure_loop_tapes(n, steps_for_count, "mock")
        counts.append(c)
        per_step[n] = int(round(c["tapes_per_step"]))
        log(f"    {c['tapes_per_step']:.0f} tapes/step (formula {c['expected_per_step']}), "
            f"{c['elapsed_s']} s")
        sampling[n] = measure_sampling_call(n)
        counts.append({"n": n, "device": "default.qubit (committed path)",
                       "steps": 1, "tapes_total": sampling[n]["sampling_tapes_committed"] + 1,
                       "sampling_tapes": sampling[n]["sampling_tapes_committed"],
                       "tapes_per_step": 1, "expected_per_step": 1,
                       "matches_formula": sampling[n]["frozen_before_qnode"],
                       "elapsed_s": ""})
        if cross_check_local and n == 8:
            log("[counts] N=8: cross-check on braket.local.qubit (slow) ...")
            c2 = measure_loop_tapes(n, steps_for_count, "braket_local")
            counts.append(c2)
            log(f"    {c2['tapes_per_step']:.0f} tapes/step, {c2['elapsed_s']} s")
        # The transform's own count, tied to the ledger at N = 8.
        _, tapes, _ = param_shift_tape_count(n, P_LAYERS, qml)
        counts.append({"n": n, "device": "param_shift transform", "steps": "",
                       "tapes_total": 1 + tapes, "sampling_tapes": 1 + tapes,
                       "tapes_per_step": "", "expected_per_step": 1 + tapes_per_gradient(n),
                       "matches_formula": tapes == tapes_per_gradient(n),
                       "elapsed_s": ""})

    cells: list[Cell] = []
    tmp_root = RESULTS_DIR / "_es2_tmp"
    for configuration in CONFIGURATIONS:
        cap_value = cap if configuration in ("cap", "both") else None
        for n in sizes:
            mock_n = NoDerivativeDevice.make(qml, n)
            cells.append(cell_qpu_optimizer(n, configuration, cap_value, per_step[n], qml, mock_n))
            cells.append(cell_trainable_sampling(n, configuration, cap_value, "sv1", sampling[n]))
            cells.append(cell_trainable_sampling(n, configuration, cap_value, "qpu", sampling[n]))
        cells.append(cell_qpu_sampling_default_shots(8, configuration, cap_value, qml,
                                                     NoDerivativeDevice.make(qml, 8)))
        cells.append(cell_untagged_cloud_run(configuration, qml, tmp_root / configuration))
        log(f"[cells] {configuration}: {len(cells)} rows so far")
    return counts, cells


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    p.add_argument("--cap", type=float, default=DEFAULT_CAP_USD,
                   help="per-QPU spending limit in USD for the cap configurations")
    p.add_argument("--no-cross-check-local", action="store_true")
    p.add_argument("--steps-for-count", type=int, default=1)
    args = p.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    counts, cells = run(tuple(args.sizes), args.cap, not args.no_cross_check_local,
                        args.steps_for_count)
    write_csv(COUNTS_CSV, counts)
    write_csv(STAGES_CSV, [c.__dict__ for c in cells])
    print(f"wrote {COUNTS_CSV} ({len(counts)} rows) and {STAGES_CSV} ({len(cells)} rows)")
    print()
    print(f"{'misconfiguration':<26}{'N':>3}  {'none':<22}{'cap':<22}{'structural':<22}{'both':<22}")
    keys = sorted({(c.misconfiguration, c.n) for c in cells},
                  key=lambda k: (list(dict.fromkeys(c.misconfiguration for c in cells)).index(k[0]), k[1]))
    by = {(c.misconfiguration, c.n, c.configuration): c for c in cells}
    for m, n in keys:
        row = f"{m:<26}{n:>3}  "
        for conf in CONFIGURATIONS:
            c = by[(m, n, conf)]
            row += f"{c.stage} {c.tasks_billed}t ${c.usd_billed:,.2f}".ljust(22)
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
