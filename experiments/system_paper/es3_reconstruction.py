# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
E-S3: what does a third party need to rebuild the results, and what does it cost?

Principle P3 of the system paper says reconstruction is guaranteed at three
levels -- tables and figures from the committed aggregates, one instance end
to end, the cross experiment from its manifest -- with the cost of each level
recorded, and with interruption handled by per-run flushing and ``--resume``.
This harness plays the third party: it clones the *public* reproduction
package (the repository the paper's availability statement names, at its
release commit), builds a fresh environment from the pinned requirements, runs
each level exactly as the README says, and records for every step the command,
the wall-clock, the exit code and the outcome (files that came back identical,
cells that agree, columns that differ and by how much).

Steps (``--only`` selects a subset)
------------------------------------
  clone      ``git clone`` of the public package; commit recorded
  venv       ``python -m venv`` + ``pip install -r requirements-repro.txt``
  tests      the package's own test suite
  level1     ``notebooks.generate_tqe_figures`` (figures regenerated, then
             compared byte for byte with the committed renders) and
             ``papers/check_tables.py`` (manuscript cells against the CSVs)
  level2     ``run_experiment.py`` on N ∈ {8, 12}, CPU, as the README's
             level-2 command; classical rows compared with the committed
             file (exact), QAOA rows compared (CPU double against the
             committed GPU single)
  level3     a block of the E1 cross on the CPU (N = 12, instances 0-1,
             seeds 42-43: 24 runs), compared column by column with the
             committed GPU rows; the per-run wall-clock is used with the
             committed ``runtime_s`` column to state the full cross's cost
  interrupt  the same block run with ``--resume``, killed with SIGKILL part
             way, resumed, and the result compared row for row with the
             uninterrupted block of ``level3`` (both CPU double on this host,
             so they must be identical); the rows that survived the kill are
             the work the per-run flush saved, and the control (no flush, no
             resume) is what a restart would have had to redo
  status     (working repository, not the clone) the driver's own status
             record around a kill and a resume: RUNNING n/total after SIGKILL
             and the aggregation refusing the partial manifest; EXIT 0 n/n
             after resume and the aggregation running
  offline    the level-2 command in a copy of the clone with ``data/derived``
             removed and the network unreachable: what a package without the
             derived statistics does for a reader without the price snapshot
  missing    which committed results the clone cannot regenerate and why
             (metered backends, the Zenodo artifacts): ``analyze_conditional``
             run without ``e1_runs`` as the measured instance

Nothing here touches the working repository or AWS; the clone lives under
``--work-dir`` (the session scratchpad by default) and is deleted only on
request. The venv step downloads packages, so it needs the network once.

Run:
    python -m experiments.system_paper.es3_reconstruction                 # ~15 min (pip from cache)
    python -m experiments.system_paper.es3_reconstruction --only clone venv tests
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.paper01_qubo_baseline.run_experiment import write_csv

RESULTS_DIR = Path(__file__).parent / "results"
SRC = Path(__file__).resolve().parents[2]
STEPS_CSV = RESULTS_DIR / "paper02_es3_steps.csv"
COLUMNS_CSV = RESULTS_DIR / "paper02_es3_column_agreement.csv"
#: What a reader has: the published repository, over the network. A local path
#: works too (``--source``), and the first runs of this harness used the local
#: mirror — which is not what a reader would clone, so the default is the URL.
PUBLIC_REPO = "https://github.com/tobitaQ/qaoa-portfolio-penalty-benchmark.git"
RELEASE_TAG = "v1.0"
E1_BLOCK = ["--n", "12", "--instances", "0", "1", "--seeds", "42", "43"]
#: Columns of a run row that derive from the exact state or the optimiser,
#: compared across hosts; the shot-derived columns are a draw and are listed
#: separately so the table can say which kind moved.
STATE_COLUMNS = ["final_expectation", "expectation_at_final_angles", "final_P_F",
                 "final_P_opt", "final_P_tau_main", "initial_P_F", "final_Q_tau_main_1000",
                 "hamiltonian_scale", "f_star", "delta_F", "a_crit", "a_heur", "a_margin"]
SHOT_COLUMNS = ["feasible_shots", "shots_optimal", "best_all_E", "g_F", "g_off"]
ANGLE_COLUMNS = ["initial_angles", "final_angles"]


@dataclass
class Step:
    step: str
    level: str
    command: str
    seconds: float
    exit_code: int | str
    outcome: str
    metric: str = ""
    manual_inputs: str = ""     # what the third party had to supply or decide


class Harness:
    def __init__(self, work_dir: Path, log=print, source: str | Path = PUBLIC_REPO,
                 tag: str = RELEASE_TAG):
        self.work = work_dir
        self.source = source
        self.tag = tag
        self.clone = work_dir / "clone"
        self.venv = work_dir / "venv"
        self.py = self.venv / "bin" / "python"
        self.log = log
        self.steps: list[Step] = []
        self.columns: list[dict] = []

    # -- plumbing ---------------------------------------------------------


    def _portable(self, text: str) -> str:
        """Paths as a reader would see them: the record ships in the package.

        The absolute paths of this host say nothing a reader can use and carry a
        home directory into a published file, so the work directory, the clone
        and the source tree are written as the placeholders they stand for.
        """
        for real, name in ((str(self.clone), "<clone>"), (str(self.work), "<work>"),
                           (str(SRC), "<repo>"), (str(Path.home()), "~")):
            text = text.replace(real, name)
        return text

    def run(self, step: str, level: str, cmd: list[str], cwd: Path | None = None,
            env: dict | None = None, timeout: int = 7200, manual: str = "",
            outcome_fn=None) -> subprocess.CompletedProcess:
        shown = self._portable(" ".join(str(c) for c in cmd))
        self.log(f"[{step}] $ {shown}")
        t0 = time.perf_counter()
        full_env = dict(os.environ)
        full_env.pop("PYTHONPATH", None)
        if env:
            full_env.update(env)
        try:
            proc = subprocess.run(cmd, cwd=cwd or self.clone, env=full_env, timeout=timeout,
                                  capture_output=True, text=True)
            code: int | str = proc.returncode
        except subprocess.TimeoutExpired as exc:
            proc = subprocess.CompletedProcess(cmd, -1, exc.stdout or "", exc.stderr or "")
            code = "timeout"
        secs = time.perf_counter() - t0
        outcome, metric = (outcome_fn(proc) if outcome_fn else
                           ("ok" if code == 0 else "failed", ""))
        self.steps.append(Step(step, level, shown, round(secs, 1), code, outcome,
                               self._portable(str(metric)), manual))
        self.log(f"    -> {outcome} ({secs:.0f} s) {metric}")
        return proc

    def _git(self, *args, cwd=None) -> str:
        return subprocess.run(["git", *args], cwd=cwd or self.clone, capture_output=True,
                              text=True, check=True).stdout.strip()

    # -- steps ------------------------------------------------------------

    def step_clone(self, source: str | Path | None = None) -> None:
        if self.clone.exists():
            shutil.rmtree(self.clone)
        self.run("clone", "0-setup", ["git", "clone", "--quiet", "--branch", self.tag,
                                      str(source or self.source), str(self.clone)], cwd=self.work,
                 manual="repository URL and release tag from the availability statement")
        commit = self._git("rev-parse", "--short", "HEAD")
        self.steps[-1].metric = f"commit {commit}"
        size = sum(f.stat().st_size for f in self.clone.rglob("*") if f.is_file()
                   and ".git" not in f.parts)
        self.steps[-1].metric += f"; {size / 1e6:.0f} MB of files"

    def step_venv(self) -> None:
        if self.venv.exists():
            shutil.rmtree(self.venv)
        self.run("venv", "0-setup", [sys.executable, "-m", "venv", str(self.venv)],
                 cwd=self.work)
        self.run("pip", "0-setup", [str(self.py), "-m", "pip", "install", "-q", "-r",
                                    "requirements-repro.txt"], timeout=3600,
                 manual="none; the README names the file (CUDA 12 Linux host)")
        proc = subprocess.run([str(self.py), "-c",
                               "import pennylane, numpy, matplotlib; "
                               "print(pennylane.__version__, numpy.__version__, matplotlib.__version__)"],
                              capture_output=True, text=True)
        self.steps[-1].metric = f"pennylane/numpy/matplotlib {proc.stdout.strip()}"

    def step_tests(self) -> None:
        """The suite, with the reason for every skip.

        A count of skips says nothing about what went unchecked, so ``-rs``
        prints the reason of each and the metric carries them grouped: a reader
        can see which of the paper's claims the clone alone does not exercise.
        """
        def outcome(proc):
            m = re.search(r"(\d+) passed(?:, (\d+) skipped)?", proc.stdout)
            why: dict[str, int] = {}
            for line in proc.stdout.splitlines():
                s = re.match(r"^SKIPPED \[(\d+)\].*?:\d+: (.*)$", line.strip())
                if s:
                    why[s.group(2)] = why.get(s.group(2), 0) + int(s.group(1))
            grouped = "; ".join(f"{v} {k}" for k, v in sorted(why.items(), key=lambda kv: -kv[1]))
            return (("ok" if proc.returncode == 0 else "failed"),
                    (f"{m.group(1)} passed, {m.group(2) or 0} skipped"
                     + (f" ({grouped})" if grouped else "") if m else proc.stdout[-200:]))
        self.run("tests", "0-setup",
                 [str(self.py), "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider"],
                 timeout=3600, outcome_fn=outcome)

    def step_level1(self) -> None:
        fig_dir = self.clone / "notebooks" / "figures"
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in fig_dir.rglob("*.png")}
        t_start = time.time()

        def fig_outcome(proc):
            if proc.returncode != 0:
                return "failed", proc.stderr[-300:]
            # Files the script rewrote (mtime after the start), then how many
            # of those came back byte-identical to the committed render.
            rewritten = [p for p in fig_dir.rglob("*.png") if p.stat().st_mtime >= t_start]
            same = [p.name for p in rewritten
                    if p in before and hashlib.sha256(p.read_bytes()).hexdigest() == before[p]]
            changed = [p.name for p in rewritten if p.name not in same]
            return ("ok", f"{len(same)}/{len(rewritten)} regenerated PNG renders byte-identical "
                          f"to the committed ones" + (f"; changed: {changed}" if changed else ""))
        self.run("level1-figures", "1-aggregates", [str(self.py), "-m", "notebooks.generate_tqe_figures"],
                 timeout=1800, outcome_fn=fig_outcome, manual="none")

        def tab_outcome(proc):
            m = re.search(r"(\d+) cell\(s\) agree with the result files, (\d+) disagree, (\d+) pending",
                          proc.stdout)
            if not m:
                return "failed", proc.stdout[-300:] + proc.stderr[-300:]
            return ("ok" if m.group(2) == "0" else "mismatch",
                    f"{m.group(1)} cells agree, {m.group(2)} disagree, {m.group(3)} pending")
        self.run("level1-tables", "1-aggregates", [str(self.py), "check_tables.py"],
                 cwd=self.clone / "papers", timeout=1800, outcome_fn=tab_outcome, manual="none")

    def step_level2(self) -> None:
        results = self.clone / "experiments" / "paper01_qubo_baseline" / "results"
        cmd = [str(self.py), "-m", "experiments.paper01_qubo_baseline.run_experiment",
               "--backend", "lightning_cpu", "--classical-n", "8", "12", "--quantum-n", "8", "12",
               "--results-tag", "es3"]

        def outcome(proc):
            if proc.returncode != 0:
                return "failed", proc.stderr[-300:]
            got = pd.read_csv(results / "paper01_results_es3.csv", float_precision="round_trip")
            ref = pd.read_csv(results / "paper01_results.csv", float_precision="round_trip")
            ref = ref[ref.n_assets.isin([8, 12])]
            parts = []
            for solver_got, solver_ref in (("classical_auto", "classical_auto"),
                                           ("lightning_cpu", "lightning_gpu")):
                g = got[got.solver == solver_got].set_index("n_assets")
                r = ref[ref.solver == solver_ref].set_index("n_assets")
                d = (g.energy_no_offset - r.energy_no_offset).abs().max()
                sh = (g.sharpe_ratio - r.sharpe_ratio).abs().max()
                parts.append(f"{solver_got} vs committed {solver_ref}: |dE| max {d:.1e}, |dSharpe| max {sh:.1e}")
            return "ok", "; ".join(parts)
        self.run("level2", "2-one-instance", cmd, timeout=3600, outcome_fn=outcome,
                 manual="a --results-tag name (the README suggests one)")

    def _compare_block(self, got_csv: Path, label: str) -> str:
        results = self.clone / "experiments" / "paper01_qubo_baseline" / "results"
        got = pd.read_csv(got_csv, float_precision="round_trip")
        ref = pd.read_csv(results / "paper01_e1_cross.csv", float_precision="round_trip")
        key = ["n_assets", "instance", "penalty_label", "qaoa_seed"]
        m = got.merge(ref, on=key, suffixes=("_cpu", "_gpu"))
        for group, cols in (("state", STATE_COLUMNS), ("shots", SHOT_COLUMNS)):
            for c in cols:
                a, b = m[f"{c}_cpu"], m[f"{c}_gpu"]
                if a.dtype.kind in "fi" and b.dtype.kind in "fi":
                    diff = (a - b).abs()
                    rel = (diff / b.abs()).where(b != 0)   # undefined against a zero reference
                    self.columns.append({"comparison": label, "group": group, "column": c,
                                         "rows": int(len(m)), "identical_rows": int((diff == 0).sum()),
                                         "max_abs_diff": float(diff.max()),
                                         "max_rel_diff": (float(rel.max()) if rel.notna().any() else "")})
        for c in ANGLE_COLUMNS:
            same = int((m[f"{c}_cpu"] == m[f"{c}_gpu"]).sum())
            self.columns.append({"comparison": label, "group": "angles", "column": c,
                                 "rows": int(len(m)), "identical_rows": same,
                                 "max_abs_diff": "", "max_rel_diff": ""})
        mine = [c for c in self.columns if c["comparison"] == label]
        state_max = max(c["max_abs_diff"] for c in mine if c["group"] == "state")
        shots_ident = [c for c in mine if c["group"] == "shots" and c["identical_rows"] == c["rows"]]
        init_same = next(c["identical_rows"] for c in mine if c["column"] == "initial_angles")
        return (f"{len(m)} rows matched; initial_angles identical in {init_same}; state columns "
                f"max |diff| {state_max:.1e}; shot columns identical in {len(shots_ident)}/{len(SHOT_COLUMNS)}")

    def step_level3(self) -> None:
        results = self.clone / "experiments" / "paper01_qubo_baseline" / "results"
        cmd = [str(self.py), "-m", "experiments.paper01_qubo_baseline.run_e1_cross",
               "--backend", "lightning_cpu", "--precision", "double", *E1_BLOCK,
               "--results-tag", "es3block"]

        def outcome(proc):
            if proc.returncode != 0:
                return "failed", proc.stderr[-400:]
            got = pd.read_csv(results / "paper01_e1_cross_es3block.csv")
            per_run = got.runtime_s.mean()
            ref = pd.read_csv(results / "paper01_e1_cross.csv")
            gpu_hours = ref.runtime_s.sum() / 3600
            note = (f"{len(got)} runs, {per_run:.1f} s/run on the CPU at N=12; the committed "
                    f"cross records {ref.runtime_s.sum() / 3600:.1f} h of run time for "
                    f"{len(ref)} runs on the GPU")
            return "ok", note + " | " + self._compare_block(results / "paper01_e1_cross_es3block.csv",
                                                             "level3 cpu-double vs committed gpu-single")
        self.run("level3-block", "3-cross", cmd, timeout=7200, outcome_fn=outcome,
                 manual="a --results-tag; the backend/precision to use (README states the GPU ones)")

    def step_interrupt(self, kill_after_runs: int = 7) -> None:
        results = self.clone / "experiments" / "paper01_qubo_baseline" / "results"
        out_csv = results / "paper01_e1_cross_es3resume.csv"
        for p in (out_csv, results / "paper01_e1_reference_es3resume.csv"):
            p.unlink(missing_ok=True)
        shutil.rmtree(results / "e1_runs_es3resume", ignore_errors=True)
        cmd = [str(self.py), "-u", "-m", "experiments.paper01_qubo_baseline.run_e1_cross",
               "--backend", "lightning_cpu", "--precision", "double", *E1_BLOCK,
               "--results-tag", "es3resume", "--resume"]
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        self.log(f"[interrupt] $ {' '.join(cmd)}   (SIGKILL after {kill_after_runs} runs)")
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, cwd=self.clone, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        completed = 0
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if re.search(r"\d+\.\ds$", line.strip()):
                completed += 1
                if completed >= kill_after_runs:
                    os.kill(proc.pid, signal.SIGKILL)
                    break
        proc.wait()
        t_kill = time.perf_counter() - t0
        survived = len(pd.read_csv(out_csv)) if out_csv.exists() else 0
        self.steps.append(Step("interrupt-kill", "3-cross",
                               self._portable(" ".join(cmd)) + "  # SIGKILL",
                               round(t_kill, 1), "SIGKILL",
                               "ok" if survived == completed else "rows lost",
                               f"{completed} runs completed before the kill, {survived} rows on disk; "
                               f"control without per-run flush: 0 rows, {t_kill:.0f} s to redo",
                               manual_inputs="none"))
        self.log(f"    -> killed after {completed} runs; {survived} rows on disk ({t_kill:.0f} s)")

        def outcome(p):
            if p.returncode != 0:
                return "failed", p.stderr[-300:]
            resumed = pd.read_csv(out_csv, float_precision="round_trip")
            straight = pd.read_csv(results / "paper01_e1_cross_es3block.csv", float_precision="round_trip")
            key = ["n_assets", "instance", "penalty_label", "qaoa_seed"]
            a = resumed.sort_values(key).reset_index(drop=True)
            b = straight.sort_values(key).reset_index(drop=True)
            skip = {"runtime_s", "optimisation_seconds", "sampling_seconds", "state_seconds",
                    "evaluation_seconds", "run_id"}
            cols = [c for c in a.columns if c in b.columns and c not in skip]
            differing = [c for c in cols if not a[c].equals(b[c])]
            m = re.search(r"(\d+) rows already done", p.stdout)
            return (("ok" if not differing and len(a) == len(b) else "differs"),
                    f"resumed with {m.group(1) if m else '?'} rows already done; "
                    f"{len(a)} rows vs {len(b)} uninterrupted; "
                    f"{len(cols) - len(differing)}/{len(cols)} non-timing columns identical"
                    + (f"; differing: {differing[:5]}" if differing else ""))
        self.run("interrupt-resume", "3-cross", cmd, timeout=7200, outcome_fn=outcome,
                 manual="re-issue the same command (README: --resume is the restart path)")

    def step_status(self, kill_after_runs: int = 5) -> None:
        """The job's own status record around a kill and a resume.

        Not in the ``v1.0`` clone: the record was added after the first E-S3
        pass, when the evaluation's own harness failed unnoticed for nine
        hours. It runs in the clone when the clone carries it (``v1.1`` and
        later) and in the working tree otherwise, and the step says which. The
        driver is killed, its status file read (RUNNING, n/total, last run), the
        aggregation asked to run (it must refuse), the driver resumed, and both
        read again.
        """
        import json
        repo = Path(__file__).resolve().parents[2]
        py = repo / ".venv" / "bin" / "python"
        where = "the working repository"
        if (self.clone / "src" / "run_status.py").exists() and self.py.exists():
            repo, py, where = self.clone, self.py, f"the {self.tag} clone"
        results = repo / "experiments" / "paper01_qubo_baseline" / "results"
        tag = "es3status"
        manifest = results / f"paper01_e1_cross_{tag}.csv"
        status_file = Path(str(manifest) + ".status.json")
        produced = [manifest, results / f"paper01_e1_reference_{tag}.csv", status_file,
                    *results.glob(f"paper01_e1_*_{tag}.csv")]
        for p in produced:
            p.unlink(missing_ok=True)
        shutil.rmtree(results / f"e1_runs_{tag}", ignore_errors=True)
        cmd = [str(py), "-u", "-m", "experiments.paper01_qubo_baseline.run_e1_cross",
               "--backend", "lightning_cpu", "--precision", "double", "--n", "12",
               "--instances", "0", "--seeds", "42", "43", "--results-tag", tag, "--resume"]
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        completed = 0
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if re.search(r"\d+\.\ds$", line.strip()):
                completed += 1
                if completed >= kill_after_runs:
                    os.kill(proc.pid, signal.SIGKILL)
                    break
        proc.wait()
        rec = json.loads(status_file.read_text())
        after_kill = f"state={rec['state']} done={rec['done']}/{rec['total']} last={rec['last']}"
        # What the kill cost, on the driver as it is now (the clone above runs v1.0,
        # whose flush rewrote the file in place). The manifest must parse, its rows
        # must be the runs the status file claims, and an artifact written for a run
        # whose row never landed is the orphan the interruption leaves behind.
        import csv as _csv
        with open(manifest, newline="", encoding="utf-8") as f:
            rows_on_disk = list(_csv.DictReader(f))
        ids = {r["run_id"] for r in rows_on_disk}
        arts = {q.stem for q in (results / f"e1_runs_{tag}").glob("*.npz")}
        after_kill += (f"; manifest parses with {len(rows_on_disk)} rows"
                       f"; {len(arts - ids)} artifact(s) without a row")
        agg = subprocess.run([str(py), "-m", "experiments.paper01_qubo_baseline.analyze_e1",
                              "--results-tag", tag], cwd=repo, env=env, capture_output=True, text=True)
        refused = agg.returncode != 0 and "not complete" in (agg.stderr + agg.stdout)
        self.steps.append(Step("status-after-kill", "3-cross",
                               self._portable(" ".join(cmd)) + "  # SIGKILL",
                               round(time.perf_counter() - t0, 1), "SIGKILL",
                               "ok" if rec["state"] == "RUNNING" and refused else "unexpected",
                               f"in {where}: status file after the kill: {after_kill}; analyze_e1 "
                               f"{'refused the partial manifest' if refused else 'DID NOT refuse'}",
                               manual_inputs="none"))
        self.log(f"    -> {after_kill}; aggregation {'refused' if refused else 'ran'}")

        def outcome(p):
            rec2 = json.loads(status_file.read_text())
            agg2 = subprocess.run([str(py), "-m", "experiments.paper01_qubo_baseline.analyze_e1",
                                   "--results-tag", tag], cwd=repo, env=env, capture_output=True, text=True)
            ok = (p.returncode == 0 and rec2["state"] == "EXIT" and rec2["exit_code"] == 0
                  and rec2["done"] == rec2["total"] and agg2.returncode == 0)
            m = re.search(r"previous job: (.*)", p.stdout)
            return ("ok" if ok else "unexpected",
                    f"resume printed '{m.group(1) if m else '?'}'; status after resume: "
                    f"state={rec2['state']} exit={rec2['exit_code']} done={rec2['done']}/{rec2['total']}; "
                    f"analyze_e1 {'ran' if agg2.returncode == 0 else 'refused'}")
        shown = self._portable(" ".join(cmd))
        t1 = time.perf_counter()
        p = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=3600)
        outc, metric = outcome(p)
        self.steps.append(Step("status-after-resume", "3-cross", shown, round(time.perf_counter() - t1, 1),
                               p.returncode, outc, metric, "re-issue the same command"))
        self.log(f"    -> {outc}: {metric}")
        for p_ in produced + list(results.glob(f"paper01_e1_*_{tag}.csv")):
            Path(p_).unlink(missing_ok=True)
        shutil.rmtree(results / f"e1_runs_{tag}", ignore_errors=True)

    def step_offline(self) -> None:
        copy = self.work / "clone_noderived"
        if copy.exists():
            shutil.rmtree(copy)
        shutil.copytree(self.clone, copy, ignore=shutil.ignore_patterns(".git", "e1_runs*", "*.npz"))
        shutil.rmtree(copy / "data" / "derived")
        cmd = [str(self.py), "-m", "experiments.paper01_qubo_baseline.run_experiment",
               "--backend", "lightning_cpu", "--classical-n", "8", "--quantum-n", "8",
               "--results-tag", "es3offline"]
        # An unreachable proxy stands in for "no network": the loader's only
        # remaining source is the market-data provider.
        env = {"HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9"}

        def outcome(proc):
            text = (proc.stdout + proc.stderr)[-400:]
            return ("cannot reproduce offline" if proc.returncode != 0 else "unexpected success",
                    text.strip().splitlines()[-1] if text.strip() else "")
        self.run("offline-no-derived", "control", cmd, cwd=copy, env=env, timeout=600,
                 outcome_fn=outcome, manual="a price snapshot from the provider (revised since)")

    def step_missing(self) -> None:
        cmd = [str(self.py), "-m", "experiments.paper01_qubo_baseline.analyze_conditional"]

        def outcome(proc):
            text = (proc.stdout + proc.stderr).strip()
            return (("needs the Zenodo artifacts" if proc.returncode != 0 or "0 " in text[:80]
                     else "ran"), text.splitlines()[-1][:200] if text else "")
        self.run("missing-npz", "missing", cmd, timeout=600, outcome_fn=outcome,
                 manual="download e1_runs*.npz from the Zenodo deposit (≈100 MB)")
        results = self.clone / "experiments" / "paper01_qubo_baseline" / "results"
        metered = sorted(p.name for p in results.glob("*sv1*.csv")) + \
            sorted(p.name for p in results.glob("*ionq*.csv"))
        self.steps.append(Step("missing-metered", "missing", "(none: listed from the README)", 0.0,
                               "n/a", "not regenerable without an AWS account and spend",
                               f"{len(metered)} result files from SV1/IonQ ($37.90 in total); the "
                               f"raw IonQ task records are committed under data/braket_results/",
                               manual_inputs="AWS account, region us-east-1, service-linked role, budget"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default_work = Path(os.environ.get("ES3_WORK_DIR", RESULTS_DIR / "_es3"))
    p.add_argument("--work-dir", type=Path, default=default_work)
    p.add_argument("--source", default=PUBLIC_REPO,
                   help="the repository to clone (default: the public URL)")
    p.add_argument("--tag", default=RELEASE_TAG, help="the release tag to clone")
    p.add_argument("--only", nargs="+", default=None,
                   choices=["clone", "venv", "tests", "level1", "level2", "level3",
                            "interrupt", "status", "offline", "missing"])
    p.add_argument("--keep", action="store_true", help="keep the work directory")
    p.add_argument("--out-suffix", default="",
                   help="write the step record to paper02_es3_steps<suffix>.csv; with it, a "
                        "partial run (--only) is recorded too")
    args = p.parse_args(argv)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    h = Harness(args.work_dir, source=args.source, tag=args.tag)
    order = ["clone", "venv", "tests", "level1", "level2", "level3", "interrupt", "status", "offline", "missing"]
    todo = [s for s in order if args.only is None or s in args.only]
    for name in todo:
        getattr(h, f"step_{name}")()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.only is None or args.out_suffix:
        steps_csv = STEPS_CSV.with_name(STEPS_CSV.stem + args.out_suffix + ".csv")
        write_csv(steps_csv, [asdict(s) for s in h.steps])
        print(f"wrote {steps_csv.name} ({len(h.steps)})")
        if h.columns:
            cols_csv = COLUMNS_CSV.with_name(COLUMNS_CSV.stem + args.out_suffix + ".csv")
            write_csv(cols_csv, h.columns)
            print(f"wrote {cols_csv.name} ({len(h.columns)})")
    else:
        for s in h.steps:
            print(asdict(s))
    if not args.keep and args.only is None:
        shutil.rmtree(args.work_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
