# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT
"""The repetition table of the account-level control is a function of the task records.

Two cells of the first repetition are not in the summary written while it ran, so the paper
reports recomputed values. This pins the recomputation: every field the later summaries do
hold must come back from the task records (to within the second the stored timestamps are
truncated to), and the two missing cells must be the ones the paper prints.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from experiments.system_paper import es2_cost_control_aggregate as agg

RUNS = agg.RUNS


@pytest.mark.parametrize("run", RUNS)
def test_recomputation_matches_every_field_the_live_summary_holds(run):
    live = pd.read_csv(agg.RESULTS / f"paper02_es2_cost_control_summary_run{run}.csv").iloc[0]
    rec = agg.recomputed(run, datetime.fromisoformat(live.alarm_to_ALARM_utc),
                         datetime.fromisoformat(live.first_denial_utc))
    for field, value in rec.items():
        if field not in live.index or pd.isna(live[field]) or live[field] == "":
            continue
        assert abs(type(value)(live[field]) - value) <= agg.TOLERANCE.get(field, 0), field


def test_the_two_cells_missing_from_run_1_are_recomputed_as_the_paper_reports_them():
    live = pd.read_csv(agg.RESULTS / "paper02_es2_cost_control_summary_run1.csv").iloc[0]
    assert "seconds_limit_to_alarm" not in live.index
    assert "tasks_created_after_first_denial" not in live.index
    rec = agg.recomputed(1, datetime.fromisoformat(live.alarm_to_ALARM_utc),
                         datetime.fromisoformat(live.first_denial_utc))
    assert rec["seconds_limit_to_alarm"] == 96
    assert rec["tasks_created_after_first_denial"] == 2


def test_the_committed_table_carries_five_runs_and_the_medians_of_table_8():
    reps = pd.read_csv(agg.RESULTS / "paper02_es2_cost_control_repetitions.csv")
    summary = pd.read_csv(agg.RESULTS / "paper02_es2_cost_control_repetitions_summary.csv").set_index("quantity")
    assert list(reps.run) == list(RUNS)
    assert reps.first_task_over_limit.eq(81).all()
    assert summary.loc["tasks_created_after_limit_crossed", "median"] == 47
    assert summary.loc["seconds_limit_to_alarm", "median"] == 94
    assert summary.loc["tasks_created_after_alarm", "median"] == 10
    assert summary.loc["tasks_created_after_first_denial", "median"] == 4
