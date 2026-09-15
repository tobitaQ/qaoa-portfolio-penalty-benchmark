# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""The billed task must be findable even when the plugin does not record it.

Measured 2026-08-10: the N = 16 IonQ run submitted a task, the task completed,
$8.30 was charged, and the run then died with ``'NoneType' object has no
attribute 'id'``. ``dev.task`` is only assigned in the plugin's ``execute``
path; a QNode returning one sample per wire goes through ``batch_execute``,
which keeps the batch in a local variable. The earlier S3 fix was written
against ``dev.task`` and had never been exercised on hardware, so it failed the
first time it mattered.

These tests exercise the recovery path without submitting anything.
"""

from datetime import datetime, timezone

import pytest

from src.solvers.braket_results import resolve_task_arn

ARN = "arn:aws:braket:us-east-1:1:quantum-task/abc"
DEVICE = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"
SINCE = datetime(2026, 8, 10, tzinfo=timezone.utc)


class _Task:
    id = ARN


class _Dev:
    """Stand-in for the PennyLane device: ``task`` set or not, plus ``_device``."""

    def __init__(self, task=None):
        self.task = task
        self._device = type("D", (), {"arn": DEVICE})()


def _fake_boto(monkeypatch, arns):
    """Make boto3 return `arns` from the paginated task search."""
    class _Paginator:
        def paginate(self, filters):
            self.filters = filters
            return [{"quantumTasks": [{"quantumTaskArn": a} for a in arns]}]

    class _Client:
        def get_paginator(self, name):
            return _Paginator()

    class _Session:
        def __init__(self, region_name=None):
            self.region_name = region_name

        def client(self, name):
            return _Client()

    import boto3
    monkeypatch.setattr(boto3, "Session", _Session)


def test_uses_the_plugin_handle_when_it_is_there(monkeypatch):
    """The cheap path stays the first one, and does not call AWS."""
    def explode(*a, **k):  # pragma: no cover - must not run
        raise AssertionError("searched AWS despite dev.task being set")

    import boto3
    monkeypatch.setattr(boto3, "Session", explode)
    assert resolve_task_arn(_Dev(_Task()), SINCE) == ARN


def test_finds_the_billed_task_when_the_plugin_forgot_it(monkeypatch):
    """batch_execute leaves dev.task None; the task is still ours to find."""
    _fake_boto(monkeypatch, [ARN])
    assert resolve_task_arn(_Dev(), SINCE) == ARN


@pytest.mark.parametrize("arns", [[], [ARN, ARN + "2"]])
def test_refuses_to_guess(monkeypatch, arns):
    """Zero or many means the one-task assumption is wrong. Say so, loudly.

    Picking one would attribute another experiment's shots to this row, which
    is worse than failing: the number would look plausible.
    """
    _fake_boto(monkeypatch, arns)
    with pytest.raises(RuntimeError, match="expected exactly 1 task"):
        resolve_task_arn(_Dev(), SINCE)
    

def test_names_the_recovery_path_so_billed_data_is_not_lost(monkeypatch):
    _fake_boto(monkeypatch, [])
    with pytest.raises(RuntimeError, match="recover_task.py"):
        resolve_task_arn(_Dev(), SINCE)
