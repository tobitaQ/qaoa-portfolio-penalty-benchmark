# SPDX-FileCopyrightText: 2026 Hiroaki Tobita
# SPDX-License-Identifier: MIT

"""
Reading measurements back out of a completed Amazon Braket task.

Braket returns two result schemas and they nest differently:

    SV1   ``braket.task_result.gate_model_task_result``
          -> ``measurements``: one row per shot, at the top level

    IonQ  ``braket.task_result.program_set_task_result``
          -> ``programs/0/results.json`` -> ``executables/0.json``
             -> ``measurementProbabilities``

Measured 2026-08-09 on IonQ Forte-1: the PennyLane-Braket plugin (1.35) does not
read the second form. It does not raise either -- it returns an array that is
not the device's measurements, and the run reports a bitstring of zeros, an
energy of exactly 0.0 and a 100 % optimality gap. That is a plausible-looking
row rather than a visible failure, and it happened on all three hardware tasks
while the correct shots sat in S3 the whole time (60, 79 and 92 distinct
bitstrings out of 100).

At $8.30 per IonQ task, "the framework could not parse it" is not an acceptable
way to lose data. ``BraketSolver`` therefore reads hardware shots from S3 itself
rather than trusting the plugin, and ``recover_task.py`` uses the same code to
rescue tasks that were already billed.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


def _read_s3_json(session, bucket: str, key: str) -> dict:
    """Fetch and parse one JSON object from S3."""
    body = session.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body)


def _counts_from_result(result: dict, shots: int, read_json) -> Counter:
    """Normalise either result schema to bitstring -> shot count.

    Args:
        result: Parsed ``results.json`` at the root of the task's output.
        shots: Shot total the task was submitted with, used to turn the
            program-set schema's probabilities back into counts.
        read_json: Callable taking a path relative to the task's output root
            and returning the parsed JSON there.

    Returns:
        Mapping from bitstring (qubit 0 first) to number of shots.

    Raises:
        RuntimeError: If the result holds no measurements.
    """
    schema = result.get("braketSchemaHeader", {}).get("name", "")

    if "program_set" in schema:
        program = read_json(result["programResults"][0])
        leaf = read_json(f"programs/0/{program['executableResults'][0]}")
        probabilities = leaf.get("measurementProbabilities")
        if not probabilities:
            raise RuntimeError("program-set result carries no measurements")
        # Probabilities are shot counts over the shot total; recover the counts
        # so both schemas hand back the same object.
        return Counter({bits: int(round(p * shots))
                        for bits, p in probabilities.items()})

    measurements = result.get("measurements")
    if not measurements:
        raise RuntimeError(f"no measurements in result of schema {schema!r}")
    return Counter("".join(str(int(b)) for b in row) for row in measurements)


def shot_counts_from_dir(task_dir: "str | Path") -> Counter:
    """Return shot counts for a task archived to a local directory.

    Reads the same two schemas as :func:`shot_counts_from_task` from a copy of
    the task's S3 prefix, so a run stays readable after the results bucket is
    torn down. The shot total comes from the archived ``metadata.json`` rather
    than from the Braket API.

    Args:
        task_dir: Directory holding the task's ``results.json``, and for the
            program-set schema the ``programs/`` tree beneath it.

    Returns:
        Mapping from bitstring (qubit 0 first) to number of shots.

    Raises:
        FileNotFoundError: If the directory holds no ``results.json``.
        RuntimeError: If the result holds no measurements.
    """
    root = Path(task_dir)
    read_json = lambda rel: json.loads((root / rel).read_text())  # noqa: E731
    result = read_json("results.json")

    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        shots = int(json.loads(metadata_path.read_text())["requestedShots"])
    else:
        shots = int(result["taskMetadata"]["shots"])
    return _counts_from_result(result, shots, read_json)


def shot_counts_from_task(task_arn: str, region: str = "us-east-1") -> Counter:
    """Return ``Counter`` of measured bitstring -> shot count for a Braket task.

    Handles both result schemas, normalising them to shot counts so callers do
    not have to know which device ran. Bit *j* of each string is qubit *j*, and
    a '1' is the ``|1>`` state.

    Args:
        task_arn: ARN of a COMPLETED quantum task.
        region: Braket region holding the task.

    Returns:
        Mapping from bitstring (qubit 0 first) to number of shots.

    Raises:
        RuntimeError: If the task is not complete or holds no measurements.
    """
    import boto3

    session = boto3.Session(region_name=region)
    task = session.client("braket").get_quantum_task(quantumTaskArn=task_arn)
    if task["status"] != "COMPLETED":
        raise RuntimeError(f"task is {task['status']}, not COMPLETED")
    bucket = task["outputS3Bucket"]
    prefix = task["outputS3Directory"]
    shots = int(task["shots"])

    read_json = lambda rel: _read_s3_json(session, bucket, f"{prefix}/{rel}")  # noqa: E731
    return _counts_from_result(read_json("results.json"), shots, read_json)


def execution_duration_ms(task_arn: str, region: str = "us-east-1") -> float:
    """Return the simulator's billed execution time for a completed task.

    SV1 bills ``executionDuration`` (milliseconds) rounded up to a 3-second
    minimum, and the number sits in the task's ``results.json`` under
    ``additionalMetadata.simulatorMetadata``. Recording it per task is what
    makes a run's bill auditable from its own record rather than from the
    monthly statement.

    Args:
        task_arn: ARN of a COMPLETED quantum task.
        region: Braket region holding the task.

    Returns:
        Execution duration in milliseconds, or ``nan`` if the result carries
        no simulator metadata (a QPU task).
    """
    import boto3

    session = boto3.Session(region_name=region)
    task = session.client("braket").get_quantum_task(quantumTaskArn=task_arn)
    result = _read_s3_json(session, task["outputS3Bucket"],
                           f"{task['outputS3Directory']}/results.json")
    meta = result.get("additionalMetadata", {}).get("simulatorMetadata", {})
    return float(meta.get("executionDuration", float("nan")))


def samples_from_counts(counts: Counter, n: int) -> np.ndarray:
    """Expand bitstring counts into one row per shot.

    Args:
        counts: Bitstring -> shot count, qubit 0 first.
        n: Expected number of qubits.

    Returns:
        Array of shape ``(total shots, n)`` holding 0/1 floats.

    Raises:
        RuntimeError: If a bitstring does not have ``n`` bits, or none were read.
    """
    rows = []
    for bits, count in sorted(counts.items()):
        if len(bits) != n:
            raise RuntimeError(
                f"measured bitstring {bits!r} has {len(bits)} bits, expected {n}")
        row = [float(c) for c in bits]
        rows.extend([row] * count)
    if not rows:
        raise RuntimeError("task returned no shots")
    return np.asarray(rows, dtype=float)


def resolve_task_arn(dev, since, region: str = "us-east-1") -> str:
    """Return the ARN of the billed task ``dev`` just ran.

    ``dev.task`` is the obvious source and it is correct when it is set, but the
    PennyLane-Braket plugin only assigns it in ``execute``. A QNode that returns
    one ``qml.sample`` per wire goes through ``batch_execute`` instead, whose
    ``_run_task_batch`` keeps the batch in a local variable and never stores it
    on the device -- so ``dev.task`` is ``None`` and reading ``dev.task.id``
    raises ``AttributeError`` *after* the task has run and been billed.
    Measured 2026-08-10: the N = 16 IonQ run submitted, completed and charged
    $8.30, then failed to read its own result.

    Rather than reach further into plugin internals, ask Braket. The task we
    want is the one created on this device since ``since``, and on a QPU a run
    submits exactly one. Anything else -- none, or more than one -- means the
    assumption is wrong, so raise with the ARNs rather than guess: picking the
    wrong task would silently attribute one experiment's shots to another.

    Args:
        dev: the PennyLane device the circuit ran on.
        since: UTC datetime taken immediately before the run.
        region: Braket region the task was submitted to.

    Returns:
        The task ARN.

    Raises:
        RuntimeError: if the device ran anything other than exactly one task.
    """
    task = getattr(dev, "task", None)
    if task is not None:
        return task.id

    import boto3

    device_arn = getattr(getattr(dev, "_device", None), "arn", None)
    if device_arn is None:
        raise RuntimeError(
            "cannot identify the device that ran this circuit, so the billed "
            "task cannot be located; recover it with recover_task.py")

    client = boto3.Session(region_name=region).client("braket")
    found = []
    paginator = client.get_paginator("search_quantum_tasks")
    for page in paginator.paginate(filters=[
        {"name": "deviceArn", "operator": "EQUAL", "values": [device_arn]},
        {"name": "createdAt", "operator": "GTE", "values": [since.isoformat()]},
    ]):
        found += [t["quantumTaskArn"] for t in page["quantumTasks"]]

    if len(found) == 1:
        return found[0]
    raise RuntimeError(
        f"expected exactly 1 task on {device_arn} since {since.isoformat()}, "
        f"found {len(found)}: {found}. The circuit has already been billed; "
        f"recover it with recover_task.py --task-arn <arn>.")
