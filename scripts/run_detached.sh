#!/bin/bash
# Run a long job detached, with a status file that says how it ended.
#
#   scripts/run_detached.sh <name> <command...>
#
# Writes to $QFR_RUN_DIR (default: /tmp/qfr-runs/<name>/):
#   log      stdout+stderr of the command
#   status   one line, rewritten on every change:
#              RUNNING <pid> <started-utc>
#              EXIT 0 <finished-utc>            the command returned 0
#              FAILED <code> <finished-utc>     the command returned non-zero
#              STALLED <minutes> <checked-utc>  RUNNING, but the log has not
#                                               grown for $QFR_STALL_MIN minutes
#                                               (default 30); cleared when it grows
#   With QFR_TIMEOUT set (e.g. 6h, 90m), the command runs under `timeout` and a
#   job that outlives it ends as FAILED 124 -- a bound on wall-clock, so a hung
#   job cannot hold a host for days.
#
# Why: on 2026-09-16 a two-hour harness crashed in its first minute inside a
# shell script without `set -e`, the script went on to print its "all done"
# marker, and nothing watched for the traceback; the failure was noticed nine
# hours later. A job's end state has to be a written fact, not an inference
# from a log line, and silence has to become a state too.
#
# Watch it (any of these fires on success, failure and stall alike):
#   until grep -qE '^(EXIT|FAILED|STALLED)' /tmp/qfr-runs/<name>/status; do sleep 30; done; cat .../status
set -uo pipefail
name="$1"; shift
dir="${QFR_RUN_DIR:-/tmp/qfr-runs}/$name"
mkdir -p "$dir"
log="$dir/log"; status="$dir/status"
stall_min="${QFR_STALL_MIN:-30}"
: > "$log"

(
  echo "RUNNING $BASHPID $(date -u +%FT%TZ)" > "$status"
  # Stall watcher: marks STALLED when the log stops growing, clears it when it resumes.
  (
    while true; do
      sleep 60
      read -r state _ < "$status" || break
      [ "$state" = RUNNING ] || [ "$state" = STALLED ] || break
      age=$(( ( $(date +%s) - $(stat -c %Y "$log") ) / 60 ))
      if [ "$age" -ge "$stall_min" ]; then
        echo "STALLED $age $(date -u +%FT%TZ)" > "$status"
      elif [ "$state" = STALLED ]; then
        echo "RUNNING $BASHPID $(date -u +%FT%TZ)" > "$status"
      fi
    done
  ) &
  watcher=$!
  # The command runs as a background child and the subshell waits on it:
  # bash defers a trap while a foreground child runs, so a TERM to the
  # wrapper would otherwise be handled only when the job ended -- and the
  # status file would say RUNNING until then.
  if [ -n "${QFR_TIMEOUT:-}" ]; then
    timeout "$QFR_TIMEOUT" "$@" >> "$log" 2>&1 &
  else
    "$@" >> "$log" 2>&1 &
  fi
  child=$!
  trap 'kill "$child" "$watcher" 2>/dev/null; echo "FAILED 143 $(date -u +%FT%TZ)  # terminated" > "$status"; exit 143' TERM INT HUP
  wait "$child"
  code=$?
  kill "$watcher" 2>/dev/null
  if [ "$code" -eq 0 ]; then
    echo "EXIT 0 $(date -u +%FT%TZ)" > "$status"
  else
    echo "FAILED $code $(date -u +%FT%TZ)" > "$status"
  fi
) </dev/null >/dev/null 2>&1 &
disown
echo "$dir"
