#!/usr/bin/env bash
# Monitors the four background live-eval jobs launched via nohup (see README
# reproduction instructions): reports whether each process is still running,
# whether it has finished (result JSON present), its latest progress line, and
# any errors/tracebacks/rate-limit messages found in its output log.
#
# Usage:
#   ./monitor_eval.sh            # one-shot status snapshot
#   ./monitor_eval.sh --watch    # refresh every 30s until Ctrl+C

set -uo pipefail

LOG_DIR="logs/$(date +%F)"

# name : script path (for pgrep -f) : nohup output file : expected result JSON
JOBS=(
  "tiger_main:tiger_pomdp/run_full_eval.py:/tmp/tiger_main.out:${LOG_DIR}/tiger_full_eval_results.json"
  "tiger_ablations:tiger_pomdp/run_ablations_eval.py:/tmp/tiger_ablations.out:${LOG_DIR}/tiger_ablations_eval_results.json"
  "attack_main:red_team_graph/run_full_eval.py:/tmp/attack_main.out:${LOG_DIR}/attack_full_eval_results.json"
  "attack_ablations:red_team_graph/run_ablations_eval.py:/tmp/attack_ablations.out:${LOG_DIR}/attack_ablations_eval_results.json"
)

check_one() {
  local name="$1" script="$2" outfile="$3" resultfile="$4"
  local pid running status progress_line last_line errors

  pid=$(pgrep -f "$script" | head -n1)
  running="no"; [[ -n "$pid" ]] && running="yes (pid $pid)"

  if [[ -f "$resultfile" ]]; then
    status="DONE"
  elif [[ -n "$pid" ]]; then
    status="RUNNING"
  else
    status="STOPPED (no result file, no process -- likely crashed or not started)"
  fi

  progress_line="(none yet)"
  last_line="(no output yet)"
  errors=""
  if [[ -f "$outfile" ]]; then
    last_line=$(tail -n1 "$outfile")
    progress_line=$(grep -E "completed [0-9]+/[0-9]+ episodes" "$outfile" | tail -n1)
    [[ -z "$progress_line" ]] && progress_line="(none yet)"
    errors=$(grep -iE "traceback|error|exception|critical|rate.?limit|\b429\b" "$outfile" | tail -n3)
  fi

  echo "== ${name} =="
  echo "  status:    ${status}"
  echo "  process:   ${running}"
  echo "  progress:  ${progress_line}"
  echo "  last line: ${last_line}"
  if [[ -n "$errors" ]]; then
    echo "  POSSIBLE ISSUES DETECTED:"
    echo "${errors}" | sed 's/^/    /'
  fi
  echo
}

run_once() {
  echo "Snapshot at $(date '+%H:%M:%S')"
  echo "-----------------------------------"
  for job in "${JOBS[@]}"; do
    IFS=":" read -r name script outfile resultfile <<< "$job"
    check_one "$name" "$script" "$outfile" "$resultfile"
  done
}

if [[ "${1:-}" == "--watch" ]]; then
  while true; do
    clear
    run_once
    sleep 30
  done
else
  run_once
fi
