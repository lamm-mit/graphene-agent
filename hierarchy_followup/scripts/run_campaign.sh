#!/bin/bash
# Launch the hierarchy follow-up: queue 1 (T1 crack arrest, T4 two-stage, T2 seams; 120 A cells) with 4 workers, then
# queue 2 (T3 scale separation; 300 A cells) with 3 workers.  Detached (nohup); progress in the queue logs and
# `python scripts/status.py`.  Re-running is safe: finished specs live in done/, failed ones in failed/.
#   ./scripts/run_campaign.sh            # both queues in sequence
#   ./scripts/run_campaign.sh T3         # only the T3 queue
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
if [ "$1" != "T3" ]; then
  echo "== $(date) queue 1: experiments/specs/hierarchy (4 workers, device auto = mps) ==" >> experiments/specs/hierarchy/queue.log
  $PY scripts/run_queue.py experiments/specs/hierarchy --workers 4 --device auto >> experiments/specs/hierarchy/queue.log 2>&1
  echo "== $(date) queue 1 finished ==" >> experiments/specs/hierarchy/queue.log
fi
echo "== $(date) queue 2: experiments/specs/hierarchy_T3 (3 workers, device auto = mps) ==" >> experiments/specs/hierarchy_T3/queue.log
$PY scripts/run_queue.py experiments/specs/hierarchy_T3 --workers 3 --device auto >> experiments/specs/hierarchy_T3/queue.log 2>&1
echo "== $(date) queue 2 finished ==" >> experiments/specs/hierarchy_T3/queue.log
