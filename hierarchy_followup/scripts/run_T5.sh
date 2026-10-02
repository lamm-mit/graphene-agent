#!/bin/bash
# Queue 3 (T5 composite hierarchies): waits until queue 2 (T3) has finished, then runs experiments/specs/hierarchy_T5
# with 4 workers.  Detached: bash scripts/run_T5.sh  (launched by the campaign runner through Popen/start_new_session).
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
LOG=experiments/specs/hierarchy_T5/queue.log
mkdir -p experiments/specs/hierarchy_T5
echo "== $(date) waiting for queue 2 (T3) to finish ==" >> $LOG
until grep -q "queue 2 finished" experiments/specs/hierarchy_T3/queue.log 2>/dev/null; do sleep 300; done
echo "== $(date) queue 3: experiments/specs/hierarchy_T5 (3 workers, device auto = mps) ==" >> $LOG
$PY scripts/run_queue.py experiments/specs/hierarchy_T5 --workers 4 --device auto >> $LOG 2>&1
echo "== $(date) queue 3 finished ==" >> $LOG
