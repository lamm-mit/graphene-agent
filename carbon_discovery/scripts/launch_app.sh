#!/bin/bash
# Launch the local backend + frontend (http://127.0.0.1:8766)
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
exec $PY -m uvicorn app.backend.server:app --host 127.0.0.1 --port ${PORT:-8766}
