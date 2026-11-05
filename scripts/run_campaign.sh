#!/usr/bin/env bash
set -euo pipefail
repository=$(cd -- "$(dirname -- "$0")/.." && pwd)
config=${1:-"$repository/configs/campaign.json"}
attempt=${2:-attempt_001}
python_binary=${COMBMOL_PYTHON:-/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python}
export PYTHONPATH="$repository/src:$repository/third_party/flowr_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
result_root=$($python_binary -c 'import json,sys;print(json.load(open(sys.argv[1]))["result_root"])' "$config")
mkdir -p "$result_root/runs/$attempt"
set +e
"$python_binary" -m pytest "$repository/tests" -q >"$result_root/runs/$attempt/cpu_preflight.log" 2>&1
preflight_exit=$?
"$python_binary" -c 'import json,pathlib,sys,time;pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(exit_code=int(sys.argv[2]),time=time.time())))' "$result_root/CPU_PREFLIGHT_EXIT.json" "$preflight_exit"
set -e
if [ "$preflight_exit" -ne 0 ]; then
    exit "$preflight_exit"
fi
"$python_binary" -m combmol.cli preflight --config "$config" >"$result_root/runs/$attempt/runtime_preflight.log" 2>&1
exec "$python_binary" -u -m combmol.cli run --config "$config" --attempt "$attempt"
