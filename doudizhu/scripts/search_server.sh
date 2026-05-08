#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"
GPU_ID="0"
GPU_ACTORS=""
SEARCH_SCOPE="narrow"
TRIAL_SECONDS="${TRIAL_SECONDS:-180}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-5}"
REPEATS="${REPEATS:-2}"
MAX_TOTAL_ACTORS="${MAX_TOTAL_ACTORS:-}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/search_outputs}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/search_server.sh [options]

Options:
  --gpu <id[,id...]>    Visible GPU list for each trial. Default: 0
  --gpu-actors          Use DouZero-style GPU actors. Default in GPU mode
  --cpu-actors          Keep actors on CPU even when learner uses GPU
  --python <path>       Python interpreter to use. Default: python3/python from PATH
  --scope <name>        Search scope: narrow, full, or saturate. Default: narrow
  --trial-seconds <n>   Per-trial runtime. Default: 180
  --cooldown-seconds <n>
                        Sleep between trials. Default: 5
  --repeats <n>         Repeat count per combo. Default: 2
  --max-total-actors <n>
                        Skip combos whose effective total actor count exceeds this cap.
                        Default: 12 for narrow/full, 18 for saturate
  --savedir <dir>       Output directory. Default: <repo>/search_outputs
  -h, --help            Show this help

Examples:
  bash scripts/search_server.sh --gpu 0
  bash scripts/search_server.sh --gpu 0,1,2,3 --gpu-actors
  bash scripts/search_server.sh --gpu 0,1,2,3 --gpu-actors --scope saturate
  bash scripts/search_server.sh --gpu 0 --scope full --trial-seconds 180

Notes:
  - `narrow` only validates the most useful current question:
    can `num_actors=2` be made stable against the known-good `num_actors=1` baseline?
  - `full` runs the larger two-stage search over actor/thread and batch/unroll/replay.
  - `saturate` is the high-throughput server preset for multi-GPU DouZero-style runs.
  - With `--gpu-actors` and multiple visible GPUs, the last visible GPU is
    used as the learner and the preceding GPUs are used for simulation.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_ID="${2:-}"
            if [[ -z "${GPU_ID}" ]]; then
                echo "error: --gpu requires a device id" >&2
                exit 1
            fi
            shift 2
            ;;
        --gpu-actors)
            GPU_ACTORS=1
            shift
            ;;
        --cpu-actors)
            GPU_ACTORS=0
            shift
            ;;
        --python)
            PYTHON_BIN="${2:-}"
            if [[ -z "${PYTHON_BIN}" ]]; then
                echo "error: --python requires a path" >&2
                exit 1
            fi
            shift 2
            ;;
        --scope)
            SEARCH_SCOPE="${2:-}"
            if [[ "${SEARCH_SCOPE}" != "narrow" && "${SEARCH_SCOPE}" != "full" && "${SEARCH_SCOPE}" != "saturate" ]]; then
                echo "error: --scope must be 'narrow', 'full', or 'saturate'" >&2
                exit 1
            fi
            shift 2
            ;;
        --trial-seconds)
            TRIAL_SECONDS="${2:-}"
            shift 2
            ;;
        --cooldown-seconds)
            COOLDOWN_SECONDS="${2:-}"
            shift 2
            ;;
        --repeats)
            REPEATS="${2:-}"
            shift 2
            ;;
        --max-total-actors)
            MAX_TOTAL_ACTORS="${2:-}"
            shift 2
            ;;
        --savedir)
            SAVE_DIR="${2:-}"
            if [[ -z "${SAVE_DIR}" ]]; then
                echo "error: --savedir requires a path" >&2
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${PYTHON_BIN}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        echo "error: no python interpreter found in PATH" >&2
        exit 1
    fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "error: python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ "${GPU_ID}" == "cpu" && "${GPU_ACTORS}" == "1" ]]; then
    echo "error: --gpu-actors requires a visible GPU list" >&2
    exit 1
fi

if [[ -z "${GPU_ACTORS}" ]]; then
    if [[ "${GPU_ID}" == "cpu" ]]; then
        GPU_ACTORS=0
    else
        GPU_ACTORS=1
    fi
fi

if [[ -z "${MAX_TOTAL_ACTORS}" ]]; then
    if [[ "${SEARCH_SCOPE}" == "saturate" ]]; then
        MAX_TOTAL_ACTORS=18
    else
        MAX_TOTAL_ACTORS=12
    fi
fi

mkdir -p "${SAVE_DIR}"

echo "Repo root : ${REPO_ROOT}"
echo "Python    : ${PYTHON_BIN}"
echo "GPU       : ${GPU_ID}"
echo "GPU actor : ${GPU_ACTORS}"
echo "Scope     : ${SEARCH_SCOPE}"
echo "Trial     : ${TRIAL_SECONDS}s"
echo "Cooldown  : ${COOLDOWN_SECONDS}s"
echo "Repeats   : ${REPEATS}"
echo "Max actor : ${MAX_TOTAL_ACTORS}"
echo "Save dir  : ${SAVE_DIR}"

echo
echo "Cleaning up any previous cli_search train.py runs..."
pkill -f "train.py --xpid cli_search_" || true
sleep 5

echo
echo "Constraining per-process CPU thread pools for stable actor/replay behavior..."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
    scripts/search_cli_params.py
    --python "${PYTHON_BIN}"
    --gpu "${GPU_ID}"
    --search-mode two-stage
    --trial-seconds "${TRIAL_SECONDS}"
    --cooldown-seconds "${COOLDOWN_SECONDS}"
    --repeats "${REPEATS}"
    --max-total-actors "${MAX_TOTAL_ACTORS}"
    --savedir "${SAVE_DIR}"
)

if [[ "${GPU_ACTORS}" == "1" ]]; then
    COMMON_ARGS+=(--gpu-actors)
else
    COMMON_ARGS+=(--cpu-actors)
fi

if [[ "${SEARCH_SCOPE}" == "narrow" ]]; then
    SEARCH_ARGS=(
        --num-actors 1,2
        --num-threads 1
        --batch-sizes 16
        --unroll-lengths 16
        --replay-warmups 8
        --replay-sizes 64
    )
elif [[ "${SEARCH_SCOPE}" == "saturate" ]]; then
    SEARCH_ARGS=(
        --num-actors 3,4,5,6
        --num-threads 1,2,4
        --batch-sizes 16,32,64
        --unroll-lengths 16,32
        --replay-warmups 8,16,32
        --replay-sizes 64,128,256
    )
else
    SEARCH_ARGS=(
        --num-actors 1,2,4
        --num-threads 1,2
        --batch-sizes 8,16,32
        --unroll-lengths 8,16
        --replay-warmups 4,8
        --replay-sizes 64,128
    )
fi

echo
echo "Running search..."
echo "${PYTHON_BIN} ${COMMON_ARGS[*]} ${SEARCH_ARGS[*]}"
echo

cd "${REPO_ROOT}"
"${PYTHON_BIN}" "${COMMON_ARGS[@]}" "${SEARCH_ARGS[@]}"

echo
echo "Search finished."
echo "Artifacts:"
echo "  ${SAVE_DIR}"
