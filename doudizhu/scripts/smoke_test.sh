#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"
GPU_ID=""
GPU_ACTORS=""
XPID=""
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/smoke_outputs}"
TOTAL_FRAMES="${TOTAL_FRAMES:-}"
NUM_ACTORS="${NUM_ACTORS:-}"
NUM_THREADS="${NUM_THREADS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
UNROLL_LENGTH="${UNROLL_LENGTH:-}"
REPLAY_BUFFER_SIZE="${REPLAY_BUFFER_SIZE:-}"
REPLAY_WARMUP_SIZE="${REPLAY_WARMUP_SIZE:-}"
PRESET="minimal"
EXTRA_TRAIN_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  bash scripts/smoke_test.sh [--cpu] [--gpu <id[,id...]>] [--cpu-actors] [--gpu-actors] [--preset <name>] [--num-actors <n>] [--num-threads <n>] [--batch-size <n>] [--unroll-length <n>] [--replay-buffer-size <n>] [--replay-warmup-size <n>] [--total-frames <n>] [--python <path>] [--xpid <name>] [--savedir <dir>] [-- <extra train.py args>]

Examples:
  bash scripts/smoke_test.sh --cpu
  bash scripts/smoke_test.sh --gpu 0
  bash scripts/smoke_test.sh --gpu 0 --gpu-actors
  bash scripts/smoke_test.sh --gpu 0,1,2,3 --gpu-actors
  bash scripts/smoke_test.sh --gpu 0,1,2,3 --gpu-actors --preset throughput
  bash scripts/smoke_test.sh --gpu 0 -- --enable_module_c false
  PYTHON_BIN=/usr/local/miniconda3/envs/py312/bin/python bash scripts/smoke_test.sh --gpu 0

Notes:
  - `minimal` is the smallest end-to-end validation.
  - `throughput` is a short high-load smoke intended for multi-GPU server checks.
  - Smoke tests inherit `train.py` defaults, so the paper-style Full setting runs
    by default: Module A/B/C and bidding are enabled.
  - To run an ablation through this wrapper, append the override after `--`, for
    example `-- --enable_module_a false --train_bidding false`.
  - CPU mode uses actor CPU + learner CPU.
  - GPU mode now follows the DouZero-style baseline: GPU actors + GPU learner.
  - `--cpu-actors` switches the rollout path back to CPU actors.
  - With `--gpu-actors` and multiple visible GPUs, the last visible GPU is
    used as the learner and the preceding GPUs are used for simulation.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            GPU_ID=""
            shift
            ;;
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
        --preset)
            PRESET="${2:-}"
            if [[ "${PRESET}" != "minimal" && "${PRESET}" != "throughput" ]]; then
                echo "error: --preset must be 'minimal' or 'throughput'" >&2
                exit 1
            fi
            shift 2
            ;;
        --num-actors)
            NUM_ACTORS="${2:-}"
            shift 2
            ;;
        --num-threads)
            NUM_THREADS="${2:-}"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="${2:-}"
            shift 2
            ;;
        --unroll-length)
            UNROLL_LENGTH="${2:-}"
            shift 2
            ;;
        --replay-buffer-size)
            REPLAY_BUFFER_SIZE="${2:-}"
            shift 2
            ;;
        --replay-warmup-size)
            REPLAY_WARMUP_SIZE="${2:-}"
            shift 2
            ;;
        --total-frames)
            TOTAL_FRAMES="${2:-}"
            shift 2
            ;;
        --xpid)
            XPID="${2:-}"
            if [[ -z "${XPID}" ]]; then
                echo "error: --xpid requires a value" >&2
                exit 1
            fi
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
        --)
            shift
            EXTRA_TRAIN_ARGS=("$@")
            break
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

MODE="cpu"
if [[ -n "${GPU_ID}" ]]; then
    MODE="gpu"
fi

if [[ "${MODE}" == "cpu" && "${GPU_ACTORS}" == "1" ]]; then
    echo "error: --gpu-actors requires --gpu <id[,id...]>" >&2
    exit 1
fi

if [[ -z "${GPU_ACTORS}" ]]; then
    if [[ "${MODE}" == "gpu" ]]; then
        GPU_ACTORS=1
    else
        GPU_ACTORS=0
    fi
fi

GPU_COUNT=0
LEARNER_DEVICE="cpu"
NUM_ACTOR_DEVICES=1
if [[ "${MODE}" == "gpu" ]]; then
    IFS=',' read -r -a GPU_LIST <<< "${GPU_ID}"
    GPU_COUNT="${#GPU_LIST[@]}"
    if (( GPU_COUNT == 0 )); then
        echo "error: --gpu requires at least one device id" >&2
        exit 1
    fi
    for gpu in "${GPU_LIST[@]}"; do
        if [[ -z "${gpu}" ]]; then
            echo "error: malformed --gpu list: ${GPU_ID}" >&2
            exit 1
        fi
    done
    LEARNER_DEVICE="$((GPU_COUNT - 1))"
    if [[ "${GPU_ACTORS}" == "1" ]]; then
        if (( GPU_COUNT > 1 )); then
            NUM_ACTOR_DEVICES="$((GPU_COUNT - 1))"
        else
            NUM_ACTOR_DEVICES=1
        fi
    fi
fi

if [[ -z "${XPID}" ]]; then
    XPID="smoke_${MODE}_$(date +%Y%m%d_%H%M%S)"
fi

if [[ "${PRESET}" == "throughput" ]]; then
    TOTAL_FRAMES="${TOTAL_FRAMES:-4096}"
    NUM_ACTORS="${NUM_ACTORS:-4}"
    NUM_THREADS="${NUM_THREADS:-2}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    UNROLL_LENGTH="${UNROLL_LENGTH:-16}"
    REPLAY_BUFFER_SIZE="${REPLAY_BUFFER_SIZE:-64}"
    REPLAY_WARMUP_SIZE="${REPLAY_WARMUP_SIZE:-8}"
fi

TOTAL_FRAMES="${TOTAL_FRAMES:-10}"
NUM_ACTORS="${NUM_ACTORS:-1}"
NUM_THREADS="${NUM_THREADS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
UNROLL_LENGTH="${UNROLL_LENGTH:-1}"
REPLAY_BUFFER_SIZE="${REPLAY_BUFFER_SIZE:-2}"
REPLAY_WARMUP_SIZE="${REPLAY_WARMUP_SIZE:-1}"

mkdir -p "${SAVE_DIR}"

echo "Repo root : ${REPO_ROOT}"
echo "Python    : ${PYTHON_BIN}"
"${PYTHON_BIN}" - <<PY
settings = {
    "preset": "${PRESET}",
    "total_frames": "${TOTAL_FRAMES}",
    "num_actors": "${NUM_ACTORS}",
    "num_threads": "${NUM_THREADS}",
    "batch_size": "${BATCH_SIZE}",
    "unroll_length": "${UNROLL_LENGTH}",
    "replay_buffer_size": "${REPLAY_BUFFER_SIZE}",
    "replay_warmup_size": "${REPLAY_WARMUP_SIZE}",
}
for key, value in settings.items():
    print(f"{key:18}: {value}")
PY
"${PYTHON_BIN}" -V

echo "Torch env :"
"${PYTHON_BIN}" - <<'PY'
import sys
try:
    import torch
    print(f"  executable={sys.executable}")
    print(f"  torch={torch.__version__}")
    print(f"  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device_count={torch.cuda.device_count()}")
except Exception as exc:
    print(f"  torch import failed: {exc}")
PY

COMMON_ARGS=(
    train.py
    --num_actor_devices "${NUM_ACTOR_DEVICES}"
    --num_actors "${NUM_ACTORS}"
    --num_threads "${NUM_THREADS}"
    --batch_size "${BATCH_SIZE}"
    --unroll_length "${UNROLL_LENGTH}"
    --replay_buffer_size "${REPLAY_BUFFER_SIZE}"
    --replay_warmup_size "${REPLAY_WARMUP_SIZE}"
    --total_frames "${TOTAL_FRAMES}"
    --save_interval 1000
    --xpid "${XPID}"
    --savedir "${SAVE_DIR}"
)

if [[ "${MODE}" == "cpu" ]]; then
    MODE_ARGS=(
        --actor_device_cpu
        --training_device cpu
        --gpu_devices ''
    )
else
    MODE_ARGS=(
        --training_device "${LEARNER_DEVICE}"
        --gpu_devices "${GPU_ID}"
    )
    if [[ "${GPU_ACTORS}" != "1" ]]; then
        MODE_ARGS=(
            --actor_device_cpu
            "${MODE_ARGS[@]}"
        )
    fi
fi

echo
echo "Running ${MODE} smoke test..."
if [[ "${MODE}" == "gpu" ]]; then
    echo "Visible GPUs      : ${GPU_ID}"
    echo "GPU actors        : ${GPU_ACTORS}"
    echo "Actor devices     : ${NUM_ACTOR_DEVICES}"
    echo "Learner device    : ${LEARNER_DEVICE}"
fi
echo "Module defaults   : Full (Module A/B/C + bidding enabled)"
if (( ${#EXTRA_TRAIN_ARGS[@]} > 0 )); then
    echo "Train overrides   : ${EXTRA_TRAIN_ARGS[*]}"
fi
if (( ${#EXTRA_TRAIN_ARGS[@]} > 0 )); then
    echo "${PYTHON_BIN} ${COMMON_ARGS[*]} ${MODE_ARGS[*]} ${EXTRA_TRAIN_ARGS[*]}"
else
    echo "${PYTHON_BIN} ${COMMON_ARGS[*]} ${MODE_ARGS[*]}"
fi
echo

cd "${REPO_ROOT}"
if (( ${#EXTRA_TRAIN_ARGS[@]} > 0 )); then
    "${PYTHON_BIN}" "${COMMON_ARGS[@]}" "${MODE_ARGS[@]}" "${EXTRA_TRAIN_ARGS[@]}"
else
    "${PYTHON_BIN}" "${COMMON_ARGS[@]}" "${MODE_ARGS[@]}"
fi

echo
echo "Smoke test finished."
echo "Artifacts:"
echo "  ${SAVE_DIR}/${XPID}"
