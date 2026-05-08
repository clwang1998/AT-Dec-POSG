#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import os
import re
import shutil
import shlex
import signal
import subprocess
import sys
import threading
import time
import numpy as np
from pathlib import Path


AFTER_RE = re.compile(
    r"After (?P<frames>\d+) .* @ (?P<fps>[0-9.]+) fps \(avg@ (?P<avg_fps>[0-9.]+) fps\)"
)

MER_PATTERNS = {
    "mean_episode_return_landlord": re.compile(
        r"'mean_episode_return_landlord':\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    ),
    "mean_episode_return_farmer": re.compile(
        r"'mean_episode_return_farmer':\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    ),
    "mean_episode_return_bidding": re.compile(
        r"'mean_episode_return_bidding':\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    ),
}

# Empirical baseline from this project:
# - good:  actors=2, threads=2, batch=16, unroll=16, warmup=8, replay=64
# - bad:   actors=8, threads=4, batch=64, unroll=32, warmup=32, replay=256
# - bad:   actors=8, threads=2, batch=16, unroll=16, warmup=8, replay=64
EMPIRICAL_TRIALS = [
    dict(num_actors=2, num_threads=2, batch_size=16, unroll_length=16, replay_warmup_size=8, replay_buffer_size=64),
    dict(num_actors=1, num_threads=1, batch_size=16, unroll_length=16, replay_warmup_size=8, replay_buffer_size=64),
    dict(num_actors=2, num_threads=1, batch_size=16, unroll_length=16, replay_warmup_size=8, replay_buffer_size=64),
    dict(num_actors=4, num_threads=1, batch_size=16, unroll_length=16, replay_warmup_size=8, replay_buffer_size=64),
    dict(num_actors=4, num_threads=2, batch_size=16, unroll_length=16, replay_warmup_size=8, replay_buffer_size=64),
    dict(num_actors=5, num_threads=2, batch_size=32, unroll_length=16, replay_warmup_size=8, replay_buffer_size=128),
    dict(num_actors=6, num_threads=2, batch_size=32, unroll_length=16, replay_warmup_size=16, replay_buffer_size=128),
    dict(num_actors=2, num_threads=2, batch_size=8, unroll_length=16, replay_warmup_size=8, replay_buffer_size=64),
    dict(num_actors=2, num_threads=2, batch_size=32, unroll_length=16, replay_warmup_size=8, replay_buffer_size=64),
    dict(num_actors=2, num_threads=2, batch_size=16, unroll_length=8, replay_warmup_size=8, replay_buffer_size=64),
    dict(num_actors=2, num_threads=2, batch_size=16, unroll_length=16, replay_warmup_size=4, replay_buffer_size=64),
    dict(num_actors=2, num_threads=2, batch_size=16, unroll_length=16, replay_warmup_size=8, replay_buffer_size=128),
]

COARSE_BASELINE = dict(
    batch_size=16,
    unroll_length=16,
    replay_warmup_size=8,
    replay_buffer_size=64,
)

PRIMARY_METRIC_FIELD = {
    "avg_fps": "mean_success_avg_fps",
    "frames": "mean_success_frames",
    "success_rate": "success_rate",
    "mer_landlord": "mean_success_mer_landlord",
    "mer_farmer": "mean_success_mer_farmer",
    "mer_bidding": "mean_success_mer_bidding",
    "gpu_util": "mean_success_gpu_util_pct",
    "gpu_power": "mean_success_gpu_power_total_w",
}

PRIMARY_METRIC_DEFAULT_GOAL = {
    "avg_fps": "maximize",
    "frames": "maximize",
    "success_rate": "maximize",
    "mer_landlord": "maximize",
    "mer_farmer": "maximize",
    "mer_bidding": "maximize",
    "gpu_util": "maximize",
    "gpu_power": "minimize",
}

RESULTS_TSV_COLUMNS = [
    "timestamp",
    "run_stamp",
    "stage",
    "combo_index",
    "status",
    "reason",
    "primary_metric",
    "primary_goal",
    "primary_value",
    "incumbent_before",
    "incumbent_after",
    "success_count",
    "repeat_count",
    "success_rate",
    "traceback_count",
    "mean_success_frames",
    "mean_success_avg_fps",
    "mean_success_mer_landlord",
    "mean_success_mer_farmer",
    "mean_success_mer_bidding",
    "num_actors",
    "num_threads",
    "batch_size",
    "unroll_length",
    "replay_warmup_size",
    "replay_buffer_size",
    "total_actors",
    "example_xpid",
    "example_log_path",
]

GPU_PER_RUN_KEYS = [
    "gpu_sample_count",
    "gpu_power_total_mean_w",
    "gpu_power_total_max_w",
    "gpu_util_mean_pct",
    "gpu_util_max_pct",
    "gpu_mem_total_mean_mib",
    "actor_power_total_mean_w",
    "actor_util_mean_pct",
    "actor_mem_total_mean_mib",
    "learner_power_mean_w",
    "learner_util_mean_pct",
    "learner_mem_mean_mib",
]

PROCESS_PER_RUN_KEYS = [
    "proc_sample_count",
    "proc_cpu_mean_pct",
    "proc_cpu_max_pct",
    "proc_mem_mean_mib",
    "proc_mem_max_mib",
    "proc_count_mean",
]

GPU_AGGREGATE_KEYS = [
    ("gpu_power_total_mean_w", "mean_success_gpu_power_total_w"),
    ("gpu_util_mean_pct", "mean_success_gpu_util_pct"),
    ("actor_power_total_mean_w", "mean_success_actor_power_w"),
    ("actor_util_mean_pct", "mean_success_actor_util_pct"),
    ("learner_power_mean_w", "mean_success_learner_power_w"),
    ("learner_util_mean_pct", "mean_success_learner_util_pct"),
]

PROCESS_AGGREGATE_KEYS = [
    ("proc_cpu_mean_pct", "mean_success_proc_cpu_pct"),
    ("proc_mem_mean_mib", "mean_success_proc_mem_mib"),
    ("proc_count_mean", "mean_success_proc_count"),
]


def parse_csv_ints(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_gpu_devices(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.lower() == "cpu":
        return []
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if not devices:
        raise argparse.ArgumentTypeError(
            "expected 'cpu' or a comma-separated GPU list such as '0' or '0,1,2,3'"
        )
    return devices


def parse_gpu_ids(raw: str) -> list[int]:
    ids: list[int] = []
    for item in parse_gpu_devices(raw):
        try:
            ids.append(int(item))
        except ValueError:
            continue
    return ids


def resolve_gpu_groups(gpu: str, gpu_actors: bool) -> dict[str, list[int]]:
    visible_gpu_ids = parse_gpu_ids(gpu)
    if not visible_gpu_ids:
        return {
            "visible_gpu_ids": [],
            "actor_gpu_ids": [],
            "learner_gpu_ids": [],
        }
    learner_gpu_ids = [visible_gpu_ids[-1]]
    if gpu_actors:
        actor_gpu_ids = visible_gpu_ids[:-1] if len(visible_gpu_ids) > 1 else visible_gpu_ids[:]
    else:
        actor_gpu_ids = []
    return {
        "visible_gpu_ids": visible_gpu_ids,
        "actor_gpu_ids": actor_gpu_ids,
        "learner_gpu_ids": learner_gpu_ids,
    }


def resolve_runtime_layout(gpu: str, gpu_actors: bool) -> dict[str, object]:
    devices = parse_gpu_devices(gpu)
    if not devices:
        return {
            "mode": "cpu",
            "gpu_devices": "",
            "training_device": "cpu",
            "num_actor_devices": 1,
            "actor_device_cpu": True,
            "actor_device_count": 1,
            "visible_gpu_count": 0,
        }

    visible_gpu_count = len(devices)
    training_device = str(visible_gpu_count - 1)
    if gpu_actors:
        actor_device_count = visible_gpu_count - 1 if visible_gpu_count > 1 else 1
        actor_device_cpu = False
    else:
        actor_device_count = 1
        actor_device_cpu = True
    return {
        "mode": "gpu",
        "gpu_devices": ",".join(devices),
        "training_device": training_device,
        "num_actor_devices": actor_device_count,
        "actor_device_cpu": actor_device_cpu,
        "actor_device_count": actor_device_count,
        "visible_gpu_count": visible_gpu_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search train.py CLI parameter combinations with short timed trials. "
            "The current train.py defaults follow the paper-style Full setting "
            "(Module A/B/C + bidding enabled); use --extra-args with explicit "
            "'false' overrides for ablations."
        )
    )
    parser.add_argument(
        "--search-mode",
        choices=("empirical", "grid", "two-stage"),
        default="two-stage",
        help=(
            "two-stage first searches actor/thread pairs with a fixed baseline, then fine-searches "
            "batch/unroll/warmup/replay around the best pair. empirical uses a small curated set; "
            "grid uses the cartesian product below."
        ),
    )
    parser.add_argument(
        "--python",
        dest="python_bin",
        default=sys.executable,
        help="Python interpreter used to launch train.py",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help=(
            "Visible GPU list for each trial, for example '0' or '0,1,2,3'. "
            "Use 'cpu' to force full CPU mode."
        ),
    )
    actor_mode_group = parser.add_mutually_exclusive_group()
    actor_mode_group.add_argument(
        "--gpu-actors",
        dest="gpu_actors",
        action="store_true",
        help=(
            "Use DouZero-style GPU actors. With multiple visible GPUs, the last visible "
            "GPU becomes the learner and the preceding GPUs become actor devices."
        ),
    )
    actor_mode_group.add_argument(
        "--cpu-actors",
        dest="gpu_actors",
        action="store_false",
        help="Keep actors on CPU even when the learner runs on GPU.",
    )
    parser.set_defaults(gpu_actors=None)
    parser.add_argument(
        "--trial-seconds",
        type=int,
        default=180,
        help="How long each trial should run before being stopped.",
    )
    parser.add_argument(
        "--sample-seconds",
        "--gpu-sample-seconds",
        dest="gpu_sample_seconds",
        type=float,
        default=5.0,
        help=(
            "Resource metric sampling interval in seconds during each timed trial. "
            "Set 0 to disable GPU/CPU/memory sampling."
        ),
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=5,
        help="Sleep between trials so the machine can cool down and release resources.",
    )
    parser.add_argument(
        "--num-actors",
        type=parse_csv_ints,
        default=parse_csv_ints("1,2,4"),
        help="Comma-separated candidate values for --num_actors",
    )
    parser.add_argument(
        "--num-threads",
        type=parse_csv_ints,
        default=parse_csv_ints("1,2"),
        help="Comma-separated candidate values for --num_threads",
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_csv_ints,
        default=parse_csv_ints("8,16,32"),
        help="Comma-separated candidate values for --batch_size",
    )
    parser.add_argument(
        "--unroll-lengths",
        type=parse_csv_ints,
        default=parse_csv_ints("8,16"),
        help="Comma-separated candidate values for --unroll_length",
    )
    parser.add_argument(
        "--replay-warmups",
        type=parse_csv_ints,
        default=parse_csv_ints("4,8"),
        help="Comma-separated candidate values for --replay_warmup_size",
    )
    parser.add_argument(
        "--replay-sizes",
        type=parse_csv_ints,
        default=parse_csv_ints("32,64,128"),
        help="Comma-separated candidate values for --replay_buffer_size",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help=(
            "Extra train.py args appended to every run. The default search inherits "
            "the Full setting (Module A/B/C + bidding enabled). Use this to run "
            "ablations, for example '--enable_module_a false --train_bidding false'."
        ),
    )
    parser.add_argument(
        "--savedir",
        default="search_outputs",
        help="Root directory for all search trial outputs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of combinations to try. 0 means all.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to rerun each parameter combination to measure stability.",
    )
    parser.add_argument(
        "--max-total-actors",
        type=int,
        default=12,
        help=(
            "Skip combinations whose effective total actor count across actor devices exceeds "
            "this limit. Set 0 to disable the cap."
        ),
    )
    parser.add_argument(
        "--coarse-batch-size",
        type=int,
        default=COARSE_BASELINE["batch_size"],
        help="Fixed batch size used during stage-1 actor/thread search in two-stage mode.",
    )
    parser.add_argument(
        "--coarse-unroll-length",
        type=int,
        default=COARSE_BASELINE["unroll_length"],
        help="Fixed unroll length used during stage-1 actor/thread search in two-stage mode.",
    )
    parser.add_argument(
        "--coarse-replay-warmup-size",
        type=int,
        default=COARSE_BASELINE["replay_warmup_size"],
        help="Fixed replay warmup size used during stage-1 actor/thread search in two-stage mode.",
    )
    parser.add_argument(
        "--coarse-replay-buffer-size",
        type=int,
        default=COARSE_BASELINE["replay_buffer_size"],
        help="Fixed replay buffer size used during stage-1 actor/thread search in two-stage mode.",
    )
    parser.add_argument(
        "--primary-metric",
        choices=tuple(PRIMARY_METRIC_FIELD.keys()),
        default="avg_fps",
        help=(
            "Primary metric used by keep/discard decisions. "
            "Use mer_* for short-horizon policy quality signals."
        ),
    )
    parser.add_argument(
        "--primary-goal",
        choices=("auto", "maximize", "minimize"),
        default="auto",
        help=(
            "Optimization goal for the primary metric. "
            "'auto' follows metric defaults."
        ),
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=1.0,
        help=(
            "Gate for keep/discard decisions. "
            "Combinations below this success ratio are discarded."
        ),
    )
    parser.add_argument(
        "--min-mean-success-frames",
        type=float,
        default=1.0,
        help=(
            "Gate for keep/discard decisions. "
            "Combinations below this mean successful frame count are discarded."
        ),
    )
    parser.add_argument(
        "--min-primary-improvement",
        type=float,
        default=0.0,
        help=(
            "Minimum primary-metric delta required for keep against incumbent."
        ),
    )
    parser.add_argument(
        "--results-tsv",
        default="results.tsv",
        help=(
            "Tab-separated keep/discard ledger path. "
            "Relative paths are resolved under --savedir. Empty string disables logging."
        ),
    )
    parser.add_argument(
        "--reset-results-tsv",
        action="store_true",
        help="Reset keep/discard ledger before this run starts.",
    )
    return parser


def should_skip_combo(
    num_actors: int,
    num_threads: int,
    batch_size: int,
    unroll_length: int,
    replay_warmup: int,
    replay_size: int,
    actor_device_count: int = 1,
    max_total_actors: int = 0,
) -> bool:
    if num_threads > num_actors:
        return True
    total_actors = num_actors * max(1, actor_device_count)
    if max_total_actors > 0 and total_actors > max_total_actors:
        return True
    if num_actors > 8:
        return True
    if num_threads > 4:
        return True
    if batch_size > 64:
        return True
    if unroll_length > 128:
        return True
    if replay_warmup > 32:
        return True
    if replay_size > 256:
        return True
    return False


def resolve_primary_goal(primary_metric: str, requested_goal: str) -> str:
    if requested_goal != "auto":
        return requested_goal
    return PRIMARY_METRIC_DEFAULT_GOAL[primary_metric]


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        casted = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(casted):
        return default
    return casted


def _primary_metric_value(item: dict[str, object], primary_metric: str) -> float:
    field = PRIMARY_METRIC_FIELD[primary_metric]
    return _as_float(item.get(field), default=0.0)


def _is_better_metric(
    candidate: float,
    incumbent: float,
    goal: str,
    min_improvement: float,
) -> bool:
    if goal == "maximize":
        return candidate > incumbent + min_improvement
    return candidate < incumbent - min_improvement


def ensure_results_tsv(results_tsv_path: Path, reset: bool = False) -> None:
    if reset and results_tsv_path.exists():
        results_tsv_path.unlink()
    if results_tsv_path.exists():
        return
    results_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with results_tsv_path.open("w", newline="", encoding="utf-8") as tsv_file:
        writer = csv.DictWriter(tsv_file, fieldnames=RESULTS_TSV_COLUMNS, delimiter="\t")
        writer.writeheader()


def append_results_tsv(
    results_tsv_path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return
    ensure_results_tsv(results_tsv_path, reset=False)
    with results_tsv_path.open("a", newline="", encoding="utf-8") as tsv_file:
        writer = csv.DictWriter(tsv_file, fieldnames=RESULTS_TSV_COLUMNS, delimiter="\t")
        for row in rows:
            normalized_row = {column: row.get(column, "") for column in RESULTS_TSV_COLUMNS}
            writer.writerow(normalized_row)


def ensure_process_stopped(proc: subprocess.Popen, grace_seconds: int = 10) -> None:
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = None
    # Give the root training process a chance to orchestrate its own shutdown
    # before we fall back to process-group termination. This avoids racing the
    # learner threads against abruptly killed actor children.
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    if pgid is not None:
        os.killpg(pgid, signal.SIGTERM)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    if pgid is not None:
        os.killpg(pgid, signal.SIGKILL)
    else:
        proc.kill()
    proc.wait(timeout=5)


def _parse_gpu_metric(raw: str) -> float:
    raw = raw.strip()
    if not raw or raw in {"[N/A]", "N/A", "Not Supported"}:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def query_gpu_snapshot(gpu_ids: list[int]) -> dict[int, dict[str, float]]:
    if not gpu_ids:
        return {}
    query_bin = shutil.which("nvidia-smi")
    if query_bin is None:
        return {}
    cmd = [
        query_bin,
        f"--id={','.join(str(gpu_id) for gpu_id in gpu_ids)}",
        "--query-gpu=index,power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {}

    snapshot: dict[int, dict[str, float]] = {}
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpu_id = int(parts[0])
        except ValueError:
            continue
        snapshot[gpu_id] = {
            "power_w": _parse_gpu_metric(parts[1]),
            "util_pct": _parse_gpu_metric(parts[2]),
            "mem_mib": _parse_gpu_metric(parts[3]),
        }
    return snapshot


def collect_gpu_samples(
    proc: subprocess.Popen,
    gpu_ids: list[int],
    sample_interval: float,
    samples: list[dict[str, object]],
    stop_event: threading.Event,
) -> None:
    if sample_interval <= 0 or not gpu_ids:
        return
    start_time = time.time()
    while not stop_event.is_set():
        if proc.poll() is not None:
            break
        snapshot = query_gpu_snapshot(gpu_ids)
        if snapshot:
            samples.append(
                {
                    "elapsed_seconds": round(time.time() - start_time, 2),
                    "gpus": snapshot,
                }
            )
        if stop_event.wait(sample_interval):
            break


def query_process_tree_snapshot(root_pid: int) -> dict[str, float]:
    try:
        proc = subprocess.run(
            ["ps", "-Ao", "pid=,ppid=,%cpu=,rss="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {}

    rows: dict[int, dict[str, float]] = {}
    children: dict[int, list[int]] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            cpu_pct = float(parts[2])
            rss_kib = float(parts[3])
        except ValueError:
            continue
        rows[pid] = {
            "ppid": ppid,
            "cpu_pct": cpu_pct,
            "rss_kib": rss_kib,
        }
        children.setdefault(ppid, []).append(pid)

    if root_pid not in rows:
        return {}

    selected = {root_pid}
    queue = [root_pid]
    while queue:
        pid = queue.pop()
        for child_pid in children.get(pid, []):
            if child_pid in selected:
                continue
            selected.add(child_pid)
            queue.append(child_pid)

    total_cpu_pct = sum(rows[pid]["cpu_pct"] for pid in selected if pid in rows)
    total_rss_mib = sum(rows[pid]["rss_kib"] for pid in selected if pid in rows) / 1024.0
    return {
        "proc_count": float(len(selected)),
        "cpu_pct": total_cpu_pct,
        "rss_mib": total_rss_mib,
    }


def collect_process_samples(
    proc: subprocess.Popen,
    sample_interval: float,
    samples: list[dict[str, float]],
    stop_event: threading.Event,
) -> None:
    if sample_interval <= 0:
        return
    start_time = time.time()
    while not stop_event.is_set():
        if proc.poll() is not None:
            break
        snapshot = query_process_tree_snapshot(proc.pid)
        if snapshot:
            samples.append(
                {
                    "elapsed_seconds": round(time.time() - start_time, 2),
                    "proc_count": snapshot["proc_count"],
                    "cpu_pct": snapshot["cpu_pct"],
                    "rss_mib": snapshot["rss_mib"],
                }
            )
        if stop_event.wait(sample_interval):
            break


def write_gpu_samples_csv(samples: list[dict[str, object]], path: Path) -> None:
    rows: list[dict[str, object]] = []
    for sample in samples:
        elapsed_seconds = float(sample["elapsed_seconds"])
        for gpu_id, metrics in sorted(sample["gpus"].items()):
            rows.append(
                {
                    "elapsed_seconds": elapsed_seconds,
                    "gpu_id": gpu_id,
                    "power_w": metrics["power_w"],
                    "util_pct": metrics["util_pct"],
                    "mem_mib": metrics["mem_mib"],
                }
            )
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_process_samples_csv(samples: list[dict[str, float]], path: Path) -> None:
    if not samples:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(samples[0].keys()))
        writer.writeheader()
        writer.writerows(samples)


def summarize_gpu_samples(
    samples: list[dict[str, object]],
    gpu_groups: dict[str, list[int]],
) -> dict[str, object]:
    if not samples:
        return {
            "gpu_sample_count": 0,
            "gpu_power_total_mean_w": 0.0,
            "gpu_power_total_max_w": 0.0,
            "gpu_util_mean_pct": 0.0,
            "gpu_util_max_pct": 0.0,
            "gpu_mem_total_mean_mib": 0.0,
            "actor_power_total_mean_w": 0.0,
            "actor_util_mean_pct": 0.0,
            "actor_mem_total_mean_mib": 0.0,
            "learner_power_mean_w": 0.0,
            "learner_util_mean_pct": 0.0,
            "learner_mem_mean_mib": 0.0,
            "gpu_power_mean_w_by_id": "",
            "gpu_util_mean_pct_by_id": "",
        }

    visible_gpu_ids = gpu_groups["visible_gpu_ids"]
    actor_gpu_ids = gpu_groups["actor_gpu_ids"]
    learner_gpu_ids = gpu_groups["learner_gpu_ids"]

    total_power_series: list[float] = []
    mean_util_series: list[float] = []
    total_mem_series: list[float] = []
    actor_power_series: list[float] = []
    actor_util_series: list[float] = []
    actor_mem_series: list[float] = []
    learner_power_series: list[float] = []
    learner_util_series: list[float] = []
    learner_mem_series: list[float] = []
    per_gpu_metrics = {
        gpu_id: {"power_w": [], "util_pct": []}
        for gpu_id in visible_gpu_ids
    }

    for sample in samples:
        snapshot = sample["gpus"]
        visible_metrics = [snapshot[gpu_id] for gpu_id in visible_gpu_ids if gpu_id in snapshot]
        if not visible_metrics:
            continue

        total_power_series.append(sum(item["power_w"] for item in visible_metrics))
        mean_util_series.append(
            sum(item["util_pct"] for item in visible_metrics) / len(visible_metrics)
        )
        total_mem_series.append(sum(item["mem_mib"] for item in visible_metrics))

        actor_metrics = [snapshot[gpu_id] for gpu_id in actor_gpu_ids if gpu_id in snapshot]
        if actor_metrics:
            actor_power_series.append(sum(item["power_w"] for item in actor_metrics))
            actor_util_series.append(
                sum(item["util_pct"] for item in actor_metrics) / len(actor_metrics)
            )
            actor_mem_series.append(sum(item["mem_mib"] for item in actor_metrics))

        learner_metrics = [snapshot[gpu_id] for gpu_id in learner_gpu_ids if gpu_id in snapshot]
        if learner_metrics:
            learner_power_series.append(
                sum(item["power_w"] for item in learner_metrics) / len(learner_metrics)
            )
            learner_util_series.append(
                sum(item["util_pct"] for item in learner_metrics) / len(learner_metrics)
            )
            learner_mem_series.append(
                sum(item["mem_mib"] for item in learner_metrics) / len(learner_metrics)
            )

        for gpu_id in visible_gpu_ids:
            if gpu_id not in snapshot:
                continue
            per_gpu_metrics[gpu_id]["power_w"].append(snapshot[gpu_id]["power_w"])
            per_gpu_metrics[gpu_id]["util_pct"].append(snapshot[gpu_id]["util_pct"])

    if not total_power_series:
        return summarize_gpu_samples([], gpu_groups)

    gpu_power_by_id = "|".join(
        f"{gpu_id}:{sum(values['power_w']) / len(values['power_w']):.1f}"
        for gpu_id, values in per_gpu_metrics.items()
        if values["power_w"]
    )
    gpu_util_by_id = "|".join(
        f"{gpu_id}:{sum(values['util_pct']) / len(values['util_pct']):.1f}"
        for gpu_id, values in per_gpu_metrics.items()
        if values["util_pct"]
    )

    def _mean_or_zero(values: list[float]) -> float:
        return round(sum(values) / len(values), 1) if values else 0.0

    def _max_or_zero(values: list[float]) -> float:
        return round(max(values), 1) if values else 0.0

    return {
        "gpu_sample_count": len(samples),
        "gpu_power_total_mean_w": _mean_or_zero(total_power_series),
        "gpu_power_total_max_w": _max_or_zero(total_power_series),
        "gpu_util_mean_pct": _mean_or_zero(mean_util_series),
        "gpu_util_max_pct": _max_or_zero(mean_util_series),
        "gpu_mem_total_mean_mib": _mean_or_zero(total_mem_series),
        "actor_power_total_mean_w": _mean_or_zero(actor_power_series),
        "actor_util_mean_pct": _mean_or_zero(actor_util_series),
        "actor_mem_total_mean_mib": _mean_or_zero(actor_mem_series),
        "learner_power_mean_w": _mean_or_zero(learner_power_series),
        "learner_util_mean_pct": _mean_or_zero(learner_util_series),
        "learner_mem_mean_mib": _mean_or_zero(learner_mem_series),
        "gpu_power_mean_w_by_id": gpu_power_by_id,
        "gpu_util_mean_pct_by_id": gpu_util_by_id,
    }


def summarize_process_samples(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        return {
            "proc_sample_count": 0,
            "proc_cpu_mean_pct": 0.0,
            "proc_cpu_max_pct": 0.0,
            "proc_mem_mean_mib": 0.0,
            "proc_mem_max_mib": 0.0,
            "proc_count_mean": 0.0,
        }

    cpu_series = [sample["cpu_pct"] for sample in samples]
    mem_series = [sample["rss_mib"] for sample in samples]
    proc_count_series = [sample["proc_count"] for sample in samples]
    return {
        "proc_sample_count": len(samples),
        "proc_cpu_mean_pct": round(sum(cpu_series) / len(cpu_series), 1),
        "proc_cpu_max_pct": round(max(cpu_series), 1),
        "proc_mem_mean_mib": round(sum(mem_series) / len(mem_series), 1),
        "proc_mem_max_mib": round(max(mem_series), 1),
        "proc_count_mean": round(sum(proc_count_series) / len(proc_count_series), 1),
    }


def _extract_last_float(pattern: re.Pattern[str], log_text: str) -> float:
    value = 0.0
    for match in pattern.finditer(log_text):
        value = _as_float(match.group("value"), default=0.0)
    return value


def parse_metrics(log_text: str) -> dict[str, float | int | bool]:
    frames = 0
    fps = 0.0
    avg_fps = 0.0
    for match in AFTER_RE.finditer(log_text):
        frames = int(match.group("frames"))
        fps = float(match.group("fps"))
        avg_fps = float(match.group("avg_fps"))
    mer_landlord = _extract_last_float(MER_PATTERNS["mean_episode_return_landlord"], log_text)
    mer_farmer = _extract_last_float(MER_PATTERNS["mean_episode_return_farmer"], log_text)
    mer_bidding = _extract_last_float(MER_PATTERNS["mean_episode_return_bidding"], log_text)
    return {
        "frames": frames,
        "fps": fps,
        "avg_fps": avg_fps,
        "mer_landlord": mer_landlord,
        "mer_farmer": mer_farmer,
        "mer_bidding": mer_bidding,
        "made_progress": frames > 0,
        "stuck_at_zero": "After 0" in log_text and frames == 0,
        "has_traceback": "Traceback" in log_text,
    }


def aggregate_results(
    results: list[dict[str, object]],
    primary_metric: str,
    primary_goal: str,
    min_success_rate: float,
    min_mean_success_frames: float,
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, int, int, int, int], list[dict[str, object]]] = {}
    for item in results:
        key = (
            int(item["num_actors"]),
            int(item["num_threads"]),
            int(item["batch_size"]),
            int(item["unroll_length"]),
            int(item["replay_warmup_size"]),
            int(item["replay_buffer_size"]),
        )
        grouped.setdefault(key, []).append(item)

    summary: list[dict[str, object]] = []
    for key, items in grouped.items():
        first_item = items[0]
        successes = [item for item in items if bool(item["made_progress"]) and not bool(item["has_traceback"])]
        avg_fps_success = (
            sum(float(item["avg_fps"]) for item in successes) / len(successes)
            if successes
            else 0.0
        )
        frames_success = (
            sum(int(item["frames"]) for item in successes) / len(successes)
            if successes
            else 0.0
        )
        mer_landlord_success = (
            sum(_as_float(item["mer_landlord"]) for item in successes) / len(successes)
            if successes
            else 0.0
        )
        mer_farmer_success = (
            sum(_as_float(item["mer_farmer"]) for item in successes) / len(successes)
            if successes
            else 0.0
        )
        mer_bidding_success = (
            sum(_as_float(item["mer_bidding"]) for item in successes) / len(successes)
            if successes
            else 0.0
        )
        summary.append(
            {
                "combo_index": int(first_item["combo_index"]),
                "num_actors": key[0],
                "num_threads": key[1],
                "gpu": str(first_item["gpu"]),
                "gpu_actors": bool(first_item["gpu_actors"]),
                "num_actor_devices": int(first_item["num_actor_devices"]),
                "training_device": str(first_item["training_device"]),
                "total_actors": int(first_item["total_actors"]),
                "batch_size": key[2],
                "unroll_length": key[3],
                "replay_warmup_size": key[4],
                "replay_buffer_size": key[5],
                "repeat_count": len(items),
                "success_count": len(successes),
                "success_rate": round(len(successes) / len(items), 3),
                "mean_success_frames": round(frames_success, 1),
                "best_frames": max(int(item["frames"]) for item in items),
                "mean_success_avg_fps": round(avg_fps_success, 1),
                "mean_success_mer_landlord": round(mer_landlord_success, 6),
                "mean_success_mer_farmer": round(mer_farmer_success, 6),
                "mean_success_mer_bidding": round(mer_bidding_success, 6),
                "traceback_count": sum(bool(item["has_traceback"]) for item in items),
                "example_xpid": str(first_item["xpid"]),
                "example_log_path": str(first_item["log_path"]),
            }
        )
        for source_key, target_key in GPU_AGGREGATE_KEYS:
            summary[-1][target_key] = (
                round(
                    sum(float(item[source_key]) for item in successes) / len(successes),
                    1,
                )
                if successes
                else 0.0
            )
        for source_key, target_key in PROCESS_AGGREGATE_KEYS:
            summary[-1][target_key] = (
                round(
                    sum(float(item[source_key]) for item in successes) / len(successes),
                    1,
                )
                if successes
                else 0.0
            )
        summary[-1]["primary_metric"] = primary_metric
        summary[-1]["primary_goal"] = primary_goal
        summary[-1]["primary_value"] = round(
            _primary_metric_value(summary[-1], primary_metric),
            6,
        )
        summary[-1]["passes_gate"] = bool(
            _as_float(summary[-1]["success_rate"]) >= min_success_rate
            and _as_float(summary[-1]["mean_success_frames"]) >= min_mean_success_frames
            and int(summary[-1]["success_count"]) > 0
        )

    metric_direction = -1.0 if primary_goal == "maximize" else 1.0
    summary.sort(
        key=lambda item: (
            int(item["traceback_count"]),
            0 if bool(item["passes_gate"]) else 1,
            -int(item["success_count"]),
            -_as_float(item["success_rate"]),
            metric_direction * _primary_metric_value(item, primary_metric),
            -_as_float(item["mean_success_frames"]),
            -_as_float(item["mean_success_avg_fps"]),
            -int(item["best_frames"]),
            int(item["combo_index"]),
        )
    )
    return summary


def apply_keep_discard(
    aggregate_rows: list[dict[str, object]],
    *,
    run_stamp: str,
    label: str,
    primary_metric: str,
    primary_goal: str,
    min_success_rate: float,
    min_mean_success_frames: float,
    min_primary_improvement: float,
    decision_state: dict[str, object],
    results_tsv_path: Path | None,
) -> list[dict[str, object]]:
    if not aggregate_rows:
        return aggregate_rows

    rows_by_execution_order = sorted(aggregate_rows, key=lambda row: int(row["combo_index"]))
    tsv_rows: list[dict[str, object]] = []
    incumbent_value = (
        _as_float(decision_state.get("incumbent_primary_value"))
        if decision_state.get("incumbent_primary_value") is not None
        else None
    )
    incumbent_ref = str(decision_state.get("incumbent_ref", ""))

    for row in rows_by_execution_order:
        success_count = int(row["success_count"])
        success_rate = _as_float(row["success_rate"])
        mean_success_frames = _as_float(row["mean_success_frames"])
        traceback_count = int(row["traceback_count"])
        primary_value = _primary_metric_value(row, primary_metric)
        incumbent_before = incumbent_value

        if success_count <= 0:
            status = "crash"
            reason = "no successful repeats"
        elif traceback_count > 0 and success_count <= 0:
            status = "crash"
            reason = "all repeats failed with traceback"
        elif success_rate < min_success_rate:
            status = "discard"
            reason = (
                f"success_rate {success_rate:.3f} below gate {min_success_rate:.3f}"
            )
        elif mean_success_frames < min_mean_success_frames:
            status = "discard"
            reason = (
                f"mean_success_frames {mean_success_frames:.1f} below gate {min_mean_success_frames:.1f}"
            )
        elif incumbent_value is None:
            status = "keep"
            reason = "baseline incumbent initialized"
            incumbent_value = primary_value
            incumbent_ref = (
                f"{label}/combo{int(row['combo_index']):03d}"
            )
        elif _is_better_metric(
            candidate=primary_value,
            incumbent=incumbent_value,
            goal=primary_goal,
            min_improvement=min_primary_improvement,
        ):
            status = "keep"
            reason = (
                f"improved {primary_metric} from {incumbent_value:.6f} to {primary_value:.6f}"
            )
            incumbent_value = primary_value
            incumbent_ref = (
                f"{label}/combo{int(row['combo_index']):03d}"
            )
        else:
            status = "discard"
            reason = (
                f"no {primary_metric} improvement over incumbent {incumbent_value:.6f}"
            )

        row["decision_status"] = status
        row["decision_reason"] = reason
        row["decision_incumbent_before"] = (
            "" if incumbent_before is None else round(float(incumbent_before), 6)
        )
        row["decision_incumbent_after"] = (
            "" if incumbent_value is None else round(float(incumbent_value), 6)
        )

        tsv_rows.append(
            {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "run_stamp": run_stamp,
                "stage": label,
                "combo_index": int(row["combo_index"]),
                "status": status,
                "reason": reason,
                "primary_metric": primary_metric,
                "primary_goal": primary_goal,
                "primary_value": round(primary_value, 6),
                "incumbent_before": "" if incumbent_before is None else round(float(incumbent_before), 6),
                "incumbent_after": "" if incumbent_value is None else round(float(incumbent_value), 6),
                "success_count": success_count,
                "repeat_count": int(row["repeat_count"]),
                "success_rate": round(success_rate, 3),
                "traceback_count": traceback_count,
                "mean_success_frames": round(mean_success_frames, 1),
                "mean_success_avg_fps": round(_as_float(row["mean_success_avg_fps"]), 1),
                "mean_success_mer_landlord": round(_as_float(row["mean_success_mer_landlord"]), 6),
                "mean_success_mer_farmer": round(_as_float(row["mean_success_mer_farmer"]), 6),
                "mean_success_mer_bidding": round(_as_float(row["mean_success_mer_bidding"]), 6),
                "num_actors": int(row["num_actors"]),
                "num_threads": int(row["num_threads"]),
                "batch_size": int(row["batch_size"]),
                "unroll_length": int(row["unroll_length"]),
                "replay_warmup_size": int(row["replay_warmup_size"]),
                "replay_buffer_size": int(row["replay_buffer_size"]),
                "total_actors": int(row["total_actors"]),
                "example_xpid": str(row["example_xpid"]),
                "example_log_path": str(row["example_log_path"]),
            }
        )

    decision_state["incumbent_primary_value"] = incumbent_value
    decision_state["incumbent_ref"] = incumbent_ref
    decision_state["primary_metric"] = primary_metric
    decision_state["primary_goal"] = primary_goal

    if results_tsv_path is not None:
        append_results_tsv(results_tsv_path, tsv_rows)

    return aggregate_rows


def build_two_stage_coarse_combinations(
    args: argparse.Namespace,
    runtime_layout: dict[str, object],
) -> list[tuple[int, int, int, int, int, int]]:
    combos = []
    for num_actors, num_threads in itertools.product(args.num_actors, args.num_threads):
        combo = (
            num_actors,
            num_threads,
            args.coarse_batch_size,
            args.coarse_unroll_length,
            args.coarse_replay_warmup_size,
            args.coarse_replay_buffer_size,
            )
        if should_skip_combo(
            *combo,
            actor_device_count=int(runtime_layout["actor_device_count"]),
            max_total_actors=args.max_total_actors,
        ):
            continue
        combos.append(combo)
    return list(dict.fromkeys(combos))


def build_two_stage_fine_combinations(
    args: argparse.Namespace,
    runtime_layout: dict[str, object],
    best_num_actors: int,
    best_num_threads: int,
) -> list[tuple[int, int, int, int, int, int]]:
    combos = [
        (
            best_num_actors,
            best_num_threads,
            batch_size,
            unroll_length,
            replay_warmup,
            replay_size,
        )
        for batch_size, unroll_length, replay_warmup, replay_size in itertools.product(
            args.batch_sizes,
            args.unroll_lengths,
            args.replay_warmups,
            args.replay_sizes,
        )
        if not should_skip_combo(
            best_num_actors,
            best_num_threads,
            batch_size,
            unroll_length,
            replay_warmup,
            replay_size,
            actor_device_count=int(runtime_layout["actor_device_count"]),
            max_total_actors=args.max_total_actors,
        )
    ]
    return list(dict.fromkeys(combos))


def build_train_command(
    python_bin: str,
    repo_root: Path,
    savedir: Path,
    xpid: str,
    runtime_layout: dict[str, object],
    num_actors: int,
    num_threads: int,
    batch_size: int,
    unroll_length: int,
    replay_warmup: int,
    replay_size: int,
    extra_args: str,
) -> list[str]:
    cmd = [
        python_bin,
        "train.py",
        "--xpid",
        xpid,
        "--savedir",
        str(savedir),
        "--num_actor_devices",
        str(runtime_layout["num_actor_devices"]),
        "--num_actors",
        str(num_actors),
        "--num_threads",
        str(num_threads),
        "--batch_size",
        str(batch_size),
        "--unroll_length",
        str(unroll_length),
        "--replay_warmup_size",
        str(replay_warmup),
        "--replay_buffer_size",
        str(replay_size),
        "--save_interval",
        "1000",
        "--disable_checkpoint",
    ]

    if runtime_layout["mode"] == "cpu":
        cmd.extend(
            [
                "--actor_device_cpu",
                "--training_device",
                "cpu",
                "--gpu_devices",
                "",
            ]
        )
    else:
        if runtime_layout["actor_device_cpu"]:
            cmd.append("--actor_device_cpu")
        cmd.extend(
            [
                "--training_device",
                str(runtime_layout["training_device"]),
                "--gpu_devices",
                str(runtime_layout["gpu_devices"]),
            ]
        )

    if extra_args.strip():
        cmd.extend(shlex.split(extra_args))

    return cmd


def run_trials(
    *,
    combinations: list[tuple[int, int, int, int, int, int]],
    args: argparse.Namespace,
    repo_root: Path,
    savedir: Path,
    run_stamp: str,
    label: str,
    runtime_layout: dict[str, object],
    decision_state: dict[str, object],
    results_tsv_path: Path | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], Path, Path]:
    if args.limit > 0:
        combinations = combinations[: args.limit]

    if not combinations:
        raise SystemExit(f"no parameter combinations left for stage {label}")

    summary_path = savedir / f"cli_search_{run_stamp}_{label}.csv"
    aggregate_path = savedir / f"cli_search_{run_stamp}_{label}_aggregate.csv"

    print(f"Stage       : {label}")
    print(f"Combos      : {len(combinations)}")
    print(f"Repeats     : {args.repeats}")
    print(f"Trial time  : {args.trial_seconds}s")
    print(f"GPU sample  : {args.gpu_sample_seconds}s")
    print()

    results: list[dict[str, object]] = []
    total_runs = len(combinations) * args.repeats
    run_index = 0
    gpu_groups = resolve_gpu_groups(args.gpu, bool(args.gpu_actors))

    for combo_index, (
        num_actors,
        num_threads,
        batch_size,
        unroll_length,
        replay_warmup,
        replay_size,
    ) in enumerate(combinations, start=1):
        for repeat_index in range(1, args.repeats + 1):
            run_index += 1
            xpid = (
                f"cli_search_{run_stamp}_{label}_{combo_index:03d}"
                f"_r{repeat_index:02d}"
                f"_a{num_actors}_t{num_threads}_b{batch_size}"
                f"_u{unroll_length}_w{replay_warmup}_r{replay_size}"
            )
            cmd = build_train_command(
                python_bin=args.python_bin,
                repo_root=repo_root,
                savedir=savedir,
                xpid=xpid,
                runtime_layout=runtime_layout,
                num_actors=num_actors,
                num_threads=num_threads,
                batch_size=batch_size,
                unroll_length=unroll_length,
                replay_warmup=replay_warmup,
                replay_size=replay_size,
                extra_args=args.extra_args,
            )
            log_path = savedir / f"{xpid}.log"
            gpu_log_path = savedir / f"{xpid}.gpu.csv"
            proc_log_path = savedir / f"{xpid}.proc.csv"

            print(f"[{run_index}/{total_runs}] {xpid}")
            print("  " + " ".join(shlex.quote(part) for part in cmd))

            start = time.time()
            gpu_samples: list[dict[str, object]] = []
            proc_samples: list[dict[str, float]] = []
            gpu_stop_event = threading.Event()
            gpu_thread: threading.Thread | None = None
            proc_thread: threading.Thread | None = None
            with log_path.open("w", encoding="utf-8") as log_file:
                proc = subprocess.Popen(
                    cmd,
                    cwd=repo_root,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                if args.gpu_sample_seconds > 0 and gpu_groups["visible_gpu_ids"]:
                    gpu_thread = threading.Thread(
                        target=collect_gpu_samples,
                        args=(
                            proc,
                            gpu_groups["visible_gpu_ids"],
                            args.gpu_sample_seconds,
                            gpu_samples,
                            gpu_stop_event,
                        ),
                        daemon=True,
                    )
                    gpu_thread.start()
                if args.gpu_sample_seconds > 0:
                    proc_thread = threading.Thread(
                        target=collect_process_samples,
                        args=(
                            proc,
                            args.gpu_sample_seconds,
                            proc_samples,
                            gpu_stop_event,
                        ),
                        daemon=True,
                    )
                    proc_thread.start()
                try:
                    proc.wait(timeout=args.trial_seconds)
                except subprocess.TimeoutExpired:
                    ensure_process_stopped(proc)
                finally:
                    gpu_stop_event.set()
                    if gpu_thread is not None:
                        gpu_thread.join(timeout=max(1.0, args.gpu_sample_seconds + 1.0))
                    if proc_thread is not None:
                        proc_thread.join(timeout=max(1.0, args.gpu_sample_seconds + 1.0))
            elapsed = time.time() - start
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            metrics = parse_metrics(log_text)
            gpu_metrics = summarize_gpu_samples(gpu_samples, gpu_groups)
            proc_metrics = summarize_process_samples(proc_samples)
            if gpu_samples:
                write_gpu_samples_csv(gpu_samples, gpu_log_path)
            if proc_samples:
                write_process_samples_csv(proc_samples, proc_log_path)
            result = {
                "stage": label,
                "xpid": xpid,
                "combo_index": combo_index,
                "repeat_index": repeat_index,
                "gpu": args.gpu,
                "gpu_actors": bool(args.gpu_actors),
                "num_actor_devices": int(runtime_layout["num_actor_devices"]),
                "training_device": str(runtime_layout["training_device"]),
                "total_actors": num_actors * int(runtime_layout["actor_device_count"]),
                "num_actors": num_actors,
                "num_threads": num_threads,
                "batch_size": batch_size,
                "unroll_length": unroll_length,
                "replay_warmup_size": replay_warmup,
                "replay_buffer_size": replay_size,
                "elapsed_seconds": round(elapsed, 1),
                "frames": int(metrics["frames"]),
                "fps": float(metrics["fps"]),
                "avg_fps": float(metrics["avg_fps"]),
                "mer_landlord": float(metrics["mer_landlord"]),
                "mer_farmer": float(metrics["mer_farmer"]),
                "mer_bidding": float(metrics["mer_bidding"]),
                "made_progress": bool(metrics["made_progress"]),
                "stuck_at_zero": bool(metrics["stuck_at_zero"]),
                "has_traceback": bool(metrics["has_traceback"]),
                "log_path": str(log_path),
                "gpu_metrics_path": str(gpu_log_path) if gpu_samples else "",
                "proc_metrics_path": str(proc_log_path) if proc_samples else "",
                "gpu_power_mean_w_by_id": str(gpu_metrics["gpu_power_mean_w_by_id"]),
                "gpu_util_mean_pct_by_id": str(gpu_metrics["gpu_util_mean_pct_by_id"]),
            }
            for key in GPU_PER_RUN_KEYS:
                result[key] = gpu_metrics[key]
            for key in PROCESS_PER_RUN_KEYS:
                result[key] = proc_metrics[key]
            results.append(result)
            print(
                "  result:"
                f" frames={result['frames']}"
                f" fps={result['fps']}"
                f" avg_fps={result['avg_fps']}"
                f" merL={result['mer_landlord']:.6f}"
                f" merF={result['mer_farmer']:.6f}"
                f" gpu_power={result['gpu_power_total_mean_w']}W"
                f" learner_util={result['learner_util_mean_pct']}%"
                f" cpu={result['proc_cpu_mean_pct']}%"
                f" rss={result['proc_mem_mean_mib']}MiB"
                f" progress={result['made_progress']}"
                f" traceback={result['has_traceback']}"
            )
            print()
            if args.cooldown_seconds > 0 and run_index != total_runs:
                time.sleep(args.cooldown_seconds)

    per_run_results = sorted(
        results,
        key=lambda item: (
            int(item["has_traceback"]),
            not bool(item["made_progress"]),
            -int(item["frames"]),
            -float(item["avg_fps"]),
        )
    )
    aggregate = aggregate_results(
        results,
        primary_metric=args.primary_metric,
        primary_goal=args.primary_goal,
        min_success_rate=args.min_success_rate,
        min_mean_success_frames=args.min_mean_success_frames,
    )
    aggregate = apply_keep_discard(
        aggregate,
        run_stamp=run_stamp,
        label=label,
        primary_metric=args.primary_metric,
        primary_goal=args.primary_goal,
        min_success_rate=args.min_success_rate,
        min_mean_success_frames=args.min_mean_success_frames,
        min_primary_improvement=args.min_primary_improvement,
        decision_state=decision_state,
        results_tsv_path=results_tsv_path,
    )

    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(per_run_results[0].keys()) if per_run_results else [])
        if per_run_results:
            writer.writeheader()
            writer.writerows(per_run_results)

    with aggregate_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(aggregate[0].keys()) if aggregate else [])
        if aggregate:
            writer.writeheader()
            writer.writerows(aggregate)

    print(f"Top results ({label}):")
    for rank, item in enumerate(aggregate[:10], start=1):
        print(
            f"{rank}. success={item['success_count']}/{item['repeat_count']}"
            f" rate={item['success_rate']} best_frames={item['best_frames']}"
            f" mean_success_frames={item['mean_success_frames']}"
            f" mean_success_avg_fps={item['mean_success_avg_fps']} "
            f" merL={item['mean_success_mer_landlord']:.6f} "
            f" merF={item['mean_success_mer_farmer']:.6f} "
            f"gpu_power={item['mean_success_gpu_power_total_w']}W "
            f"learner_power={item['mean_success_learner_power_w']}W "
            f"learner_util={item['mean_success_learner_util_pct']}% "
            f"cpu={item['mean_success_proc_cpu_pct']}% "
            f"rss={item['mean_success_proc_mem_mib']}MiB "
            f"actors/device={item['num_actors']} total_actors={item['total_actors']} "
            f"threads={item['num_threads']} "
            f"batch={item['batch_size']} unroll={item['unroll_length']} "
            f"warmup={item['replay_warmup_size']} replay={item['replay_buffer_size']} "
            f"status={item.get('decision_status', '')}"
        )
    print()
    print(f"Summary CSV ({label}): {summary_path}")
    print(f"Aggregate CSV ({label}): {aggregate_path}")
    print()
    return per_run_results, aggregate, summary_path, aggregate_path


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    savedir = (repo_root / args.savedir).resolve()
    savedir.mkdir(parents=True, exist_ok=True)
    args.primary_goal = resolve_primary_goal(args.primary_metric, args.primary_goal)
    results_tsv_path: Path | None = None
    if args.results_tsv.strip():
        raw_results_path = Path(args.results_tsv)
        if not raw_results_path.is_absolute():
            raw_results_path = savedir / raw_results_path
        results_tsv_path = raw_results_path.resolve()
        ensure_results_tsv(results_tsv_path, reset=args.reset_results_tsv)
    decision_state: dict[str, object] = {
        "incumbent_primary_value": None,
        "incumbent_ref": "",
        "primary_metric": args.primary_metric,
        "primary_goal": args.primary_goal,
    }
    if args.gpu_actors is None:
        args.gpu_actors = args.gpu.strip().lower() != "cpu"
    try:
        runtime_layout = resolve_runtime_layout(args.gpu, bool(args.gpu_actors))
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if runtime_layout["mode"] == "cpu" and args.gpu_actors:
        raise SystemExit("--gpu-actors requires a visible GPU list; use --gpu cpu --cpu-actors for full CPU mode")

    if args.search_mode == "empirical":
        combinations = [
            (
                item["num_actors"],
                item["num_threads"],
                item["batch_size"],
                item["unroll_length"],
                item["replay_warmup_size"],
                item["replay_buffer_size"],
            )
            for item in EMPIRICAL_TRIALS
        ]
    else:
        combinations = [
            combo
            for combo in itertools.product(
                args.num_actors,
                args.num_threads,
                args.batch_sizes,
                args.unroll_lengths,
                args.replay_warmups,
                args.replay_sizes,
            )
            if not should_skip_combo(
                *combo,
                actor_device_count=int(runtime_layout["actor_device_count"]),
                max_total_actors=args.max_total_actors,
            )
        ]

    run_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Repo root   : {repo_root}")
    print(f"Python      : {args.python_bin}")
    print(f"Search dir  : {savedir}")
    print(f"Mode        : {args.search_mode}")
    print(f"Visible GPU : {args.gpu}")
    print(f"GPU actors  : {bool(args.gpu_actors)}")
    print(f"Actor devs  : {runtime_layout['num_actor_devices']}")
    print(f"Learner dev : {runtime_layout['training_device']}")
    print(f"GPU sample  : {args.gpu_sample_seconds}s")
    print(f"Metric      : {args.primary_metric} ({args.primary_goal})")
    print(
        "Gate        : "
        f"success_rate>={args.min_success_rate:.3f}, "
        f"mean_success_frames>={args.min_mean_success_frames:.1f}, "
        f"min_improvement={args.min_primary_improvement}"
    )
    if results_tsv_path is not None:
        print(f"Results TSV : {results_tsv_path}")
    print("Modules     : default Full (Module A/B/C + bidding enabled)")
    if args.extra_args.strip():
        print(f"Overrides   : {args.extra_args}")
    print()

    if args.search_mode == "empirical":
        print("Using empirical search set centered on the known-good baseline")
        print("actors=2, threads=2, batch=16, unroll=16, warmup=8, replay=64")
        print()
    if args.search_mode == "two-stage":
        coarse_combinations = build_two_stage_coarse_combinations(args, runtime_layout)
        print(
            "Stage 1 will search actor/thread pairs with fixed baseline "
            f"batch={args.coarse_batch_size} unroll={args.coarse_unroll_length} "
            f"warmup={args.coarse_replay_warmup_size} replay={args.coarse_replay_buffer_size}"
        )
        print()
        _, coarse_aggregate, _, _ = run_trials(
            combinations=coarse_combinations,
            args=args,
            repo_root=repo_root,
            savedir=savedir,
            run_stamp=run_stamp,
            label="stage1_actor_thread",
            runtime_layout=runtime_layout,
            decision_state=decision_state,
            results_tsv_path=results_tsv_path,
        )
        best_stage1 = coarse_aggregate[0]
        best_num_actors = int(best_stage1["num_actors"])
        best_num_threads = int(best_stage1["num_threads"])
        print(
            f"Stage 1 winner: actors={best_num_actors} threads={best_num_threads} "
            f"success={best_stage1['success_count']}/{best_stage1['repeat_count']} "
            f"rate={best_stage1['success_rate']} "
            f"status={best_stage1.get('decision_status', '')}"
        )
        print()

        fine_combinations = build_two_stage_fine_combinations(
            args=args,
            runtime_layout=runtime_layout,
            best_num_actors=best_num_actors,
            best_num_threads=best_num_threads,
        )
        print(
            "Stage 2 will fix the winning actor/thread pair and search "
            "batch/unroll/warmup/replay."
        )
        print()
        _, fine_aggregate, fine_summary_path, fine_aggregate_path = run_trials(
            combinations=fine_combinations,
            args=args,
            repo_root=repo_root,
            savedir=savedir,
            run_stamp=run_stamp,
            label="stage2_batch_unroll_replay",
            runtime_layout=runtime_layout,
            decision_state=decision_state,
            results_tsv_path=results_tsv_path,
        )
        best_final = fine_aggregate[0]
        print("Final recommendation:")
        print(
            f"actors={best_final['num_actors']} threads={best_final['num_threads']} "
            f"batch={best_final['batch_size']} unroll={best_final['unroll_length']} "
            f"warmup={best_final['replay_warmup_size']} replay={best_final['replay_buffer_size']} "
            f"success={best_final['success_count']}/{best_final['repeat_count']} "
            f"rate={best_final['success_rate']} "
            f"mean_success_avg_fps={best_final['mean_success_avg_fps']} "
            f"mean_success_mer_landlord={best_final['mean_success_mer_landlord']} "
            f"mean_success_mer_farmer={best_final['mean_success_mer_farmer']} "
            f"mean_success_gpu_power_total_w={best_final['mean_success_gpu_power_total_w']} "
            f"mean_success_proc_cpu_pct={best_final['mean_success_proc_cpu_pct']} "
            f"mean_success_proc_mem_mib={best_final['mean_success_proc_mem_mib']} "
            f"mean_success_learner_util_pct={best_final['mean_success_learner_util_pct']} "
            f"status={best_final.get('decision_status', '')}"
        )
        print()
        print(f"Final summary CSV: {fine_summary_path}")
        print(f"Final aggregate CSV: {fine_aggregate_path}")
        if results_tsv_path is not None:
            print(f"Keep/discard TSV: {results_tsv_path}")
        return 0

    combinations = list(dict.fromkeys(combinations))
    _, aggregate, summary_path, aggregate_path = run_trials(
        combinations=combinations,
        args=args,
        repo_root=repo_root,
        savedir=savedir,
        run_stamp=run_stamp,
        label=args.search_mode,
        runtime_layout=runtime_layout,
        decision_state=decision_state,
        results_tsv_path=results_tsv_path,
    )
    if aggregate:
        best = aggregate[0]
        print("Recommended combination:")
        print(
            f"actors={best['num_actors']} threads={best['num_threads']} "
            f"batch={best['batch_size']} unroll={best['unroll_length']} "
            f"warmup={best['replay_warmup_size']} replay={best['replay_buffer_size']} "
            f"status={best.get('decision_status', '')}"
        )
        print()
    print(f"Summary CSV: {summary_path}")
    print(f"Aggregate CSV: {aggregate_path}")
    if results_tsv_path is not None:
        print(f"Keep/discard TSV: {results_tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
