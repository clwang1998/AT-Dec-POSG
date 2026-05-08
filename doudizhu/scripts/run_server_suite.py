#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from search_cli_params import (
    build_train_command,
    collect_gpu_samples,
    collect_process_samples,
    ensure_process_stopped,
    parse_metrics,
    resolve_gpu_groups,
    resolve_runtime_layout,
    summarize_gpu_samples,
    summarize_process_samples,
    write_gpu_samples_csv,
    write_process_samples_csv,
)


SMOKE_PRESETS = {
    "minimal": dict(
        num_actors=1,
        num_threads=1,
        batch_size=1,
        unroll_length=1,
        replay_warmup_size=1,
        replay_buffer_size=2,
        total_frames=10,
    ),
    "throughput": dict(
        num_actors=4,
        num_threads=2,
        batch_size=16,
        unroll_length=16,
        replay_warmup_size=8,
        replay_buffer_size=64,
        total_frames=4096,
    ),
}

HOTRUN_PROFILES = {
    "balanced": dict(
        num_actors=4,
        num_threads=2,
        batch_size=32,
        unroll_length=32,
        replay_warmup_size=16,
        replay_buffer_size=128,
        total_frames=50000,
    ),
    "dmc15": dict(
        num_actors=5,
        num_threads=2,
        batch_size=32,
        unroll_length=16,
        replay_warmup_size=16,
        replay_buffer_size=128,
        total_frames=50000,
    ),
    "dmc_default": dict(
        num_actors=5,
        num_threads=4,
        batch_size=32,
        unroll_length=100,
        replay_warmup_size=16,
        replay_buffer_size=64,
        total_frames=50000,
        extra_args="--epsilon 1e-5",
    ),
    "learner_push": dict(
        num_actors=5,
        num_threads=2,
        batch_size=64,
        unroll_length=32,
        replay_warmup_size=32,
        replay_buffer_size=256,
        total_frames=50000,
    ),
    "learner_push_threads": dict(
        num_actors=4,
        num_threads=4,
        batch_size=64,
        unroll_length=32,
        replay_warmup_size=32,
        replay_buffer_size=256,
        total_frames=50000,
    ),
}

SEARCH_SCOPES = {
    "narrow": {
        "num_actors": "1,2",
        "num_threads": "1",
        "batch_sizes": "16",
        "unroll_lengths": "16",
        "replay_warmups": "8",
        "replay_sizes": "64",
        "max_total_actors": 12,
    },
    "full": {
        "num_actors": "1,2,4",
        "num_threads": "1,2",
        "batch_sizes": "8,16,32",
        "unroll_lengths": "8,16",
        "replay_warmups": "4,8",
        "replay_sizes": "64,128",
        "max_total_actors": 12,
    },
    "saturate": {
        "num_actors": "3,4,5,6",
        "num_threads": "1,2,4",
        "batch_sizes": "16,32,64",
        "unroll_lengths": "16,32",
        "replay_warmups": "8,16,32",
        "replay_sizes": "64,128,256",
        "max_total_actors": 18,
    },
    "dmc": {
        "num_actors": "4,5,6",
        "num_threads": "2,4",
        "batch_sizes": "24,32,48",
        "unroll_lengths": "64,100,128",
        "replay_warmups": "8,16,32",
        "replay_sizes": "64,128,256",
        "max_total_actors": 18,
        "coarse_batch_size": 32,
        "coarse_unroll_length": 100,
        "coarse_replay_warmup_size": 16,
        "coarse_replay_buffer_size": 64,
        "fixed_extra_args": "--epsilon 1e-5",
    },
}

CLEANUP_PATTERNS = [
    "scripts/search_cli_params.py",
    "train.py --xpid cli_search_",
    "train.py --xpid suite_4gpu_",
]


def parse_csv_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def apply_runtime_env() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"


def cleanup_processes() -> None:
    for pattern in CLEANUP_PATTERNS:
        subprocess.run(["pkill", "-f", pattern], check=False)


def write_summary_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def monitor_single_run(
    *,
    cmd: list[str],
    cwd: Path,
    timeout_seconds: int,
    sample_seconds: float,
    gpu_groups: dict[str, list[int]],
    log_path: Path,
    gpu_log_path: Path,
    proc_log_path: Path,
) -> dict[str, object]:
    start = time.time()
    gpu_samples: list[dict[str, object]] = []
    proc_samples: list[dict[str, float]] = []
    stop_event = threading.Event()

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        gpu_thread = None
        proc_thread = None
        if sample_seconds > 0 and gpu_groups["visible_gpu_ids"]:
            gpu_thread = threading.Thread(
                target=collect_gpu_samples,
                args=(proc, gpu_groups["visible_gpu_ids"], sample_seconds, gpu_samples, stop_event),
                daemon=True,
            )
            gpu_thread.start()
        if sample_seconds > 0:
            proc_thread = threading.Thread(
                target=collect_process_samples,
                args=(proc, sample_seconds, proc_samples, stop_event),
                daemon=True,
            )
            proc_thread.start()

        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            ensure_process_stopped(proc)
        finally:
            stop_event.set()
            if gpu_thread is not None:
                gpu_thread.join(timeout=max(1.0, sample_seconds + 1.0))
            if proc_thread is not None:
                proc_thread.join(timeout=max(1.0, sample_seconds + 1.0))

    elapsed = round(time.time() - start, 1)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics = parse_metrics(log_text)
    gpu_metrics = summarize_gpu_samples(gpu_samples, gpu_groups)
    proc_metrics = summarize_process_samples(proc_samples)
    if gpu_samples:
        write_gpu_samples_csv(gpu_samples, gpu_log_path)
    if proc_samples:
        write_process_samples_csv(proc_samples, proc_log_path)

    result: dict[str, object] = {
        "elapsed_seconds": elapsed,
        "frames": int(metrics["frames"]),
        "fps": float(metrics["fps"]),
        "avg_fps": float(metrics["avg_fps"]),
        "made_progress": bool(metrics["made_progress"]),
        "stuck_at_zero": bool(metrics["stuck_at_zero"]),
        "has_traceback": bool(metrics["has_traceback"]),
        "log_path": str(log_path),
        "gpu_metrics_path": str(gpu_log_path) if gpu_samples else "",
        "proc_metrics_path": str(proc_log_path) if proc_samples else "",
        "gpu_power_mean_w_by_id": str(gpu_metrics["gpu_power_mean_w_by_id"]),
        "gpu_util_mean_pct_by_id": str(gpu_metrics["gpu_util_mean_pct_by_id"]),
    }
    result.update(gpu_metrics)
    result.update(proc_metrics)
    return result


def build_common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--gpu", default="0,1,2,3")
    actor_mode_group = parser.add_mutually_exclusive_group()
    actor_mode_group.add_argument("--gpu-actors", dest="gpu_actors", action="store_true")
    actor_mode_group.add_argument("--cpu-actors", dest="gpu_actors", action="store_false")
    parser.set_defaults(gpu_actors=True)
    parser.add_argument("--sample-seconds", type=float, default=5.0)
    parser.add_argument("--trial-seconds", type=int, default=300)
    parser.add_argument("--savedir", default="suite_outputs")
    parser.add_argument("--extra-args", default="")
    return parser


def build_parser() -> argparse.ArgumentParser:
    common = build_common_parser()
    parser = argparse.ArgumentParser(
        description="Unified server-side smoke/search/hotrun test entry for multi-GPU Doudizhu runs."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser("kill", parents=[common], help="Stop suite/search test processes.")

    smoke_parser = subparsers.add_parser("smoke", parents=[common], help="Run one smoke test.")
    smoke_parser.add_argument("--preset", choices=sorted(SMOKE_PRESETS.keys()), default="throughput")
    smoke_parser.add_argument("--total-frames", type=int, default=0)

    search_parser = subparsers.add_parser("search", parents=[common], help="Run a timed hyperparameter search.")
    search_parser.add_argument("--scope", choices=sorted(SEARCH_SCOPES.keys()), default="saturate")
    search_parser.add_argument("--repeats", type=int, default=2)
    search_parser.add_argument("--cooldown-seconds", type=int, default=5)
    search_parser.add_argument("--max-total-actors", type=int, default=0)

    hotrun_parser = subparsers.add_parser("hotrun", parents=[common], help="Run one or more learner-pressure profiles.")
    hotrun_parser.add_argument(
        "--profiles",
        default="balanced",
        help=f"Comma-separated hotrun profiles: {','.join(sorted(HOTRUN_PROFILES.keys()))}",
    )
    hotrun_parser.add_argument("--total-frames", type=int, default=0)

    all_parser = subparsers.add_parser("all", parents=[common], help="Run kill + smoke + search.")
    all_parser.add_argument("--smoke-preset", choices=sorted(SMOKE_PRESETS.keys()), default="throughput")
    all_parser.add_argument("--scope", choices=sorted(SEARCH_SCOPES.keys()), default="saturate")
    all_parser.add_argument("--repeats", type=int, default=2)
    all_parser.add_argument("--cooldown-seconds", type=int, default=5)
    all_parser.add_argument("--max-total-actors", type=int, default=0)
    return parser


def resolve_savedirs(repo_root: Path, savedir: str) -> dict[str, Path]:
    root = (repo_root / savedir).resolve()
    return {
        "root": root,
        "smoke": root / "smoke",
        "search": root / "search",
        "hotrun": root / "hotrun",
    }


def run_smoke_mode(args: argparse.Namespace, repo_root: Path) -> int:
    savedirs = resolve_savedirs(repo_root, args.savedir)
    savedirs["smoke"].mkdir(parents=True, exist_ok=True)
    runtime_layout = resolve_runtime_layout(args.gpu, bool(args.gpu_actors))
    gpu_groups = resolve_gpu_groups(args.gpu, bool(args.gpu_actors))
    preset = dict(SMOKE_PRESETS[args.preset])
    if args.total_frames > 0:
        preset["total_frames"] = args.total_frames

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    xpid = f"suite_4gpu_smoke_{args.preset}_{stamp}"
    cmd = build_train_command(
        python_bin=args.python_bin,
        repo_root=repo_root,
        savedir=savedirs["smoke"],
        xpid=xpid,
        runtime_layout=runtime_layout,
        num_actors=int(preset["num_actors"]),
        num_threads=int(preset["num_threads"]),
        batch_size=int(preset["batch_size"]),
        unroll_length=int(preset["unroll_length"]),
        replay_warmup=int(preset["replay_warmup_size"]),
        replay_size=int(preset["replay_buffer_size"]),
        extra_args=args.extra_args,
    )
    cmd.extend(["--total_frames", str(int(preset["total_frames"]))])

    print(f"Smoke preset : {args.preset}")
    print("Command      : " + " ".join(shlex.quote(part) for part in cmd))
    result = monitor_single_run(
        cmd=cmd,
        cwd=repo_root,
        timeout_seconds=args.trial_seconds,
        sample_seconds=args.sample_seconds,
        gpu_groups=gpu_groups,
        log_path=savedirs["smoke"] / f"{xpid}.log",
        gpu_log_path=savedirs["smoke"] / f"{xpid}.gpu.csv",
        proc_log_path=savedirs["smoke"] / f"{xpid}.proc.csv",
    )
    result.update(
        {
            "mode": "smoke",
            "preset": args.preset,
            "xpid": xpid,
            "gpu": args.gpu,
            "gpu_actors": bool(args.gpu_actors),
            "num_actor_devices": int(runtime_layout["num_actor_devices"]),
        }
    )
    summary_path = savedirs["smoke"] / f"{xpid}.summary.csv"
    write_summary_csv([result], summary_path)
    print(
        f"Smoke result: frames={result['frames']} avg_fps={result['avg_fps']} "
        f"gpu_power={result['gpu_power_total_mean_w']}W learner_util={result['learner_util_mean_pct']}% "
        f"cpu={result['proc_cpu_mean_pct']}% rss={result['proc_mem_mean_mib']}MiB"
    )
    print(f"Summary CSV : {summary_path}")
    return 0


def run_hotrun_mode(args: argparse.Namespace, repo_root: Path) -> int:
    savedirs = resolve_savedirs(repo_root, args.savedir)
    savedirs["hotrun"].mkdir(parents=True, exist_ok=True)
    runtime_layout = resolve_runtime_layout(args.gpu, bool(args.gpu_actors))
    gpu_groups = resolve_gpu_groups(args.gpu, bool(args.gpu_actors))
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    profiles = parse_csv_values(args.profiles)
    rows: list[dict[str, object]] = []

    for profile_name in profiles:
        if profile_name not in HOTRUN_PROFILES:
            raise SystemExit(f"unknown hotrun profile: {profile_name}")
        profile = dict(HOTRUN_PROFILES[profile_name])
        if args.total_frames > 0:
            profile["total_frames"] = args.total_frames
        xpid = f"suite_4gpu_hotrun_{profile_name}_{stamp}"
        merged_extra_args = " ".join(
            part for part in [str(profile.get("extra_args", "")).strip(), args.extra_args.strip()] if part
        )
        cmd = build_train_command(
            python_bin=args.python_bin,
            repo_root=repo_root,
            savedir=savedirs["hotrun"],
            xpid=xpid,
            runtime_layout=runtime_layout,
            num_actors=int(profile["num_actors"]),
            num_threads=int(profile["num_threads"]),
            batch_size=int(profile["batch_size"]),
            unroll_length=int(profile["unroll_length"]),
            replay_warmup=int(profile["replay_warmup_size"]),
            replay_size=int(profile["replay_buffer_size"]),
            extra_args=merged_extra_args,
        )
        cmd.extend(["--total_frames", str(int(profile["total_frames"]))])
        print(f"Hotrun profile: {profile_name}")
        print("Command       : " + " ".join(shlex.quote(part) for part in cmd))
        result = monitor_single_run(
            cmd=cmd,
            cwd=repo_root,
            timeout_seconds=args.trial_seconds,
            sample_seconds=args.sample_seconds,
            gpu_groups=gpu_groups,
            log_path=savedirs["hotrun"] / f"{xpid}.log",
            gpu_log_path=savedirs["hotrun"] / f"{xpid}.gpu.csv",
            proc_log_path=savedirs["hotrun"] / f"{xpid}.proc.csv",
        )
        result.update(
            {
                "mode": "hotrun",
                "profile": profile_name,
                "xpid": xpid,
                "gpu": args.gpu,
                "gpu_actors": bool(args.gpu_actors),
                "num_actor_devices": int(runtime_layout["num_actor_devices"]),
            }
        )
        rows.append(result)
        print(
            f"Hotrun result: frames={result['frames']} avg_fps={result['avg_fps']} "
            f"gpu_power={result['gpu_power_total_mean_w']}W learner_util={result['learner_util_mean_pct']}% "
            f"cpu={result['proc_cpu_mean_pct']}% rss={result['proc_mem_mean_mib']}MiB"
        )
        print()

    summary_path = savedirs["hotrun"] / f"suite_4gpu_hotrun_{stamp}.csv"
    write_summary_csv(rows, summary_path)
    print(f"Hotrun summary CSV: {summary_path}")
    return 0


def build_search_command(args: argparse.Namespace, repo_root: Path) -> list[str]:
    scope = SEARCH_SCOPES[args.scope]
    max_total_actors = args.max_total_actors if args.max_total_actors > 0 else int(scope["max_total_actors"])
    merged_extra_args = " ".join(
        part for part in [str(scope.get("fixed_extra_args", "")).strip(), args.extra_args.strip()] if part
    )
    cmd = [
        args.python_bin,
        str((repo_root / "scripts" / "search_cli_params.py").resolve()),
        "--python",
        args.python_bin,
        "--gpu",
        args.gpu,
        "--search-mode",
        "two-stage",
        "--trial-seconds",
        str(args.trial_seconds),
        "--sample-seconds",
        str(args.sample_seconds),
        "--cooldown-seconds",
        str(args.cooldown_seconds),
        "--repeats",
        str(args.repeats),
        "--max-total-actors",
        str(max_total_actors),
        "--savedir",
        str((resolve_savedirs(repo_root, args.savedir)["search"]).resolve()),
        "--num-actors",
        scope["num_actors"],
        "--num-threads",
        scope["num_threads"],
        "--batch-sizes",
        scope["batch_sizes"],
        "--unroll-lengths",
        scope["unroll_lengths"],
        "--replay-warmups",
        scope["replay_warmups"],
        "--replay-sizes",
        scope["replay_sizes"],
        "--coarse-batch-size",
        str(scope.get("coarse_batch_size", 16)),
        "--coarse-unroll-length",
        str(scope.get("coarse_unroll_length", 16)),
        "--coarse-replay-warmup-size",
        str(scope.get("coarse_replay_warmup_size", 8)),
        "--coarse-replay-buffer-size",
        str(scope.get("coarse_replay_buffer_size", 64)),
    ]
    if bool(args.gpu_actors):
        cmd.append("--gpu-actors")
    else:
        cmd.append("--cpu-actors")
    if merged_extra_args:
        cmd.extend(["--extra-args", merged_extra_args])
    return cmd


def run_search_mode(args: argparse.Namespace, repo_root: Path) -> int:
    search_dir = resolve_savedirs(repo_root, args.savedir)["search"]
    search_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_search_command(args, repo_root)
    print("Search command: " + " ".join(shlex.quote(part) for part in cmd))
    proc = subprocess.run(cmd, cwd=repo_root, check=False)
    return proc.returncode


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    apply_runtime_env()

    if args.mode == "kill":
        cleanup_processes()
        time.sleep(2)
        print("Stopped suite/search test processes.")
        return 0

    print(f"Repo root  : {repo_root}")
    print(f"Python     : {args.python_bin}")
    print(f"GPU        : {args.gpu}")
    print(f"GPU actors : {bool(args.gpu_actors)}")
    print(f"Sample     : {args.sample_seconds}s")
    print(f"Trial      : {args.trial_seconds}s")
    print(f"Save dir   : {resolve_savedirs(repo_root, args.savedir)['root']}")
    print()

    if args.mode == "smoke":
        return run_smoke_mode(args, repo_root)
    if args.mode == "search":
        return run_search_mode(args, repo_root)
    if args.mode == "hotrun":
        return run_hotrun_mode(args, repo_root)
    if args.mode == "all":
        cleanup_processes()
        time.sleep(2)
        smoke_args = argparse.Namespace(**vars(args))
        smoke_args.preset = args.smoke_preset
        smoke_args.total_frames = 0
        run_smoke_mode(smoke_args, repo_root)
        print()
        search_args = argparse.Namespace(**vars(args))
        return run_search_mode(search_args, repo_root)
    raise SystemExit(f"unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
