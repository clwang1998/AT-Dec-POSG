#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path


PLAY_ROLES = ("landlord", "landlord_up", "landlord_down")
CHECKPOINT_RE = re.compile(
    r"^general_(landlord|landlord_up|landlord_down)_(\d+)\.ckpt$"
)
LANDLORD_FARMERS_RE = re.compile(
    r"landlord : Farmers - ([+-]?\d+(?:\.\d+)?) : ([+-]?\d+(?:\.\d+)?)"
)
DRAW_RE = re.compile(r"number of draw: - (\d+)")
BID_HIST_RE = re.compile(r"bid count histogram: \[([^\]]*)\]")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Watch a training checkpoint directory and automatically evaluate "
            "each new landlord/landlord_up/landlord_down checkpoint set."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory that receives general_landlord*.ckpt files.",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        required=True,
        help="Path to eval_data pickle used by evaluate.py.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used to run evaluate.py. Default: current interpreter.",
    )
    parser.add_argument(
        "--gpu-device",
        default="3",
        help="GPU id passed to evaluate.py. Default: 3 (the fourth GPU).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of evaluation workers per match. Default: 4.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Evaluation seed. Default: 2026.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="How often to scan for new checkpoints. Default: 60.",
    )
    parser.add_argument(
        "--min-age-seconds",
        type=int,
        default=30,
        help="Ignore checkpoint files newer than this age to avoid partial saves. Default: 30.",
    )
    parser.add_argument(
        "--opponents",
        default="douzero_WP,resnet_model",
        help=(
            "Comma-separated opponent presets. Available: douzero_WP, "
            "resnet_model, douzero_ADP. Default: douzero_WP,resnet_model."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for auto-eval logs and summary CSV. "
            "Default: <checkpoint-dir>/auto_eval."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repo root containing evaluate.py and baselines/. Default: inferred from this script.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process currently visible checkpoints once and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned evaluation commands without executing them.",
    )
    return parser.parse_args()


def opponent_presets(repo_root: Path) -> dict[str, dict[str, Path]]:
    base = repo_root / "baselines"
    return {
        "douzero_WP": {
            "landlord": base / "douzero_WP" / "landlord.ckpt",
            "landlord_down": base / "douzero_WP" / "landlord_down.ckpt",
            "landlord_up": base / "douzero_WP" / "landlord_up.ckpt",
        },
        "resnet_model": {
            "landlord": base / "resnet_model" / "resnet_landlord.ckpt",
            "landlord_down": base / "resnet_model" / "resnet_landlord_down.ckpt",
            "landlord_up": base / "resnet_model" / "resnet_landlord_up.ckpt",
        },
        "douzero_ADP": {
            "landlord": base / "douzero_ADP" / "landlord.ckpt",
            "landlord_down": base / "douzero_ADP" / "landlord_down.ckpt",
            "landlord_up": base / "douzero_ADP" / "landlord_up.ckpt",
        },
    }


def scan_ready_frames(checkpoint_dir: Path, min_age_seconds: int) -> dict[int, dict[str, Path]]:
    now = time.time()
    grouped: dict[int, dict[str, Path]] = {}
    for path in checkpoint_dir.glob("general_*.ckpt"):
        match = CHECKPOINT_RE.match(path.name)
        if match is None:
            continue
        role, frame_raw = match.groups()
        frame = int(frame_raw)
        grouped.setdefault(frame, {})[role] = path

    ready: dict[int, dict[str, Path]] = {}
    for frame, role_map in grouped.items():
        if not all(role in role_map for role in PLAY_ROLES):
            continue
        if any(now - role_map[role].stat().st_mtime < min_age_seconds for role in PLAY_ROLES):
            continue
        ready[frame] = role_map
    return dict(sorted(ready.items()))


def summary_path(output_dir: Path) -> Path:
    return output_dir / "summary.csv"


def load_completed_runs(path: Path) -> set[tuple[int, str, str]]:
    completed: set[tuple[int, str, str]] = set()
    if not path.exists():
        return completed
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row.get("status") != "success":
                continue
            completed.add((
                int(row["frame"]),
                row["opponent"],
                row["candidate_side"],
            ))
    return completed


def append_summary_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "frame",
        "opponent",
        "candidate_side",
        "status",
        "returncode",
        "wp_landlord",
        "wp_farmers",
        "adp_landlord",
        "adp_farmers",
        "draws",
        "bid_count_histogram",
        "log_path",
        "command",
    ]
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_eval_metrics(log_text: str) -> dict[str, object]:
    matches = LANDLORD_FARMERS_RE.findall(log_text)
    if len(matches) < 2:
        raise ValueError("Could not find both WP and ADP landlord/farmers lines in evaluation log.")
    wp_landlord, wp_farmers = (float(value) for value in matches[0])
    adp_landlord, adp_farmers = (float(value) for value in matches[1])
    draws_match = DRAW_RE.search(log_text)
    bid_hist_match = BID_HIST_RE.search(log_text)
    return {
        "wp_landlord": wp_landlord,
        "wp_farmers": wp_farmers,
        "adp_landlord": adp_landlord,
        "adp_farmers": adp_farmers,
        "draws": int(draws_match.group(1)) if draws_match else "",
        "bid_count_histogram": bid_hist_match.group(1).replace(" ", "") if bid_hist_match else "",
    }


def timestamp_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def matchup_log_path(output_dir: Path, frame: int, opponent: str, candidate_side: str) -> Path:
    frame_dir = output_dir / str(frame)
    frame_dir.mkdir(parents=True, exist_ok=True)
    suffix = "candidate_landlord" if candidate_side == "landlord" else "candidate_farmers"
    return frame_dir / f"{opponent}_{suffix}.log"


def build_eval_command(
    *,
    python_bin: str,
    evaluate_py: Path,
    eval_data: Path,
    gpu_device: str,
    num_workers: int,
    seed: int,
    candidate_paths: dict[str, Path],
    opponent_paths: dict[str, Path],
    candidate_side: str,
) -> list[str]:
    if candidate_side == "landlord":
        landlord = candidate_paths["landlord"]
        landlord_down = opponent_paths["landlord_down"]
        landlord_up = opponent_paths["landlord_up"]
    else:
        landlord = opponent_paths["landlord"]
        landlord_down = candidate_paths["landlord_down"]
        landlord_up = candidate_paths["landlord_up"]
    return [
        python_bin,
        str(evaluate_py),
        "--gpu_device",
        gpu_device,
        "--eval_data",
        str(eval_data),
        "--num_workers",
        str(num_workers),
        "--seed",
        str(seed),
        "--player_1_bid",
        "random",
        "--player_2_bid",
        "random",
        "--player_3_bid",
        "random",
        "--landlord",
        str(landlord),
        "--landlord_down",
        str(landlord_down),
        "--landlord_up",
        str(landlord_up),
    ]


def run_evaluation(
    *,
    command: list[str],
    cwd: Path,
    log_path: Path,
    dry_run: bool,
) -> tuple[int, str]:
    if dry_run:
        log_path.write_text("DRY RUN\n" + " ".join(command) + "\n", encoding="utf-8")
        return 0, ""
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    return result.returncode, log_text


def main() -> int:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    repo_root = args.repo_root.resolve()
    eval_data = args.eval_data.resolve()
    output_dir = (args.output_dir or (checkpoint_dir / "auto_eval")).resolve()
    evaluate_py = repo_root / "evaluate.py"

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if not evaluate_py.exists():
        raise FileNotFoundError(f"evaluate.py not found at: {evaluate_py}")
    if not eval_data.exists():
        raise FileNotFoundError(f"Eval data not found at: {eval_data}")

    preset_map = opponent_presets(repo_root)
    opponent_names = [item.strip() for item in args.opponents.split(",") if item.strip()]
    if not opponent_names:
        raise ValueError("At least one opponent preset is required.")
    for opponent_name in opponent_names:
        if opponent_name not in preset_map:
            raise ValueError(
                f"Unknown opponent preset `{opponent_name}`. "
                f"Available: {', '.join(sorted(preset_map))}."
            )
        for role, path in preset_map[opponent_name].items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing opponent checkpoint for {opponent_name} {role}: {path}"
                )

    completed_runs = load_completed_runs(summary_path(output_dir))
    print(
        f"[{timestamp_now()}] Watching {checkpoint_dir} on GPU {args.gpu_device} "
        f"against {', '.join(opponent_names)}",
        flush=True,
    )

    while True:
        ready_frames = scan_ready_frames(checkpoint_dir, args.min_age_seconds)
        pending_work = []
        for frame, candidate_paths in ready_frames.items():
            for opponent_name in opponent_names:
                for candidate_side in ("landlord", "farmers"):
                    run_key = (frame, opponent_name, candidate_side)
                    if run_key in completed_runs:
                        continue
                    pending_work.append((frame, candidate_paths, opponent_name, candidate_side))

        if not pending_work:
            if args.once:
                print(f"[{timestamp_now()}] No pending checkpoint evaluations.", flush=True)
                return 0
            time.sleep(args.poll_seconds)
            continue

        for frame, candidate_paths, opponent_name, candidate_side in pending_work:
            log_path = matchup_log_path(output_dir, frame, opponent_name, candidate_side)
            command = build_eval_command(
                python_bin=args.python_bin,
                evaluate_py=evaluate_py,
                eval_data=eval_data,
                gpu_device=args.gpu_device,
                num_workers=args.num_workers,
                seed=args.seed,
                candidate_paths=candidate_paths,
                opponent_paths=preset_map[opponent_name],
                candidate_side=candidate_side,
            )
            print(
                f"[{timestamp_now()}] Evaluating frame {frame} vs {opponent_name} "
                f"as {candidate_side}.",
                flush=True,
            )
            returncode, log_text = run_evaluation(
                command=command,
                cwd=repo_root,
                log_path=log_path,
                dry_run=args.dry_run,
            )
            row: dict[str, object] = {
                "timestamp": timestamp_now(),
                "frame": frame,
                "opponent": opponent_name,
                "candidate_side": candidate_side,
                "status": "success" if returncode == 0 else "failed",
                "returncode": returncode,
                "wp_landlord": "",
                "wp_farmers": "",
                "adp_landlord": "",
                "adp_farmers": "",
                "draws": "",
                "bid_count_histogram": "",
                "log_path": str(log_path),
                "command": " ".join(command),
            }
            if returncode == 0 and not args.dry_run:
                metrics = parse_eval_metrics(log_text)
                row.update(metrics)
                completed_runs.add((frame, opponent_name, candidate_side))
            append_summary_row(summary_path(output_dir), row)

        if args.once:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
