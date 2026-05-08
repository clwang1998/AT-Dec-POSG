from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import subprocess
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .benchmark import format_markdown_table, run_solver_research_eval
from .config import OptionExecutionConfig, OptionTrainingConfig
from .hardware import detect_option_hardware, format_hardware_report, resolve_runtime_device

DEFAULT_ALLOWED_FILES = (
    "option/config.py",
    "option/models.py",
    "option/solver.py",
)

PRIMARY_METRIC_GOALS = {
    "action_public_mi_lb": "maximize",
    "completion_rate": "maximize",
    "implementation_shortfall": "minimize",
    "intent_public_mi_lb": "maximize",
    "inventory_penalty": "minimize",
    "public_trace_pnl_drop": "minimize",
    "public_trace_reward_drop": "minimize",
    "public_trace_shortfall_increase": "minimize",
    "receiver_accuracy": "maximize",
    "reward": "maximize",
    "risk_adjusted_pnl": "maximize",
    "sender_accuracy": "maximize",
    "slippage": "minimize",
}

RESULTS_TSV_COLUMNS = [
    "timestamp",
    "run_id",
    "branch",
    "head_commit",
    "solver",
    "method",
    "episodes",
    "eval_episodes",
    "num_contracts",
    "base_seed",
    "research_seeds",
    "include_public_intervention",
    "selected_device",
    "visible_gpu_count",
    "single_run_recommendation",
    "throughput_recommendation",
    "primary_metric",
    "primary_goal",
    "primary_value",
    "reward",
    "risk_adjusted_pnl",
    "implementation_shortfall",
    "completion_rate",
    "public_trace_reward_drop",
    "status",
    "incumbent_value",
    "reason",
    "description",
    "allowed_files",
    "changed_files",
    "summary_json",
]


def metric_goal(metric: str) -> str:
    try:
        return PRIMARY_METRIC_GOALS[metric]
    except KeyError as exc:
        supported = ", ".join(sorted(PRIMARY_METRIC_GOALS))
        raise ValueError(f"unsupported primary metric '{metric}'. Expected one of: {supported}") from exc


def is_metric_better(metric: str, candidate: float, incumbent: float) -> bool:
    goal = metric_goal(metric)
    if math.isnan(incumbent):
        return True
    if math.isnan(candidate):
        return False
    if goal == "maximize":
        return candidate > incumbent
    return candidate < incumbent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed-budget autoresearch-style loop for the option benchmark."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run the fixed-budget option evaluation and print the aggregate summary.",
    )
    add_common_args(eval_parser)
    eval_parser.add_argument(
        "--output-format",
        choices=("markdown", "json", "both"),
        default="both",
        help="How to print the aggregate result.",
    )

    hardware_parser = subparsers.add_parser(
        "hardware",
        help="Detect the current hardware profile and print the option workload recommendation.",
    )
    hardware_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Requested runtime device. Use 'auto', 'cpu', 'cuda', or 'cuda:N'.",
    )
    hardware_parser.add_argument(
        "--output-format",
        choices=("markdown", "json", "both"),
        default="both",
        help="How to print the detected hardware profile.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Evaluate the current code, validate the editable scope, and append results.tsv.",
    )
    add_common_args(run_parser)
    run_parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root used for git metadata and result paths.",
    )
    run_parser.add_argument(
        "--results-tsv",
        type=Path,
        default=Path("option/research_runs/results.tsv"),
        help="TSV ledger that stores keep/discard decisions.",
    )
    run_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("option/research_runs"),
        help="Directory that stores per-run JSON artifacts.",
    )
    run_parser.add_argument(
        "--primary-metric",
        choices=tuple(sorted(PRIMARY_METRIC_GOALS)),
        default="reward",
        help="Metric used for automatic keep/discard decisions.",
    )
    run_parser.add_argument(
        "--allowed-files",
        nargs="*",
        default=list(DEFAULT_ALLOWED_FILES),
        help="Relative files that experiments are allowed to edit under option/.",
    )
    run_parser.add_argument(
        "--allow-any-files",
        action="store_true",
        help="Disable the allowed-files guard and record whatever changed under option/.",
    )
    run_parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Short experiment description recorded in the TSV ledger.",
    )
    run_parser.add_argument(
        "--output-format",
        choices=("markdown", "json", "both"),
        default="both",
        help="How to print the aggregate result after logging.",
    )
    return parser


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--solver",
        choices=("full", "ppo"),
        default="full",
        help="Solver family to evaluate under the fixed research budget.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Fixed training budget in episodes for each research trial.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=30,
        help="Evaluation episodes per seed.",
    )
    parser.add_argument(
        "--num-contracts",
        type=int,
        default=3,
        help="Number of option agents/contracts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base seed used to generate the fixed seed schedule.",
    )
    parser.add_argument(
        "--research-seeds",
        type=int,
        default=3,
        help="How many train/eval seeds to aggregate for each candidate.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level used for bootstrap intervals.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap samples used for confidence intervals.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for solver runs. Use 'auto', 'cpu', 'cuda', or 'cuda:N'.",
    )
    parser.add_argument(
        "--no-public-intervention",
        action="store_false",
        dest="include_public_intervention",
        help="Skip the teammate-public-trace corruption evaluation.",
    )
    parser.set_defaults(include_public_intervention=True)


def run_fixed_budget_eval(args: argparse.Namespace) -> Dict[str, Any]:
    hardware_profile = detect_option_hardware(args.device)
    selected_device = resolve_runtime_device(args.device)
    env_config = OptionExecutionConfig(num_contracts=args.num_contracts, seed=args.seed)
    training_config = OptionTrainingConfig(
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        log_interval=max(args.episodes + 1, 1000),
    )
    result = run_solver_research_eval(
        args.solver,
        env_config=env_config,
        training_config=training_config,
        device=selected_device,
        seed=args.seed,
        num_seeds=args.research_seeds,
        confidence_level=args.confidence_level,
        bootstrap_samples=args.bootstrap_samples,
        include_public_intervention=args.include_public_intervention,
    )
    result["hardware"] = hardware_profile
    return result


def emit_summary(result: Dict[str, Any], output_format: str) -> None:
    summary = result["summary"]
    hardware_profile = result.get("hardware")
    markdown = format_markdown_table([summary])
    if output_format in {"markdown", "both"}:
        if hardware_profile is not None:
            print(format_hardware_report(hardware_profile))
            print(hardware_profile["analysis"])
        print(markdown)
    if output_format in {"json", "both"}:
        payload = {
            "solver": result["solver"],
            "method": result["method"],
            "seed_schedule": result["seed_schedule"],
            "hardware": hardware_profile,
            "summary": summary,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))


def emit_hardware_summary(profile: Dict[str, Any], output_format: str) -> None:
    if output_format in {"markdown", "both"}:
        print(format_hardware_report(profile))
        print(profile["analysis"])
    if output_format in {"json", "both"}:
        print(json.dumps(profile, indent=2, sort_keys=True))


def git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.rstrip("\n")


def git_branch(repo_root: Path) -> str:
    return git_output(repo_root, "rev-parse", "--abbrev-ref", "HEAD")


def git_head_commit(repo_root: Path) -> str:
    return git_output(repo_root, "rev-parse", "--short", "HEAD")


def changed_option_files(repo_root: Path) -> List[str]:
    status = git_output(repo_root, "status", "--porcelain", "--untracked-files=all", "--", "option")
    changed: List[str] = []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        path_text = line[3:]
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        changed.append(path_text)
    return sorted(dict.fromkeys(changed))


def validate_allowed_files(changed_files: Sequence[str], allowed_files: Sequence[str]) -> List[str]:
    allowed = set(allowed_files)
    return [path for path in changed_files if path not in allowed]


def ensure_results_tsv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_TSV_COLUMNS, delimiter="\t")
        writer.writeheader()


def append_results_tsv(path: Path, row: Dict[str, object]) -> None:
    ensure_results_tsv(path)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_TSV_COLUMNS, delimiter="\t")
        writer.writerow(row)


def load_results_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def find_incumbent_value(
    rows: Iterable[Dict[str, str]],
    *,
    solver: str,
    primary_metric: str,
    episodes: int,
    eval_episodes: int,
    num_contracts: int,
    research_seeds: int,
    include_public_intervention: bool,
) -> float | None:
    incumbent: float | None = None
    for row in rows:
        if row.get("status") not in {"baseline", "keep"}:
            continue
        if row.get("solver") != solver:
            continue
        if row.get("primary_metric") != primary_metric:
            continue
        if int(row.get("episodes", 0)) != episodes:
            continue
        if int(row.get("eval_episodes", 0)) != eval_episodes:
            continue
        if int(row.get("num_contracts", 0)) != num_contracts:
            continue
        if int(row.get("research_seeds", 0)) != research_seeds:
            continue
        recorded_public_intervention = row.get("include_public_intervention", "True")
        if recorded_public_intervention.lower() != str(include_public_intervention).lower():
            continue
        try:
            value = float(row["primary_value"])
        except (KeyError, TypeError, ValueError):
            continue
        if incumbent is None or is_metric_better(primary_metric, value, incumbent):
            incumbent = value
    return incumbent


def decide_trial_status(
    *,
    primary_metric: str,
    primary_value: float,
    incumbent_value: float | None,
) -> tuple[str, str]:
    if incumbent_value is None:
        return "baseline", "no incumbent for this fixed-budget configuration"
    if is_metric_better(primary_metric, primary_value, incumbent_value):
        goal = metric_goal(primary_metric)
        relation = "above" if goal == "maximize" else "below"
        return "keep", f"{primary_metric} improved {relation} incumbent"
    return "discard", f"{primary_metric} did not beat incumbent"


def timestamp_now() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat()


def run_id_now() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def write_artifact(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_tsv_row(
    *,
    args: argparse.Namespace,
    run_id: str,
    branch: str,
    head_commit: str,
    summary_path: Path,
    changed_files: Sequence[str],
    allowed_files: Sequence[str],
    result: Dict[str, Any] | None,
    hardware_profile: Dict[str, Any] | None,
    status: str,
    incumbent_value: float | None,
    reason: str,
) -> Dict[str, object]:
    summary = {} if result is None else result["summary"]
    primary_metric = args.primary_metric
    primary_value = float("nan") if result is None else float(summary[primary_metric])
    return {
        "timestamp": timestamp_now(),
        "run_id": run_id,
        "branch": branch,
        "head_commit": head_commit,
        "solver": args.solver,
        "method": summary.get("method", ""),
        "episodes": args.episodes,
        "eval_episodes": args.eval_episodes,
        "num_contracts": args.num_contracts,
        "base_seed": args.seed,
        "research_seeds": args.research_seeds,
        "include_public_intervention": args.include_public_intervention,
        "selected_device": "" if hardware_profile is None else hardware_profile.get("selected_device", ""),
        "visible_gpu_count": "" if hardware_profile is None else hardware_profile.get("visible_gpu_count", ""),
        "single_run_recommendation": (
            "" if hardware_profile is None else hardware_profile.get("single_run_recommendation", "")
        ),
        "throughput_recommendation": (
            "" if hardware_profile is None else hardware_profile.get("throughput_recommendation", "")
        ),
        "primary_metric": primary_metric,
        "primary_goal": metric_goal(primary_metric),
        "primary_value": primary_value,
        "reward": summary.get("reward", ""),
        "risk_adjusted_pnl": summary.get("risk_adjusted_pnl", ""),
        "implementation_shortfall": summary.get("implementation_shortfall", ""),
        "completion_rate": summary.get("completion_rate", ""),
        "public_trace_reward_drop": summary.get("public_trace_reward_drop", ""),
        "status": status,
        "incumbent_value": "" if incumbent_value is None else incumbent_value,
        "reason": reason,
        "description": args.description,
        "allowed_files": ",".join(allowed_files),
        "changed_files": ",".join(changed_files),
        "summary_json": str(summary_path),
    }


def run_logged_trial(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    results_tsv = (repo_root / args.results_tsv).resolve()
    artifacts_dir = (repo_root / args.artifacts_dir).resolve()
    allowed_files = [str(Path(path).as_posix()) for path in args.allowed_files]
    changed_files = changed_option_files(repo_root)

    if not args.allow_any_files:
        disallowed = validate_allowed_files(changed_files, allowed_files)
        if disallowed:
            raise RuntimeError(
                "disallowed edits under option/: "
                + ", ".join(disallowed)
                + ". Allowed files: "
                + ", ".join(allowed_files)
            )

    branch = git_branch(repo_root)
    head_commit = git_head_commit(repo_root)
    run_id = run_id_now()
    summary_path = artifacts_dir / run_id / "summary.json"
    ensure_results_tsv(results_tsv)
    incumbent_value = find_incumbent_value(
        load_results_rows(results_tsv),
        solver=args.solver,
        primary_metric=args.primary_metric,
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        num_contracts=args.num_contracts,
        research_seeds=args.research_seeds,
        include_public_intervention=args.include_public_intervention,
    )

    try:
        result = run_fixed_budget_eval(args)
        hardware_profile = result["hardware"]
        primary_value = float(result["summary"][args.primary_metric])
        status, reason = decide_trial_status(
            primary_metric=args.primary_metric,
            primary_value=primary_value,
            incumbent_value=incumbent_value,
        )
        artifact = {
            "timestamp": timestamp_now(),
            "run_id": run_id,
            "git": {
                "branch": branch,
                "head_commit": head_commit,
            },
            "budget": {
                "solver": args.solver,
                "episodes": args.episodes,
                "eval_episodes": args.eval_episodes,
                "num_contracts": args.num_contracts,
                "base_seed": args.seed,
                "research_seeds": args.research_seeds,
                "include_public_intervention": args.include_public_intervention,
                "requested_device": args.device,
            },
            "hardware": hardware_profile,
            "scope": {
                "allowed_files": allowed_files,
                "changed_files": changed_files,
            },
            "decision": {
                "status": status,
                "reason": reason,
                "primary_metric": args.primary_metric,
                "primary_goal": metric_goal(args.primary_metric),
                "primary_value": primary_value,
                "incumbent_value": incumbent_value,
            },
            "result": result,
            "markdown_table": format_markdown_table([result["summary"]]),
        }
        write_artifact(summary_path, artifact)
        row = build_tsv_row(
            args=args,
            run_id=run_id,
            branch=branch,
            head_commit=head_commit,
            summary_path=summary_path,
            changed_files=changed_files,
            allowed_files=allowed_files,
            result=result,
            hardware_profile=hardware_profile,
            status=status,
            incumbent_value=incumbent_value,
            reason=reason,
        )
        append_results_tsv(results_tsv, row)
        emit_summary(result, args.output_format)
        print(
            f"decision={status} primary_metric={args.primary_metric} "
            f"primary_value={primary_value:.6f} "
            f"summary_json={summary_path}"
        )
        return 0
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        try:
            failure_hardware = detect_option_hardware(args.device)
        except Exception:
            failure_hardware = None
        artifact = {
            "timestamp": timestamp_now(),
            "run_id": run_id,
            "git": {
                "branch": branch,
                "head_commit": head_commit,
            },
            "hardware": failure_hardware,
            "scope": {
                "allowed_files": allowed_files,
                "changed_files": changed_files,
            },
            "status": "crash",
            "reason": reason,
            "traceback": traceback.format_exc(),
        }
        write_artifact(summary_path, artifact)
        row = build_tsv_row(
            args=args,
            run_id=run_id,
            branch=branch,
            head_commit=head_commit,
            summary_path=summary_path,
            changed_files=changed_files,
            allowed_files=allowed_files,
            result=None,
            hardware_profile=failure_hardware,
            status="crash",
            incumbent_value=incumbent_value,
            reason=reason,
        )
        append_results_tsv(results_tsv, row)
        raise


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "hardware":
        emit_hardware_summary(detect_option_hardware(args.device), args.output_format)
        return 0
    if args.command == "eval":
        result = run_fixed_budget_eval(args)
        emit_summary(result, args.output_format)
        return 0
    return run_logged_trial(args)


if __name__ == "__main__":
    raise SystemExit(main())
