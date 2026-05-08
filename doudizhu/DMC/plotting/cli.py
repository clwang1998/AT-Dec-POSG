"""CLI for Doudizhu experiment plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .figures import (
    export_latex_table,
    fig_ablation_bars,
    fig_resource_timeline,
    fig_search_tradeoff,
    fig_training_curves,
)
from .loaders import (
    load_metrics_manifest,
    load_resource_series,
    load_search_aggregate,
    load_training_log,
)


def _parse_run_spec(raw: str) -> dict[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("run specs must look like LABEL=PATH")
    label, path = raw.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("run specs must look like LABEL=PATH")
    return {"label": label, "path": path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper-aligned figures from Doudizhu experiment outputs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    training = subparsers.add_parser("training-curves")
    training.add_argument("--run", action="append", required=True, type=_parse_run_spec)
    training.add_argument("--x-key", default="frames")
    training.add_argument("--y-key", default="mean_episode_return_landlord")
    training.add_argument("--title", default="Training Curves")
    training.add_argument("--smoothing", type=int, default=1)
    training.add_argument("--output", default="plotting/output/training_curves.pdf")

    search = subparsers.add_parser("search-tradeoff")
    search.add_argument("--aggregate", required=True)
    search.add_argument("--x-key", default="total_actors")
    search.add_argument("--y-key", default="mean_success_avg_fps")
    search.add_argument("--color-key", default="mean_success_learner_util_pct")
    search.add_argument("--size-key", default="mean_success_gpu_power_total_w")
    search.add_argument("--title", default="Search Trade-Off")
    search.add_argument("--output", default="plotting/output/search_tradeoff.pdf")

    resource = subparsers.add_parser("resource-timeline")
    resource.add_argument("--gpu-csv", required=True)
    resource.add_argument("--proc-csv", required=True)
    resource.add_argument("--learner-gpu-id", type=int, default=None)
    resource.add_argument("--title", default="Runtime Resource Timeline")
    resource.add_argument("--output", default="plotting/output/resource_timeline.pdf")

    ablation = subparsers.add_parser("ablation-bars")
    ablation.add_argument("--metrics", required=True)
    ablation.add_argument("--domain", default="Doudizhu")
    ablation.add_argument("--metric-name", default="normalized_win_rate")
    ablation.add_argument("--title", default="Ablation Summary")
    ablation.add_argument("--output", default="plotting/output/ablation_bars.pdf")
    ablation.add_argument("--latex-output", default="")

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--config", required=True)

    return parser


def _run_training_curves(args: argparse.Namespace) -> None:
    run_specs = []
    for spec in args.run:
        run_specs.append(
            {
                "label": spec["label"],
                "rows": load_training_log(spec["path"]),
            }
        )
    output = fig_training_curves(
        run_specs,
        args.output,
        x_key=args.x_key,
        y_key=args.y_key,
        title=args.title,
        smoothing=args.smoothing,
    )
    print(f"Saved training curves: {output}")


def _run_search_tradeoff(args: argparse.Namespace) -> None:
    output = fig_search_tradeoff(
        load_search_aggregate(args.aggregate),
        args.output,
        x_key=args.x_key,
        y_key=args.y_key,
        color_key=args.color_key,
        size_key=args.size_key,
        title=args.title,
    )
    print(f"Saved search trade-off figure: {output}")


def _run_resource_timeline(args: argparse.Namespace) -> None:
    gpu_rows, proc_rows = load_resource_series(args.gpu_csv, args.proc_csv)
    output = fig_resource_timeline(
        gpu_rows,
        proc_rows,
        args.output,
        learner_gpu_id=args.learner_gpu_id,
        title=args.title,
    )
    print(f"Saved resource timeline: {output}")


def _run_ablation_bars(args: argparse.Namespace) -> None:
    rows = load_metrics_manifest(args.metrics)
    output = fig_ablation_bars(
        rows,
        args.output,
        domain=args.domain,
        metric_name=args.metric_name,
        title=args.title,
    )
    print(f"Saved ablation bar chart: {output}")
    if args.latex_output:
        table = export_latex_table(
            rows,
            args.latex_output,
            domain=args.domain,
            metric_name=args.metric_name,
        )
        print(f"Saved LaTeX table: {table}")


def _run_all(config_path: str | Path) -> None:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    if "training_curves" in config:
        section = config["training_curves"]
        run_specs = [
            {"label": item["label"], "rows": load_training_log(item["path"])}
            for item in section["runs"]
        ]
        fig_training_curves(
            run_specs,
            section["output"],
            x_key=section.get("x_key", "frames"),
            y_key=section.get("y_key", "mean_episode_return_landlord"),
            title=section.get("title", "Training Curves"),
            smoothing=int(section.get("smoothing", 1)),
        )

    if "search_tradeoff" in config:
        section = config["search_tradeoff"]
        fig_search_tradeoff(
            load_search_aggregate(section["aggregate"]),
            section["output"],
            x_key=section.get("x_key", "total_actors"),
            y_key=section.get("y_key", "mean_success_avg_fps"),
            color_key=section.get("color_key", "mean_success_learner_util_pct"),
            size_key=section.get("size_key", "mean_success_gpu_power_total_w"),
            title=section.get("title", "Search Trade-Off"),
        )

    if "resource_timeline" in config:
        section = config["resource_timeline"]
        gpu_rows, proc_rows = load_resource_series(section["gpu_csv"], section["proc_csv"])
        fig_resource_timeline(
            gpu_rows,
            proc_rows,
            section["output"],
            learner_gpu_id=section.get("learner_gpu_id"),
            title=section.get("title", "Runtime Resource Timeline"),
        )

    if "ablation_bars" in config:
        section = config["ablation_bars"]
        rows = load_metrics_manifest(section["metrics"])
        fig_ablation_bars(
            rows,
            section["output"],
            domain=section.get("domain", "Doudizhu"),
            metric_name=section.get("metric_name", "normalized_win_rate"),
            title=section.get("title", "Ablation Summary"),
        )
        if section.get("latex_output"):
            export_latex_table(
                rows,
                section["latex_output"],
                domain=section.get("domain", "Doudizhu"),
                metric_name=section.get("metric_name", "normalized_win_rate"),
            )

    print(f"Finished plotting bundle from config: {config_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "training-curves":
        _run_training_curves(args)
    elif args.command == "search-tradeoff":
        _run_search_tradeoff(args)
    elif args.command == "resource-timeline":
        _run_resource_timeline(args)
    elif args.command == "ablation-bars":
        _run_ablation_bars(args)
    elif args.command == "all":
        _run_all(args.config)
