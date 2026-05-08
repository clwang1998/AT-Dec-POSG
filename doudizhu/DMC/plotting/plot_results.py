#!/usr/bin/env python3
"""Project-facing plotting CLI for Doudizhu experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - import guard
    mpl = None
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:
    MATPLOTLIB_IMPORT_ERROR = None


COLORS = {
    "base": "#4878CF",
    "accent": "#D65F5F",
    "gold": "#C4AD66",
    "green": "#6ACC65",
    "purple": "#B47CC7",
}


def require_matplotlib() -> None:
    if MATPLOTLIB_IMPORT_ERROR is not None:
        raise SystemExit(
            "matplotlib is required for plotting.\n"
            "Install it with:\n"
            "  python3 -m pip install matplotlib\n"
            f"Original import error: {MATPLOTLIB_IMPORT_ERROR}"
        )


def setup_style() -> None:
    require_matplotlib()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def float_or_zero(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def smooth(values: list[float], width: int) -> list[float]:
    if width <= 1 or not values:
        return values
    out: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - width + 1)
        window = values[start : idx + 1]
        out.append(sum(window) / len(window))
    return out


def resolve_logs_csv(path: Path) -> Path:
    if path.is_dir():
        return path / "logs.csv"
    return path


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"Expected LABEL=PATH run spec, got: {spec}")
    label, raw_path = spec.split("=", 1)
    return label, Path(raw_path)


def plot_training_curves(
    runs: list[str],
    output: Path,
    y_key: str,
    x_key: str,
    title: str,
    smoothing: int,
) -> Path:
    ensure_dir(output)
    fig, ax = plt.subplots(figsize=(6.75, 2.4))
    palette = [COLORS["base"], COLORS["accent"], COLORS["green"], COLORS["gold"], COLORS["purple"]]
    for idx, spec in enumerate(runs):
        label, run_path = parse_run_spec(spec)
        rows = read_csv(resolve_logs_csv(run_path))
        xs = [float_or_zero(row.get(x_key)) for row in rows]
        ys = [float_or_zero(row.get(y_key)) for row in rows]
        ys = smooth(ys, smoothing)
        ax.plot(xs, ys, label=label, color=palette[idx % len(palette)])
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.legend()
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_search_tradeoff(
    aggregate: Path,
    output: Path,
    x_key: str,
    y_key: str,
    color_key: str,
    size_key: str,
    title: str,
) -> Path:
    ensure_dir(output)
    rows = read_csv(aggregate)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    xs = [float_or_zero(row.get(x_key)) for row in rows]
    ys = [float_or_zero(row.get(y_key)) for row in rows]
    colors = [float_or_zero(row.get(color_key)) for row in rows]
    sizes = [max(30.0, float_or_zero(row.get(size_key)) / 6.0) for row in rows]
    scatter = ax.scatter(xs, ys, c=colors, s=sizes, cmap="viridis", edgecolors="black", linewidths=0.4)
    for idx, row in enumerate(rows[:12]):
        label = f"a{row.get('num_actors','?')}-t{row.get('num_threads','?')}"
        ax.annotate(label, (xs[idx], ys[idx]), fontsize=6.8)
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(color_key)
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_resource_timeline(
    gpu_csv: Path,
    proc_csv: Path,
    output: Path,
    learner_gpu_id: int,
    title: str,
) -> Path:
    ensure_dir(output)
    gpu_rows = read_csv(gpu_csv)
    proc_rows = read_csv(proc_csv)

    learner_rows = [row for row in gpu_rows if int(float_or_zero(row.get("gpu_id"))) == learner_gpu_id]
    total_rows: dict[float, dict[str, float]] = {}
    for row in gpu_rows:
        elapsed = float_or_zero(row.get("elapsed_seconds"))
        total_rows.setdefault(elapsed, {"power_w": 0.0, "util_pct": 0.0, "count": 0.0})
        total_rows[elapsed]["power_w"] += float_or_zero(row.get("power_w"))
        total_rows[elapsed]["util_pct"] += float_or_zero(row.get("util_pct"))
        total_rows[elapsed]["count"] += 1.0

    xs = sorted(total_rows.keys())
    total_power = [total_rows[x]["power_w"] for x in xs]
    total_util = [
        total_rows[x]["util_pct"] / max(1.0, total_rows[x]["count"])
        for x in xs
    ]
    learner_x = [float_or_zero(row.get("elapsed_seconds")) for row in learner_rows]
    learner_util = [float_or_zero(row.get("util_pct")) for row in learner_rows]
    proc_x = [float_or_zero(row.get("elapsed_seconds")) for row in proc_rows]
    proc_cpu = [float_or_zero(row.get("cpu_pct")) for row in proc_rows]
    proc_rss = [float_or_zero(row.get("rss_mib")) for row in proc_rows]

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.6))
    axes[0].plot(xs, total_power, label="Total GPU power", color=COLORS["gold"])
    axes[0].plot(xs, total_util, label="Mean GPU util", color=COLORS["base"])
    if learner_x:
        axes[0].plot(learner_x, learner_util, label="Learner util", color=COLORS["accent"])
    axes[0].set_title(title)
    axes[0].set_xlabel("elapsed_seconds")
    axes[0].legend()

    axes[1].plot(proc_x, proc_cpu, label="CPU %", color=COLORS["green"])
    axes[1].plot(proc_x, proc_rss, label="RSS MiB", color=COLORS["purple"])
    axes[1].set_title("Process Footprint")
    axes[1].set_xlabel("elapsed_seconds")
    axes[1].legend()
    fig.savefig(output)
    plt.close(fig)
    return output


def write_latex_table(rows: list[dict[str, str]], output: Path, domain: str, metric_name: str) -> Path:
    ensure_dir(output)
    filtered = [
        row for row in rows
        if row.get("domain") == domain and row.get("metric") == metric_name
    ]
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Mean & Std & N \\\\",
        "\\midrule",
    ]
    for row in filtered:
        lines.append(
            f"{row['method']} & {row['mean']} & {row['std']} & {row['n']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def plot_ablation_bars(
    metrics: Path,
    output: Path,
    latex_output: Path | None,
    domain: str,
    metric_name: str,
    title: str,
) -> list[Path]:
    ensure_dir(output)
    rows = read_csv(metrics)
    filtered = [
        row for row in rows
        if row.get("domain") == domain and row.get("metric") == metric_name
    ]
    labels = [row["method"] for row in filtered]
    means = [float_or_zero(row.get("mean")) for row in filtered]
    stds = [float_or_zero(row.get("std")) for row in filtered]

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    xs = list(range(len(filtered)))
    ax.bar(xs, means, yerr=stds, color=COLORS["base"], alpha=0.92, capsize=3)
    ax.set_xticks(xs, labels)
    ax.set_title(title)
    ax.set_ylabel(metric_name)
    fig.savefig(output)
    plt.close(fig)

    outputs = [output]
    if latex_output is not None:
        outputs.append(write_latex_table(rows, latex_output, domain, metric_name))
    return outputs


def run_from_config(config_path: Path) -> list[Path]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    outputs: list[Path] = []
    if "training_curves" in payload:
        cfg = payload["training_curves"]
        outputs.append(
            plot_training_curves(
                runs=[f"{item['label']}={item['path']}" for item in cfg["runs"]],
                output=Path(cfg["output"]),
                x_key=cfg.get("x_key", "frames"),
                y_key=cfg.get("y_key", "mean_episode_return_landlord"),
                title=cfg.get("title", "Training Curves"),
                smoothing=int(cfg.get("smoothing", 1)),
            )
        )
    if "search_tradeoff" in payload:
        cfg = payload["search_tradeoff"]
        outputs.append(
            plot_search_tradeoff(
                aggregate=Path(cfg["aggregate"]),
                output=Path(cfg["output"]),
                x_key=cfg.get("x_key", "total_actors"),
                y_key=cfg.get("y_key", "mean_success_avg_fps"),
                color_key=cfg.get("color_key", "mean_success_learner_util_pct"),
                size_key=cfg.get("size_key", "mean_success_gpu_power_total_w"),
                title=cfg.get("title", "Search Trade-Off"),
            )
        )
    if "resource_timeline" in payload:
        cfg = payload["resource_timeline"]
        outputs.append(
            plot_resource_timeline(
                gpu_csv=Path(cfg["gpu_csv"]),
                proc_csv=Path(cfg["proc_csv"]),
                output=Path(cfg["output"]),
                learner_gpu_id=int(cfg.get("learner_gpu_id", 0)),
                title=cfg.get("title", "Runtime Timeline"),
            )
        )
    if "ablation_bars" in payload:
        cfg = payload["ablation_bars"]
        outputs.extend(
            plot_ablation_bars(
                metrics=Path(cfg["metrics"]),
                output=Path(cfg["output"]),
                latex_output=Path(cfg["latex_output"]) if cfg.get("latex_output") else None,
                domain=cfg.get("domain", "Doudizhu"),
                metric_name=cfg.get("metric_name", "normalized_win_rate"),
                title=cfg.get("title", "Ablation Summary"),
            )
        )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Doudizhu experiment results.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    curve_parser = subparsers.add_parser("training-curves")
    curve_parser.add_argument("--run", action="append", required=True, help="LABEL=PATH to a run directory or logs.csv")
    curve_parser.add_argument("--x-key", default="frames")
    curve_parser.add_argument("--y-key", default="mean_episode_return_landlord")
    curve_parser.add_argument("--title", default="Training Curves")
    curve_parser.add_argument("--smoothing", type=int, default=1)
    curve_parser.add_argument("--output", type=Path, default=Path("plotting/output/training_curves.pdf"))

    search_parser = subparsers.add_parser("search-tradeoff")
    search_parser.add_argument("--aggregate", required=True, type=Path)
    search_parser.add_argument("--x-key", default="total_actors")
    search_parser.add_argument("--y-key", default="mean_success_avg_fps")
    search_parser.add_argument("--color-key", default="mean_success_learner_util_pct")
    search_parser.add_argument("--size-key", default="mean_success_gpu_power_total_w")
    search_parser.add_argument("--title", default="Search Trade-Off")
    search_parser.add_argument("--output", type=Path, default=Path("plotting/output/search_tradeoff.pdf"))

    timeline_parser = subparsers.add_parser("resource-timeline")
    timeline_parser.add_argument("--gpu-csv", required=True, type=Path)
    timeline_parser.add_argument("--proc-csv", required=True, type=Path)
    timeline_parser.add_argument("--learner-gpu-id", type=int, default=0)
    timeline_parser.add_argument("--title", default="Runtime Timeline")
    timeline_parser.add_argument("--output", type=Path, default=Path("plotting/output/resource_timeline.pdf"))

    bar_parser = subparsers.add_parser("ablation-bars")
    bar_parser.add_argument("--metrics", required=True, type=Path)
    bar_parser.add_argument("--domain", default="Doudizhu")
    bar_parser.add_argument("--metric-name", default="normalized_win_rate")
    bar_parser.add_argument("--title", default="Ablation Summary")
    bar_parser.add_argument("--output", type=Path, default=Path("plotting/output/ablation_bars.pdf"))
    bar_parser.add_argument("--latex-output", type=Path, default=None)

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--config", required=True, type=Path)

    return parser


def main() -> None:
    setup_style()
    args = build_parser().parse_args()
    outputs: list[Path]
    if args.command == "training-curves":
        outputs = [
            plot_training_curves(
                runs=args.run,
                output=args.output,
                y_key=args.y_key,
                x_key=args.x_key,
                title=args.title,
                smoothing=args.smoothing,
            )
        ]
    elif args.command == "search-tradeoff":
        outputs = [
            plot_search_tradeoff(
                aggregate=args.aggregate,
                output=args.output,
                x_key=args.x_key,
                y_key=args.y_key,
                color_key=args.color_key,
                size_key=args.size_key,
                title=args.title,
            )
        ]
    elif args.command == "resource-timeline":
        outputs = [
            plot_resource_timeline(
                gpu_csv=args.gpu_csv,
                proc_csv=args.proc_csv,
                output=args.output,
                learner_gpu_id=args.learner_gpu_id,
                title=args.title,
            )
        ]
    elif args.command == "ablation-bars":
        outputs = plot_ablation_bars(
            metrics=args.metrics,
            output=args.output,
            latex_output=args.latex_output,
            domain=args.domain,
            metric_name=args.metric_name,
            title=args.title,
        )
    else:
        outputs = run_from_config(args.config)

    print("Generated outputs:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()

