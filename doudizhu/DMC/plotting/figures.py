"""Figure builders for Doudizhu experiment outputs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .styles import COLORS, FIG_SIZES, apply_style, ensure_output_dir, require_matplotlib

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _sorted_numeric_series(rows: list[dict[str, object]], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        x_value = row.get(x_key)
        y_value = row.get(y_key)
        if isinstance(x_value, (int, float)) and isinstance(y_value, (int, float)):
            pairs.append((float(x_value), float(y_value)))
    if not pairs:
        raise ValueError(f"No numeric pairs found for x={x_key}, y={y_key}")
    pairs.sort(key=lambda item: item[0])
    xs = np.asarray([item[0] for item in pairs], dtype=float)
    ys = np.asarray([item[1] for item in pairs], dtype=float)
    return xs, ys


def save_pdf(fig, output_path: str | Path) -> Path:
    path = ensure_output_dir(output_path)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def fig_training_curves(
    run_specs: list[dict[str, object]],
    output_path: str | Path,
    *,
    x_key: str = "frames",
    y_key: str = "mean_episode_return_landlord",
    title: str = "Training Curves",
    smoothing: int = 1,
) -> Path:
    require_matplotlib()
    apply_style()

    fig, ax = plt.subplots(figsize=FIG_SIZES["full"])
    for index, spec in enumerate(run_specs):
        rows = spec["rows"]
        label = str(spec["label"])
        color = spec.get("color") or list(COLORS.values())[index % len(COLORS)]
        xs, ys = _sorted_numeric_series(rows, x_key, y_key)
        ys = _smooth(ys, smoothing)
        ax.plot(xs, ys, label=label, color=color)

    ax.set_title(title)
    ax.set_xlabel(x_key.replace("_", " ").title())
    ax.set_ylabel(y_key.replace("_", " ").title())
    ax.legend(ncol=min(3, max(1, len(run_specs))))
    return save_pdf(fig, output_path)


def fig_search_tradeoff(
    rows: list[dict[str, object]],
    output_path: str | Path,
    *,
    x_key: str = "total_actors",
    y_key: str = "mean_success_avg_fps",
    color_key: str = "mean_success_learner_util_pct",
    size_key: str = "mean_success_gpu_power_total_w",
    title: str = "Search Trade-Off",
) -> Path:
    require_matplotlib()
    apply_style()

    filtered = []
    for row in rows:
        if not isinstance(row.get(x_key), (int, float)):
            continue
        if not isinstance(row.get(y_key), (int, float)):
            continue
        filtered.append(row)
    if not filtered:
        raise ValueError("No numeric rows available for search-tradeoff figure.")

    xs = np.asarray([float(row[x_key]) for row in filtered], dtype=float)
    ys = np.asarray([float(row[y_key]) for row in filtered], dtype=float)
    colors = np.asarray([float(row.get(color_key, 0.0) or 0.0) for row in filtered], dtype=float)
    sizes = np.asarray([float(row.get(size_key, 0.0) or 0.0) for row in filtered], dtype=float)
    marker_sizes = 60.0 + 0.45 * sizes

    fig, ax = plt.subplots(figsize=FIG_SIZES["full"])
    scatter = ax.scatter(xs, ys, c=colors, s=marker_sizes, cmap="viridis", alpha=0.88, edgecolors="black", linewidths=0.4)
    for row, x_value, y_value in zip(filtered, xs, ys):
        ax.annotate(
            f"a{row['num_actors']}/t{row['num_threads']}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6.8,
        )

    ax.set_title(title)
    ax.set_xlabel(x_key.replace("_", " ").title())
    ax.set_ylabel(y_key.replace("_", " ").title())
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(color_key.replace("_", " ").title())
    return save_pdf(fig, output_path)


def fig_resource_timeline(
    gpu_rows: list[dict[str, object]],
    proc_rows: list[dict[str, object]],
    output_path: str | Path,
    *,
    learner_gpu_id: int | None = None,
    title: str = "Runtime Resource Timeline",
) -> Path:
    require_matplotlib()
    apply_style()

    if not gpu_rows and not proc_rows:
        raise ValueError("No GPU or process samples supplied.")

    grouped_gpu: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in gpu_rows:
        elapsed = row.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            grouped_gpu[float(elapsed)].append(row)

    gpu_times = sorted(grouped_gpu.keys())
    total_power = np.asarray([
        sum(float(item.get("power_w", 0.0) or 0.0) for item in grouped_gpu[timestamp])
        for timestamp in gpu_times
    ], dtype=float) if gpu_times else np.asarray([], dtype=float)

    total_util = np.asarray([
        sum(float(item.get("util_pct", 0.0) or 0.0) for item in grouped_gpu[timestamp])
        for timestamp in gpu_times
    ], dtype=float) if gpu_times else np.asarray([], dtype=float)

    if learner_gpu_id is None and gpu_rows:
        learner_gpu_id = max(int(row["gpu_id"]) for row in gpu_rows if isinstance(row.get("gpu_id"), (int, float)))

    learner_util = np.asarray([
        next(
            (
                float(item.get("util_pct", 0.0) or 0.0)
                for item in grouped_gpu[timestamp]
                if int(item.get("gpu_id", -1)) == learner_gpu_id
            ),
            0.0,
        )
        for timestamp in gpu_times
    ], dtype=float) if gpu_times else np.asarray([], dtype=float)

    proc_times = np.asarray([
        float(row["elapsed_seconds"])
        for row in proc_rows
        if isinstance(row.get("elapsed_seconds"), (int, float))
    ], dtype=float)
    proc_cpu = np.asarray([
        float(row["cpu_pct"])
        for row in proc_rows
        if isinstance(row.get("cpu_pct"), (int, float))
    ], dtype=float)
    proc_mem = np.asarray([
        float(row["rss_mib"])
        for row in proc_rows
        if isinstance(row.get("rss_mib"), (int, float))
    ], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=FIG_SIZES["tall"], sharex=False)
    if len(gpu_times) > 0:
        axes[0].plot(gpu_times, total_power, color=COLORS["Actors"], label="Total GPU Power")
        axes[0].set_ylabel("Power (W)")
        axes[0].legend(loc="upper left")

        axes[1].plot(gpu_times, total_util, color=COLORS["Base"], label="Total GPU Util")
        axes[1].plot(gpu_times, learner_util, color=COLORS["Learner"], label="Learner Util")
        axes[1].set_ylabel("Util (%)")
        axes[1].legend(loc="upper left")

    if len(proc_times) > 0:
        axes[2].plot(proc_times, proc_cpu, color=COLORS["CPU"], label="CPU %")
        ax_mem = axes[2].twinx()
        ax_mem.plot(proc_times, proc_mem, color=COLORS["RSS"], label="RSS MiB")
        axes[2].set_ylabel("CPU (%)")
        ax_mem.set_ylabel("RSS (MiB)")
        lines = axes[2].get_lines() + ax_mem.get_lines()
        axes[2].legend(lines, [line.get_label() for line in lines], loc="upper left")

    axes[2].set_xlabel("Elapsed Seconds")
    fig.suptitle(title)
    fig.tight_layout()
    return save_pdf(fig, output_path)


def export_latex_table(
    rows: list[dict[str, object]],
    output_path: str | Path,
    *,
    domain: str,
    metric_name: str,
) -> Path:
    filtered = [
        row for row in rows
        if str(row.get("domain", "")).lower() == domain.lower()
        and str(row.get("metric", "")) == metric_name
    ]
    if not filtered:
        raise ValueError(f"No rows found for domain={domain}, metric={metric_name}")

    path = ensure_output_dir(output_path)
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Method & Mean & Std \\\\",
        "\\midrule",
    ]
    for row in filtered:
        mean = row.get("mean", "")
        std = row.get("std", "")
        lines.append(f"{row['method']} & {mean} & {std} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def fig_ablation_bars(
    rows: list[dict[str, object]],
    output_path: str | Path,
    *,
    domain: str,
    metric_name: str,
    title: str = "Ablation Summary",
) -> Path:
    require_matplotlib()
    apply_style()

    filtered = [
        row for row in rows
        if str(row.get("domain", "")).lower() == domain.lower()
        and str(row.get("metric", "")) == metric_name
    ]
    if not filtered:
        raise ValueError(f"No rows found for domain={domain}, metric={metric_name}")

    labels = [str(row["method"]) for row in filtered]
    means = np.asarray([float(row.get("mean", 0.0) or 0.0) for row in filtered], dtype=float)
    stds = np.asarray([float(row.get("std", 0.0) or 0.0) for row in filtered], dtype=float)
    colors = [COLORS.get(label, COLORS["Base"]) for label in labels]

    fig, ax = plt.subplots(figsize=FIG_SIZES["full"])
    positions = np.arange(len(labels))
    ax.bar(positions, means, yerr=stds, color=colors, edgecolor="black", linewidth=0.4, alpha=0.9, capsize=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title)
    return save_pdf(fig, output_path)
