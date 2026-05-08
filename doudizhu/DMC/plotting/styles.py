"""Paper-aligned plotting styles for Doudizhu experiments."""

from __future__ import annotations

from pathlib import Path

try:
    import matplotlib as mpl
except Exception as exc:  # pragma: no cover
    mpl = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:
    MATPLOTLIB_IMPORT_ERROR = None


COLORS = {
    "Base": "#4878CF",
    "+Module A": "#6ACC65",
    "+Module B": "#D65F5F",
    "+Module C": "#B47CC7",
    "Full": "#C4AD66",
    "Learner": "#D65F5F",
    "Actors": "#4878CF",
    "CPU": "#6ACC65",
    "RSS": "#B47CC7",
}


FIG_SIZES = {
    "full": (6.75, 2.35),
    "half": (4.50, 2.50),
    "tall": (6.75, 4.20),
}


DESIGN_RULES = [
    "Use one reviewer-facing claim per figure.",
    "Keep typography compact, serif, and paper-friendly.",
    "Prefer vector PDF outputs for paper-ready artifacts.",
    "Use the same method colors across all figures.",
    "For runtime plots, make the actor-vs-learner imbalance immediately visible.",
    "For ablations, preserve the Base -> +A -> +B -> +C -> Full ordering.",
]


def require_matplotlib() -> None:
    if MATPLOTLIB_IMPORT_ERROR is not None:
        raise SystemExit(
            "Plotting requires matplotlib.\n"
            "Install it first, for example:\n"
            "  python3 -m pip install matplotlib\n"
            f"Original import error: {MATPLOTLIB_IMPORT_ERROR}"
        )


def apply_style() -> None:
    require_matplotlib()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.4,
            "legend.framealpha": 0.96,
            "legend.edgecolor": "#d0d0d0",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.5,
            "grid.color": "#d9d9d9",
            "lines.linewidth": 1.8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_output_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
