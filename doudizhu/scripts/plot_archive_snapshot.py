from __future__ import annotations

import argparse
import math
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont


SEAT_ORDER = ("landlord", "landlord_up", "landlord_down")
SEAT_LABELS = {
    "landlord": "Landlord",
    "landlord_up": "Farmer Up",
    "landlord_down": "Farmer Down",
}
SEAT_COLORS = {
    "landlord": "#D55C4B",
    "landlord_up": "#4F7CAC",
    "landlord_down": "#6A9F58",
}


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
        )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _build_checkpoint_zip(archive_dir: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    handle.close()
    zip_path = Path(handle.name)
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_STORED, strict_timestamps=False
    ) as zf:
        for path in archive_dir.rglob("*"):
            if path.is_file() and path.name != ".DS_Store":
                zf.write(path, arcname=str(Path("archive") / path.relative_to(archive_dir)))
    return zip_path


def _load_archive_snapshot(archive_dir: Path) -> Tuple[int, Dict[str, int], Dict[str, float]]:
    zip_path = _build_checkpoint_zip(archive_dir)
    try:
        checkpoint = torch.load(zip_path, map_location="cpu", weights_only=False)
    finally:
        zip_path.unlink(missing_ok=True)
    return checkpoint["frames"], checkpoint["position_frames"], checkpoint["stats"]


def _format_frames(frames: int) -> str:
    if frames >= 1_000_000_000:
        return f"{frames / 1_000_000_000:.2f}B"
    if frames >= 1_000_000:
        return f"{frames / 1_000_000:.2f}M"
    if frames >= 1_000:
        return f"{frames / 1_000:.2f}K"
    return str(frames)


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _draw_bar_panel(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    title: str,
    values: Dict[str, float],
    formatter,
    baseline_zero: bool = False,
) -> None:
    x0, y0, x1, y1 = box
    panel_pad = 18
    title_font = _font(22, bold=True)
    label_font = _font(17)
    value_font = _font(15, bold=True)

    draw.rounded_rectangle(box, radius=18, fill="#F8FAFC", outline="#D9E2EC", width=2)
    draw.text((x0 + panel_pad, y0 + panel_pad), title, fill="#102A43", font=title_font)

    title_w, title_h = _measure_text(draw, title, title_font)
    chart_left = x0 + 70
    chart_right = x1 - 24
    chart_top = y0 + panel_pad + title_h + 28
    chart_bottom = y1 - 82
    chart_height = chart_bottom - chart_top

    numeric_values = [float(values[seat]) for seat in SEAT_ORDER]
    min_value = min(numeric_values)
    max_value = max(numeric_values)
    if baseline_zero:
        min_value = min(min_value, 0.0)
        max_value = max(max_value, 0.0)
    if math.isclose(min_value, max_value):
        pad = 1.0 if math.isclose(max_value, 0.0) else abs(max_value) * 0.2
        min_value -= pad
        max_value += pad

    def value_to_y(value: float) -> float:
        ratio = (value - min_value) / (max_value - min_value)
        return chart_bottom - ratio * chart_height

    zero_y = value_to_y(0.0 if baseline_zero else min_value)
    if baseline_zero:
        draw.line((chart_left, zero_y, chart_right, zero_y), fill="#7B8794", width=2)
    else:
        draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill="#7B8794", width=2)

    bar_gap = 26
    slot_width = (chart_right - chart_left) / len(SEAT_ORDER)
    bar_width = max(34, int(slot_width - bar_gap))

    for idx, seat in enumerate(SEAT_ORDER):
        value = float(values[seat])
        cx = chart_left + slot_width * idx + slot_width / 2
        bar_left = int(cx - bar_width / 2)
        bar_right = int(cx + bar_width / 2)
        top_y = value_to_y(value)
        bar_top = int(min(top_y, zero_y))
        bar_bottom = int(max(top_y, zero_y))
        if bar_bottom == bar_top:
            bar_bottom += 1
        draw.rounded_rectangle(
            (bar_left, bar_top, bar_right, bar_bottom),
            radius=10,
            fill=SEAT_COLORS[seat],
            outline=None,
        )

        seat_label = SEAT_LABELS[seat]
        label_w, label_h = _measure_text(draw, seat_label, label_font)
        label_y = y1 - 42
        draw.text((cx - label_w / 2, label_y), seat_label, fill="#334E68", font=label_font)

        value_text = formatter(value)
        value_w, value_h = _measure_text(draw, value_text, value_font)
        text_y = bar_top - value_h - 8 if value >= (0.0 if baseline_zero else min_value) else bar_bottom + 6
        draw.text((cx - value_w / 2, text_y), value_text, fill="#102A43", font=value_font)


def render_archive_snapshot(archive_dir: Path, output_path: Path) -> Path:
    total_frames, position_frames, stats = _load_archive_snapshot(archive_dir)

    image = Image.new("RGB", (1600, 920), "#EEF2F6")
    draw = ImageDraw.Draw(image)
    title_font = _font(34, bold=True)
    subtitle_font = _font(20)
    note_font = _font(17)

    draw.rounded_rectangle((32, 28, 1568, 892), radius=28, fill="white", outline="#D9E2EC", width=2)
    draw.text((64, 56), "Archive Checkpoint Snapshot", fill="#102A43", font=title_font)
    draw.text(
        (64, 106),
        f"Source: {archive_dir} | Total frames: {_format_frames(total_frames)}",
        fill="#486581",
        font=subtitle_font,
    )
    draw.text(
        (64, 140),
        "This archive stores a final training snapshot, so the figure shows terminal metrics by seat rather than a full training curve.",
        fill="#7B8794",
        font=note_font,
    )

    returns = {
        "landlord": float(stats["mean_episode_return_landlord"]),
        "landlord_up": float(stats["mean_episode_return_landlord_up"]),
        "landlord_down": float(stats["mean_episode_return_landlord_down"]),
    }
    losses = {
        "landlord": float(stats["loss_landlord"]),
        "landlord_up": float(stats["loss_landlord_up"]),
        "landlord_down": float(stats["loss_landlord_down"]),
    }
    frames = {seat: float(position_frames[seat]) / 1_000_000_000 for seat in SEAT_ORDER}

    panels = [
        ((64, 210, 510, 844), "Mean Episode Return", returns, lambda v: f"{v:+.3f}", True),
        ((576, 210, 1022, 844), "Final Loss", losses, lambda v: f"{v:.3f}", False),
        ((1088, 210, 1534, 844), "Position Frames (Billions)", frames, lambda v: f"{v:.2f}B", False),
    ]
    for box, title, values, formatter, baseline_zero in panels:
        _draw_bar_panel(draw, box, title, values, formatter, baseline_zero=baseline_zero)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a summary figure for a checkpoint archive directory.")
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("doudizhu/baselines/archive"),
        help="Path to the exploded PyTorch archive directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("doudizhu/plotting/output/archive_checkpoint_snapshot.png"),
        help="Where to save the rendered PNG figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_archive_snapshot(args.archive_dir.resolve(), args.output.resolve())
    print(output)


if __name__ == "__main__":
    main()
