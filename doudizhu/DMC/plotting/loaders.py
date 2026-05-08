"""CSV loaders for training, search, and runtime plotting."""

from __future__ import annotations

import csv
from pathlib import Path


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _coerce_value(raw: str) -> object:
    value = raw.strip()
    if value == "":
        return value
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_standard_csv(path: str | Path) -> list[dict[str, object]]:
    csv_path = _resolve_path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return [
            {key: _coerce_value(value or "") for key, value in row.items()}
            for row in reader
        ]


def _resolve_training_files(path: str | Path) -> tuple[Path, Path | None]:
    target = _resolve_path(path)
    if target.is_dir():
        log_path = target / "logs.csv"
        fields_path = target / "fields.csv"
    else:
        log_path = target
        fields_path = target.with_name("fields.csv")
    return log_path, fields_path if fields_path.exists() else None


def _read_training_fieldnames(log_path: Path, fields_path: Path | None) -> list[str]:
    if fields_path is not None:
        with fields_path.open("r", encoding="utf-8", newline="") as csv_file:
            rows = list(csv.reader(csv_file))
        if rows:
            return rows[0]

    with log_path.open("r", encoding="utf-8") as log_file:
        first_line = log_file.readline().strip()
    if first_line.startswith("# "):
        return [field.strip() for field in first_line[2:].split(",")]
    raise ValueError(f"Could not infer field names for training log: {log_path}")


def load_training_log(path: str | Path) -> list[dict[str, object]]:
    log_path, fields_path = _resolve_training_files(path)
    fieldnames = _read_training_fieldnames(log_path, fields_path)
    rows: list[dict[str, object]] = []
    with log_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                continue
            padded = list(row) + [""] * max(0, len(fieldnames) - len(row))
            rows.append(
                {
                    fieldnames[index]: _coerce_value(padded[index])
                    for index in range(len(fieldnames))
                }
            )
    return rows


def load_search_summary(path: str | Path) -> list[dict[str, object]]:
    return _load_standard_csv(path)


def load_search_aggregate(path: str | Path) -> list[dict[str, object]]:
    return _load_standard_csv(path)


def load_suite_summary(path: str | Path) -> list[dict[str, object]]:
    return _load_standard_csv(path)


def load_resource_series(gpu_csv: str | Path, proc_csv: str | Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return _load_standard_csv(gpu_csv), _load_standard_csv(proc_csv)


def load_metrics_manifest(path: str | Path) -> list[dict[str, object]]:
    return _load_standard_csv(path)
