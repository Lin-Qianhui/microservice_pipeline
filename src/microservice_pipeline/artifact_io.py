"""Small file-writing helpers for generated script artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_csv_value(value: Any) -> Any:
    return value


def write_csv_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    extrasaction: Literal["raise", "ignore"] = "ignore",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction=extrasaction)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_csv_value(row.get(field, "")) for field in fieldnames})


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")


def write_markdown(path: Path, lines: Sequence[str], *, trailing_newline: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    if trailing_newline:
        text += "\n"
    path.write_text(text, encoding="utf-8")
