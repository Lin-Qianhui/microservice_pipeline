"""JSON-with-comments config loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def strip_jsonc_comments(text: str) -> str:
    """Remove // and /* */ comments while preserving string literals."""

    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue

        if char == "/" and next_char == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue

        if char == "/" and next_char == "*":
            index += 2
            while index + 1 < len(text) and not (text[index] == "*" and text[index + 1] == "/"):
                if text[index] in "\r\n":
                    output.append(text[index])
                index += 1
            index = min(index + 2, len(text))
            continue

        output.append(char)
        index += 1

    return "".join(output)


def strip_jsonc_trailing_commas(text: str) -> str:
    """Remove trailing commas before object/array closers outside strings."""

    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]

        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue

        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
                continue

        output.append(char)
        index += 1

    return "".join(output)


def loads_jsonc(text: str, *, path: Path | str = "<string>") -> Any:
    prepared = strip_jsonc_trailing_commas(strip_jsonc_comments(text))
    try:
        return json.loads(prepared)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSONC config {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def load_jsonc(path: Path | str) -> Any:
    config_path = Path(path)
    return loads_jsonc(config_path.read_text(encoding="utf-8"), path=config_path)
