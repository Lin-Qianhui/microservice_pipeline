"""Heading classification policy for notebook task analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_IGNORED_PATTERNS = (
    r"\bimports?\b",
    r"\binstall(?:ation)?\b",
    r"\bsetup environment\b",
    r"\benvironment setup\b",
)
DEFAULT_SUPPORT_PATTERNS = (
    r"\bload(?:ing)?\s+(?:the\s+)?(?:default\s+)?(?:configuration|config|data)\b",
    r"\bdata files?\b",
    r"\boutput and results?\b",
    r"\bprocess rate constants?\b",
    r"\brate constants?\b",
    r"\bgeneral results?\b",
    r"\bresults by compartment\b",
    r"\bheatmaps?\b",
    r"\bplots?\b",
    r"\bvisual(?:ize|ization)?\b",
    r"\bsav(?:e|ing)\b",
    r"\bexport\b",
    r"\bdisplay\b",
)


@dataclass(frozen=True)
class HeadingClassification:
    ignored_patterns: tuple[str, ...] = DEFAULT_IGNORED_PATTERNS
    support_patterns: tuple[str, ...] = DEFAULT_SUPPORT_PATTERNS


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def heading_classification_from_config(config: Mapping[str, Any]) -> HeadingClassification:
    heading_config = config.get("heading_classification")
    if not isinstance(heading_config, Mapping):
        heading_config = {}
    ignored = heading_config.get("ignored_patterns", DEFAULT_IGNORED_PATTERNS)
    support = heading_config.get("support_patterns", DEFAULT_SUPPORT_PATTERNS)
    return HeadingClassification(
        ignored_patterns=tuple(_text(value) for value in ignored if _text(value)),
        support_patterns=tuple(_text(value) for value in support if _text(value)),
    )


def classify_heading(text: str, rules: HeadingClassification) -> str:
    normalized = text.lower()
    for pattern in rules.ignored_patterns:
        if re.search(pattern, normalized):
            return "ignored"
    for pattern in rules.support_patterns:
        if re.search(pattern, normalized):
            return "support"
    return "domain"
