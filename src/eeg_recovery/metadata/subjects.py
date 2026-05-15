from __future__ import annotations

import re
from typing import Any


_SUBJECT_ID_PATTERN = re.compile(r"sub\s*(\d+)", re.IGNORECASE)


def normalize_subject_id(value: Any) -> str:
    """Normalize subject identifiers such as sub013 to canonical sub13."""

    match = _SUBJECT_ID_PATTERN.search(str(value).strip())
    if match is None:
        raise ValueError(f"Could not parse subject ID from value: {value!r}")

    subject_number = int(match.group(1))
    if subject_number <= 0:
        raise ValueError(f"Subject ID must be positive: {value!r}")
    return f"sub{subject_number:02d}"
