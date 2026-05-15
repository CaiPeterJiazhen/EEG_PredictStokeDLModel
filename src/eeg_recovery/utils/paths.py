from __future__ import annotations

from pathlib import Path


def resolve_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    """Return a normalized absolute Path without creating filesystem entries."""

    resolved = Path(path).expanduser()
    if not resolved.is_absolute() and base_dir is not None:
        resolved = Path(base_dir).expanduser() / resolved
    return resolved.resolve()
