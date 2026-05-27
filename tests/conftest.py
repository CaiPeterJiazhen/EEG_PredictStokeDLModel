from __future__ import annotations

from pathlib import Path
import re
import tempfile
import uuid

import pytest


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name).strip("_")
    path = Path(tempfile.gettempdir()).resolve() / "codex_pytest_tmp" / f"{safe_name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
