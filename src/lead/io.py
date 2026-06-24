"""JSON/provenance helpers."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def current_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json_with_provenance(
    path: Path,
    payload: dict[str, Any],
    *,
    root: Path,
    provenance: dict[str, Any] | None = None,
) -> None:
    payload = dict(payload)
    payload.setdefault("provenance", {})
    payload["provenance"].update(
        {
            "commit": current_commit(root),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            **(provenance or {}),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
