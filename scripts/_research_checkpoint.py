"""Atomic, provenance-checked checkpoints for repository research programs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


class CheckpointError(RuntimeError):
    """Raised when stored work is incomplete, corrupt, or incompatible."""


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_unit_id(*parts: object) -> str:
    value = "__".join(str(part) for part in parts)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    if not value or value in {".", ".."}:
        raise ValueError("checkpoint unit identifier is empty or unsafe")
    return value


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_bytes(
        path, json.dumps(payload, indent=2, sort_keys=True, default=str).encode()
    )


class ResearchCheckpointStore:
    """A phase-scoped store that never reuses mismatched scientific work."""

    def __init__(self, root: Path, phase: str, provenance: dict[str, Any]) -> None:
        self.root = root
        self.phase = canonical_unit_id(phase)
        self.phase_root = root / "checkpoints" / self.phase
        self.provenance = provenance
        self.provenance_digest = digest_bytes(
            json.dumps(
                provenance, sort_keys=True, separators=(",", ":"), default=str
            ).encode()
        )
        self._write_or_validate_provenance()

    @property
    def provenance_path(self) -> Path:
        return self.phase_root / "provenance.json"

    def _write_or_validate_provenance(self) -> None:
        payload = {"digest": self.provenance_digest, "provenance": self.provenance}
        if self.provenance_path.exists():
            existing = json.loads(self.provenance_path.read_text(encoding="utf-8"))
            if existing.get("digest") != self.provenance_digest:
                raise CheckpointError(
                    f"{self.phase}: result-affecting provenance mismatch"
                )
            return
        atomic_write_json(self.provenance_path, payload)

    def unit_path(self, unit_id: str) -> Path:
        return self.phase_root / "units" / f"{canonical_unit_id(unit_id)}.json"

    def write_unit(
        self, unit_id: str, payload: dict[str, Any], *, validate: bool = True
    ) -> Path:
        identifier = canonical_unit_id(unit_id)
        record = {
            "schema_version": 1,
            "unit_id": identifier,
            "provenance_digest": self.provenance_digest,
            "payload": payload,
        }
        body = json.dumps(
            record, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        record["content_sha256"] = digest_bytes(body)
        atomic_write_json(self.unit_path(identifier), record)
        if validate:
            self.read_unit(
                identifier
            )  # validate before the atomic result counts as complete
        return self.unit_path(identifier)

    def read_unit(self, unit_id: str) -> dict[str, Any]:
        path = self.unit_path(unit_id)
        if not path.exists():
            raise CheckpointError(f"missing checkpoint unit: {unit_id}")
        record = json.loads(path.read_text(encoding="utf-8"))
        digest = record.pop("content_sha256", None)
        actual = digest_bytes(
            json.dumps(
                record, sort_keys=True, separators=(",", ":"), default=str
            ).encode()
        )
        if (
            digest != actual
            or record.get("provenance_digest") != self.provenance_digest
        ):
            raise CheckpointError(f"corrupt or incompatible checkpoint unit: {unit_id}")
        if record.get("unit_id") != canonical_unit_id(unit_id) or not isinstance(
            record.get("payload"), dict
        ):
            raise CheckpointError(f"invalid checkpoint schema: {unit_id}")
        return record["payload"]

    def completed_units(self, expected: Iterable[str]) -> set[str]:
        return {
            unit
            for unit in expected
            if self.unit_path(unit).exists() and self.read_unit(unit) is not None
        }

    def missing_units(self, expected: Iterable[str]) -> list[str]:
        return [unit for unit in expected if unit not in self.completed_units(expected)]

    def complete_phase(self, expected: Iterable[str]) -> Path:
        expected = list(expected)
        missing = self.missing_units(expected)
        if missing:
            raise CheckpointError(
                f"{self.phase}: cannot finalize with {len(missing)} missing units"
            )
        marker = self.phase_root / "COMPLETE.json"
        payload = {
            "phase": self.phase,
            "expected_units": expected,
            "provenance_digest": self.provenance_digest,
        }
        if marker.exists():
            existing = json.loads(marker.read_text(encoding="utf-8"))
            if existing != payload:
                raise CheckpointError(f"{self.phase}: incompatible completion marker")
            return marker  # validate/reuse a completed phase without touching its mtime
        atomic_write_json(marker, payload)
        return marker

    def heartbeat(self, **status: Any) -> None:
        atomic_write_json(
            self.phase_root / "heartbeat.json",
            {
                "phase": self.phase,
                "provenance_digest": self.provenance_digest,
                "timestamp": time.time(),
                **status,
            },
        )

    def write_runtime(self, unit_id: str, payload: dict[str, Any]) -> Path:
        path = self.phase_root / "runtime" / f"{canonical_unit_id(unit_id)}.json"
        atomic_write_json(
            path,
            {"phase": self.phase, "unit_id": canonical_unit_id(unit_id), **payload},
        )
        return path

    def log(self, message: str) -> None:
        path = self.phase_root / "progress.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    def assemble_json_records(self, expected: Iterable[str]) -> list[dict[str, Any]]:
        missing = self.missing_units(expected)
        if missing:
            raise CheckpointError(f"{self.phase}: finalization refused; units missing")
        return [self.read_unit(unit) for unit in expected]

    def assemble_table(self, expected: Iterable[str], destination: Path) -> Path:
        records = self.assemble_json_records(expected)
        frame = pd.DataFrame(records)
        content = frame.to_csv(index=False).encode()
        atomic_write_bytes(destination, content)
        return destination
