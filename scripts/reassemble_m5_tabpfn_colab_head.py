from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/content/lead_tabpfn_head")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def assemble(name: str) -> None:
    parts = sorted(ROOT.glob(f"{name}.part*"))
    if not parts:
        return
    temporary = ROOT / f"{name}.assembling"
    with temporary.open("wb") as target:
        for part in parts:
            with part.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(block)
    temporary.replace(ROOT / name)


manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
expected = {
    "features.float32.npy": manifest["features"]["sha256"],
    "metadata.npz": manifest["metadata"]["sha256"],
    "model.portable.tabpfn_fit": manifest["fit_state"]["sha256"],
    "tabpfn-v3-classifier-v3_default.ckpt": manifest["foundation_checkpoint"]["sha256"],
}
for filename in (
    "features.float32.npy",
    "tabpfn-v3-classifier-v3_default.ckpt",
):
    assemble(filename)
observed = {name: digest(ROOT / name) for name in expected}
if observed != expected:
    raise RuntimeError(
        "SHA-256 mismatch: " + json.dumps({"expected": expected, "observed": observed})
    )
print(json.dumps({"sha256": observed, "verified": True}, sort_keys=True))
