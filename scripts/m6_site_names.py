"""GEPIII site_id -> BDG2 site name.

The M6 artifacts key everything on the GEPIII `site_id` (0-15), which is what
the frozen M3 frame carries. BDG2 ships the human name for the same site, and
its metadata holds both keys, so the mapping is read from there rather than
hardcoded from memory.

Verified on load: every GEPIII site resolves to exactly one BDG2 name and every
BDG2 name resolves back to exactly one GEPIII site.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from lead import ROOT

BDG2_METADATA = ROOT / "data" / "raw" / "bdg2" / "metadata.csv"


@lru_cache(maxsize=1)
def site_names() -> dict[int, str]:
    """Return {gepiii_site_id: bdg2_site_name} for all 16 sites."""
    meta = pd.read_csv(BDG2_METADATA, usecols=["site_id", "site_id_kaggle"])
    meta = meta.dropna(subset=["site_id_kaggle"])
    meta["site_id_kaggle"] = meta["site_id_kaggle"].astype(int)

    forward = meta.groupby("site_id_kaggle")["site_id"].agg(["nunique", "first"])
    if not (forward["nunique"] == 1).all():
        raise SystemExit(
            "a GEPIII site maps to more than one BDG2 name; refusing to guess"
        )
    if not (meta.groupby("site_id")["site_id_kaggle"].nunique() == 1).all():
        raise SystemExit(
            "a BDG2 name maps to more than one GEPIII site; refusing to guess"
        )

    mapping = {int(k): str(v) for k, v in forward["first"].items()}
    missing = sorted(set(range(16)) - set(mapping))
    if missing:
        raise SystemExit(f"GEPIII sites absent from BDG2 metadata: {missing}")
    return mapping


def label(site_id: int, *, with_id: bool = False) -> str:
    """Figure label for a site: its BDG2 name, optionally with the GEPIII id."""
    name = site_names()[int(site_id)]
    return f"{name} (site {int(site_id)})" if with_id else name
