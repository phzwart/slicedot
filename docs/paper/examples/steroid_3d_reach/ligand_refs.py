"""Load steroid ligand topologies from ``ligands/<slug>/``."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from slicedot.restraints import load_restraint_cif

ROOT = Path(__file__).resolve().parent
LIGANDS_DIR = ROOT / "ligands"
LIGANDS_JSON = ROOT / "ligands.json"


def list_ligands() -> list[dict]:
    return json.loads(LIGANDS_JSON.read_text())


def load_ligand(slug: str) -> dict:
    """Return a topology dict compatible with the leucine ensemble runner."""
    d = LIGANDS_DIR / slug
    meta_path = d / "meta.json"
    npz_path = d / "topology.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"missing {npz_path}; run build_ligand_refs.py first"
        )
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {"slug": slug}
    data = np.load(npz_path, allow_pickle=False)
    names = tuple(str(n) for n in data["names"])
    X = np.asarray(data["X"], dtype=np.float64)
    Z = np.asarray(data["Z"], dtype=np.float64)
    bonds = [(int(a), int(b)) for a, b in np.asarray(data["bonds"])]
    rot_raw = np.asarray(data["rotatable_bonds"])
    rotatable = (
        [(int(a), int(b)) for a, b in rot_raw.reshape(-1, 2)]
        if rot_raw.size else []
    )
    chir_raw = np.asarray(data["chiral_centres"])
    chiral = (
        [tuple(int(x) for x in row) for row in chir_raw.reshape(-1, 4)]
        if chir_raw.size else []
    )
    planar_path = d / "planar.json"
    planar = json.loads(planar_path.read_text()) if planar_path.is_file() else []
    n = X.shape[0]
    identity = np.arange(n, dtype=np.int64)
    out = {
        "X_ref": X.copy(),
        "elements": Z.astype(np.int64),
        "Z": Z.copy(),
        "W": (Z / Z.sum()).copy(),
        "names": names,
        "bonds": bonds,
        "rotatable_bonds": rotatable,
        "chiral_centres": chiral,
        "planar_groups": planar,
        "idx": {n: i for i, n in enumerate(names)},
        "automorphism_generators": [identity.copy()],
        "n": int(n),
        "label": meta.get("label", slug),
        "slug": slug,
        "source": meta,
    }
    cif = d / "restraints.cif"
    if cif.is_file():
        out["restraint_set"] = load_restraint_cif(cif)
    return out
