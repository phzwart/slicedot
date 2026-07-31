"""CIF restraint dictionary for naming."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slicedot.restraints import (
    AngleRestraint,
    BondRestraint,
    RestraintSet,
    load_restraint_cif,
    pair_dev,
    write_restraint_cif,
)

PHENOL_CIF = (
    Path(__file__).resolve().parents[1]
    / "docs/paper/examples/phenol_2d_reach/phenol_restraints.cif"
)


def test_cif_round_trip(tmp_path):
    rs = RestraintSet(
        comp_id="TST",
        atom_ids=["C1", "C2", "O"],
        elements=np.array([6, 6, 8], dtype=np.int64),
        bonds=[
            BondRestraint("C1", "C2", 1.39, 0.35),
            BondRestraint("C1", "O", 1.36, 0.35),
        ],
        angles=[
            AngleRestraint("C2", "C1", "O", 120.0, 5.0),
            AngleRestraint(
                "C1", "C2", "O", 120.0, 15.0,
                value_min=108.0, value_max=180.0,
            ),
        ],
    )
    path = tmp_path / "tst.cif"
    write_restraint_cif(path, rs)
    loaded = load_restraint_cif(path)
    assert loaded.comp_id == "TST"
    assert loaded.atom_ids == ["C1", "C2", "O"]
    assert list(loaded.elements) == [6, 6, 8]
    assert len(loaded.bonds) == 2
    assert len(loaded.angles) == 2
    flat = [a for a in loaded.angles if a.value_min is not None][0]
    assert flat.value_min == pytest.approx(108.0)
    assert flat.value_max == pytest.approx(180.0)


def test_phenol_cif_loads():
    rs = load_restraint_cif(PHENOL_CIF)
    assert rs.comp_id == "PHNL"
    assert len(rs.atom_ids) == 12
    assert len(rs.bonds) == 12
    flats = [a for a in rs.angles if a.value_min is not None]
    assert len(flats) == 4
    assert len(rs.planes) == 1
    assert rs.planes[0].plane_id == "ring"
    assert len(rs.extra_distances) == 3  # para 1–4
    pairs = rs.to_naming_pairs()
    assert any(r.kind == "bond" for r in pairs)
    assert any(r.d_lo < r.d_hi for r in pairs)


def test_geometry_compat_planar_14_only():
    from slicedot.fixtures import leucine_topology
    from slicedot.restraints import restraint_set_from_geometry

    topo = leucine_topology()
    rs = restraint_set_from_geometry(
        topo["X_ref"],
        topo["elements"],
        topo["bonds"],
        rotatable_bonds=topo["rotatable_bonds"],
        planar_groups=topo["planar_groups"],
        atom_ids=topo.get("names"),
        torsion14="planar",
    )
    assert rs.bonds
    assert rs.angles
    assert rs.planes
    # No bulk rigid 1–4 inventory — only plane-scoped diagonals (often none
    # for 4-atom peptide planes).
    kinds = {r.kind for r in rs.to_naming_pairs()}
    assert "bond" in kinds and "angle" in kinds
    rs_all = restraint_set_from_geometry(
        topo["X_ref"],
        topo["elements"],
        topo["bonds"],
        rotatable_bonds=topo["rotatable_bonds"],
        planar_groups=topo["planar_groups"],
        atom_ids=topo.get("names"),
        torsion14="all",
    )
    assert len(rs_all.extra_distances) >= len(rs.extra_distances)


def test_phenol_cif_names_zigzag_cloud():
    """Zigzag cloud + extended unary prior must name under the CIF dictionary."""
    import importlib.util
    import sys

    phenol_path = PHENOL_CIF.parent / "phenol.py"
    spec = importlib.util.spec_from_file_location("phenol_reach", phenol_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phenol_reach"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    Xe, _ = mod.build_phenol(chain_style="extended")
    Xz, _ = mod.build_phenol(chain_style="zigzag")
    namer = mod.phenol_namer(Xe)
    Ye = mod.embed3(Xz)
    Xp = mod.embed3(Xe)
    Ye = Ye - Ye.mean(0) + Xp.mean(0)
    rng = np.random.default_rng(0)
    shuf = rng.permutation(len(Ye))
    Y = Ye[shuf]
    inv = np.empty_like(shuf)
    inv[shuf] = np.arange(len(shuf))
    asg = namer.assign(Y, Xp)
    assert int((asg.perm == inv).sum()) == len(Ye)
    # Both conformers sit inside the dictionary wells.
    assert namer.restraint_rms(Ye, np.arange(len(Ye))) < 0.05
    assert namer.restraint_rms(Xp, np.arange(len(Xp))) < 0.05


def test_flat_angle_pair_dev():
    rs = RestraintSet(
        comp_id="X",
        atom_ids=["A", "B", "C"],
        elements=np.array([6, 6, 6]),
        bonds=[
            BondRestraint("A", "B", 1.54, 0.35),
            BondRestraint("B", "C", 1.54, 0.35),
        ],
        angles=[
            AngleRestraint(
                "A", "B", "C", 120.0, 15.0,
                value_min=108.0, value_max=180.0,
            ),
        ],
    )
    pairs = [p for p in rs.to_naming_pairs() if p.kind == "angle"]
    assert len(pairs) == 1
    r = pairs[0]
    mid = 0.5 * (r.d_lo + r.d_hi)
    assert pair_dev(mid, r.d_lo, r.d_hi) == 0.0
    assert pair_dev(r.d_lo - 0.2, r.d_lo, r.d_hi) > 0.0
