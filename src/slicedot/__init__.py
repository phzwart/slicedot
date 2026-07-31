"""slicedot -- sliced Wasserstein-1 against a sampled density map, in reciprocal space."""

from slicedot.core import (
    CrystalSlicedOT,
    SlicedOT,
    SlicedOTConfig,
    b_from_sigma,
    fibonacci_directions,
    form_factor_qmax,
    orthogonalization_matrix,
    sigma_from_resolution,
)
from slicedot.geometry import Geometry, build_distance_pairs, topo_distance_matrix
from slicedot.namer import Assignment, Namer
from slicedot.restraints import (
    AngleRestraint,
    BondRestraint,
    PairRestraint,
    PlaneRestraint,
    RestraintSet,
    load_restraint_cif,
    restraint_set_from_geometry,
    write_restraint_cif,
)

__version__ = "0.1.0"

__all__ = [
    "AngleRestraint",
    "Assignment",
    "BondRestraint",
    "CrystalSlicedOT",
    "Geometry",
    "Namer",
    "PairRestraint",
    "PlaneRestraint",
    "RestraintSet",
    "SlicedOT",
    "SlicedOTConfig",
    "b_from_sigma",
    "build_distance_pairs",
    "fibonacci_directions",
    "form_factor_qmax",
    "load_restraint_cif",
    "orthogonalization_matrix",
    "restraint_set_from_geometry",
    "sigma_from_resolution",
    "topo_distance_matrix",
    "write_restraint_cif",
    "__version__",
]
