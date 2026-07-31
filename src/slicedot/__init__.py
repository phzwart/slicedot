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

__version__ = "0.1.0"

__all__ = [
    "CrystalSlicedOT",
    "Geometry",
    "SlicedOT",
    "SlicedOTConfig",
    "b_from_sigma",
    "build_distance_pairs",
    "fibonacci_directions",
    "form_factor_qmax",
    "orthogonalization_matrix",
    "sigma_from_resolution",
    "topo_distance_matrix",
    "__version__",
]
