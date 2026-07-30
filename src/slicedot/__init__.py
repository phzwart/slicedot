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

__version__ = "0.1.0"

__all__ = [
    "CrystalSlicedOT",
    "SlicedOT",
    "SlicedOTConfig",
    "b_from_sigma",
    "fibonacci_directions",
    "form_factor_qmax",
    "orthogonalization_matrix",
    "sigma_from_resolution",
    "__version__",
]
