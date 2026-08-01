<p align="center">
  <img src="slicedot.png" alt="SliceDOT" width="520">
</p>

# slicedot

Sliced Wasserstein-1 targets for fitting atomic models into density maps, computed as a structure-factor problem in reciprocal space.

Real-space refinement scores go blind once model and map stop overlapping. Sliced $W_1$ does not: its gradient is a smoothed sign, bounded away from zero wherever mass is misplaced. This package is a reference PyTorch implementation of that target — differentiable in atomic coordinates and weights, with direct and FFT/gridded backends, and with crystallographic (non-orthogonal) maps supported exactly.

> **Role.** Optimal transport here is a *search* device. Use it to deliver a model into the correct basin, then hand off to a conventional forward-model refinement with proper scaling, bulk solvent, and stereochemistry.

Paper draft: [`docs/paper/`](docs/paper/) ([PDF](docs/paper/paper.pdf), [TeX](docs/paper/paper.tex)).
Visual guide (1-D OT → FFT → phenol 2-D → peptide 3-D): [`docs/paper/guide/overview.md`](docs/paper/guide/overview.md).

## Install

Requires Python ≥ 3.10 and a working PyTorch.

```bash
# with uv (recommended)
uv sync --extra dev

# or pip
pip install -e ".[dev]"
```

## Quick start

```python
import torch
from slicedot import SlicedOT, SlicedOTConfig, sigma_from_resolution

target_map = ...          # (Nz, Ny, Nx) non-negative density
origin = (x0, y0, z0)     # Cartesian origin of voxel (0,0,0)
spacing = 0.5             # Å, or length-3 tensor for anisotropic steps
sigma = sigma_from_resolution(2.5)

ot = SlicedOT(
    torch.as_tensor(target_map),
    origin,
    spacing,
    sigma_data=sigma,
    config=SlicedOTConfig(n_dirs=48, backend="auto"),
)

x = torch.randn(N, 3, requires_grad=True)   # atomic coordinates (Å)
w = torch.full((N,), 1.0 / N)               # normalized masses / Z-weights
loss = ot(x, w, sigma)
loss.backward()                             # ∂/∂x via autograd

# length-valued Monge step (vanishes at the solution)
delta = ot.deformation(x.detach(), w, sigma)
```

For a general crystallographic cell, use `CrystalSlicedOT(map, cell, sigma_data, ...)` with `cell = (a, b, c, α, β, γ)` in Å / degrees.

## Backends

| backend | cost | notes |
|---------|------|-------|
| `direct` | $O(B L N K)$ | exact structure-factor sum; best for $N \lesssim 50$ |
| `grid` | $O(B L (N W + P\log P))$ | Agarwal / Ten Eyck scatter + FFT; ~150× faster at $N=8000$ |
| `auto` | — | chooses `grid` when $N \ge 64$ |
| `grid_custom` | same as `grid` | hand-derived backward (lower activation memory) |

## Geometry operator (`P_restr`)

OT moves atoms independently and breaks chemistry. Alternating projection repairs that:

```python
from slicedot import Geometry
from slicedot.fixtures import leucine_topology

topo = leucine_topology()
geom = Geometry(
    topo["X_ref"], topo["bonds"],
    rotatable_bonds=topo["rotatable_bonds"],   # χ bonds: no 1–4 across these
    chiral_centres=topo["chiral_centres"],     # signed Cα volumes
    planar_groups=topo["planar_groups"],       # peptide planes
)

# Stage A: idealise. Loop: over-relaxed P_data, then P_restr. End on P_restr.
X_proj, weighted_rms, nfev = geom.project(X, tol=1e-4, max_iter=200)
```

`Geometry.project` is a nonlinear least-squares idealisation onto distance / chiral /
planar / antibump restraints — an *approximate* projection (not Dykstra). Distance,
planar, and antibump terms use a ReLU flat-bottom of width ``slack`` (Å); anneal
``slack`` from loose → tight during Stage A. Mark rotatable bonds explicitly so
1–4 restraints do not freeze rotamers.

## Validation

The paper's end-to-end protocol lives in `tests/`:

```bash
uv run pytest
# or: pytest
```

Checks floor, analytic translation anchor, autograd vs finite differences, reach at 12 Å, one-step deformation recovery, and backend agreement — on cubic, anisotropic, monoclinic, and triclinic grids.

A short demo:

```bash
uv run python examples/score_leucine.py
```

## Package layout

```
src/slicedot/
  core.py       SlicedOT, CrystalSlicedOT, helpers
  geometry.py   P_restr / Geometry idealisation
  perturb.py    torsion / backrub start generators
  fixtures.py   capped leucine fragment for tests/examples
tests/          OT + geometry acceptance battery
docs/paper/     manuscript (TeX + PDF)
examples/       runnable demos
```

## Citation

If you use this code, please cite the accompanying manuscript (draft in `docs/paper/`):

> P. H. Zwart. *Optimal-transport targets for macromolecular model fitting: non-vanishing gradients from a structure-factor computation.*

## License

MIT — see [LICENSE](LICENSE).
