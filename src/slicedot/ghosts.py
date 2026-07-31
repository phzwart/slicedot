"""Ghost marking and prune-by-geometry/L1 for overcomplete free-atom clouds.

When free OT uses more atoms than the labelled topology (``N_model > N_true``),
surplus atoms are marked as ghosts. Candidate keep-sets of size ``N_true`` are
scored by ``Namer`` restraint residual plus map L1 (chemical weights, resolution
σ), and the best model is returned for the fixed-composition refine path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from slicedot.namer import Assignment, Namer

__all__ = [
    "PruneResult",
    "seed_kept_hungarian",
    "map_density_at",
    "prune_ghosts",
    "LAM_GEOM_DEFAULT",
]

LAM_GEOM_DEFAULT = 0.2
_REJECT_SCORE = 1e6


@dataclass
class PruneResult:
    """Outcome of ``prune_ghosts`` (weights/σ already reset for downstream)."""

    kept_idx: np.ndarray          # (N_true,) free-cloud index per label
    ghost_idx: np.ndarray         # (N_model - N_true,)
    ghost_mask: np.ndarray        # (N_model,) bool; True = ghost
    Y_named: np.ndarray           # (N_true, 3) label-ordered
    assignment: Assignment
    l1: float
    restraint_rms: float
    score: float
    n_models: int
    w_chem: np.ndarray            # chemical weights used for L1 / refine
    sigma: float                  # resolution σ restored for refine


def seed_kept_hungarian(X_free: np.ndarray, X_prior: np.ndarray) -> np.ndarray:
    """Rectangular Hungarian: each label gets a unique free-cloud slot."""
    X_free = np.asarray(X_free, dtype=np.float64)
    X_prior = np.asarray(X_prior, dtype=np.float64)
    n_m, n_t = X_free.shape[0], X_prior.shape[0]
    if n_m < n_t:
        raise ValueError(
            f"need at least N_true={n_t} free atoms, got N_model={n_m}"
        )
    cost = ((X_prior[:, None, :] - X_free[None, :, :]) ** 2).sum(-1)
    _ri, cj = linear_sum_assignment(cost)
    return np.asarray(cj, dtype=np.int64)


def map_density_at(l1_oracle: Any, X: np.ndarray) -> np.ndarray:
    """Sample target map density at atom centres (nearest voxel if gridded)."""
    X = np.asarray(X, dtype=np.float64)
    if hasattr(l1_oracle, "density_at"):
        return np.asarray(l1_oracle.density_at(X), dtype=np.float64).ravel()
    T = getattr(l1_oracle, "T", None)
    V = getattr(l1_oracle, "V", None)
    shape = getattr(l1_oracle, "shape", None)
    origin = getattr(l1_oracle, "origin", None)
    spacing = getattr(l1_oracle, "spacing", None)
    if T is None or origin is None or spacing is None or shape is None:
        return np.zeros(X.shape[0], dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(shape)
    origin = np.asarray(origin, dtype=np.float64).ravel()
    spacing = np.atleast_1d(spacing).astype(np.float64) * np.ones(3)
    out = np.empty(X.shape[0], dtype=np.float64)
    for i, p in enumerate(X):
        ijk = np.rint((p - origin) / spacing).astype(np.int64)
        ijk = np.clip(ijk, 0, np.asarray(shape) - 1)
        out[i] = float(T[tuple(ijk)])
    return out


def _unique_models(models: Sequence[np.ndarray]) -> list[np.ndarray]:
    seen: set[tuple[int, ...]] = set()
    out: list[np.ndarray] = []
    for m in models:
        key = tuple(sorted(int(i) for i in m))
        if key in seen:
            continue
        seen.add(key)
        out.append(np.asarray(sorted(int(i) for i in m), dtype=np.int64))
    return out


def _generate_models(
    X_free: np.ndarray,
    X_prior: np.ndarray,
    seed_kept: np.ndarray,
    dens: np.ndarray,
    *,
    max_swap_models: int,
) -> list[np.ndarray]:
    n_m = X_free.shape[0]
    n_t = X_prior.shape[0]
    models: list[np.ndarray] = [seed_kept.copy()]

    # Density-greedy: top N_true by map density.
    order = np.argsort(-dens)
    models.append(order[:n_t].astype(np.int64))

    ghost = np.array(
        sorted(set(range(n_m)) - set(int(i) for i in seed_kept)),
        dtype=np.int64,
    )
    if ghost.size == 0 or max_swap_models <= 0:
        return _unique_models(models)

    # Unary distance of each kept slot to nearest prior atom.
    d2 = ((X_free[seed_kept, None, :] - X_prior[None, :, :]) ** 2).sum(-1)
    unary = np.sqrt(d2.min(axis=1))
    # Prefer swapping out low-density / high-unary kept atoms.
    kept_rank = np.argsort(dens[seed_kept] - 0.1 * unary)
    ghost_rank = np.argsort(-dens[ghost])  # high-density ghosts first

    n_swap = 0
    for gi in ghost_rank:
        if n_swap >= max_swap_models:
            break
        g = int(ghost[gi])
        for ki in kept_rank:
            if n_swap >= max_swap_models:
                break
            k = int(seed_kept[ki])
            # Only swap if ghost is denser or kept is a clear unary outlier.
            if dens[g] + 1e-15 < dens[k] and unary[ki] < 1.5:
                continue
            trial = seed_kept.copy()
            trial[ki] = g
            models.append(trial)
            n_swap += 1
    return _unique_models(models)


def _score_model(
    kept: np.ndarray,
    X_free: np.ndarray,
    *,
    namer: Namer,
    X_prior: np.ndarray,
    l1_oracle: Any,
    w_chem: np.ndarray,
    lam_geom: float,
) -> tuple[float, float, float, Optional[Assignment]]:
    Y = X_free[kept]
    try:
        asn = namer.assign(Y, X_prior, weights=None)
    except Exception:
        return _REJECT_SCORE, float("nan"), float("nan"), None
    l1_val, _ = l1_oracle.value_grad(asn.Y_named, w_chem)
    l1_val = float(l1_val)
    restr = float(asn.restraint_rms)
    score = l1_val + float(lam_geom) * restr
    return score, l1_val, restr, asn


def prune_ghosts(
    X_free: np.ndarray,
    *,
    namer: Namer,
    X_prior: np.ndarray,
    l1_oracle: Any,
    w_chem: np.ndarray,
    sigma: float,
    max_swap_models: int = 32,
    lam_geom: float = LAM_GEOM_DEFAULT,
    verbose: bool = False,
) -> PruneResult:
    """Mark ghosts, score prune models by geometry + L1, return best keep-set.

    ``w_chem`` and ``sigma`` are the restored chemical weights and map-resolution
    atomic width used for L1 scoring and intended for downstream refine (not the
    uniform 1/N_model weights from overcomplete free OT).
    """
    X_free = np.asarray(X_free, dtype=np.float64)
    X_prior = np.asarray(X_prior, dtype=np.float64)
    w_chem = np.asarray(w_chem, dtype=np.float64).ravel()
    w_chem = w_chem / w_chem.sum()
    sigma = float(sigma)
    n_m = X_free.shape[0]
    n_t = int(namer.n)
    if X_prior.shape[0] != n_t:
        raise ValueError("X_prior length must match namer.n")
    if w_chem.shape[0] != n_t:
        raise ValueError("w_chem length must match namer.n")
    if n_m < n_t:
        raise ValueError(
            f"need at least N_true={n_t} free atoms, got N_model={n_m}"
        )

    if n_m == n_t:
        kept = np.arange(n_t, dtype=np.int64)
        models = [kept]
    else:
        seed = seed_kept_hungarian(X_free, X_prior)
        dens = map_density_at(l1_oracle, X_free)
        models = _generate_models(
            X_free, X_prior, seed, dens, max_swap_models=max_swap_models,
        )

    best_score = float("inf")
    best: Optional[tuple] = None
    n_models = len(models)
    if verbose:
        print(
            f"  [prune] scoring {n_models} keep-sets "
            f"(N_model={n_m} → N_true={n_t})",
            flush=True,
        )
    for mi, kept in enumerate(models):
        score, l1_val, restr, asn = _score_model(
            kept, X_free,
            namer=namer, X_prior=X_prior, l1_oracle=l1_oracle,
            w_chem=w_chem, lam_geom=lam_geom,
        )
        if asn is None:
            if verbose:
                print(f"  [prune] model {mi + 1}/{n_models}  REJECT", flush=True)
            continue
        if score < best_score:
            best_score = score
            best = (kept, asn, l1_val, restr, score)
        if verbose and (
            mi == 0 or (mi + 1) % 5 == 0 or mi + 1 == n_models
        ):
            print(
                f"  [prune] model {mi + 1}/{n_models}  "
                f"score={score:.5g}  L1={l1_val:.5g}  "
                f"restr={restr:.4f}  best={best_score:.5g}",
                flush=True,
            )

    if best is None:
        raise RuntimeError("prune_ghosts: every candidate model failed naming")

    kept, asn, l1_val, restr, score = best
    kept = np.asarray(kept, dtype=np.int64)
    # Label → free-cloud index (Y was X_free[kept] in that order).
    kept_label = kept[np.asarray(asn.perm, dtype=np.int64)]
    ghost_mask = np.ones(n_m, dtype=bool)
    ghost_mask[kept] = False
    ghost_idx = np.flatnonzero(ghost_mask).astype(np.int64)
    return PruneResult(
        kept_idx=kept_label,
        ghost_idx=ghost_idx,
        ghost_mask=ghost_mask,
        Y_named=np.asarray(asn.Y_named, dtype=np.float64).copy(),
        assignment=asn,
        l1=float(l1_val),
        restraint_rms=float(restr),
        score=float(score),
        n_models=len(models),
        w_chem=w_chem.copy(),
        sigma=sigma,
    )
