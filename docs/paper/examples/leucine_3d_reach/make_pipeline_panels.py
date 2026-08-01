#!/usr/bin/env python3
"""Three-panel pipeline figure: free OT → named → refined.

Vertical stack of 3-D panels (white paper background): translucent density
isosurface, true-model wire overlay, and the working pose. Named / refined
atoms use standard CPK element colours; free atoms stay unlabelled orange.

Usage
-----
  uv run --extra paper python make_pipeline_panels.py \\
      --path out/path_AFSSFN_3A_seed0.npz --sequence AFSSFN
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image
from skimage.measure import marching_cubes

from make_ot_name_refine_ensemble import OUT_DIR, build_scene

C_BG = "#ffffff"
C_ISO = "#3cb371"
C_ISO_EDGE = "#2f8f56"
C_TRUE = "#4a7ab5"
C_FREE = "#e85d04"
C_FREE_EDGE = "#5a1a00"
C_TEXT = "#1a1a1a"
C_MUTED = "#666666"

# Jmol / CPK element colours.
CPK = {
    "H": "#ffffff",
    "C": "#909090",
    "N": "#3050f8",
    "O": "#ff0d0d",
    "S": "#ffff30",
    "P": "#ff8000",
}
Z_TO_ELEM = {1: "H", 6: "C", 7: "N", 8: "O", 15: "P", 16: "S"}


def _last_stage_index(stages: np.ndarray, name: str) -> int:
    idx = [i for i, s in enumerate(stages) if str(s) == name]
    if not idx:
        raise KeyError(f"stage {name!r} missing from trajectory")
    return int(idx[-1])


def _element_list(scene: dict) -> list[str]:
    """Return element symbols; ``topo['elements']`` may store Z numbers."""
    topo = scene.get("topo") or {}
    raw = topo.get("elements", topo.get("Z"))
    if raw is None:
        return ["C"] * int(scene["n_atoms"])
    out = []
    for e in raw:
        if isinstance(e, (int, np.integer)) or (isinstance(e, str) and e.isdigit()):
            out.append(Z_TO_ELEM.get(int(e), "C"))
        else:
            out.append(str(e))
    return out


def _cpk_colors(elems: list[str]) -> list[str]:
    return [CPK.get(e, "#cccccc") for e in elems]


def _crop_density(T, origin, spacing, center, half):
    """Axis-aligned crop around ``center`` ± ``half`` (Å)."""
    sp = np.atleast_1d(spacing).astype(np.float64) * np.ones(3)
    org = np.asarray(origin, dtype=np.float64)
    lo = center - half
    hi = center + half
    i0 = np.maximum(0, np.floor((lo - org) / sp).astype(int))
    i1 = np.minimum(np.asarray(T.shape) - 1, np.ceil((hi - org) / sp).astype(int))
    sl = tuple(slice(int(a), int(b) + 1) for a, b in zip(i0, i1))
    Tc = T[sl]
    org_c = org + i0 * sp
    return Tc, org_c, sp


def _iso_mesh(T, origin, spacing, level: float):
    sp = tuple(float(x) for x in np.atleast_1d(spacing) * np.ones(3))
    verts, faces, _n, _v = marching_cubes(
        np.asarray(T, dtype=np.float64),
        level=float(level),
        spacing=sp,
        allow_degenerate=False,
    )
    verts = verts + np.asarray(origin, dtype=np.float64)
    return verts, faces


def _triangle_edges(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    e = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    e.sort(axis=1)
    e = np.unique(e, axis=0)
    return verts[e]


def _add_isosurface(
    ax,
    verts,
    faces,
    *,
    color=C_ISO,
    alpha=0.14,
    edge_alpha=0.35,
    lw=0.22,
):
    tris = verts[faces]
    mesh = Poly3DCollection(
        tris,
        facecolors=color,
        edgecolors=color,
        linewidths=0.0,
        alpha=alpha,
        shade=False,
    )
    mesh.set_clip_on(False)
    ax.add_collection3d(mesh)
    segs = _triangle_edges(verts, faces)
    if len(segs) > 4500:
        rng = np.random.default_rng(0)
        segs = segs[rng.choice(len(segs), size=4500, replace=False)]
    edges = Line3DCollection(
        segs,
        colors=C_ISO_EDGE,
        linewidths=lw,
        alpha=edge_alpha,
    )
    edges.set_clip_on(False)
    ax.add_collection3d(edges)


def _add_wire(ax, X, bonds, *, color, lw=1.35, alpha=0.95):
    X = np.asarray(X, dtype=np.float64)
    segs = [[X[i], X[j]] for i, j in np.asarray(bonds, dtype=int)]
    if segs:
        coll = Line3DCollection(
            segs, colors=color, linewidths=lw, alpha=alpha,
        )
        coll.set_clip_on(False)
        ax.add_collection3d(coll)


def _add_cpk_model(ax, X, bonds, colors, *, lw=1.7, s=38, alpha=0.98):
    """Ball-and-stick with half-bond CPK colours."""
    X = np.asarray(X, dtype=np.float64)
    segs = []
    cols = []
    for i, j in np.asarray(bonds, dtype=int):
        mid = 0.5 * (X[i] + X[j])
        segs.append([X[i], mid])
        cols.append(colors[i])
        segs.append([mid, X[j]])
        cols.append(colors[j])
    if segs:
        coll = Line3DCollection(segs, colors=cols, linewidths=lw, alpha=alpha)
        coll.set_clip_on(False)
        ax.add_collection3d(coll)
    ax.scatter(
        X[:, 0], X[:, 1], X[:, 2],
        c=colors, s=s, alpha=alpha, depthshade=True,
        edgecolors="#333333", linewidths=0.25, zorder=5, clip_on=False,
    )


def _add_atoms(ax, X, *, color, s=28, alpha=0.95, edgecolors="none", linewidths=0.0):
    X = np.asarray(X, dtype=np.float64)
    ax.scatter(
        X[:, 0], X[:, 1], X[:, 2],
        c=color, s=s, alpha=alpha, depthshade=True,
        edgecolors=edgecolors, linewidths=linewidths, zorder=5, clip_on=False,
    )


def _frame_density(ax, cloud: np.ndarray, *, pad: float = 0.12):
    """Tight AABB around the density/atoms (data aspect, no zoom-clip)."""
    cloud = np.asarray(cloud, dtype=np.float64)
    lo = cloud.min(0) - pad
    hi = cloud.max(0) + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    aspect = np.maximum(hi - lo, 1e-3)
    try:
        ax.set_box_aspect(aspect, zoom=1.0)
    except TypeError:
        ax.set_box_aspect(tuple(aspect))
        ax.dist = 9.0


def _style_axes(ax):
    ax.set_facecolor(C_BG)
    ax.set_axis_off()
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(C_BG)
        axis.pane.set_alpha(0.0)
        axis.line.set_color((0, 0, 0, 0))
        axis.set_ticklabels([])
        axis.set_ticks([])


def _pca_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frame with the *short* PCA axis along +y (view direction at azim=-90).

    Matplotlib ``view_init(elev=0, azim=-90)`` looks along ±y (x horizontal,
    z vertical). So: longest → +x (width), shortest → +y (depth / view),
    middle → +z (height).
    """
    pts = np.asarray(points, dtype=np.float64)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    proj = (pts - c) @ vt.T
    extents = proj.max(axis=0) - proj.min(axis=0)
    i_short = int(np.argmin(extents))
    rest = [i for i in range(3) if i != i_short]
    # Among the in-plane axes, put the longer one on x (width).
    if extents[rest[0]] < extents[rest[1]]:
        rest = [rest[1], rest[0]]
    i_long, i_mid = rest[0], rest[1]
    # Rows of R: new +x, +y (view/depth), +z
    R = np.stack([vt[i_long], vt[i_short], vt[i_mid]], axis=0)
    if np.linalg.det(R) < 0:
        R[2] *= -1.0
    # Prefer density "up" with positive mean z after transform.
    z_sign = float(((pts - c) @ R[2]).mean())
    if z_sign < 0:
        R[2] *= -1.0
        R[1] *= -1.0  # keep right-handed
    return c, R


def _xf(P: np.ndarray, c: np.ndarray, R: np.ndarray) -> np.ndarray:
    return (np.asarray(P, dtype=np.float64) - c) @ R.T


def _crop_white(arr: np.ndarray, *, pad: int = 6, thr: int = 248) -> np.ndarray:
    """Trim near-white margins so content spans the image width."""
    ink = (arr < thr).any(axis=2)
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return arr
    r0 = max(0, int(rows[0]) - pad)
    r1 = min(arr.shape[0], int(rows[-1]) + pad + 1)
    c0 = max(0, int(cols[0]) - pad)
    c1 = min(arr.shape[1], int(cols[-1]) + pad + 1)
    return arr[r0:r1, c0:c1]


def _render_panel_rgb(
    *,
    X,
    style,
    X_true,
    bonds,
    cpk,
    verts_hi,
    faces_hi,
    verts_lo,
    faces_lo,
    have_lo,
    cloud,
    elev,
    azim,
    iso_alpha,
    dpi: int = 220,
) -> np.ndarray:
    """Render one 3-D panel and return a tightly cropped RGB array."""
    # Canvas aspect ≈ in-plane density aspect (x/z) so width is used fully.
    ext = cloud.max(0) - cloud.min(0)
    aspect = float(max(ext[0], 1e-3) / max(ext[2], 1e-3))
    fig_w = 9.0
    fig_h = max(2.2, min(5.0, fig_w / aspect))
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=C_BG)
    ax = fig.add_subplot(1, 1, 1, projection="3d", facecolor=C_BG)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    if have_lo:
        _add_isosurface(
            ax, verts_lo, faces_lo,
            color=C_ISO, alpha=0.55 * iso_alpha, edge_alpha=0.20, lw=0.16,
        )
    _add_isosurface(
        ax, verts_hi, faces_hi,
        color=C_ISO, alpha=iso_alpha, edge_alpha=0.32, lw=0.22,
    )
    _add_wire(ax, X_true, bonds, color=C_TRUE, lw=1.6, alpha=0.85)
    _add_atoms(
        ax, X_true, color=C_TRUE, s=22, alpha=0.85,
        edgecolors="#2a4a72", linewidths=0.3,
    )
    if style == "cpk":
        _add_cpk_model(ax, X, bonds, cpk, lw=2.1, s=52, alpha=0.98)
    else:
        _add_atoms(
            ax, X, color=C_FREE, s=52, alpha=0.98,
            edgecolors=C_FREE_EDGE, linewidths=0.35,
        )

    # elev≈0 keeps the short PCA axis (±y) as the viewing direction.
    ax.view_init(elev=elev, azim=azim)
    _frame_density(ax, cloud, pad=0.10)
    _style_axes(ax)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=C_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    rgb = np.asarray(Image.open(buf).convert("RGB"))
    return _crop_white(rgb, pad=4)


def _stack_panels(
    panel_rgbs: list[np.ndarray],
    titles: list[tuple[str, str]],
    *,
    suptitle: str,
    out_stem: str,
) -> Path:
    """Assemble cropped panels to a common width with titles + legend."""
    dpi = 220
    # Scale every panel up to at least this width so density spans the page
    # and titles have room (avoids side-cropping labels).
    min_w = int(7.2 * dpi)
    target_w = max(min_w, max(im.shape[1] for im in panel_rgbs))
    resized = []
    for im in panel_rgbs:
        if im.shape[1] == target_w:
            resized.append(im)
            continue
        h = max(1, int(round(im.shape[0] * target_w / im.shape[1])))
        resized.append(
            np.asarray(Image.fromarray(im).resize((target_w, h), Image.Resampling.LANCZOS))
        )

    # Title strip height in pixels (at 220 dpi ≈ 11 pt).
    title_h = int(0.55 * dpi)
    gap = int(0.12 * dpi)
    legend_h = int(0.55 * dpi)
    sup_h = int(0.55 * dpi)

    rows = []
    # Suptitle
    fig_t = plt.figure(figsize=(target_w / dpi, sup_h / dpi), dpi=dpi, facecolor=C_BG)
    fig_t.text(0.5, 0.45, suptitle, ha="center", va="center", fontsize=12, color=C_TEXT)
    buf = io.BytesIO()
    fig_t.savefig(buf, format="png", dpi=dpi, facecolor=C_BG)
    plt.close(fig_t)
    buf.seek(0)
    rows.append(np.asarray(Image.open(buf).convert("RGB")))

    for (title, sub), im in zip(titles, resized):
        fig_t = plt.figure(figsize=(target_w / dpi, title_h / dpi), dpi=dpi, facecolor=C_BG)
        fig_t.text(
            0.5, 0.55, title, ha="center", va="center",
            fontsize=11, color=C_TEXT,
        )
        fig_t.text(0.5, 0.18, sub, ha="center", va="center", fontsize=9.5, color=C_MUTED)
        buf = io.BytesIO()
        fig_t.savefig(buf, format="png", dpi=dpi, facecolor=C_BG)
        plt.close(fig_t)
        buf.seek(0)
        rows.append(np.asarray(Image.open(buf).convert("RGB")))
        rows.append(im)
        rows.append(np.full((gap, target_w, 3), 255, dtype=np.uint8))

    # Legend strip
    fig_l = plt.figure(figsize=(target_w / dpi, legend_h / dpi), dpi=dpi, facecolor=C_BG)
    fig_l.legend(
        handles=[
            Line2D([0], [0], color=C_ISO, lw=4, alpha=0.55, label="density"),
            Line2D([0], [0], color=C_TRUE, lw=1.8, label="true model"),
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=C_FREE,
                markeredgecolor=C_FREE_EDGE, markersize=7, label="free atoms",
            ),
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=CPK["C"],
                markeredgecolor="#333333", markersize=7, label="C",
            ),
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=CPK["N"],
                markeredgecolor="#333333", markersize=7, label="N",
            ),
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=CPK["O"],
                markeredgecolor="#333333", markersize=7, label="O",
            ),
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=CPK["S"],
                markeredgecolor="#333333", markersize=7, label="S",
            ),
        ],
        loc="center",
        ncol=7,
        frameon=False,
        fontsize=8.5,
        labelcolor=C_TEXT,
    )
    buf = io.BytesIO()
    fig_l.savefig(buf, format="png", dpi=dpi, facecolor=C_BG)
    plt.close(fig_l)
    buf.seek(0)
    rows.append(np.asarray(Image.open(buf).convert("RGB")))

    # Match widths (text strips may differ by 1 px).
    rows = [
        r if r.shape[1] == target_w
        else np.asarray(Image.fromarray(r).resize((target_w, r.shape[0]), Image.Resampling.NEAREST))
        for r in rows
    ]
    canvas = np.concatenate(rows, axis=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{out_stem}.png"
    pdf = OUT_DIR / f"{out_stem}.pdf"
    Image.fromarray(canvas).save(png)
    # Vector-ish PDF via matplotlib imshow of the assembled raster.
    h_in = canvas.shape[0] / dpi
    w_in = canvas.shape[1] / dpi
    fig = plt.figure(figsize=(w_in, h_in), facecolor=C_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas)
    ax.set_axis_off()
    fig.savefig(pdf, dpi=dpi, facecolor=C_BG, edgecolor="none")
    plt.close(fig)
    return png


def draw_panels(
    scene: dict,
    path: dict,
    *,
    out_stem: str,
    elev: float = 8.0,
    azim: float = -90.0,
    iso_frac: float = 0.42,
    iso_alpha: float = 0.14,
) -> Path:
    X_true = np.asarray(scene["X_true"], dtype=np.float64)
    bonds = np.asarray(scene["bonds"], dtype=int)
    T = np.asarray(scene["T"], dtype=np.float64)
    org = scene["origin"]
    sp = scene["spacing"]
    cpk = _cpk_colors(_element_list(scene))

    coords = np.asarray(path["coords"], dtype=np.float64)
    stages = np.asarray(path["stages"])

    i_free = _last_stage_index(stages, "free")
    i_named = _last_stage_index(stages, "named")
    try:
        i_ref = _last_stage_index(stages, "polish")
        refined_tag = "refined (polish)"
    except KeyError:
        try:
            i_ref = _last_stage_index(stages, "cleanup")
            refined_tag = "refined (cleanup)"
        except KeyError:
            i_ref = i_named
            refined_tag = "named"

    X_free = coords[i_free]
    X_named = coords[i_named]
    X_ref = coords[i_ref]

    pts_all = np.vstack([X_true, X_free[: len(X_true)], X_named, X_ref])
    center = 0.5 * (pts_all.max(0) + pts_all.min(0))
    half = 0.5 * float((pts_all.max(0) - pts_all.min(0)).max()) + 2.8
    Tc, org_c, sp_c = _crop_density(T, org, sp, center, half)
    level = float(iso_frac * Tc.max())
    level_lo = float(0.55 * level)
    verts_hi, faces_hi = _iso_mesh(Tc, org_c, sp_c, level)
    try:
        verts_lo, faces_lo = _iso_mesh(Tc, org_c, sp_c, level_lo)
        have_lo = True
    except (ValueError, RuntimeError):
        have_lo = False
        verts_lo = faces_lo = None

    mesh_pts = [pts_all, verts_hi]
    if have_lo:
        mesh_pts.append(verts_lo)
    cloud = np.vstack(mesh_pts)

    # Put the long molecular axis along +x so density fills the panel width.
    c0, R = _pca_basis(cloud)
    X_true = _xf(X_true, c0, R)
    X_free = _xf(X_free, c0, R)
    X_named = _xf(X_named, c0, R)
    X_ref = _xf(X_ref, c0, R)
    verts_hi = _xf(verts_hi, c0, R)
    if have_lo:
        verts_lo = _xf(verts_lo, c0, R)
    cloud = _xf(cloud, c0, R)

    free_nn = float(path["free_nn"]) if "free_nn" in path.files else float("nan")
    named_rmsd = (
        float(path["named_rmsd"]) if "named_rmsd" in path.files else float("nan")
    )
    polish_rmsd = (
        float(path["polish_rmsd"]) if "polish_rmsd" in path.files else float("nan")
    )
    cleanup_rmsd = (
        float(path["cleanup_rmsd"]) if "cleanup_rmsd" in path.files else float("nan")
    )
    refined_rmsd = polish_rmsd if np.isfinite(polish_rmsd) else cleanup_rmsd

    panels = [
        (
            X_free,
            "free",
            "Free-atom OT placement",
            f"NN-RMSD {free_nn:.2f} Å" if np.isfinite(free_nn) else "",
        ),
        (
            X_named,
            "cpk",
            "Named free atoms",
            f"label-RMSD {named_rmsd:.2f} Å" if np.isfinite(named_rmsd) else "",
        ),
        (
            X_ref,
            "cpk",
            "Refined structure",
            f"RMSD {refined_rmsd:.3f} Å · {refined_tag}"
            if np.isfinite(refined_rmsd)
            else refined_tag,
        ),
    ]

    panel_rgbs = []
    titles = []
    for X, style, title, sub in panels:
        panel_rgbs.append(
            _render_panel_rgb(
                X=X,
                style=style,
                X_true=X_true,
                bonds=bonds,
                cpk=cpk,
                verts_hi=verts_hi,
                faces_hi=faces_hi,
                verts_lo=verts_lo,
                faces_lo=faces_lo,
                have_lo=have_lo,
                cloud=cloud,
                elev=elev,
                azim=azim,
                iso_alpha=iso_alpha,
            )
        )
        titles.append((title, sub))

    res = float(scene["resolution"])
    suptitle = (
        f"{scene['label']} @ {res:g} Å · density isosurface with true model overlay"
    )
    return _stack_panels(panel_rgbs, titles, suptitle=suptitle, out_stem=out_stem)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--path",
        type=Path,
        default=OUT_DIR / "path_AFSSFN_3A_seed0.npz",
        help="trajectory npz from export_single_trajectory / ensemble",
    )
    ap.add_argument("--sequence", default="AFSSFN")
    ap.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="override resolution (default: read from path npz)",
    )
    ap.add_argument(
        "--out-stem",
        default=None,
        help="output stem under out/ (default: derived from sequence/resolution)",
    )
    ap.add_argument(
        "--elev",
        type=float,
        default=8.0,
        help="elevation; keep small so short PCA axis (~y) stays the view dir",
    )
    ap.add_argument(
        "--azim",
        type=float,
        default=-90.0,
        help="azimuth after PCA align (-90 looks along ±y = short axis)",
    )
    ap.add_argument(
        "--iso-frac",
        type=float,
        default=0.42,
        help="isosurface level as a fraction of cropped density max",
    )
    ap.add_argument("--iso-alpha", type=float, default=0.14)
    args = ap.parse_args()

    path = np.load(args.path, allow_pickle=True)
    res = float(args.resolution if args.resolution is not None else path["resolution"])
    seq = tuple(args.sequence.strip().upper())
    scene = build_scene(resolution=res, sequence=seq)

    stem = args.out_stem or (
        f"peptide_{''.join(seq)}_pipeline_panels_{res:g}A".replace(".", "p")
    )
    out = draw_panels(
        scene,
        path,
        out_stem=stem,
        elev=args.elev,
        azim=args.azim,
        iso_frac=args.iso_frac,
        iso_alpha=args.iso_alpha,
    )
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

