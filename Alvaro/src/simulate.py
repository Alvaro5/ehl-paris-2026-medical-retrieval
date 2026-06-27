"""Synthesize dataset2-like and dataset3-like conditions from our LABELED
dataset1 internal-val split.

We have no labels for the real dataset2 / dataset3, so we can never measure
generalization on them locally. This module manufactures faithful *proxies* from
the one split we DO have labels for (splits/internal_val_*), letting us watch the
MRR drop a change causes on broken-geometry (ds2) and broken-content (ds3)
conditions without burning Kaggle submissions. The exact same transforms can
later be reused as training-time augmentation.

Two modes (see CLAUDE.md for how the real datasets differ):

  ds2 — breaks GEOMETRY on BOTH sides, INDEPENDENTLY. Every image (query and
        gallery) gets its own random rigid affine (rotation + translation) plus a
        non-linear elastic deformation. A query and its true target are warped
        with DIFFERENT random draws, so they do NOT stay aligned to each other —
        which is the whole point, otherwise the proxy is trivial.

  ds3 — keeps geometry ~shared, breaks CONTENT on the TARGET only. Queries stay
        geometrically intact (re-saved unchanged). Each target gets a simulated
        resection (a contiguous region inside the brain zeroed out) plus a
        scanner/intensity shift (bias field + gamma/intensity change). No global
        affine, because real ds3 is resampled to ~the same space. The asymmetry
        (query intact, target altered) mirrors pre-op -> intra-op.

Independence & reproducibility: each image's RNG seed is derived from a stable
hash of its globally-unique image_id combined with --seed. Because query_id and
target_id are distinct strings, query and target automatically get independent
draws, and the whole thing is deterministic across runs and machines.

CPU only. Output is written with .nii.gz both on disk and in the CSVs (no
.nii/.nii.gz remap hack) and is loadable by slice_clip_baseline.py unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandAffined,
    RandBiasFieldd,
    RandScaleIntensityd,
)
from scipy.ndimage import distance_transform_edt

# --- Column schemas (must match CLAUDE.md / local_eval.py exactly) -----------
QUERIES_COLS = ["query_id", "query_image", "query_modality", "dataset"]
GALLERY_COLS = ["target_id", "target_image", "target_modality", "dataset"]

# --- Default magnitudes (a STARTING point; calibrate by eye + by MRR drop) ---
DEFAULTS = {
    # ds2 rigid
    "rotate_deg": 15.0,      # max abs rotation per axis (degrees)
    "translate": 8.0,        # max abs translation per axis (voxels)
    # ds2 elastic
    "elastic_sigma": (6.0, 9.0),       # smoothness of the displacement field
    "elastic_magnitude": (60.0, 120.0),  # peak displacement (voxels, pre-smooth)
    # ds3 resection + scanner shift
    "resection_mm": 20.0,    # sphere radius of resected tissue (mm)
    "bias_coeff": 0.3,       # max bias-field coefficient
    "gamma": (0.7, 1.5),     # contrast/gamma range
    "intensity_scale": 0.1,  # +/- fractional intensity scaling
}


# ---------------------------------------------------------------------------
# Seeding: stable, per-image, ID-derived
# ---------------------------------------------------------------------------
def stable_seed(image_id: str, base_seed: int) -> int:
    """Deterministic 32-bit seed from an image_id and the base seed.

    Uses sha256 (NOT Python's salted built-in hash) so the value is identical
    across runs and machines. Distinct IDs -> distinct seeds, which is what makes
    a query and its target warp independently in ds2.
    """
    digest = hashlib.sha256(f"{base_seed}:{image_id}".encode()).hexdigest()
    return int(digest[:8], 16)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def resolve(data_root: str, rel_path: str) -> str:
    """Join a manifest image path to the data root (absolute paths pass through)."""
    return rel_path if os.path.isabs(rel_path) else os.path.join(data_root, rel_path)


def load_volume(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a NIfTI volume -> (array[H,W,D] float32, affine 4x4)."""
    img = nib.load(path)
    return np.asarray(img.get_fdata(), dtype=np.float32), img.affine


def save_volume(arr: np.ndarray, affine: np.ndarray, out_path: str) -> None:
    """Write array[H,W,D] as a .nii.gz, preserving the original affine.

    We keep the SOURCE affine on purpose: ds2 warps move content within the grid
    (so the grid metadata stays the original 1 mm space), and ds3 must stay in
    ~the same space. Either way the baseline's Spacing/Orientation behave sanely.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(arr, dtype=np.float32), affine), out_path)


# ---------------------------------------------------------------------------
# ds2: independent geometry break (rigid + elastic), both sides
# ---------------------------------------------------------------------------
def make_ds2_compose(args: argparse.Namespace) -> Compose:
    """Build the rigid+elastic geometry-breaking pipeline (prob=1.0 always on)."""
    rot = np.deg2rad(args.rotate_deg)
    return Compose(
        [
            LoadImaged(keys="image", image_only=True),
            EnsureChannelFirstd(keys="image"),
            RandAffined(
                keys="image",
                prob=1.0,
                rotate_range=(rot, rot, rot),
                translate_range=(args.translate, args.translate, args.translate),
                mode="bilinear",
                padding_mode="zeros",  # keep background 0
            ),
            Rand3DElasticd(
                keys="image",
                prob=1.0,
                sigma_range=tuple(args.elastic_sigma),
                magnitude_range=tuple(args.elastic_magnitude),
                mode="bilinear",
                padding_mode="zeros",
            ),
        ]
    )


def perturb_ds2(path: str, seed: int, compose: Compose) -> np.ndarray:
    """Apply the seeded ds2 geometry break -> array[H,W,D]."""
    compose.set_random_state(seed=seed)  # propagates to every Rand* child
    out = compose({"image": path})
    return np.asarray(out["image"][0])  # drop channel dim


# ---------------------------------------------------------------------------
# ds3: content break on the target (resection + scanner shift)
# ---------------------------------------------------------------------------
def make_ds3_compose(args: argparse.Namespace) -> Compose:
    """Bias field + gamma + mild intensity scale. NO geometric warp."""
    return Compose(
        [
            LoadImaged(keys="image", image_only=True),
            EnsureChannelFirstd(keys="image"),
            RandBiasFieldd(
                keys="image", prob=1.0, coeff_range=(0.0, args.bias_coeff), degree=3
            ),
            RandAdjustContrastd(keys="image", prob=1.0, gamma=tuple(args.gamma)),
            RandScaleIntensityd(keys="image", prob=1.0, factors=args.intensity_scale),
        ]
    )


def apply_resection(
    arr: np.ndarray,
    affine: np.ndarray,
    rng: np.random.Generator,
    radius_mm: float,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Zero a contiguous ellipsoid of tissue strictly INSIDE the brain.

    Returns the modified array and the (x,y,z) voxel center of the resection
    (used by the verification PNG to show the right slice).

    Placement guarantee: we sample the center only among voxels whose mm-distance
    to background is >= radius_mm, so the whole sphere fits inside real tissue --
    never a black box floating in the background.
    """
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))  # mm per voxel, per axis
    mask = arr > 0
    # EDT distance (in mm) from each foreground voxel to nearest background voxel.
    dist = distance_transform_edt(mask, sampling=spacing)
    valid = np.argwhere(dist >= radius_mm)
    if len(valid) == 0:
        # Radius too big for this brain: fall back to deepest interior voxel.
        valid = np.argwhere(dist >= dist.max() * 0.5)
    center = valid[rng.integers(len(valid))]

    radius_vox = radius_mm / spacing
    # Slight per-axis jitter so the hole is an irregular ellipsoid, not a sphere.
    radius_vox = radius_vox * rng.uniform(0.85, 1.15, size=3)

    # Work in a bounding box around the center for efficiency.
    lo = np.maximum(center - np.ceil(radius_vox).astype(int), 0)
    hi = np.minimum(center + np.ceil(radius_vox).astype(int) + 1, arr.shape)
    gx, gy, gz = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    ell = (
        ((gx - center[0]) / radius_vox[0]) ** 2
        + ((gy - center[1]) / radius_vox[1]) ** 2
        + ((gz - center[2]) / radius_vox[2]) ** 2
    ) <= 1.0
    arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]][ell] = 0.0
    return arr, tuple(int(c) for c in center)


def perturb_ds3_target(
    path: str, seed: int, compose: Compose, radius_mm: float
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """Scanner shift + resection on a target -> (array, affine, resection center)."""
    _, affine = load_volume(path)  # original affine for spacing + save
    compose.set_random_state(seed=seed)
    out = compose({"image": path})
    arr = np.asarray(out["image"][0])
    rng = np.random.default_rng(seed)  # same seed -> reproducible placement
    arr, center = apply_resection(arr, affine, rng, radius_mm)
    return arr, affine, center


# ---------------------------------------------------------------------------
# Verification PNGs (the trust gate)
# ---------------------------------------------------------------------------
def _axial(arr: np.ndarray, z: int) -> np.ndarray:
    """Display-oriented middle/− axial slice (last axis is axial for BraTS)."""
    z = int(np.clip(z, 0, arr.shape[2] - 1))
    return np.rot90(arr[:, :, z])


def verify_ds2(
    orig_q, pert_q, orig_t, pert_t, out_png: str, qid: str, tid: str
) -> None:
    """2x2 grid: [orig query | perturbed query] / [orig target | perturbed target]."""
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    panels = [
        (orig_q, f"orig query\n{qid}"),
        (pert_q, "perturbed query"),
        (orig_t, f"orig target\n{tid}"),
        (pert_t, "perturbed target"),
    ]
    for a, (vol, title) in zip(ax.flat, panels):
        a.imshow(_axial(vol, vol.shape[2] // 2), cmap="gray")
        a.set_title(title, fontsize=9)
        a.axis("off")
    fig.suptitle("ds2 proxy: independent rigid+elastic warp (query vs target)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def verify_ds3(orig_t, pert_t, center, out_png: str, tid: str) -> None:
    """[orig target | perturbed target] at the resection-center slice."""
    z = center[2]
    fig, ax = plt.subplots(1, 2, figsize=(8, 4.5))
    ax[0].imshow(_axial(orig_t, z), cmap="gray")
    ax[0].set_title(f"orig target\n{tid} (z={z})", fontsize=9)
    ax[1].imshow(_axial(pert_t, z), cmap="gray")
    ax[1].set_title("perturbed target\nresection + scanner shift", fontsize=9)
    for a in ax:
        a.axis("off")
    fig.suptitle("ds3 proxy: resection (hole) inside brain + intensity shift")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def _truth_pairs(splits_dir: str) -> Dict[str, str]:
    df = pd.read_csv(os.path.join(splits_dir, "internal_val_truth.csv"))
    return dict(zip(df["query_id"].astype(str), df["target_id"].astype(str)))


def run_ds2(args: argparse.Namespace) -> None:
    """Warp every val query and gallery image independently; write proxy_ds2."""
    sd, out_dir = args.splits_dir, os.path.join(args.splits_dir, "proxy_ds2")
    img_dir = os.path.join(out_dir, "images")
    queries = pd.read_csv(os.path.join(sd, "internal_val_queries.csv"))
    gallery = pd.read_csv(os.path.join(sd, "internal_val_gallery.csv"))
    truth = _truth_pairs(sd)
    compose = make_ds2_compose(args)

    if args.limit:
        queries, gallery = queries.head(args.limit), gallery.head(args.limit)

    def process(df, id_col, path_col, rel_prefix):
        new_paths = {}
        for _, row in df.iterrows():
            iid = str(row[id_col])
            src = resolve(args.data_root, row[path_col])
            arr = perturb_ds2(src, stable_seed(iid, args.seed), compose)
            _, affine = load_volume(src)
            rel = os.path.join(rel_prefix, f"{iid}.nii.gz")
            save_volume(arr, affine, os.path.join(args.data_root, rel))
            new_paths[iid] = rel
            print(f"  ds2 {id_col}={iid} -> {rel}")
        return new_paths

    rel_prefix = os.path.join(out_dir, "images")
    q_paths = process(queries, "query_id", "query_image", rel_prefix)
    g_paths = process(gallery, "target_id", "target_image", rel_prefix)

    # CSVs: identical schema/IDs, only the image path swapped to the .nii.gz copy.
    queries = queries.copy()
    queries["query_image"] = queries["query_id"].astype(str).map(q_paths)
    queries[QUERIES_COLS].to_csv(os.path.join(out_dir, "val_queries.csv"), index=False)
    gallery = gallery.copy()
    gallery["target_image"] = gallery["target_id"].astype(str).map(g_paths)
    gallery[GALLERY_COLS].to_csv(os.path.join(out_dir, "val_gallery.csv"), index=False)

    _verify_ds2_pair(args, queries, gallery, truth, out_dir, compose)
    print(f"ds2 proxy written to {out_dir}")


def _verify_ds2_pair(args, queries, gallery, truth, out_dir, compose):
    """Independence check + PNG for the first matched pair present in the run."""
    qrow = queries.iloc[0]
    qid = str(qrow["query_id"])
    tid = truth[qid]
    q_src = resolve(args.data_root, _orig_path(args, "query", qid))
    t_src = resolve(args.data_root, _orig_path(args, "target", tid))

    # Numeric independence guard: the two RNG seeds MUST differ, else the warps
    # would be identical and the pair would stay aligned -> proxy meaningless.
    sq, st = stable_seed(qid, args.seed), stable_seed(tid, args.seed)
    assert sq != st, f"query/target share a seed ({sq}); independence broken"
    print(f"  independence ok: seed(query)={sq} != seed(target)={st}")

    orig_q, _ = load_volume(q_src)
    orig_t, _ = load_volume(t_src)
    pert_q = perturb_ds2(q_src, sq, compose)
    pert_t = perturb_ds2(t_src, st, compose)
    verify_ds2(orig_q, pert_q, orig_t, pert_t,
               os.path.join(out_dir, "verify_ds2.png"), qid, tid)


def _orig_path(args, kind, iid):
    """Look the ORIGINAL (clean) source path up from the internal_val CSVs."""
    sd = args.splits_dir
    if kind == "query":
        df = pd.read_csv(os.path.join(sd, "internal_val_queries.csv"))
        return df.loc[df["query_id"].astype(str) == iid, "query_image"].iloc[0]
    df = pd.read_csv(os.path.join(sd, "internal_val_gallery.csv"))
    return df.loc[df["target_id"].astype(str) == iid, "target_image"].iloc[0]


def run_ds3(args: argparse.Namespace) -> None:
    """Re-save queries intact; resect + shift every target. Write proxy_ds3."""
    sd, out_dir = args.splits_dir, os.path.join(args.splits_dir, "proxy_ds3")
    queries = pd.read_csv(os.path.join(sd, "internal_val_queries.csv"))
    gallery = pd.read_csv(os.path.join(sd, "internal_val_gallery.csv"))
    compose = make_ds3_compose(args)
    rel_prefix = os.path.join(out_dir, "images")

    if args.limit:
        queries, gallery = queries.head(args.limit), gallery.head(args.limit)

    # Queries: geometrically intact, just copied to .nii.gz (same format).
    q_paths = {}
    for _, row in queries.iterrows():
        qid = str(row["query_id"])
        arr, affine = load_volume(resolve(args.data_root, row["query_image"]))
        rel = os.path.join(rel_prefix, f"{qid}.nii.gz")
        save_volume(arr, affine, os.path.join(args.data_root, rel))
        q_paths[qid] = rel
        print(f"  ds3 query={qid} (intact) -> {rel}")

    # Targets: scanner shift + resection.
    g_paths, demo = {}, None
    for _, row in gallery.iterrows():
        tid = str(row["target_id"])
        src = resolve(args.data_root, row["target_image"])
        arr, affine, center = perturb_ds3_target(
            src, stable_seed(tid, args.seed), compose, args.resection_mm
        )
        rel = os.path.join(rel_prefix, f"{tid}.nii.gz")
        save_volume(arr, affine, os.path.join(args.data_root, rel))
        g_paths[tid] = rel
        if demo is None:
            demo = (tid, src, center)  # first target drives the PNG
        print(f"  ds3 target={tid} resect@{center} -> {rel}")

    queries = queries.copy()
    queries["query_image"] = queries["query_id"].astype(str).map(q_paths)
    queries[QUERIES_COLS].to_csv(os.path.join(out_dir, "val_queries.csv"), index=False)
    gallery = gallery.copy()
    gallery["target_image"] = gallery["target_id"].astype(str).map(g_paths)
    gallery[GALLERY_COLS].to_csv(os.path.join(out_dir, "val_gallery.csv"), index=False)

    if demo is not None:
        tid, src, center = demo
        orig_t, _ = load_volume(src)
        pert_t, _ = load_volume(os.path.join(args.data_root, g_paths[tid]))
        verify_ds3(orig_t, pert_t, center,
                   os.path.join(out_dir, "verify_ds3.png"), tid)
    print(f"ds3 proxy written to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=["ds2", "ds3"])
    p.add_argument("--splits-dir", default="splits")
    p.add_argument("--data-root", default=".",
                   help="root that image paths in the CSVs are relative to")
    p.add_argument("--seed", type=int, default=0, help="base seed (mixed per-image)")
    p.add_argument("--limit", type=int, default=0,
                   help="process only the first N rows (0 = all; for quick checks)")
    # magnitudes (defaults are a starting point, not a final answer)
    p.add_argument("--rotate-deg", type=float, default=DEFAULTS["rotate_deg"])
    p.add_argument("--translate", type=float, default=DEFAULTS["translate"])
    p.add_argument("--elastic-sigma", type=float, nargs=2,
                   default=DEFAULTS["elastic_sigma"])
    p.add_argument("--elastic-magnitude", type=float, nargs=2,
                   default=DEFAULTS["elastic_magnitude"])
    p.add_argument("--resection-mm", type=float, default=DEFAULTS["resection_mm"])
    p.add_argument("--bias-coeff", type=float, default=DEFAULTS["bias_coeff"])
    p.add_argument("--gamma", type=float, nargs=2, default=DEFAULTS["gamma"])
    p.add_argument("--intensity-scale", type=float,
                   default=DEFAULTS["intensity_scale"])
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    if args.mode == "ds2":
        run_ds2(args)
    else:
        run_ds3(args)


if __name__ == "__main__":
    main()
