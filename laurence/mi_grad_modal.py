"""Gradient-magnitude NMI cross-modal MRI retrieval on Modal.

Key insight over mattes_mi_modal.py (score ~0.52):

  Raw NMI between T1ce and T2 has two problems:
    1. T1/T2 intensity INVERSION: WM is bright in T1ce but dark in T2.
       The joint histogram is anti-correlated, which still has MI, but the
       signal is weaker than expected and confused by tissue-specific exceptions.
    2. ds2 INDEPENDENT DEFORMATIONS: spatial NMI requires alignment.
       CoM translation correction doesn't fix rotations (up to ~30°).

  Gradient magnitude NMI solves both:
    1. Modality-independent: tissue boundaries (∂WM/∂GM etc.) appear at the
       same spatial locations in T1 and T2. ||∇f|| is always non-negative and
       has similar statistical structure regardless of intensity polarity.
    2. Rotation-invariant: ||∇(Rf)|| = ||∇f|| for any rigid rotation R.
       Independently deformed query and target (ds2) still produce similar
       gradient magnitude DISTRIBUTIONS for the same subject.

  Even without spatial alignment, gradient-NMI works as a subject-level
  discriminator: same-subject brains have similar tissue composition →
  similar gradient magnitude distribution → high mutual information.

Algorithm:
  1. Load + trilinear-interpolate to TARGET_DIM³ (48 ³, up from 32³).
  2. Translate so foreground CoM sits at image centre (translation correction).
  3. Compute 3D gradient magnitude: sqrt(gx² + gy² + gz²).
  4. Compute NMI from 2-D joint histogram between gradient magnitudes.
     - ds1: union mask (same as before, but gradients now cross-modal safe)
     - ds2: union mask (gradient NMI is rotation-invariant, so alignment ok)
     - ds3: intersection mask (avoid zero-signal surgical cavity)
  5. For ds1 only: also compute raw intensity NMI and ensemble 50/50 (the
     registered grid makes spatial NMI meaningful there as well).

Run:
    modal run laurence/mi_grad_modal.py
    modal run laurence/mi_grad_modal.py --out my_submission.csv
"""

from __future__ import annotations

import io
import csv
from pathlib import Path

import modal

app = modal.App("ehl-mi-grad")
vol = modal.Volume.from_name("ehl-2026-vol-2")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "nibabel>=5.3",
        "numpy>=2.0",
        "scipy>=1.13",   # ndimage for cleaner gradient
    )
)

TARGET_DIM = 48   # up from 32: more spatial detail, still fast
NMI_BINS   = 48   # bins per axis for joint histogram


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=600,
    memory=16384,
    cpu=8,
)
def run_mi_grad() -> str:
    import csv as _csv
    import io as _io
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path as _Path

    import nibabel as nib
    import numpy as np
    from scipy.ndimage import zoom

    # ---------------------------------------------------------------- helpers

    def find_data_root(mount: _Path) -> _Path:
        for p in sorted(mount.rglob("dataset1")):
            if p.is_dir():
                found = p.parent
                print(f"Data root: {found}")
                return found
        raise RuntimeError(f"No dataset1/ under {mount}")

    def read_csv(path: _Path) -> list[dict[str, str]]:
        with path.open(newline="") as f:
            return list(_csv.DictReader(f))

    def resolve(rel: str, root: _Path) -> _Path:
        p = _Path(rel)
        if not p.is_absolute():
            p = root / p
        if not p.exists() and p.suffix == ".gz":
            alt = p.with_suffix("")
            if alt.exists():
                return alt
        return p

    data_root = find_data_root(_Path("/data"))

    # ---------------------------------------------------------------- manifests

    SPECS = [
        ("dataset1", "val"),
        ("dataset1", "test"),
        ("dataset2", "val"),
        ("dataset2", "test"),
        ("dataset3", "val"),
        ("dataset3", "test"),
    ]

    prediction_sets: list[dict] = []
    for ds, split in SPECS:
        qcsv = data_root / ds / f"{split}_queries.csv"
        gcsv = data_root / ds / f"{split}_gallery.csv"
        if not qcsv.exists() or not gcsv.exists():
            print(f"Skipping {ds}/{split}: CSV missing")
            continue
        queries = {r["query_id"]: resolve(r["query_image"], data_root) for r in read_csv(qcsv)}
        targets = {r["target_id"]: resolve(r["target_image"], data_root) for r in read_csv(gcsv)}
        queries = {k: v for k, v in queries.items() if v.exists()}
        targets = {k: v for k, v in targets.items() if v.exists()}
        if queries and targets:
            prediction_sets.append({"ds": ds, "split": split, "queries": queries, "targets": targets})
            print(f"  {ds}/{split}: {len(queries)} queries, {len(targets)} targets")

    # ----------------------------------------------------------------
    # Load, downsample, CoM-align, then compute gradient magnitude
    # ----------------------------------------------------------------

    all_paths: dict[str, _Path] = {}
    for ps in prediction_sets:
        all_paths.update(ps["queries"])
        all_paths.update(ps["targets"])

    print(f"\nLoading {len(all_paths)} unique volumes...")

    def _gradient_magnitude(arr: np.ndarray) -> np.ndarray:
        """3-D gradient magnitude using central differences (numpy.gradient)."""
        gx = np.gradient(arr, axis=0)
        gy = np.gradient(arr, axis=1)
        gz = np.gradient(arr, axis=2)
        return np.sqrt(gx * gx + gy * gy + gz * gz).astype(np.float32)

    def _load(item: tuple) -> tuple[str, np.ndarray, np.ndarray]:
        """Returns (img_id, raw_arr, grad_arr) both shaped (TARGET_DIM,)³."""
        img_id, path = item
        arr = nib.load(str(path)).get_fdata(dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Trilinear downsample to TARGET_DIM³ using zoom
        factors = tuple(TARGET_DIM / s for s in arr.shape)
        arr = zoom(arr, factors, order=1, prefilter=False).astype(np.float32)
        # Ensure exact shape (zoom can be off by 1)
        out = np.zeros((TARGET_DIM, TARGET_DIM, TARGET_DIM), dtype=np.float32)
        s = tuple(min(arr.shape[i], TARGET_DIM) for i in range(3))
        out[:s[0], :s[1], :s[2]] = arr[:s[0], :s[1], :s[2]]
        arr = out

        # Translate so the foreground CoM sits at the image centre.
        fg = arr > 0
        if fg.any():
            cm     = np.argwhere(fg).mean(axis=0)
            centre = np.array(arr.shape) / 2.0
            shift  = np.round(centre - cm).astype(int)
            arr    = np.roll(arr, (shift[0], shift[1], shift[2]), axis=(0, 1, 2))

        grad = _gradient_magnitude(arr)
        return img_id, arr, grad

    raw_cache:  dict[str, np.ndarray] = {}
    grad_cache: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for img_id, raw, grad in ex.map(_load, sorted(all_paths.items())):
            raw_cache[img_id]  = raw
            grad_cache[img_id] = grad

    example = next(iter(raw_cache.values()))
    print(f"Loaded {len(raw_cache)} volumes. Downsampled shape: {example.shape}")

    # ----------------------------------------------------------------
    # NMI scoring
    # ----------------------------------------------------------------

    def nmi(a: np.ndarray, b: np.ndarray, intersect_mask: bool = False) -> float:
        """Normalised mutual information from a 2-D joint histogram."""
        a_f = a.ravel()
        b_f = b.ravel()
        if intersect_mask:
            mask = (a_f != 0) & (b_f != 0)
        else:
            mask = (a_f != 0) | (b_f != 0)
        if mask.sum() < 50:
            return 0.0
        h, _, _ = np.histogram2d(a_f[mask], b_f[mask], bins=NMI_BINS)
        h /= h.sum() + 1e-10
        pa  = h.sum(axis=1)
        pb  = h.sum(axis=0)
        ha  = -(pa  * np.log(pa  + 1e-10)).sum()
        hb  = -(pb  * np.log(pb  + 1e-10)).sum()
        hab = -(h   * np.log(h   + 1e-10)).sum()
        v   = float((ha + hb) / (hab + 1e-10))
        return v if np.isfinite(v) else 0.0

    # ---------------------------------------------------------------- rank and write

    print("\n=== Scoring all retrieval pools ===")

    rows: list[dict[str, str]] = []
    for ps in prediction_sets:
        ds    = ps["ds"]
        split = ps["split"]
        # ds3 uses intersection mask: only foreground voxels in BOTH volumes.
        # This prevents the surgical cavity (signal=0) from inflating NMI.
        use_intersect = (ds == "dataset3")
        is_ds1        = (ds == "dataset1")

        q_ids = sorted(k for k in ps["queries"] if k in grad_cache)
        t_ids = sorted(k for k in ps["targets"] if k in grad_cache)
        nq, nt = len(q_ids), len(t_ids)
        print(f"{ds}/{split}: {nq}×{nt} pairs")

        scores = np.zeros((nq, nt), dtype=np.float32)
        for i, qid in enumerate(q_ids):
            if i % 20 == 0:
                print(f"  row {i}/{nq}")
            g_q = grad_cache[qid]
            r_q = raw_cache[qid]
            for j, tid in enumerate(t_ids):
                g_t = grad_cache[tid]
                # Primary: gradient-magnitude NMI (modality-independent, rotation-invariant)
                score = nmi(g_q, g_t, intersect_mask=use_intersect)
                if is_ds1:
                    # dataset1 is spatially registered, so raw intensity NMI also adds signal.
                    # Ensemble 50/50.
                    r_t = raw_cache[tid]
                    raw_score = nmi(r_q, r_t, intersect_mask=False)
                    score = 0.5 * score + 0.5 * raw_score
                scores[i, j] = score

        for i, qid in enumerate(q_ids):
            ranked = [t_ids[j] for j in np.argsort(-scores[i])]
            rows.append({"query_id": qid, "target_id_ranking": " ".join(ranked)})

    buf = _io.StringIO()
    writer = _csv.DictWriter(buf, fieldnames=["query_id", "target_id_ranking"])
    writer.writeheader()
    writer.writerows(rows)
    print(f"\nGenerated {len(rows)} submission rows.")
    return buf.getvalue()


@app.local_entrypoint()
def main(out: str = "mi_grad_submission.csv") -> None:
    print("Running gradient-magnitude NMI retrieval on Modal...")
    csv_content = run_mi_grad.remote()
    out_path = Path(out)
    out_path.write_text(csv_content, encoding="utf-8")
    print(f"Saved {len(csv_content.splitlines()) - 1} rows to {out_path}")
