"""Lightweight NMI cross-modal MRI retrieval on Modal.

Algorithm (pure numpy, no registration library):
  1. Load every unique volume once with ThreadPoolExecutor.
  2. Downsample to ~32 voxels per dimension (stride).
  3. Translate each volume so its foreground centre-of-mass sits at the image
     centre.  This corrects the dominant translation component of ds2's rigid
     deformations without any iterative optimisation.
  4. Compute NMI pairwise from 2-D joint histograms (32 bins).
     - ds1/ds2: restrict to the non-zero union mask of each pair.
     - ds3:     restrict to the non-zero intersection mask so the surgical
                cavity (signal=0) doesn't drag NMI down.

Dependencies: nibabel, numpy only.  No SimpleITK.
Expected runtime on cpu=8: ~45-60 s total.

Run:
    modal run laurence/mattes_mi_modal.py
    modal run laurence/mattes_mi_modal.py --out my_submission.csv
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import modal

app = modal.App("ehl-mattes-mi")
vol = modal.Volume.from_name("ehl-2026-vol-2")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "nibabel>=5.3",
        "numpy>=2.0",
    )
)

TARGET_DIM = 32   # downsample each axis to at most this many voxels
NMI_BINS   = 32   # histogram bins per axis


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=600,
    memory=8192,
    cpu=8,
)
def run_mattes_mi() -> str:
    import csv as _csv
    import io as _io
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path as _Path

    import nibabel as nib
    import numpy as np

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
    # Load, downsample, and centre-of-mass align all unique volumes
    # ----------------------------------------------------------------

    all_paths: dict[str, _Path] = {}
    for ps in prediction_sets:
        all_paths.update(ps["queries"])
        all_paths.update(ps["targets"])

    print(f"\nLoading {len(all_paths)} unique volumes...")

    def _load(item: tuple) -> tuple[str, np.ndarray]:
        img_id, path = item
        arr = nib.load(str(path)).get_fdata(dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Stride downsample then crop/pad to exactly (TARGET_DIM,) per axis.
        # Volumes across datasets have different shapes, so we normalise here
        # to guarantee all cached arrays have the same flat length for NMI.
        steps = tuple(max(1, s // TARGET_DIM) for s in arr.shape)
        arr = arr[::steps[0], ::steps[1], ::steps[2]]
        out = np.zeros((TARGET_DIM, TARGET_DIM, TARGET_DIM), dtype=np.float32)
        s = tuple(min(arr.shape[i], TARGET_DIM) for i in range(3))
        out[:s[0], :s[1], :s[2]] = arr[:s[0], :s[1], :s[2]]
        arr = out

        # Translate so the foreground centre-of-mass sits at the image centre.
        # Corrects the dominant translation component of ds2 rigid deformations.
        fg = arr > 0
        if fg.any():
            cm     = np.argwhere(fg).mean(axis=0)
            centre = np.array(arr.shape) / 2.0
            shift  = np.round(centre - cm).astype(int)
            arr = np.roll(arr, (shift[0], shift[1], shift[2]), axis=(0, 1, 2))

        return img_id, arr

    vol_cache: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for img_id, arr in ex.map(_load, sorted(all_paths.items())):
            vol_cache[img_id] = arr

    example = next(iter(vol_cache.values()))
    print(f"Loaded {len(vol_cache)} volumes. Downsampled shape: {example.shape}")

    # ----------------------------------------------------------------
    # NMI scoring
    # ----------------------------------------------------------------

    def nmi(a: np.ndarray, b: np.ndarray, intersect_mask: bool = False) -> float:
        """Normalised mutual information from a 2-D joint histogram."""
        a_f = a.ravel(); b_f = b.ravel()
        if intersect_mask:
            mask = (a_f != 0) & (b_f != 0)   # ds3: ignore zero-signal cavity
        else:
            mask = (a_f != 0) | (b_f != 0)   # ds1/ds2: use any foreground
        if mask.sum() < 50:
            return 0.0
        h, _, _ = np.histogram2d(a_f[mask], b_f[mask], bins=NMI_BINS)
        h /= h.sum() + 1e-10
        pa  = h.sum(axis=1); pb = h.sum(axis=0)
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
        use_intersect = ds == "dataset3"

        q_ids = sorted(k for k in ps["queries"] if k in vol_cache)
        t_ids = sorted(k for k in ps["targets"] if k in vol_cache)
        nq, nt = len(q_ids), len(t_ids)
        print(f"{ds}/{split}: {nq}×{nt} pairs")

        scores = np.zeros((nq, nt), dtype=np.float32)
        for i, qid in enumerate(q_ids):
            if i % 20 == 0:
                print(f"  row {i}/{nq}")
            q_arr = vol_cache[qid]
            for j, tid in enumerate(t_ids):
                scores[i, j] = nmi(q_arr, vol_cache[tid], intersect_mask=use_intersect)

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
def main(out: str = "mattes_mi_submission.csv") -> None:
    print("Running lightweight NMI retrieval on Modal...")
    csv_content = run_mattes_mi.remote()
    out_path = Path(out)
    out_path.write_text(csv_content, encoding="utf-8")
    print(f"Saved {len(csv_content.splitlines()) - 1} rows to {out_path}")
