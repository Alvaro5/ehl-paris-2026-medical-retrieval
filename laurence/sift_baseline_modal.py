"""SIFT-based cross-modal MRI retrieval baseline running on Modal.

For each image, extracts 3 representative 2D slices, runs OpenCV SIFT, and
computes a mean-pooled 128-dim descriptor as the global image feature. Gallery
targets are ranked by cosine similarity to the query feature.

Run with:
    modal run laurence/sift_baseline_modal.py

The submission CSV is written to ./sift_submission.csv by default. Pass a
different path with:
    modal run laurence/sift_baseline_modal.py --out my_submission.csv
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import modal

app = modal.App("ehl-sift-baseline")
vol = modal.Volume.from_name("ehl-2026-vol-2")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "nibabel>=5.3",
        "numpy>=2.0",
        "opencv-python-headless>=4.8",
    )
)

SLICE_POSITIONS = (0.35, 0.50, 0.65)
SIFT_NFEATURES = 500
IMAGE_SIZE = 192  # resize each slice before SIFT


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=3600,
    memory=16384,
    cpu=4,
)
def run_sift_baseline() -> str:
    """Extract SIFT features, rank galleries, return submission CSV as a string."""
    import csv as _csv
    import io as _io
    from pathlib import Path as _Path

    import cv2
    import nibabel as nib
    import numpy as np

    def find_data_root(mount: _Path) -> _Path:
        """Walk the volume to find the directory that contains dataset1/."""
        for p in sorted(mount.rglob("dataset1")):
            if p.is_dir():
                found = p.parent
                print(f"Found data root: {found}")
                return found
        raise RuntimeError(f"Could not find dataset1/ anywhere under {mount}. Volume contents: {list(mount.iterdir())}")

    data_root = find_data_root(_Path("/data"))

    def read_csv_rows(path: _Path) -> list[dict[str, str]]:
        with path.open(newline="") as f:
            return list(_csv.DictReader(f))

    def resolve(rel: str) -> _Path:
        p = _Path(rel)
        p = p if p.is_absolute() else data_root / p
        if not p.exists() and p.suffix == ".gz":
            nii = p.with_suffix("")  # strip .gz → try .nii
            if nii.exists():
                return nii
        return p

    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)

    def compute_feature(nii_path: _Path) -> np.ndarray:
        """Return a normalised mean SIFT descriptor (128,) for one volume."""
        img = nib.load(str(nii_path))
        vol_data = img.get_fdata(dtype=np.float32)
        if vol_data.ndim == 4:
            vol_data = vol_data[..., 0]

        # find occupied z range
        nonzero = np.count_nonzero(
            np.isfinite(vol_data) & (vol_data != 0), axis=(0, 1)
        )
        occupied = np.where(nonzero > 0)[0]
        z_min = int(occupied[0]) if len(occupied) else 0
        z_max = int(occupied[-1]) if len(occupied) else vol_data.shape[2] - 1

        all_descs: list[np.ndarray] = []
        for pos in SLICE_POSITIONS:
            z = int(np.clip(round(z_min + pos * (z_max - z_min)), 0, vol_data.shape[2] - 1))
            sl = np.nan_to_num(vol_data[:, :, z], nan=0.0, posinf=0.0, neginf=0.0)
            mn, mx = sl.min(), sl.max()
            sl = ((sl - mn) / (mx - mn) * 255).astype(np.float32) if mx > mn else np.zeros_like(sl)
            sl_u8 = cv2.resize(sl, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.uint8)
            _, desc = sift.detectAndCompute(sl_u8, None)
            if desc is not None and len(desc) > 0:
                all_descs.append(desc)

        if not all_descs:
            return np.zeros(128, dtype=np.float32)

        feat = np.vstack(all_descs).mean(axis=0).astype(np.float32)
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat

    prediction_specs = [
        ("dataset1", "val"),
        ("dataset1", "test"),
        ("dataset2", "val"),
        ("dataset2", "test"),
        ("dataset3", "val"),
        ("dataset3", "test"),
    ]

    all_image_paths: dict[str, _Path] = {}
    prediction_sets: list[dict] = []

    for ds, split in prediction_specs:
        qcsv = data_root / ds / f"{split}_queries.csv"
        gcsv = data_root / ds / f"{split}_gallery.csv"
        if not qcsv.exists() or not gcsv.exists():
            print(f"Skipping {ds}/{split}: CSV not found")
            continue
        queries: dict[str, _Path] = {}
        for row in read_csv_rows(qcsv):
            p = resolve(row["query_image"])
            if p.exists():
                queries[row["query_id"]] = p
                all_image_paths[row["query_id"]] = p
            else:
                print(f"  Missing query image: {p}")
        targets: dict[str, _Path] = {}
        for row in read_csv_rows(gcsv):
            p = resolve(row["target_image"])
            if p.exists():
                targets[row["target_id"]] = p
                all_image_paths[row["target_id"]] = p
            else:
                print(f"  Missing target image: {p}")
        if queries and targets:
            prediction_sets.append({"queries": queries, "targets": targets})
            print(f"{ds}/{split}: {len(queries)} queries, {len(targets)} targets")
        else:
            print(f"Skipping {ds}/{split}: no images found on disk")

    print(f"Computing SIFT features for {len(all_image_paths)} images...")
    features: dict[str, np.ndarray] = {}
    for i, (img_id, img_path) in enumerate(sorted(all_image_paths.items())):
        if i % 50 == 0:
            print(f"  {i}/{len(all_image_paths)}")
        features[img_id] = compute_feature(img_path)
    print("Done computing features.")

    submission_rows: list[dict[str, str]] = []
    for pred_set in prediction_sets:
        query_ids = [qid for qid in sorted(pred_set["queries"]) if qid in features]
        target_ids = [tid for tid in sorted(pred_set["targets"]) if tid in features]
        if not query_ids or not target_ids:
            continue
        Q = np.stack([features[qid] for qid in query_ids])  # (Nq, 128)
        T = np.stack([features[tid] for tid in target_ids])  # (Nt, 128)
        scores = Q @ T.T  # (Nq, Nt)
        for i, qid in enumerate(query_ids):
            ranked = [target_ids[j] for j in np.argsort(-scores[i])]
            submission_rows.append({"query_id": qid, "target_id_ranking": " ".join(ranked)})

    buf = _io.StringIO()
    writer = _csv.DictWriter(buf, fieldnames=["query_id", "target_id_ranking"])
    writer.writeheader()
    writer.writerows(submission_rows)
    print(f"Generated {len(submission_rows)} submission rows.")
    return buf.getvalue()


@app.local_entrypoint()
def main(out: str = "sift_submission.csv") -> None:
    """Call the remote function and save the returned CSV locally."""
    print("Running SIFT baseline on Modal...")
    csv_content = run_sift_baseline.remote()
    out_path = Path(out)
    out_path.write_text(csv_content, encoding="utf-8")
    print(f"Saved {len(csv_content.splitlines()) - 1} rows to {out_path}")
