"""DINOv2 cross-modal MRI retrieval baseline running on Modal.

For each volume, extracts 3 representative axial slices, converts them to
3-channel uint8, and passes them through a pretrained DINOv2 ViT-B/14 model.
The CLS-token embeddings are averaged across slices to form a global feature.
Gallery targets are ranked by cosine similarity to the query feature.

DINOv2 weights are baked into the Modal image at build time so they are not
re-downloaded on every run.

Run with:
    modal run laurence/dinov2_baseline_modal.py

Custom output path:
    modal run laurence/dinov2_baseline_modal.py --out my_submission.csv
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import modal

app = modal.App("ehl-dinov2-baseline")
vol = modal.Volume.from_name("ehl-2026-vol-2")

MODEL_ID = "facebook/dinov2-base"
SLICE_POSITIONS = (0.35, 0.50, 0.65)
IMAGE_SIZE = 224  # DINOv2 ViT-B/14 canonical input size
BATCH_SIZE = 32


def _download_model():
    from transformers import AutoImageProcessor, AutoModel
    AutoImageProcessor.from_pretrained(MODEL_ID)
    AutoModel.from_pretrained(MODEL_ID)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.3",
        "torchvision>=0.18",
        "transformers>=4.40",
        "nibabel>=5.3",
        "numpy>=2.0",
        "Pillow>=10.0",
    )
    .run_function(_download_model)
)


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="T4",
    timeout=3600,
    memory=16384,
)
def run_dinov2_baseline() -> str:
    """Embed all images with DINOv2 and return a Kaggle submission CSV string."""
    import csv as _csv
    import io as _io
    from pathlib import Path as _Path

    import nibabel as nib
    import numpy as np
    import torch
    from PIL import Image as PILImage
    from transformers import AutoImageProcessor, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    data_root = _find_data_root(_Path("/data"))

    def read_csv_rows(path: _Path) -> list[dict[str, str]]:
        with path.open(newline="") as f:
            return list(_csv.DictReader(f))

    def resolve(rel: str) -> _Path:
        p = _Path(rel)
        p = p if p.is_absolute() else data_root / p
        if not p.exists() and p.suffix == ".gz":
            nii = p.with_suffix("")
            if nii.exists():
                return nii
        return p

    def load_slices(nii_path: _Path) -> list[np.ndarray]:
        """Return 3 uint8 HWC RGB arrays at representative z positions."""
        img = nib.load(str(nii_path))
        vol_data = img.get_fdata(dtype=np.float32)
        if vol_data.ndim == 4:
            vol_data = vol_data[..., 0]

        nonzero = np.count_nonzero(
            np.isfinite(vol_data) & (vol_data != 0), axis=(0, 1)
        )
        occupied = np.where(nonzero > 0)[0]
        z_min = int(occupied[0]) if len(occupied) else 0
        z_max = int(occupied[-1]) if len(occupied) else vol_data.shape[2] - 1

        slices = []
        for pos in SLICE_POSITIONS:
            z = int(np.clip(round(z_min + pos * (z_max - z_min)), 0, vol_data.shape[2] - 1))
            sl = np.nan_to_num(vol_data[:, :, z], nan=0.0, posinf=0.0, neginf=0.0)
            mn, mx = float(sl.min()), float(sl.max())
            sl = ((sl - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else np.zeros_like(sl, dtype=np.uint8)
            # PIL resize + convert to RGB
            pil = PILImage.fromarray(sl).resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR).convert("RGB")
            slices.append(np.array(pil))
        return slices  # list of 3 x (H, W, 3) uint8

    @torch.no_grad()
    def embed_batch(pil_images: list) -> np.ndarray:
        """Run DINOv2 on a batch of PIL images; return (N, 768) float32."""
        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        outputs = model(**inputs)
        cls_tokens = outputs.last_hidden_state[:, 0, :]  # CLS token
        return cls_tokens.cpu().float().numpy()

    def compute_features(image_paths: dict[str, _Path]) -> dict[str, np.ndarray]:
        """Embed all images; returns {id: (768,) unit vector}."""
        ids = sorted(image_paths)
        # flatten: each volume → 3 slices, track index back to volume
        all_pil: list = []
        slice_to_id: list[str] = []
        print(f"Loading {len(ids)} volumes...")
        for i, img_id in enumerate(ids):
            if i % 50 == 0:
                print(f"  loading {i}/{len(ids)}")
            try:
                slices = load_slices(image_paths[img_id])
            except Exception as e:
                print(f"  ERROR loading {image_paths[img_id]}: {e}")
                slices = [np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)] * 3
            for sl in slices:
                all_pil.append(PILImage.fromarray(sl))
                slice_to_id.append(img_id)

        print(f"Embedding {len(all_pil)} slices in batches of {BATCH_SIZE}...")
        all_embs = []
        for start in range(0, len(all_pil), BATCH_SIZE):
            batch = all_pil[start : start + BATCH_SIZE]
            all_embs.append(embed_batch(batch))
            if start % (BATCH_SIZE * 10) == 0:
                print(f"  embedded {start}/{len(all_pil)} slices")
        all_embs_np = np.concatenate(all_embs, axis=0)  # (N_slices, 768)

        # average slices per volume
        features: dict[str, np.ndarray] = {}
        accum: dict[str, list[np.ndarray]] = {img_id: [] for img_id in ids}
        for emb, img_id in zip(all_embs_np, slice_to_id):
            accum[img_id].append(emb)
        for img_id, embs in accum.items():
            feat = np.mean(embs, axis=0).astype(np.float32)
            norm = np.linalg.norm(feat)
            features[img_id] = feat / norm if norm > 0 else feat
        return features

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
                print(f"  Missing query: {p}")
        targets: dict[str, _Path] = {}
        for row in read_csv_rows(gcsv):
            p = resolve(row["target_image"])
            if p.exists():
                targets[row["target_id"]] = p
                all_image_paths[row["target_id"]] = p
            else:
                print(f"  Missing target: {p}")
        if queries and targets:
            prediction_sets.append({"queries": queries, "targets": targets})
            print(f"{ds}/{split}: {len(queries)} queries, {len(targets)} targets")
        else:
            print(f"Skipping {ds}/{split}: no images found on disk")

    features = compute_features(all_image_paths)

    submission_rows: list[dict[str, str]] = []
    for pred_set in prediction_sets:
        query_ids = [qid for qid in sorted(pred_set["queries"]) if qid in features]
        target_ids = [tid for tid in sorted(pred_set["targets"]) if tid in features]
        if not query_ids or not target_ids:
            continue
        Q = np.stack([features[qid] for qid in query_ids])
        T = np.stack([features[tid] for tid in target_ids])
        scores = Q @ T.T
        for i, qid in enumerate(query_ids):
            ranked = [target_ids[j] for j in np.argsort(-scores[i])]
            submission_rows.append({"query_id": qid, "target_id_ranking": " ".join(ranked)})

    buf = _io.StringIO()
    writer = _csv.DictWriter(buf, fieldnames=["query_id", "target_id_ranking"])
    writer.writeheader()
    writer.writerows(submission_rows)
    print(f"Generated {len(submission_rows)} submission rows.")
    return buf.getvalue()


def _find_data_root(mount: Path) -> Path:
    for p in sorted(mount.rglob("dataset1")):
        if p.is_dir():
            found = p.parent
            print(f"Found data root: {found}")
            return found
    raise RuntimeError(
        f"Could not find dataset1/ anywhere under {mount}. "
        f"Top-level contents: {list(mount.iterdir())}"
    )


@app.local_entrypoint()
def main(out: str = "dinov2_submission.csv") -> None:
    """Call the remote function and save the returned CSV locally."""
    print("Running DINOv2 baseline on Modal...")
    csv_content = run_dinov2_baseline.remote()
    out_path = Path(out)
    out_path.write_text(csv_content, encoding="utf-8")
    print(f"Saved {len(csv_content.splitlines()) - 1} rows to {out_path}")
