"""Extract per-slice DINOv2 (ViT-B/14) CLS embeddings for all dataset images.

Each image gets a sidecar file saved alongside it on the Modal Volume:
    {stem}_dinov2.f32  –  8-byte header (uint32 nz, uint32 dim=768)
                          followed by nz × 768 float32 L2-normalised row vectors

Run:
    modal run laurence/extract_dinov2_embeddings_modal.py

Re-run and overwrite already-processed files:
    modal run laurence/extract_dinov2_embeddings_modal.py --overwrite

After it finishes, sync the new sidecar files to your local data directory:
    modal volume get ehl-2026-vol-2 / ./sync_tmp --force
    # then copy the *.f32 files from sync_tmp into your local data tree
"""
from __future__ import annotations

import struct
from pathlib import Path

import modal

app = modal.App("ehl-dinov2-embeddings")
vol = modal.Volume.from_name("ehl-2026-vol-2")

MODEL_ID = "facebook/dinov2-base"
IMAGE_SIZE = 224
DIM = 768
BATCH_SIZE = 64   # slices per GPU batch; tune down if T4 OOMs


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


def _find_data_root(mount: Path) -> Path:
    for p in sorted(mount.rglob("dataset1")):
        if p.is_dir():
            return p.parent
    raise RuntimeError(f"dataset1/ not found under {mount}")


def _resolve(data_root: Path, rel: str) -> Path | None:
    p = data_root / rel
    if p.exists():
        return p
    if p.suffix == ".gz":
        nii = p.with_suffix("")
        if nii.exists():
            return nii
    return None


def _sidecar_path(nii_path: Path) -> Path:
    """Return the .f32 sidecar path for a given .nii or .nii.gz path."""
    stem = nii_path.stem
    if stem.endswith(".nii"):       # handles .nii.gz → strip inner .nii
        stem = stem[:-4]
    return nii_path.parent / f"{stem}_dinov2.f32"


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="A100",
    timeout=7200,
    memory=16384,
    max_containers=1,
)
def extract_all_embeddings(overwrite: bool = False) -> dict:
    import csv
    import nibabel as nib
    import numpy as np
    import torch
    from PIL import Image as PILImage
    from transformers import AutoImageProcessor, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    data_root = _find_data_root(Path("/data"))
    print(f"Data root: {data_root}")

    # ── Collect all unique image paths from every CSV ──────────────────────────
    all_paths: dict[str, Path] = {}   # image_id → absolute path

    csv_specs = [
        ("train_pairs.csv",  [("query_id", "query_image"), ("target_id", "target_image")]),
        ("val_queries.csv",  [("query_id", "query_image")]),
        ("val_gallery.csv",  [("target_id", "target_image")]),
        ("test_queries.csv", [("query_id", "query_image")]),
        ("test_gallery.csv", [("target_id", "target_image")]),
    ]

    for ds in ("dataset1", "dataset2", "dataset3"):
        for csv_name, id_path_pairs in csv_specs:
            csv_file = data_root / ds / csv_name
            if not csv_file.exists():
                continue
            with csv_file.open() as f:
                for row in csv.DictReader(f):
                    for id_key, path_key in id_path_pairs:
                        if id_key in row and path_key in row:
                            p = _resolve(data_root, row[path_key])
                            if p:
                                all_paths[row[id_key]] = p

    print(f"Total unique images: {len(all_paths)}")

    # ── Process each image ─────────────────────────────────────────────────────
    saved = skipped = errors = 0

    for img_id, nii_path in sorted(all_paths.items()):
        out_f32 = _sidecar_path(nii_path)

        if not overwrite and out_f32.exists():
            skipped += 1
            continue

        try:
            img = nib.load(str(nii_path))
            data = img.get_fdata(dtype=np.float32)
            if data.ndim == 4:
                data = data[..., 0]
        except Exception as e:
            print(f"  ERROR loading {nii_path}: {e}")
            errors += 1
            continue

        nz_full = data.shape[2]

        # 3 representative slices at 25 / 50 / 75 % depth (mirrors baseline)
        z_indices = [int(nz_full * q) for q in (0.25, 0.50, 0.75)]
        pil_slices = []
        for z in z_indices:
            sl = np.nan_to_num(data[:, :, z], nan=0.0, posinf=0.0, neginf=0.0)
            mn, mx = float(sl.min()), float(sl.max())
            sl_u8 = (
                ((sl - mn) / (mx - mn) * 255).astype(np.uint8)
                if mx > mn else np.zeros_like(sl, dtype=np.uint8)
            )
            pil = PILImage.fromarray(sl_u8).resize(
                (IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR
            ).convert("RGB")
            pil_slices.append(pil)

        # Embed all 3 slices in one batch
        with torch.no_grad():
            inputs = processor(images=pil_slices, return_tensors="pt").to(device)
            cls = model(**inputs).last_hidden_state[:, 0, :].cpu().float().numpy()

        nz = 3
        embs = cls.astype(np.float32)   # (3, 768)

        # L2-normalise rows so cosine sim = dot product
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.where(norms > 0, norms, 1)

        # Write: 8-byte header (num_slices=3, dim as uint32 LE) + raw float32 body
        header = struct.pack("<II", nz, DIM)
        out_f32.write_bytes(header + embs.tobytes())
        saved += 1

        if saved % 20 == 0:
            print(f"  saved {saved}, skipped {skipped}, errors {errors}")
            vol.commit()

    vol.commit()
    result = {"saved": saved, "skipped": skipped, "errors": errors}
    print(f"Done: {result}")
    return result


@app.local_entrypoint()
def main(overwrite: bool = False) -> None:
    result = extract_all_embeddings.remote(overwrite=overwrite)
    print(f"Result: {result}")
    print("\nTo sync sidecar files to local disk:")
    print("  modal volume get ehl-2026-vol-2 / ./sync_tmp --force")
    print("Then copy the _dinov2.f32 files into your local data directory.")
