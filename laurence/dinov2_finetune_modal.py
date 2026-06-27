"""DINOv2 + contrastive fine-tuning cross-modal MRI retrieval baseline on Modal.

Pipeline:
  1. Load pretrained DINOv2-base (frozen backbone).
  2. Extract CLS-token features for every training/val/test image once.
  3. Train two small linear projection heads (one for T1 queries, one for T2
     targets) with InfoNCE loss on the 350 dataset1 training pairs.
  4. At inference, project all image features through the appropriate head and
     rank gallery targets by cosine similarity to each query.

For inference, slices are extracted from all three anatomical planes (axial,
coronal, sagittal) and their DINOv2 features are averaged, giving richer
coverage than axial-only extraction.

Run with:
    modal run laurence/dinov2_finetune_modal.py

Custom output path:
    modal run laurence/dinov2_finetune_modal.py --out my_submission.csv
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import modal

app = modal.App("ehl-dinov2-finetune")
vol = modal.Volume.from_name("ehl-2026-vol-2")

MODEL_ID = "facebook/dinov2-base"
BACKBONE_DIM = 768
PROJ_DIM = 256          # projection head output dimension
TEMPERATURE = 0.1       # InfoNCE temperature; slightly above 0.07 helps small datasets
EPOCHS = 150
LR = 3e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64         # for projection-head training (no backbone grad, so very fast)
SLICE_POSITIONS = (0.35, 0.50, 0.65)
IMAGE_SIZE = 224
EMBED_BATCH = 64        # DINOv2 inference batch size


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


# ---------------------------------------------------------------------------
# Remote function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="T4",
    timeout=7200,
    memory=16384,
)
def run_dinov2_finetune() -> str:
    """Fine-tune projection heads on dataset1 pairs, rank all query/gallery sets."""
    import csv as _csv
    import io as _io
    from pathlib import Path as _Path

    import nibabel as nib
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image as PILImage
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoImageProcessor, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- data root -------------------------------------------------------
    data_root = _find_data_root(_Path("/data"))

    # -- helpers ---------------------------------------------------------
    def read_csv_rows(path: _Path) -> list[dict[str, str]]:
        with path.open(newline="") as f:
            return list(_csv.DictReader(f))

    def resolve(rel: str) -> _Path:
        p = _Path(rel)
        p = p if p.is_absolute() else data_root / p
        if not p.exists() and p.suffix == ".gz":
            alt = p.with_suffix("")
            if alt.exists():
                return alt
        return p

    def load_pil_slices(nii_path: _Path, all_planes: bool = False) -> list:
        """Return PIL RGB slices from a NIfTI volume.

        all_planes=False → 3 axial slices
        all_planes=True  → 9 slices (3 per anatomical plane)
        """
        img = nib.load(str(nii_path))
        vol_data = img.get_fdata(dtype=np.float32)
        if vol_data.ndim == 4:
            vol_data = vol_data[..., 0]

        axes = [0, 1, 2] if all_planes else [2]
        pils = []
        for axis in axes:
            arr = np.moveaxis(vol_data, axis, -1)  # target axis becomes last
            nz = np.count_nonzero(np.isfinite(arr) & (arr != 0), axis=(0, 1))
            occupied = np.where(nz > 0)[0]
            a_min = int(occupied[0]) if len(occupied) else 0
            a_max = int(occupied[-1]) if len(occupied) else arr.shape[2] - 1
            for pos in SLICE_POSITIONS:
                idx = int(np.clip(round(a_min + pos * (a_max - a_min)), 0, arr.shape[2] - 1))
                sl = np.nan_to_num(arr[:, :, idx], nan=0.0, posinf=0.0, neginf=0.0)
                lo, hi = np.percentile(sl[sl > 0], (1, 99)) if sl.any() else (0.0, 1.0)
                sl = np.clip((sl - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
                pil = PILImage.fromarray(sl).resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR).convert("RGB")
                pils.append(pil)
        return pils

    # -- DINOv2 backbone (frozen) ----------------------------------------
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    backbone = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def embed_images(image_paths: dict[str, _Path], all_planes: bool) -> dict[str, np.ndarray]:
        """Return {id: (BACKBONE_DIM,) float32 unit vector} for every image."""
        ids = sorted(image_paths)
        all_pils: list = []
        slice_owners: list[str] = []
        print(f"  Loading {len(ids)} volumes (all_planes={all_planes})...")
        for i, img_id in enumerate(ids):
            if i % 50 == 0:
                print(f"    {i}/{len(ids)}")
            try:
                pils = load_pil_slices(image_paths[img_id], all_planes=all_planes)
            except Exception as e:
                print(f"    ERROR {image_paths[img_id]}: {e}")
                n = 9 if all_planes else 3
                pils = [PILImage.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))] * n
            all_pils.extend(pils)
            slice_owners.extend([img_id] * len(pils))

        print(f"  Embedding {len(all_pils)} slices...")
        all_cls: list[np.ndarray] = []
        for start in range(0, len(all_pils), EMBED_BATCH):
            batch = all_pils[start : start + EMBED_BATCH]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            out = backbone(**inputs)
            cls = out.last_hidden_state[:, 0, :].cpu().float().numpy()
            all_cls.append(cls)

        all_cls_np = np.concatenate(all_cls, axis=0)  # (N_slices, 768)

        # average slices belonging to the same volume
        accum: dict[str, list[np.ndarray]] = {img_id: [] for img_id in ids}
        for feat, img_id in zip(all_cls_np, slice_owners):
            accum[img_id].append(feat)

        features: dict[str, np.ndarray] = {}
        for img_id, feats in accum.items():
            f = np.mean(feats, axis=0).astype(np.float32)
            norm = np.linalg.norm(f)
            features[img_id] = f / norm if norm > 0 else f
        return features

    # -- projection heads ------------------------------------------------
    class ProjectionHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(BACKBONE_DIM, BACKBONE_DIM),
                nn.GELU(),
                nn.Linear(BACKBONE_DIM, PROJ_DIM),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.normalize(self.net(x), dim=-1)

    query_head = ProjectionHead().to(device)
    target_head = ProjectionHead().to(device)

    # -- load training pairs ---------------------------------------------
    train_csv = data_root / "dataset1" / "train_pairs.csv"
    train_pairs = read_csv_rows(train_csv) if train_csv.exists() else []
    print(f"Training pairs: {len(train_pairs)}")

    train_query_paths: dict[str, _Path] = {}
    train_target_paths: dict[str, _Path] = {}
    valid_pairs: list[tuple[str, str]] = []
    for row in train_pairs:
        qp = resolve(row["query_image"])
        tp = resolve(row["target_image"])
        if qp.exists() and tp.exists():
            train_query_paths[row["query_id"]] = qp
            train_target_paths[row["target_id"]] = tp
            valid_pairs.append((row["query_id"], row["target_id"]))
    print(f"Valid training pairs on disk: {len(valid_pairs)}")

    # -- collect all inference images ------------------------------------
    prediction_specs = [
        ("dataset1", "val"),
        ("dataset1", "test"),
        ("dataset2", "val"),
        ("dataset2", "test"),
        ("dataset3", "val"),
        ("dataset3", "test"),
    ]
    infer_query_paths: dict[str, _Path] = {}
    infer_target_paths: dict[str, _Path] = {}
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
                infer_query_paths[row["query_id"]] = p
            else:
                print(f"  Missing query: {p}")
        targets: dict[str, _Path] = {}
        for row in read_csv_rows(gcsv):
            p = resolve(row["target_image"])
            if p.exists():
                targets[row["target_id"]] = p
                infer_target_paths[row["target_id"]] = p
            else:
                print(f"  Missing target: {p}")
        if queries and targets:
            prediction_sets.append({"queries": queries, "targets": targets})
            print(f"{ds}/{split}: {len(queries)} queries, {len(targets)} targets")

    # -- stage 1: extract frozen features --------------------------------
    print("\n=== Stage 1: Extracting frozen DINOv2 features ===")

    # Training images: axial-only (faster, sufficient for learning projections)
    print("Training images (axial):")
    train_feats = embed_images({**train_query_paths, **train_target_paths}, all_planes=False)

    # Inference images: all 3 planes (richer at inference time)
    print("Query inference images (all planes):")
    infer_query_feats = embed_images(infer_query_paths, all_planes=True)
    print("Target inference images (all planes):")
    infer_target_feats = embed_images(infer_target_paths, all_planes=True)

    # -- stage 2: train projection heads ---------------------------------
    print("\n=== Stage 2: Training projection heads ===")

    # Build training tensors from valid pairs
    q_vecs = np.stack([train_feats[qid] for qid, _ in valid_pairs])  # (N, 768)
    t_vecs = np.stack([train_feats[tid] for _, tid in valid_pairs])  # (N, 768)
    q_tensor = torch.from_numpy(q_vecs).to(device)
    t_tensor = torch.from_numpy(t_vecs).to(device)

    optimizer = torch.optim.AdamW(
        list(query_head.parameters()) + list(target_head.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    dataset = TensorDataset(q_tensor, t_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    query_head.train()
    target_head.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        n_seen = 0
        for q_batch, t_batch in loader:
            q_emb = query_head(q_batch)   # (B, PROJ_DIM)
            t_emb = target_head(t_batch)  # (B, PROJ_DIM)
            logits = q_emb @ t_emb.T / TEMPERATURE  # (B, B)
            labels = torch.arange(len(q_emb), device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(query_head.parameters()) + list(target_head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            total_loss += loss.item() * len(q_emb)
            n_seen += len(q_emb)
        scheduler.step()
        if epoch % 25 == 0 or epoch == 1:
            print(f"  epoch {epoch:03d}/{EPOCHS}  loss={total_loss / n_seen:.4f}")

    # -- stage 3: project inference features & rank ----------------------
    print("\n=== Stage 3: Projecting and ranking ===")
    query_head.eval()
    target_head.eval()

    @torch.no_grad()
    def project(feats: dict[str, np.ndarray], head: nn.Module) -> dict[str, np.ndarray]:
        ids = sorted(feats)
        vecs = torch.from_numpy(np.stack([feats[i] for i in ids])).to(device)
        embs = head(vecs).cpu().numpy()
        return {img_id: embs[j] for j, img_id in enumerate(ids)}

    proj_query = project(infer_query_feats, query_head)
    proj_target = project(infer_target_feats, target_head)

    submission_rows: list[dict[str, str]] = []
    for pred_set in prediction_sets:
        query_ids = [qid for qid in sorted(pred_set["queries"]) if qid in proj_query]
        target_ids = [tid for tid in sorted(pred_set["targets"]) if tid in proj_target]
        if not query_ids or not target_ids:
            continue
        Q = np.stack([proj_query[qid] for qid in query_ids])
        T = np.stack([proj_target[tid] for tid in target_ids])
        scores = Q @ T.T
        for i, qid in enumerate(query_ids):
            ranked = [target_ids[j] for j in np.argsort(-scores[i])]
            submission_rows.append({"query_id": qid, "target_id_ranking": " ".join(ranked)})

    buf = _io.StringIO()
    writer = _csv.DictWriter(buf, fieldnames=["query_id", "target_id_ranking"])
    writer.writeheader()
    writer.writerows(submission_rows)
    print(f"\nGenerated {len(submission_rows)} submission rows.")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Helpers (also used locally for type-checking)
# ---------------------------------------------------------------------------

def _find_data_root(mount: Path) -> Path:
    for p in sorted(mount.rglob("dataset1")):
        if p.is_dir():
            found = p.parent
            print(f"Found data root: {found}")
            return found
    raise RuntimeError(
        f"Could not find dataset1/ under {mount}. "
        f"Top-level: {list(mount.iterdir())}"
    )


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(out: str = "dinov2_finetune_submission.csv") -> None:
    """Run the fine-tuned DINOv2 baseline on Modal and save the CSV locally."""
    print("Running DINOv2 fine-tune baseline on Modal...")
    csv_content = run_dinov2_finetune.remote()
    out_path = Path(out)
    out_path.write_text(csv_content, encoding="utf-8")
    print(f"Saved {len(csv_content.splitlines()) - 1} rows to {out_path}")
