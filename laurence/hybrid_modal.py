"""Hybrid cross-modal MRI retrieval baseline on Modal.

Strategy per dataset:
  dataset1 — all images share a common registered grid, so we exploit this
              with 3D normalised cross-correlation (NCC) on GPU-downsampled
              volumes.  NCC scores are ensembled with DINOv2 cosine similarity
              from fine-tuned projection heads.
  dataset2/3 — geometric alignment is broken (independent deformations /
              pre-op to intra-op shift), so we rely on DINOv2 fine-tuned
              projection heads trained on the 350 dataset1 pairs.

Run:
    modal run laurence/hybrid_modal.py
    modal run laurence/hybrid_modal.py --out my_submission.csv
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import modal

app = modal.App("ehl-hybrid")
vol = modal.Volume.from_name("ehl-2026-vol-2")

MODEL_ID = "facebook/dinov2-base"
BACKBONE_DIM = 768

# NCC config (dataset1 only)
NCC_SIZE = (48, 48, 48)   # downsample resolution for 3D NCC
NCC_WEIGHT = 0.7           # ensemble weight for NCC score
DINO_WEIGHT = 0.3          # ensemble weight for DINOv2 score

# DINOv2 fine-tune config
PROJ_DIM = 256
TEMPERATURE = 0.1
EPOCHS = 150
LR = 3e-3
WEIGHT_DECAY = 1e-4
TRAIN_BATCH = 64
EMBED_BATCH = 48

# Slice extraction
SLICE_POSITIONS = (0.35, 0.50, 0.65)
IMAGE_SIZE = 224


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
    timeout=7200,
    memory=16384,
)
def run_hybrid() -> str:
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

    # ------------------------------------------------------------------ helpers
    data_root = _find_data_root(_Path("/data"))

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

    def filter_exists(id_path: dict[str, _Path], label: str) -> dict[str, _Path]:
        out = {k: v for k, v in id_path.items() if v.exists()}
        missing = len(id_path) - len(out)
        if missing:
            print(f"  {label}: {missing} missing paths skipped")
        return out

    # ------------------------------------------------------------------ volume loaders

    def load_volume_ncc(nii_path: _Path) -> np.ndarray:
        """Load NIfTI, downsample to NCC_SIZE, flatten, return normalised float32 vector."""
        img = nib.load(str(nii_path))
        arr = img.get_fdata(dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # trilinear downsample via torch
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,X,Y,Z)
        t = F.interpolate(t, size=NCC_SIZE, mode="trilinear", align_corners=False)
        v = t.squeeze().numpy().flatten().astype(np.float32)
        mu, sigma = v.mean(), v.std()
        return (v - mu) / (sigma + 1e-6)

    def load_pil_slices(nii_path: _Path, all_planes: bool = True) -> list:
        """Return PIL RGB slices at representative positions."""
        img = nib.load(str(nii_path))
        arr = img.get_fdata(dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        axes = [0, 1, 2] if all_planes else [2]
        pils = []
        for axis in axes:
            a = np.moveaxis(arr, axis, -1)
            nz = np.count_nonzero(np.isfinite(a) & (a != 0), axis=(0, 1))
            occ = np.where(nz > 0)[0]
            lo = int(occ[0]) if len(occ) else 0
            hi = int(occ[-1]) if len(occ) else a.shape[2] - 1
            for pos in SLICE_POSITIONS:
                idx = int(np.clip(round(lo + pos * (hi - lo)), 0, a.shape[2] - 1))
                sl = np.nan_to_num(a[:, :, idx], nan=0.0, posinf=0.0, neginf=0.0)
                p1, p99 = (np.percentile(sl[sl > 0], (1, 99)) if sl.any() else (0.0, 1.0))
                sl = np.clip((sl - p1) / max(p99 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)
                pil = PILImage.fromarray(sl).resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR).convert("RGB")
                pils.append(pil)
        return pils

    # ------------------------------------------------------------------ collect manifests

    prediction_specs = [
        ("dataset1", "val"),
        ("dataset1", "test"),
        ("dataset2", "val"),
        ("dataset2", "test"),
        ("dataset3", "val"),
        ("dataset3", "test"),
    ]

    all_query_paths: dict[str, _Path] = {}
    all_target_paths: dict[str, _Path] = {}
    prediction_sets: list[dict] = []

    for ds, split in prediction_specs:
        qcsv = data_root / ds / f"{split}_queries.csv"
        gcsv = data_root / ds / f"{split}_gallery.csv"
        if not qcsv.exists() or not gcsv.exists():
            print(f"Skipping {ds}/{split}: CSV not found")
            continue
        queries = filter_exists(
            {r["query_id"]: resolve(r["query_image"]) for r in read_csv_rows(qcsv)},
            f"{ds}/{split} queries",
        )
        targets = filter_exists(
            {r["target_id"]: resolve(r["target_image"]) for r in read_csv_rows(gcsv)},
            f"{ds}/{split} targets",
        )
        if queries and targets:
            prediction_sets.append({"ds": ds, "split": split, "queries": queries, "targets": targets})
            all_query_paths.update(queries)
            all_target_paths.update(targets)
            print(f"{ds}/{split}: {len(queries)} queries, {len(targets)} targets")

    # ------------------------------------------------------------------ stage 1: 3D NCC for dataset1

    print("\n=== Stage 1: 3D NCC for dataset1 ===")

    dataset1_sets = [ps for ps in prediction_sets if ps["ds"] == "dataset1"]

    # collect unique dataset1 image paths
    d1_query_paths: dict[str, _Path] = {}
    d1_target_paths: dict[str, _Path] = {}
    for ps in dataset1_sets:
        d1_query_paths.update(ps["queries"])
        d1_target_paths.update(ps["targets"])

    def load_ncc_matrix(id_path: dict[str, _Path], label: str) -> tuple[list[str], np.ndarray]:
        ids = sorted(id_path)
        vecs = []
        for i, img_id in enumerate(ids):
            if i % 20 == 0:
                print(f"  {label} {i}/{len(ids)}")
            try:
                vecs.append(load_volume_ncc(id_path[img_id]))
            except Exception as e:
                print(f"  ERROR {id_path[img_id]}: {e}")
                vecs.append(np.zeros(int(np.prod(NCC_SIZE)), dtype=np.float32))
        return ids, np.stack(vecs)  # (N, NCC_SIZE^3)

    q1_ids, Q1_ncc = load_ncc_matrix(d1_query_paths, "dataset1 queries")
    t1_ids, T1_ncc = load_ncc_matrix(d1_target_paths, "dataset1 targets")

    # GPU matrix multiply for NCC scores
    Q1_gpu = torch.from_numpy(Q1_ncc).to(device)
    T1_gpu = torch.from_numpy(T1_ncc).to(device)
    ncc_scores_d1 = (Q1_gpu @ T1_gpu.T / Q1_ncc.shape[1]).cpu().numpy()  # (Nq, Nt)
    ncc_q_index = {img_id: i for i, img_id in enumerate(q1_ids)}
    ncc_t_index = {img_id: i for i, img_id in enumerate(t1_ids)}
    del Q1_gpu, T1_gpu
    print(f"NCC matrix: {ncc_scores_d1.shape}, range [{ncc_scores_d1.min():.3f}, {ncc_scores_d1.max():.3f}]")

    # ------------------------------------------------------------------ stage 2: DINOv2 backbone

    print("\n=== Stage 2: DINOv2 frozen feature extraction ===")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    backbone = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def embed_image_paths(id_path: dict[str, _Path], label: str, all_planes: bool) -> dict[str, np.ndarray]:
        ids = sorted(id_path)
        all_pils: list = []
        owners: list[str] = []
        print(f"  Loading slices for {len(ids)} {label} images...")
        for i, img_id in enumerate(ids):
            if i % 50 == 0:
                print(f"    {i}/{len(ids)}")
            try:
                pils = load_pil_slices(id_path[img_id], all_planes=all_planes)
            except Exception as e:
                print(f"    ERROR {id_path[img_id]}: {e}")
                n = 9 if all_planes else 3
                pils = [PILImage.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))] * n
            all_pils.extend(pils)
            owners.extend([img_id] * len(pils))

        print(f"  Embedding {len(all_pils)} slices...")
        cls_list: list[np.ndarray] = []
        for start in range(0, len(all_pils), EMBED_BATCH):
            batch = all_pils[start : start + EMBED_BATCH]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            out = backbone(**inputs)
            cls_list.append(out.last_hidden_state[:, 0, :].cpu().float().numpy())

        all_cls = np.concatenate(cls_list, axis=0)
        accum: dict[str, list[np.ndarray]] = {img_id: [] for img_id in ids}
        for feat, img_id in zip(all_cls, owners):
            accum[img_id].append(feat)

        features: dict[str, np.ndarray] = {}
        for img_id, feats in accum.items():
            f = np.mean(feats, axis=0).astype(np.float32)
            norm = np.linalg.norm(f)
            features[img_id] = f / norm if norm > 0 else f
        return features

    # Training images: axial only (fast)
    train_csv = data_root / "dataset1" / "train_pairs.csv"
    train_pairs = read_csv_rows(train_csv) if train_csv.exists() else []
    train_q_paths: dict[str, _Path] = {}
    train_t_paths: dict[str, _Path] = {}
    valid_pairs: list[tuple[str, str]] = []
    for row in train_pairs:
        qp, tp = resolve(row["query_image"]), resolve(row["target_image"])
        if qp.exists() and tp.exists():
            train_q_paths[row["query_id"]] = qp
            train_t_paths[row["target_id"]] = tp
            valid_pairs.append((row["query_id"], row["target_id"]))
    print(f"Training pairs on disk: {len(valid_pairs)}")

    train_feats = embed_image_paths({**train_q_paths, **train_t_paths}, "train", all_planes=False)

    # Inference images: all 3 planes
    infer_q_feats = embed_image_paths(all_query_paths, "query", all_planes=True)
    infer_t_feats = embed_image_paths(all_target_paths, "target", all_planes=True)

    # ------------------------------------------------------------------ stage 3: fine-tune projections

    print("\n=== Stage 3: Training projection heads ===")

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

    q_vecs = torch.from_numpy(np.stack([train_feats[qid] for qid, _ in valid_pairs])).to(device)
    t_vecs = torch.from_numpy(np.stack([train_feats[tid] for _, tid in valid_pairs])).to(device)
    loader = DataLoader(TensorDataset(q_vecs, t_vecs), batch_size=TRAIN_BATCH, shuffle=True)

    optimizer = torch.optim.AdamW(
        list(query_head.parameters()) + list(target_head.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        query_head.train(); target_head.train()
        total_loss = 0.0; n_seen = 0
        for qb, tb in loader:
            q_emb = query_head(qb)
            t_emb = target_head(tb)
            logits = q_emb @ t_emb.T / TEMPERATURE
            labels = torch.arange(len(q_emb), device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(query_head.parameters()) + list(target_head.parameters()), 1.0
            )
            optimizer.step()
            total_loss += loss.item() * len(qb); n_seen += len(qb)
        scheduler.step()
        if epoch % 25 == 0 or epoch == 1:
            print(f"  epoch {epoch:03d}/{EPOCHS}  loss={total_loss / n_seen:.4f}")

    # ------------------------------------------------------------------ stage 4: project + rank

    print("\n=== Stage 4: Projecting and ranking ===")
    query_head.eval(); target_head.eval()

    @torch.no_grad()
    def project(feats: dict[str, np.ndarray], head: nn.Module) -> dict[str, np.ndarray]:
        ids = sorted(feats)
        vecs = torch.from_numpy(np.stack([feats[i] for i in ids])).to(device)
        embs = head(vecs).cpu().numpy()
        return {img_id: embs[j] for j, img_id in enumerate(ids)}

    proj_q = project(infer_q_feats, query_head)
    proj_t = project(infer_t_feats, target_head)

    submission_rows: list[dict[str, str]] = []

    for pred_set in prediction_sets:
        ds = pred_set["ds"]
        query_ids = [qid for qid in sorted(pred_set["queries"]) if qid in proj_q]
        target_ids = [tid for tid in sorted(pred_set["targets"]) if tid in proj_t]
        if not query_ids or not target_ids:
            continue

        Q = np.stack([proj_q[qid] for qid in query_ids])
        T = np.stack([proj_t[tid] for tid in target_ids])
        dino_scores = Q @ T.T  # (Nq, Nt), cosine similarity

        if ds == "dataset1":
            # Build NCC score sub-matrix for this prediction set
            ncc_q_idx = [ncc_q_index[qid] for qid in query_ids]
            ncc_t_idx = [ncc_t_index[tid] for tid in target_ids]
            ncc_sub = ncc_scores_d1[np.ix_(ncc_q_idx, ncc_t_idx)]

            # Normalise each score matrix to [0,1] per query before ensembling
            def minmax_rows(m: np.ndarray) -> np.ndarray:
                lo = m.min(axis=1, keepdims=True)
                hi = m.max(axis=1, keepdims=True)
                return (m - lo) / np.where(hi > lo, hi - lo, 1.0)

            scores = NCC_WEIGHT * minmax_rows(ncc_sub) + DINO_WEIGHT * minmax_rows(dino_scores)
        else:
            scores = dino_scores

        for i, qid in enumerate(query_ids):
            ranked = [target_ids[j] for j in np.argsort(-scores[i])]
            submission_rows.append({"query_id": qid, "target_id_ranking": " ".join(ranked)})

    buf = _io.StringIO()
    writer = _csv.DictWriter(buf, fieldnames=["query_id", "target_id_ranking"])
    writer.writeheader()
    writer.writerows(submission_rows)
    print(f"\nGenerated {len(submission_rows)} submission rows.")
    return buf.getvalue()


def _find_data_root(mount: Path) -> Path:
    for p in sorted(mount.rglob("dataset1")):
        if p.is_dir():
            found = p.parent
            print(f"Found data root: {found}")
            return found
    raise RuntimeError(f"Could not find dataset1/ under {mount}. Top-level: {list(mount.iterdir())}")


@app.local_entrypoint()
def main(out: str = "hybrid_submission.csv") -> None:
    print("Running hybrid baseline on Modal...")
    csv_content = run_hybrid.remote()
    out_path = Path(out)
    out_path.write_text(csv_content, encoding="utf-8")
    print(f"Saved {len(csv_content.splitlines()) - 1} rows to {out_path}")
