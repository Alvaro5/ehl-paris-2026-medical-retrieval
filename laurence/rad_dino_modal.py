"""RadDINO + augmented contrastive fine-tuning + alpha-QE re-ranking on Modal.

Three improvements over hybrid_modal.py:

  1. RadDINO (microsoft/rad-dino) — same ViT-B architecture as DINOv2 but
     pre-trained on 1.6M radiology images with DINO objective.  Much better
     starting representations for MRI than natural-image DINOv2.

  2. Augmented training — for each training image, K independent augmented
     versions are pre-computed through the frozen backbone (random flips,
     90°-multiples rotation, intensity jitter applied to the 2D slices before
     passing through RadDINO).  Each (aug_query, aug_target) combination becomes
     a valid positive pair.  This multiplies the effective training set by K²
     and teaches the projection heads to produce geometry-invariant embeddings,
     directly addressing dataset2's independent deformations.

  3. Alpha-QE re-ranking — after initial ranking, each query embedding is
     expanded with the mean of its top-k matched target embeddings.  The
     expanded (re-normalised) query is used for a final re-rank.  Free
     accuracy gain on top of any embedding.

  Dataset1 still gets the 3D NCC + DINOv2 ensemble.

Run:
    modal run laurence/rad_dino_modal.py
    modal run laurence/rad_dino_modal.py --out my_submission.csv
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import modal

app = modal.App("ehl-rad-dino")
vol = modal.Volume.from_name("ehl-2026-vol-2")

MODEL_ID = "microsoft/rad-dino"
BACKBONE_DIM = 768

# projection head
PROJ_DIM = 256
TEMPERATURE = 0.1
EPOCHS = 150
LR = 3e-3
WEIGHT_DECAY = 1e-4
TRAIN_BATCH = 64
EMBED_BATCH = 48

# augmentation
K_AUG = 4          # independent augmented views per training image

# alpha-QE re-ranking
AQE_K = 5          # top-k neighbours used to expand the query
AQE_ALPHA = 0.5    # weight of original query vs neighbour mean

# 3D NCC (dataset1 only)
NCC_SIZE = (48, 48, 48)
NCC_WEIGHT = 0.7
DINO_WEIGHT = 0.3

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
def run_rad_dino() -> str:
    import csv as _csv
    import io as _io
    import random as _random
    from pathlib import Path as _Path

    import nibabel as nib
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
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
        if len(out) < len(id_path):
            print(f"  {label}: {len(id_path) - len(out)} paths missing")
        return out

    # ------------------------------------------------------------------ slice loader

    def volume_to_pils(nii_path: _Path, all_planes: bool) -> list:
        """Load NIfTI and return PIL RGB slices."""
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

    def augment_pil(pil: PILImage.Image) -> PILImage.Image:
        """Random flip, 90°-multiple rotation, and intensity jitter."""
        if _random.random() > 0.5:
            pil = TF.hflip(pil)
        if _random.random() > 0.5:
            pil = TF.vflip(pil)
        pil = TF.rotate(pil, _random.choice([0, 90, 180, 270]))
        pil = TF.adjust_brightness(pil, _random.uniform(0.75, 1.25))
        pil = TF.adjust_contrast(pil, _random.uniform(0.75, 1.25))
        return pil

    # ------------------------------------------------------------------ backbone

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    backbone = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def embed_pils(pils: list, owners: list[str]) -> dict[str, np.ndarray]:
        """Run RadDINO on a flat list of PIL images, average per owner id."""
        all_cls: list[np.ndarray] = []
        for start in range(0, len(pils), EMBED_BATCH):
            batch = pils[start : start + EMBED_BATCH]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            out = backbone(**inputs)
            all_cls.append(out.last_hidden_state[:, 0, :].cpu().float().numpy())
        all_cls_np = np.concatenate(all_cls, axis=0)

        accum: dict[str, list[np.ndarray]] = {}
        for feat, owner in zip(all_cls_np, owners):
            accum.setdefault(owner, []).append(feat)

        features: dict[str, np.ndarray] = {}
        for img_id, feats in accum.items():
            f = np.mean(feats, axis=0).astype(np.float32)
            norm = np.linalg.norm(f)
            features[img_id] = f / norm if norm > 0 else f
        return features

    def extract_features(
        id_path: dict[str, _Path], label: str, all_planes: bool, n_aug: int = 1
    ) -> dict[str, np.ndarray]:
        """
        Extract and average RadDINO CLS features.
        n_aug=1 → no augmentation (inference mode).
        n_aug>1 → n_aug independent augmented passes averaged (training mode).
        """
        ids = sorted(id_path)
        all_pils: list = []
        owners: list[str] = []
        print(f"  Slicing {len(ids)} {label} volumes (planes={'all' if all_planes else 'axial'}, aug={n_aug})...")
        for i, img_id in enumerate(ids):
            if i % 50 == 0:
                print(f"    {i}/{len(ids)}")
            try:
                base_pils = volume_to_pils(id_path[img_id], all_planes=all_planes)
            except Exception as e:
                print(f"    ERROR {id_path[img_id]}: {e}")
                n = (9 if all_planes else 3) * n_aug
                base_pils = [PILImage.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))] * n
                all_pils.extend(base_pils)
                owners.extend([img_id] * n)
                continue

            for _ in range(n_aug):
                for pil in base_pils:
                    all_pils.append(augment_pil(pil) if n_aug > 1 else pil)
                    owners.append(img_id)

        print(f"  Embedding {len(all_pils)} slices through RadDINO...")
        return embed_pils(all_pils, owners)

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

    # ------------------------------------------------------------------ stage 1: 3D NCC (dataset1)

    print("\n=== Stage 1: 3D NCC for dataset1 ===")

    d1_query_paths: dict[str, _Path] = {}
    d1_target_paths: dict[str, _Path] = {}
    for ps in prediction_sets:
        if ps["ds"] == "dataset1":
            d1_query_paths.update(ps["queries"])
            d1_target_paths.update(ps["targets"])

    def load_ncc_vecs(id_path: dict[str, _Path], label: str) -> tuple[list[str], np.ndarray]:
        ids = sorted(id_path)
        vecs = []
        for i, img_id in enumerate(ids):
            if i % 20 == 0:
                print(f"  {label} {i}/{len(ids)}")
            try:
                img = nib.load(str(id_path[img_id]))
                arr = img.get_fdata(dtype=np.float32)
                if arr.ndim == 4:
                    arr = arr[..., 0]
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                t = F.interpolate(t, size=NCC_SIZE, mode="trilinear", align_corners=False)
                v = t.squeeze().numpy().flatten().astype(np.float32)
            except Exception as e:
                print(f"  ERROR {id_path[img_id]}: {e}")
                v = np.zeros(int(np.prod(NCC_SIZE)), dtype=np.float32)
            mu, sigma = v.mean(), v.std()
            vecs.append((v - mu) / (sigma + 1e-6))
        return ids, np.stack(vecs)

    q1_ids, Q1_ncc = load_ncc_vecs(d1_query_paths, "dataset1 queries")
    t1_ids, T1_ncc = load_ncc_vecs(d1_target_paths, "dataset1 targets")

    Q1_gpu = torch.from_numpy(Q1_ncc).to(device)
    T1_gpu = torch.from_numpy(T1_ncc).to(device)
    ncc_scores_d1 = (Q1_gpu @ T1_gpu.T / Q1_ncc.shape[1]).cpu().numpy()
    ncc_q_idx = {img_id: i for i, img_id in enumerate(q1_ids)}
    ncc_t_idx = {img_id: i for i, img_id in enumerate(t1_ids)}
    del Q1_gpu, T1_gpu
    print(f"NCC matrix: {ncc_scores_d1.shape}")

    # ------------------------------------------------------------------ stage 2: RadDINO features

    print("\n=== Stage 2: RadDINO feature extraction ===")

    train_csv = data_root / "dataset1" / "train_pairs.csv"
    train_pairs_rows = read_csv_rows(train_csv) if train_csv.exists() else []
    train_q_paths: dict[str, _Path] = {}
    train_t_paths: dict[str, _Path] = {}
    valid_pairs: list[tuple[str, str]] = []
    for row in train_pairs_rows:
        qp, tp = resolve(row["query_image"]), resolve(row["target_image"])
        if qp.exists() and tp.exists():
            train_q_paths[row["query_id"]] = qp
            train_t_paths[row["target_id"]] = tp
            valid_pairs.append((row["query_id"], row["target_id"]))
    print(f"Training pairs: {len(valid_pairs)}")

    # Training: K augmented views per image, axial slices only
    print("Extracting augmented training features...")
    train_q_aug_feats: list[dict[str, np.ndarray]] = []
    train_t_aug_feats: list[dict[str, np.ndarray]] = []
    for k in range(K_AUG):
        print(f"  Augmentation {k + 1}/{K_AUG}")
        _random.seed(k)
        train_q_aug_feats.append(extract_features(train_q_paths, f"train-query-aug{k}", all_planes=False, n_aug=1))
        _random.seed(k + 100)
        train_t_aug_feats.append(extract_features(train_t_paths, f"train-target-aug{k}", all_planes=False, n_aug=1))

    # Inference: no augmentation, all 3 planes
    print("Extracting inference features (all planes, no aug)...")
    infer_q_feats = extract_features(all_query_paths, "queries", all_planes=True, n_aug=1)
    infer_t_feats = extract_features(all_target_paths, "targets", all_planes=True, n_aug=1)

    # ------------------------------------------------------------------ stage 3: projection heads

    print("\n=== Stage 3: Training projection heads ===")

    class ProjectionHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(BACKBONE_DIM, BACKBONE_DIM),
                nn.GELU(),
                nn.LayerNorm(BACKBONE_DIM),
                nn.Linear(BACKBONE_DIM, PROJ_DIM),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.normalize(self.net(x), dim=-1)

    query_head = ProjectionHead().to(device)
    target_head = ProjectionHead().to(device)

    # Build augmented training tensors: all K_AUG combinations of (q_aug_i, t_aug_j)
    q_vecs_list, t_vecs_list = [], []
    for q_feats in train_q_aug_feats:
        for t_feats in train_t_aug_feats:
            for qid, tid in valid_pairs:
                if qid in q_feats and tid in t_feats:
                    q_vecs_list.append(q_feats[qid])
                    t_vecs_list.append(t_feats[tid])

    q_tensor = torch.from_numpy(np.stack(q_vecs_list)).to(device)
    t_tensor = torch.from_numpy(np.stack(t_vecs_list)).to(device)
    print(f"Effective training pairs (with augmentation): {len(q_vecs_list)}")

    loader = DataLoader(TensorDataset(q_tensor, t_tensor), batch_size=TRAIN_BATCH, shuffle=True)
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

    # ------------------------------------------------------------------ stage 4: project + alpha-QE + rank

    print("\n=== Stage 4: Projecting, re-ranking, and generating submission ===")
    query_head.eval(); target_head.eval()

    @torch.no_grad()
    def project(feats: dict[str, np.ndarray], head: nn.Module) -> dict[str, np.ndarray]:
        ids = sorted(feats)
        vecs = torch.from_numpy(np.stack([feats[i] for i in ids])).to(device)
        embs = head(vecs).cpu().numpy()
        return {img_id: embs[j] for j, img_id in enumerate(ids)}

    proj_q = project(infer_q_feats, query_head)
    proj_t = project(infer_t_feats, target_head)

    def alpha_qe(scores: np.ndarray, Q: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Expand each query with mean of its top-k matched targets, then re-rank."""
        new_scores = np.empty_like(scores)
        for i in range(len(Q)):
            top_k = np.argsort(-scores[i])[:AQE_K]
            expanded = AQE_ALPHA * Q[i] + (1 - AQE_ALPHA) * T[top_k].mean(axis=0)
            norm = np.linalg.norm(expanded)
            expanded = expanded / norm if norm > 0 else expanded
            new_scores[i] = expanded @ T.T
        return new_scores

    def minmax_rows(m: np.ndarray) -> np.ndarray:
        lo = m.min(axis=1, keepdims=True)
        hi = m.max(axis=1, keepdims=True)
        return (m - lo) / np.where(hi > lo, hi - lo, 1.0)

    submission_rows: list[dict[str, str]] = []

    for pred_set in prediction_sets:
        ds = pred_set["ds"]
        query_ids = [qid for qid in sorted(pred_set["queries"]) if qid in proj_q]
        target_ids = [tid for tid in sorted(pred_set["targets"]) if tid in proj_t]
        if not query_ids or not target_ids:
            continue

        Q = np.stack([proj_q[qid] for qid in query_ids])
        T = np.stack([proj_t[tid] for tid in target_ids])

        dino_scores = Q @ T.T
        dino_scores = alpha_qe(dino_scores, Q, T)  # alpha-QE re-rank

        if ds == "dataset1":
            q_ncc_idx = [ncc_q_idx[qid] for qid in query_ids]
            t_ncc_idx = [ncc_t_idx[tid] for tid in target_ids]
            ncc_sub = ncc_scores_d1[np.ix_(q_ncc_idx, t_ncc_idx)]
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
def main(out: str = "rad_dino_submission.csv") -> None:
    print("Running RadDINO + alpha-QE baseline on Modal...")
    csv_content = run_rad_dino.remote()
    out_path = Path(out)
    out_path.write_text(csv_content, encoding="utf-8")
    print(f"Saved {len(csv_content.splitlines()) - 1} rows to {out_path}")
