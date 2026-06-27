"""SigLIP2 cross-modal MRI retrieval on Modal.

Uses google/siglip2-so400m-patch14-384 as the vision backbone instead of RadDINO.

Why SigLIP2 differs from RadDINO:
  • 4.6× larger (400 M vs 86 M params) → more capacity for nuanced cross-modal features.
  • 384 × 384 input vs 224 × 224 → ~2.9× more pixels per slice, better fine detail.
  • 729 patch tokens per slice (27 × 27 at patch-14) vs 196 for ViT-B/16.
  • No explicit CLS token — all tokens are patch tokens; we use mean+max pooling
    concatenation (2304-dim) to obtain a richer per-slice descriptor.
  • Pre-trained with sigmoid image-text contrastive loss (different inductive bias
    from DINO's self-distillation) → orthogonal to RadDINO, good for ensembling.

Pipeline (mirrors strong_modal.py):
  1. 3D NCC score matrix for dataset1 (unchanged).
  2. SigLIP2 frozen backbone — mean+max pool over patch tokens → 2304-dim feature.
  3. Augmented training (K=4 views), axial slices only.
  4. Projection heads (InfoNCE + triplet hard-negative mining, 200 epochs + warmup).
  5. Self-training on dataset2/3 using confident pseudo-pairs.
  6. Inference: 15 slices per volume (5 positions × 3 planes) with TTA × 3.
  7. Re-ranking: alpha-QE → k-reciprocal encoding → ColBERT MaxSim.

Ensemble hint:
  score_ensemble = 0.5 * siglip_scores + 0.5 * strong_scores
  can be computed offline by loading both submission CSVs.

Run:
    modal run laurence/siglip_modal.py
    modal run laurence/siglip_modal.py --out my_submission.csv
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import modal

app = modal.App("ehl-siglip2")
vol = modal.Volume.from_name("ehl-2026-vol-2")

MODEL_ID = "google/siglip2-so400m-patch14-384"
# 1152 mean-pool + 1152 max-pool concatenated
BACKBONE_DIM = 2304

# Projection head
PROJ_DIM = 512
TEMPERATURE = 0.07
EPOCHS = 200
LR = 2e-3
WEIGHT_DECAY = 1e-4
LR_WARMUP_EPOCHS = 10
TRAIN_BATCH = 32           # smaller batch: larger model + 384-px inputs
EMBED_BATCH = 12           # ~400 M params at 384 px needs conservative batch size

# Augmentation
K_AUG = 4

# Slice positions
SLICE_POSITIONS_TRAIN = (0.35, 0.50, 0.65)                        # 3 per axis
SLICE_POSITIONS_INFER = (0.25, 0.375, 0.50, 0.625, 0.75)          # 5 per axis
IMAGE_SIZE = 384           # native SigLIP2 resolution

# TTA
TTA_K = 3

# Hard negative mining
HN_INTERVAL = 10
HN_PER_ANCHOR = 3
TRIPLET_MARGIN = 0.2
TRIPLET_WEIGHT = 0.4

# alpha-QE
AQE_K = 5
AQE_ALPHA = 0.5

# k-reciprocal re-ranking
KRR_K1 = 20
KRR_K2 = 6
KRR_LAMBDA = 0.35

# ColBERT MaxSim re-ranking
COLBERT_TOP_K = 50
COLBERT_WEIGHT = 0.4

# Self-training
ST_CONF_THRESH = 0.65
ST_EPOCHS = 50
ST_WEIGHT = 0.3

# 3D NCC (dataset1 only)
NCC_SIZE = (48, 48, 48)
NCC_WEIGHT = 0.45
SIGLIP_WEIGHT = 0.55       # SigLIP2 richer features → slightly higher weight vs NCC


def _download_model():
    from transformers import AutoProcessor, AutoModel
    AutoProcessor.from_pretrained(MODEL_ID)
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
    gpu="A10G",
    timeout=14400,         # 4 h: larger model + 384-px processing takes longer
    memory=32768,
)
def run_siglip() -> str:
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
    from transformers import AutoProcessor, AutoModel

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
            print(f"  {label}: {len(id_path) - len(out)} missing")
        return out

    # ------------------------------------------------------------------ slice helpers

    def volume_to_pils(nii_path: _Path, positions: tuple) -> list:
        img = nib.load(str(nii_path))
        arr = img.get_fdata(dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        pils = []
        for axis in [0, 1, 2]:
            a = np.moveaxis(arr, axis, -1)
            nz = np.count_nonzero(np.isfinite(a) & (a != 0), axis=(0, 1))
            occ = np.where(nz > 0)[0]
            lo = int(occ[0]) if len(occ) else 0
            hi = int(occ[-1]) if len(occ) else a.shape[2] - 1
            for pos in positions:
                idx = int(np.clip(round(lo + pos * (hi - lo)), 0, a.shape[2] - 1))
                sl = np.nan_to_num(a[:, :, idx], nan=0.0, posinf=0.0, neginf=0.0)
                p1, p99 = (np.percentile(sl[sl > 0], (1, 99)) if sl.any() else (0.0, 1.0))
                sl = np.clip((sl - p1) / max(p99 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)
                pil = PILImage.fromarray(sl).resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR).convert("RGB")
                pils.append(pil)
        return pils

    def augment_pil(pil: PILImage.Image) -> PILImage.Image:
        if _random.random() > 0.5:
            pil = TF.hflip(pil)
        if _random.random() > 0.5:
            pil = TF.vflip(pil)
        pil = TF.rotate(pil, _random.choice([0, 90, 180, 270]))
        pil = TF.adjust_brightness(pil, _random.uniform(0.75, 1.25))
        pil = TF.adjust_contrast(pil, _random.uniform(0.75, 1.25))
        return pil

    # ------------------------------------------------------------------ backbone

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    full_model = AutoModel.from_pretrained(MODEL_ID)
    backbone = full_model.vision_model.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    def _siglip_feat(pixel_values: torch.Tensor) -> torch.Tensor:
        """Mean + max pool over all patch tokens → 2304-dim descriptor."""
        out = backbone(pixel_values=pixel_values)
        hs = out.last_hidden_state          # (B, 729, 1152)
        mean_f = hs.mean(dim=1)             # (B, 1152)
        max_f = hs.max(dim=1).values        # (B, 1152)  — max over spatial positions
        return torch.cat([mean_f, max_f], dim=-1)   # (B, 2304)

    @torch.no_grad()
    def embed_pils(pils: list, owners: list[str]) -> dict[str, np.ndarray]:
        all_feats: list[np.ndarray] = []
        for start in range(0, len(pils), EMBED_BATCH):
            batch = pils[start : start + EMBED_BATCH]
            inputs = processor(images=batch, return_tensors="pt")
            pv = inputs["pixel_values"].to(device)
            feat = _siglip_feat(pv).cpu().float().numpy()
            all_feats.append(feat)
        all_feats_np = np.concatenate(all_feats, axis=0)

        accum: dict[str, list[np.ndarray]] = {}
        for f, owner in zip(all_feats_np, owners):
            accum.setdefault(owner, []).append(f)

        features: dict[str, np.ndarray] = {}
        for img_id, feats in accum.items():
            f = np.mean(feats, axis=0).astype(np.float32)
            norm = np.linalg.norm(f)
            features[img_id] = f / norm if norm > 0 else f
        return features

    def extract_features(
        id_path: dict[str, _Path], label: str, positions: tuple, n_aug: int = 1,
    ) -> dict[str, np.ndarray]:
        ids = sorted(id_path)
        n_slices = len(positions) * 3
        all_pils: list = []
        owners: list[str] = []
        print(f"  Slicing {len(ids)} {label} ({n_slices} slices × {n_aug} aug @ {IMAGE_SIZE}px)...")
        for i, img_id in enumerate(ids):
            if i % 50 == 0:
                print(f"    {i}/{len(ids)}")
            try:
                base_pils = volume_to_pils(id_path[img_id], positions)
            except Exception as e:
                print(f"    ERROR {id_path[img_id]}: {e}")
                base_pils = [PILImage.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))] * n_slices
            for _ in range(n_aug):
                for pil in base_pils:
                    all_pils.append(augment_pil(pil) if n_aug > 1 else pil)
                    owners.append(img_id)
        print(f"  Embedding {len(all_pils)} slices through SigLIP2...")
        return embed_pils(all_pils, owners)

    @torch.no_grad()
    def extract_per_slice(
        id_path: dict[str, _Path], positions: tuple,
    ) -> dict[str, np.ndarray]:
        """Return {img_id: (n_slices, 2304)} backbone features (no averaging)."""
        ids = sorted(id_path)
        n_slices = len(positions) * 3
        all_pils: list = []
        owners: list[str] = []
        for img_id in ids:
            try:
                pils = volume_to_pils(id_path[img_id], positions)
            except Exception:
                pils = [PILImage.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))] * n_slices
            all_pils.extend(pils)
            owners.extend([img_id] * len(pils))

        all_feats: list[np.ndarray] = []
        for start in range(0, len(all_pils), EMBED_BATCH):
            batch = all_pils[start : start + EMBED_BATCH]
            inputs = processor(images=batch, return_tensors="pt")
            pv = inputs["pixel_values"].to(device)
            feat = _siglip_feat(pv).cpu().float().numpy()
            all_feats.append(feat)
        all_feats_np = np.concatenate(all_feats, axis=0)

        accum: dict[str, list[np.ndarray]] = {img_id: [] for img_id in ids}
        for feat, owner in zip(all_feats_np, owners):
            accum[owner].append(feat)
        return {img_id: np.stack(feats).astype(np.float32) for img_id, feats in accum.items()}

    # ------------------------------------------------------------------ projection head

    class ProjectionHead(nn.Module):
        def __init__(self, in_dim: int = BACKBONE_DIM, out_dim: int = PROJ_DIM) -> None:
            super().__init__()
            hidden = 1024
            self.fc1 = nn.Linear(in_dim, hidden)
            self.ln1 = nn.LayerNorm(hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.ln2 = nn.LayerNorm(hidden)
            self.fc3 = nn.Linear(hidden, out_dim)
            self.skip = nn.Linear(in_dim, hidden)
            self.drop = nn.Dropout(0.1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.drop(F.gelu(self.ln1(self.fc1(x))))
            h = self.drop(F.gelu(self.ln2(self.fc2(h) + self.skip(x))))
            return F.normalize(self.fc3(h), dim=-1)

    # ------------------------------------------------------------------ re-ranking

    def alpha_qe(scores: np.ndarray, Q: np.ndarray, T: np.ndarray) -> np.ndarray:
        new_scores = np.empty_like(scores)
        for i in range(len(Q)):
            top_k = np.argsort(-scores[i])[:AQE_K]
            expanded = AQE_ALPHA * Q[i] + (1 - AQE_ALPHA) * T[top_k].mean(axis=0)
            norm = np.linalg.norm(expanded)
            expanded = expanded / norm if norm > 0 else expanded
            new_scores[i] = expanded @ T.T
        return new_scores

    def k_reciprocal_rerank(
        q_scores: np.ndarray, T: np.ndarray,
        k1: int = KRR_K1, k2: int = KRR_K2, lam: float = KRR_LAMBDA,
    ) -> np.ndarray:
        Nq, Nt = q_scores.shape
        t_sims = T @ T.T
        np.fill_diagonal(t_sims, -1.0)
        t_knn = np.argsort(-t_sims, axis=1)[:, :k1]

        t_mem = np.zeros((Nt, Nt), dtype=np.float32)
        t_mem[np.repeat(np.arange(Nt), k1), t_knn.flatten()] = 1.0
        t_counts = t_mem.sum(axis=1)

        final_scores = np.empty_like(q_scores)
        for i in range(Nq):
            q_nn = np.argsort(-q_scores[i])[:k1]
            q_nn_set = set(q_nn.tolist())
            R = [t for t in q_nn if len(set(t_knn[t].tolist()) & q_nn_set) >= k1 / 2]
            if not R:
                R = q_nn[:k2].tolist()
            R_expanded = set(R)
            for t in R:
                R_expanded.update(t_knn[t, :k2].tolist())
            R_vec = np.zeros(Nt, dtype=np.float32)
            R_vec[list(R_expanded)] = 1.0
            inter = t_mem @ R_vec
            union = R_vec.sum() + t_counts - inter
            final_scores[i] = lam * q_scores[i] + (1.0 - lam) * inter / (union + 1e-10)
        return final_scores

    def colbert_rerank(
        initial_scores: np.ndarray,
        q_slice_embs: list[np.ndarray],
        t_slice_embs: list[np.ndarray],
        top_k: int = COLBERT_TOP_K,
        blend: float = COLBERT_WEIGHT,
    ) -> np.ndarray:
        """MaxSim over per-slice projected embeddings (arXiv:2507.17412)."""
        reranked = initial_scores.copy()
        for i in range(len(q_slice_embs)):
            top_k_idx = np.argsort(-initial_scores[i])[:top_k]
            Qs = q_slice_embs[i]
            for j in top_k_idx:
                Ts = t_slice_embs[j]
                maxsim = float((Qs @ Ts.T).max(axis=1).mean())
                reranked[i, j] = (1 - blend) * initial_scores[i, j] + blend * maxsim
        return reranked

    def minmax_rows(m: np.ndarray) -> np.ndarray:
        lo = m.min(axis=1, keepdims=True)
        hi = m.max(axis=1, keepdims=True)
        return (m - lo) / np.where(hi > lo, hi - lo, 1.0)

    # ------------------------------------------------------------------ manifests

    prediction_specs = [
        ("dataset1", "val"), ("dataset1", "test"),
        ("dataset2", "val"), ("dataset2", "test"),
        ("dataset3", "val"), ("dataset3", "test"),
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

    # ------------------------------------------------------------------ stage 2: SigLIP2 features

    print("\n=== Stage 2: SigLIP2 feature extraction ===")

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

    # K_AUG augmented views for training (axial only, faster)
    train_q_aug: list[dict[str, np.ndarray]] = []
    train_t_aug: list[dict[str, np.ndarray]] = []
    for k in range(K_AUG):
        print(f"  Training aug {k + 1}/{K_AUG}")
        _random.seed(k)
        train_q_aug.append(extract_features(train_q_paths, f"train-q-aug{k}", SLICE_POSITIONS_TRAIN))
        _random.seed(k + 100)
        train_t_aug.append(extract_features(train_t_paths, f"train-t-aug{k}", SLICE_POSITIONS_TRAIN))

    # Inference: TTA_K augmented views, 15 slices per volume
    print("Extracting inference features (all planes, 5 positions, TTA)...")
    _random.seed(42)
    infer_q_list = [extract_features(all_query_paths, f"q-tta{k}", SLICE_POSITIONS_INFER) for k in range(TTA_K)]
    _random.seed(200)
    infer_t_list = [extract_features(all_target_paths, f"t-tta{k}", SLICE_POSITIONS_INFER) for k in range(TTA_K)]

    def average_tta(feat_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        avg: dict[str, np.ndarray] = {}
        for img_id in feat_list[0]:
            vecs = np.stack([fl[img_id] for fl in feat_list if img_id in fl])
            f = vecs.mean(axis=0).astype(np.float32)
            norm = np.linalg.norm(f)
            avg[img_id] = f / norm if norm > 0 else f
        return avg

    infer_q_feats = average_tta(infer_q_list)
    infer_t_feats = average_tta(infer_t_list)

    print("Extracting per-slice features for ColBERT re-ranking...")
    per_slice_q = extract_per_slice(all_query_paths, SLICE_POSITIONS_INFER)
    per_slice_t = extract_per_slice(all_target_paths, SLICE_POSITIONS_INFER)

    # ------------------------------------------------------------------ stage 3: projection heads

    print("\n=== Stage 3: Training projection heads ===")

    query_head = ProjectionHead().to(device)
    target_head = ProjectionHead().to(device)

    # All K_AUG × K_AUG augmented pair combinations
    q_vecs_list, t_vecs_list = [], []
    for qf in train_q_aug:
        for tf in train_t_aug:
            for qid, tid in valid_pairs:
                if qid in qf and tid in tf:
                    q_vecs_list.append(qf[qid])
                    t_vecs_list.append(tf[tid])

    q_base = torch.from_numpy(np.stack(q_vecs_list)).to(device)
    t_base = torch.from_numpy(np.stack(t_vecs_list)).to(device)
    print(f"Augmented training pairs: {len(q_vecs_list)}")

    def make_loader(q_t: torch.Tensor, t_t: torch.Tensor, bs: int = TRAIN_BATCH) -> DataLoader:
        return DataLoader(TensorDataset(q_t, t_t), batch_size=bs, shuffle=True)

    optimizer = torch.optim.AdamW(
        list(query_head.parameters()) + list(target_head.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    def warmup_cosine(epoch: int) -> float:
        if epoch < LR_WARMUP_EPOCHS:
            return (epoch + 1) / LR_WARMUP_EPOCHS
        progress = (epoch - LR_WARMUP_EPOCHS) / max(1, EPOCHS - LR_WARMUP_EPOCHS)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine)
    loader = make_loader(q_base, t_base)

    hn_q: torch.Tensor | None = None
    hn_t_pos: torch.Tensor | None = None
    hn_t_neg: torch.Tensor | None = None

    for epoch in range(1, EPOCHS + 1):
        query_head.train(); target_head.train()

        if epoch % HN_INTERVAL == 1:
            query_head.eval(); target_head.eval()
            with torch.no_grad():
                q_embs = {qid: query_head(torch.from_numpy(train_q_aug[0][qid]).unsqueeze(0).to(device)).cpu().numpy()[0]
                          for qid in train_q_paths if qid in train_q_aug[0]}
                t_embs = {tid: target_head(torch.from_numpy(train_t_aug[0][tid]).unsqueeze(0).to(device)).cpu().numpy()[0]
                          for tid in train_t_paths if tid in train_t_aug[0]}
            pairs_for_hn = [(qid, tid) for qid, tid in valid_pairs if qid in q_embs and tid in t_embs]
            qa = np.stack([q_embs[qid] for qid, _ in pairs_for_hn])
            ta = np.stack([t_embs[tid] for _, tid in pairs_for_hn])
            sims = qa @ ta.T
            q_list, tp_list, tn_list = [], [], []
            for i in range(len(pairs_for_hn)):
                row = sims[i].copy(); row[i] = -1.0
                for j in np.argsort(-row)[:HN_PER_ANCHOR]:
                    q_list.append(qa[i]); tp_list.append(ta[i]); tn_list.append(ta[j])
            hn_q = torch.from_numpy(np.stack(q_list)).to(device)
            hn_t_pos = torch.from_numpy(np.stack(tp_list)).to(device)
            hn_t_neg = torch.from_numpy(np.stack(tn_list)).to(device)
            query_head.train(); target_head.train()

        total_loss = 0.0; n_seen = 0
        for qb, tb in loader:
            q_emb = query_head(qb)
            t_emb = target_head(tb)
            logits = q_emb @ t_emb.T / TEMPERATURE
            labels = torch.arange(len(q_emb), device=device)
            info_nce = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

            triplet_loss = torch.tensor(0.0, device=device)
            if hn_q is not None:
                idx = torch.randperm(len(hn_q), device=device)[:min(len(qb), len(hn_q))]
                qhn = query_head(hn_q[idx])
                tpos = target_head(hn_t_pos[idx])
                tneg = target_head(hn_t_neg[idx])
                triplet_loss = F.relu((qhn * tneg).sum(-1) - (qhn * tpos).sum(-1) + TRIPLET_MARGIN).mean()

            loss = info_nce + TRIPLET_WEIGHT * triplet_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(query_head.parameters()) + list(target_head.parameters()), 1.0)
            optimizer.step()
            total_loss += info_nce.item() * len(qb); n_seen += len(qb)

        scheduler.step()
        if epoch % 25 == 0 or epoch == 1:
            print(f"  epoch {epoch:03d}/{EPOCHS}  loss={total_loss / n_seen:.4f}")

    # ------------------------------------------------------------------ stage 4: project

    print("\n=== Stage 4: Projecting inference features ===")
    query_head.eval(); target_head.eval()

    @torch.no_grad()
    def project(feats: dict[str, np.ndarray], head: nn.Module) -> dict[str, np.ndarray]:
        ids = sorted(feats)
        vecs = torch.from_numpy(np.stack([feats[i] for i in ids])).to(device)
        embs = head(vecs).cpu().numpy()
        return {img_id: embs[j] for j, img_id in enumerate(ids)}

    @torch.no_grad()
    def project_slices(per_slice: dict[str, np.ndarray], head: nn.Module) -> dict[str, np.ndarray]:
        return {img_id: head(torch.from_numpy(slices).to(device)).cpu().numpy()
                for img_id, slices in per_slice.items()}

    proj_q = project(infer_q_feats, query_head)
    proj_t = project(infer_t_feats, target_head)
    proj_slice_q = project_slices(per_slice_q, query_head)
    proj_slice_t = project_slices(per_slice_t, target_head)

    # ------------------------------------------------------------------ stage 5: self-training

    print("\n=== Stage 5: Self-training with pseudo-pairs from dataset2/3 ===")

    pseudo_q: list[np.ndarray] = []
    pseudo_t: list[np.ndarray] = []
    for pred_set in prediction_sets:
        if pred_set["ds"] == "dataset1":
            continue
        qids = [qid for qid in sorted(pred_set["queries"]) if qid in proj_q]
        tids = [tid for tid in sorted(pred_set["targets"]) if tid in proj_t]
        if not qids or not tids:
            continue
        scores_ps = np.stack([proj_q[qid] for qid in qids]) @ np.stack([proj_t[tid] for tid in tids]).T
        for i in range(len(qids)):
            best_j = int(np.argmax(scores_ps[i]))
            if scores_ps[i, best_j] >= ST_CONF_THRESH:
                pseudo_q.append(infer_q_feats[qids[i]])
                pseudo_t.append(infer_t_feats[tids[best_j]])

    print(f"Pseudo-pairs accepted: {len(pseudo_q)}")

    if pseudo_q:
        pq_t = torch.from_numpy(np.stack(pseudo_q)).to(device)
        pt_t = torch.from_numpy(np.stack(pseudo_t)).to(device)
        pseudo_loader = make_loader(pq_t, pt_t, bs=max(8, TRAIN_BATCH // 2))

        st_opt = torch.optim.AdamW(
            list(query_head.parameters()) + list(target_head.parameters()),
            lr=LR * 0.2, weight_decay=WEIGHT_DECAY,
        )
        st_sched = torch.optim.lr_scheduler.CosineAnnealingLR(st_opt, T_max=ST_EPOCHS)

        for epoch in range(1, ST_EPOCHS + 1):
            query_head.train(); target_head.train()
            total_loss = 0.0; n_seen = 0
            for (qr, tr), (qp, tp) in zip(loader, pseudo_loader):
                q_r = query_head(qr); t_r = target_head(tr)
                lg_r = q_r @ t_r.T / TEMPERATURE
                lb_r = torch.arange(len(q_r), device=device)
                loss_r = (F.cross_entropy(lg_r, lb_r) + F.cross_entropy(lg_r.T, lb_r)) / 2

                q_p = query_head(qp); t_p = target_head(tp)
                lg_p = q_p @ t_p.T / TEMPERATURE
                lb_p = torch.arange(len(q_p), device=device)
                loss_p = (F.cross_entropy(lg_p, lb_p) + F.cross_entropy(lg_p.T, lb_p)) / 2

                loss = loss_r + ST_WEIGHT * loss_p
                st_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(query_head.parameters()) + list(target_head.parameters()), 1.0)
                st_opt.step()
                total_loss += loss_r.item() * len(qr); n_seen += len(qr)
            st_sched.step()
            if epoch % 10 == 0 or epoch == 1:
                print(f"  ST epoch {epoch:03d}/{ST_EPOCHS}  loss={total_loss / n_seen:.4f}")

        proj_q = project(infer_q_feats, query_head)
        proj_t = project(infer_t_feats, target_head)
        proj_slice_q = project_slices(per_slice_q, query_head)
        proj_slice_t = project_slices(per_slice_t, target_head)

    # ------------------------------------------------------------------ stage 6: rank + re-rank

    print("\n=== Stage 6: Ranking with alpha-QE + k-reciprocal + ColBERT MaxSim ===")
    query_head.eval(); target_head.eval()

    submission_rows: list[dict[str, str]] = []

    for pred_set in prediction_sets:
        ds = pred_set["ds"]
        query_ids = [qid for qid in sorted(pred_set["queries"]) if qid in proj_q]
        target_ids = [tid for tid in sorted(pred_set["targets"]) if tid in proj_t]
        if not query_ids or not target_ids:
            continue

        Q = np.stack([proj_q[qid] for qid in query_ids])
        T = np.stack([proj_t[tid] for tid in target_ids])

        scores = Q @ T.T
        scores = alpha_qe(scores, Q, T)
        scores = k_reciprocal_rerank(scores, T)

        q_sl = [proj_slice_q[qid] for qid in query_ids if qid in proj_slice_q]
        t_sl = [proj_slice_t[tid] for tid in target_ids if tid in proj_slice_t]
        if len(q_sl) == len(query_ids) and len(t_sl) == len(target_ids):
            scores = colbert_rerank(scores, q_sl, t_sl)

        if ds == "dataset1":
            q_ncc = [ncc_q_idx[qid] for qid in query_ids]
            t_ncc = [ncc_t_idx[tid] for tid in target_ids]
            ncc_sub = ncc_scores_d1[np.ix_(q_ncc, t_ncc)]
            scores = NCC_WEIGHT * minmax_rows(ncc_sub) + SIGLIP_WEIGHT * minmax_rows(scores)

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
def main(out: str = "siglip_submission.csv") -> None:
    print("Running SigLIP2 pipeline on Modal...")
    csv_content = run_siglip.remote()
    out_path = Path(out)
    out_path.write_text(csv_content, encoding="utf-8")
    print(f"Saved {len(csv_content.splitlines()) - 1} rows to {out_path}")
