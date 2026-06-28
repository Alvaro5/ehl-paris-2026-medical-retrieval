"""Strong cross-modal MRI retrieval on Modal.

Improvements over rad_dino_modal.py:

1. CLS + mean-patch feature (1536-dim) — richer than CLS-only; patch tokens
   carry local texture that differs between T1/T2 modalities.

2. More slice coverage — 5 positions × 3 planes = 15 slices at inference
   (vs 9 in baseline), 3 positions × 1 plane = 3 slices at training (fast).

3. Triplet hard-negative mining — every HN_INTERVAL epochs, find the hardest
   non-matching target for each query (the closest gallery item that is NOT the
   true match) and add a triplet loss to push it further away.  Regularises the
   embedding space much better than purely random in-batch negatives.

4. Test-time augmentation (TTA) — at inference, embed TTA_K independently
   augmented versions of each slice and average the embeddings.  Reduces
   variance from a single unlucky crop or intensity jitter.

5. k-reciprocal encoding re-ranking — Zhong et al. 2017 re-ranking using
   gallery-to-gallery Jaccard similarity.  Substantially stronger than
   alpha-QE alone (applied on top of QE).

6. Self-training on dataset2/3 — after the main training loop, use the
   current model's confident top-1 pseudo-matches on dataset2/3 val as
   extra training pairs and fine-tune for ST_EPOCHS more epochs.

7. Larger / deeper projection head with residual block and LayerNorm.

8. LR warmup (cosine schedule with linear warmup).

Run:
    modal run laurence/strong_modal.py
    modal run laurence/strong_modal.py --out my_submission.csv
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import modal

app = modal.App("ehl-strong")
vol = modal.Volume.from_name("ehl-2026-vol-2")

MODEL_ID = "microsoft/rad-dino"
BACKBONE_DIM = 1536        # 768 CLS + 768 mean-patch tokens

# Projection head
PROJ_DIM = 512
TEMPERATURE = 0.07
EPOCHS = 200
LR = 2e-3
WEIGHT_DECAY = 1e-4
LR_WARMUP_EPOCHS = 10
TRAIN_BATCH = 48
EMBED_BATCH = 32

# Augmentation
K_AUG = 4                  # augmented views per training image

# Slice positions
SLICE_POSITIONS_TRAIN = (0.35, 0.50, 0.65)                        # 3 per axis
SLICE_POSITIONS_INFER = (0.25, 0.375, 0.50, 0.625, 0.75)          # 5 per axis
IMAGE_SIZE = 224

# Test-time augmentation
TTA_K = 3                  # augmented views at inference (averaged)

# Hard negative mining
HN_INTERVAL = 10           # mine hard negatives every N epochs
HN_PER_ANCHOR = 3          # hard negatives per query
TRIPLET_MARGIN = 0.2
TRIPLET_WEIGHT = 0.4

# alpha-QE re-ranking (applied before k-reciprocal)
AQE_K = 5
AQE_ALPHA = 0.5

# k-reciprocal re-ranking
KRR_K1 = 20
KRR_K2 = 6
KRR_LAMBDA = 0.35

# ColBERT-inspired MaxSim re-ranking (Inspired by arXiv:2507.17412)
#   Stage 1: fast average-embedding retrieval → top-K candidates
#   Stage 2: re-rank using full per-slice interaction matrix (MaxSim)
COLBERT_TOP_K = 50         # number of candidates to re-rank
COLBERT_WEIGHT = 0.4       # blend weight: COLBERT_WEIGHT * MaxSim + (1-w) * stage-1-score

# Self-training
ST_CONF_THRESH = 0.65      # min cosine similarity to accept a pseudo-pair
ST_EPOCHS = 50             # fine-tuning epochs with pseudo-pairs
ST_WEIGHT = 0.3            # loss weight on pseudo-pairs

# 3D NCC (dataset1 only)
NCC_SIZE = (48, 48, 48)
NCC_WEIGHT = 0.5
DINO_WEIGHT = 0.5

# Gradient-magnitude NMI (all datasets)
# For ds2/ds3: DINO alone previously; now blended with grad-NMI.
# Grad-NMI is rotation-invariant and modality-independent, giving a
# volumetric signal where NCC (which requires alignment) cannot help.
GRAD_NMI_SIZE = 48          # per-axis target after downsample
GRAD_NMI_BINS = 48          # joint-histogram bins
GRAD_NMI_WEIGHT_DS1  = 0.15 # small extra boost on top of NCC+DINO blend
GRAD_NMI_WEIGHT_DS23 = 0.30 # meaningful volumetric signal for unaligned ds


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
        "scipy>=1.13",
    )
    .run_function(_download_model)
)


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="A10G",              # more VRAM than T4 for larger feature dim + TTA
    timeout=10800,
    memory=32768,
)
def run_strong() -> str:
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
            print(f"  {label}: {len(id_path) - len(out)} missing")
        return out

    # ------------------------------------------------------------------ slice loader

    def volume_to_pils(nii_path: _Path, positions: tuple) -> list:
        img = nib.load(str(nii_path))
        arr = img.get_fdata(dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        axes = [0, 1, 2]
        pils = []
        for axis in axes:
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

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    backbone = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def embed_pils(pils: list, owners: list[str]) -> dict[str, np.ndarray]:
        """Run RadDINO on a flat list of PILs, return CLS+patch-mean per owner."""
        all_feats: list[np.ndarray] = []
        for start in range(0, len(pils), EMBED_BATCH):
            batch = pils[start : start + EMBED_BATCH]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            out = backbone(**inputs)
            hs = out.last_hidden_state                    # (B, 197, 768)
            cls = hs[:, 0, :]                             # (B, 768)
            patch_mean = hs[:, 1:, :].mean(dim=1)        # (B, 768)
            feat = torch.cat([cls, patch_mean], dim=-1)   # (B, 1536)
            all_feats.append(feat.cpu().float().numpy())
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
        id_path: dict[str, _Path],
        label: str,
        positions: tuple,
        n_aug: int = 1,
    ) -> dict[str, np.ndarray]:
        ids = sorted(id_path)
        all_pils: list = []
        owners: list[str] = []
        n_slices = len(positions) * 3  # 3 planes
        print(f"  Slicing {len(ids)} {label} volumes ({n_slices} slices × {n_aug} aug)...")
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
        print(f"  Embedding {len(all_pils)} slices through RadDINO...")
        return embed_pils(all_pils, owners)

    @torch.no_grad()
    def extract_per_slice(
        id_path: dict[str, _Path], positions: tuple
    ) -> dict[str, np.ndarray]:
        """Return {img_id: (n_slices, 1536)} backbone features, one row per slice."""
        ids = sorted(id_path)
        n_slices = len(positions) * 3
        per_slice: dict[str, np.ndarray] = {}
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
            inputs = processor(images=batch, return_tensors="pt").to(device)
            out = backbone(**inputs)
            hs = out.last_hidden_state
            cls = hs[:, 0, :]
            patch_mean = hs[:, 1:, :].mean(dim=1)
            feat = torch.cat([cls, patch_mean], dim=-1).cpu().float().numpy()
            all_feats.append(feat)
        all_feats_np = np.concatenate(all_feats, axis=0)

        accum: dict[str, list[np.ndarray]] = {img_id: [] for img_id in ids}
        for feat, owner in zip(all_feats_np, owners):
            accum[owner].append(feat)
        for img_id, feats in accum.items():
            per_slice[img_id] = np.stack(feats).astype(np.float32)   # (n_slices, 1536)
        return per_slice

    # ------------------------------------------------------------------ projection head

    class ProjectionHead(nn.Module):
        """Two-layer MLP with residual skip, LayerNorm, and dropout."""
        def __init__(self, in_dim: int = BACKBONE_DIM, out_dim: int = PROJ_DIM) -> None:
            super().__init__()
            hidden = max(in_dim, out_dim * 2)
            self.fc1 = nn.Linear(in_dim, hidden)
            self.ln1 = nn.LayerNorm(hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.ln2 = nn.LayerNorm(hidden)
            self.fc3 = nn.Linear(hidden, out_dim)
            self.skip = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()
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
        q_scores: np.ndarray,
        T: np.ndarray,
        k1: int = KRR_K1,
        k2: int = KRR_K2,
        lam: float = KRR_LAMBDA,
    ) -> np.ndarray:
        """k-reciprocal encoding re-ranking (Zhong et al. 2017, adapted for cross-modal)."""
        Nq, Nt = q_scores.shape

        # Gallery-to-gallery similarity
        t_sims = T @ T.T                           # (Nt, Nt)
        np.fill_diagonal(t_sims, -1.0)             # exclude self

        # Precompute k1-NN of each gallery item
        t_knn = np.argsort(-t_sims, axis=1)[:, :k1]   # (Nt, k1)

        # Binary membership matrix: t_mem[i, j] = 1 iff j in knn(i)
        t_mem = np.zeros((Nt, Nt), dtype=np.float32)
        rows = np.repeat(np.arange(Nt), k1)
        cols = t_knn.flatten()
        t_mem[rows, cols] = 1.0

        # Per-gallery k-NN count (for Jaccard denominator)
        t_counts = t_mem.sum(axis=1)               # (Nt,)

        final_scores = np.empty_like(q_scores)

        for i in range(Nq):
            q_nn = np.argsort(-q_scores[i])[:k1]
            q_nn_set = set(q_nn.tolist())

            # Build k-reciprocal set R: gallery items that are "mutually" close
            R = []
            for t in q_nn:
                t_nn_set = set(t_knn[t].tolist())
                if len(t_nn_set & q_nn_set) >= k1 / 2:
                    R.append(t)
            if not R:
                R = q_nn[:k2].tolist()

            # Expand R with k2-NN of each member
            R_expanded = set(R)
            for t in R:
                R_expanded.update(t_knn[t, :k2].tolist())

            R_vec = np.zeros(Nt, dtype=np.float32)
            R_vec[list(R_expanded)] = 1.0
            r_count = R_vec.sum()

            # Jaccard similarity: R_vec vs each gallery item's k-NN set
            inter = t_mem @ R_vec               # (Nt,): dot product = |R ∩ knn(j)|
            union = r_count + t_counts - inter
            jaccard = inter / (union + 1e-10)

            final_scores[i] = lam * q_scores[i] + (1.0 - lam) * jaccard

        return final_scores

    def minmax_rows(m: np.ndarray) -> np.ndarray:
        lo = m.min(axis=1, keepdims=True)
        hi = m.max(axis=1, keepdims=True)
        return (m - lo) / np.where(hi > lo, hi - lo, 1.0)

    def colbert_rerank(
        initial_scores: np.ndarray,
        q_slice_embs: list[np.ndarray],
        t_slice_embs: list[np.ndarray],
        top_k: int = COLBERT_TOP_K,
        blend: float = COLBERT_WEIGHT,
    ) -> np.ndarray:
        """ColBERT-inspired MaxSim re-ranking (arXiv:2507.17412).

        For each query, re-scores the top-K gallery candidates by computing
        the full per-slice interaction matrix and taking MaxSim:
            score(Q, T) = mean_over_q_slices( max_over_t_slices( sim(q_i, t_j) ) )

        Significantly more discriminative than dot-product on averaged embeddings.
        """
        Nq = initial_scores.shape[0]
        reranked = initial_scores.copy()
        for i in range(Nq):
            top_k_idx = np.argsort(-initial_scores[i])[:top_k]
            Qs = q_slice_embs[i]                    # (n_q_sl, dim)
            for j in top_k_idx:
                Ts = t_slice_embs[j]                # (n_t_sl, dim)
                sim_mat = Qs @ Ts.T                 # (n_q_sl, n_t_sl)
                maxsim = float(sim_mat.max(axis=1).mean())
                reranked[i, j] = (1 - blend) * initial_scores[i, j] + blend * maxsim
        return reranked

    # ------------------------------------------------------------------ manifests

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

    # ------------------------------------------------------------------ stage 1b: gradient-magnitude NMI
    # Rotation-invariant, modality-independent volumetric signal.
    # Provides a complementary score to NCC (ds1) and fills the volumetric
    # gap for ds2/ds3 where NCC cannot help due to broken spatial alignment.

    print("\n=== Stage 1b: Gradient-magnitude NMI (all datasets) ===")
    from concurrent.futures import ThreadPoolExecutor as _TPE
    from scipy.ndimage import zoom as _zoom

    def _load_grad(item: tuple) -> tuple[str, np.ndarray]:
        img_id, path = item
        try:
            arr = nib.load(str(path)).get_fdata(dtype=np.float32)
            if arr.ndim == 4:
                arr = arr[..., 0]
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            factors = tuple(GRAD_NMI_SIZE / s for s in arr.shape)
            arr = _zoom(arr, factors, order=1, prefilter=False).astype(np.float32)
        except Exception as e:
            print(f"  GRAD load error {path}: {e}")
            arr = np.zeros((GRAD_NMI_SIZE,) * 3, dtype=np.float32)
        # CoM translation correction
        fg = arr > 0
        if fg.any():
            cm     = np.argwhere(fg).mean(axis=0)
            centre = np.array(arr.shape) / 2.0
            shift  = np.round(centre - cm).astype(int)
            arr    = np.roll(arr, (int(shift[0]), int(shift[1]), int(shift[2])), axis=(0, 1, 2))
        # Gradient magnitude
        gx = np.gradient(arr, axis=0)
        gy = np.gradient(arr, axis=1)
        gz = np.gradient(arr, axis=2)
        return img_id, np.sqrt(gx * gx + gy * gy + gz * gz).astype(np.float32)

    # Collect all unique paths across all datasets
    _all_grad_paths: dict[str, _Path] = {}
    for ps in prediction_sets:
        _all_grad_paths.update(ps["queries"])
        _all_grad_paths.update(ps["targets"])
    print(f"  Loading gradient magnitudes for {len(_all_grad_paths)} volumes...")
    grad_cache: dict[str, np.ndarray] = {}
    with _TPE(max_workers=16) as ex:
        for img_id, g in ex.map(_load_grad, sorted(_all_grad_paths.items())):
            grad_cache[img_id] = g
    print(f"  Loaded {len(grad_cache)} gradient volumes, shape {next(iter(grad_cache.values())).shape}")

    def _grad_nmi(a: np.ndarray, b: np.ndarray, intersect: bool = False) -> float:
        af = a.ravel(); bf = b.ravel()
        mask = (af != 0) & (bf != 0) if intersect else (af != 0) | (bf != 0)
        if mask.sum() < 50:
            return 0.0
        h, _, _ = np.histogram2d(af[mask], bf[mask], bins=GRAD_NMI_BINS)
        h /= h.sum() + 1e-10
        pa = h.sum(axis=1); pb = h.sum(axis=0)
        ha  = -(pa * np.log(pa + 1e-10)).sum()
        hb  = -(pb * np.log(pb + 1e-10)).sum()
        hab = -(h  * np.log(h  + 1e-10)).sum()
        v = float((ha + hb) / (hab + 1e-10))
        return v if np.isfinite(v) else 0.0

    # Compute per-dataset gradient-NMI score matrices
    grad_nmi_scores: dict[str, np.ndarray] = {}   # key: "ds/split"
    grad_nmi_qids:   dict[str, list[str]]  = {}
    grad_nmi_tids:   dict[str, list[str]]  = {}
    for ps in prediction_sets:
        ds    = ps["ds"]
        split = ps["split"]
        key   = f"{ds}/{split}"
        use_intersect = (ds == "dataset3")
        q_ids = [qid for qid in sorted(ps["queries"]) if qid in grad_cache]
        t_ids = [tid for tid in sorted(ps["targets"]) if tid in grad_cache]
        nq, nt = len(q_ids), len(t_ids)
        print(f"  {key}: {nq}×{nt} grad-NMI pairs")
        S = np.zeros((nq, nt), dtype=np.float32)
        for i, qid in enumerate(q_ids):
            gq = grad_cache[qid]
            for j, tid in enumerate(t_ids):
                S[i, j] = _grad_nmi(gq, grad_cache[tid], intersect=use_intersect)
        grad_nmi_scores[key] = S
        grad_nmi_qids[key]   = q_ids
        grad_nmi_tids[key]   = t_ids
    print("  Gradient-NMI matrices computed.")

    # ------------------------------------------------------------------ stage 2: feature extraction

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

    # Training: K_AUG augmented views, axial-only (fast)
    print("Extracting augmented training features (axial, K_AUG views)...")
    train_q_aug: list[dict[str, np.ndarray]] = []
    train_t_aug: list[dict[str, np.ndarray]] = []
    for k in range(K_AUG):
        print(f"  Aug {k + 1}/{K_AUG}")
        _random.seed(k)
        train_q_aug.append(extract_features(
            train_q_paths, f"train-q-aug{k}", SLICE_POSITIONS_TRAIN, n_aug=1
        ))
        _random.seed(k + 100)
        train_t_aug.append(extract_features(
            train_t_paths, f"train-t-aug{k}", SLICE_POSITIONS_TRAIN, n_aug=1
        ))

    # Inference: TTA_K augmented views, all 5 positions × 3 planes
    print("Extracting inference features (all planes, 5 positions, TTA)...")
    _random.seed(42)
    infer_q_feats_list = [
        extract_features(all_query_paths, f"query-tta{k}", SLICE_POSITIONS_INFER, n_aug=1)
        for k in range(TTA_K)
    ]
    _random.seed(200)
    infer_t_feats_list = [
        extract_features(all_target_paths, f"target-tta{k}", SLICE_POSITIONS_INFER, n_aug=1)
        for k in range(TTA_K)
    ]

    # Average TTA views: mean of TTA_K L2-normalised embeddings, then re-normalise
    def average_tta(feat_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        ids = list(feat_list[0].keys())
        avg: dict[str, np.ndarray] = {}
        for img_id in ids:
            vecs = np.stack([fl[img_id] for fl in feat_list if img_id in fl])
            f = vecs.mean(axis=0).astype(np.float32)
            norm = np.linalg.norm(f)
            avg[img_id] = f / norm if norm > 0 else f
        return avg

    infer_q_feats = average_tta(infer_q_feats_list)
    infer_t_feats = average_tta(infer_t_feats_list)

    # Also store per-slice backbone features for ColBERT re-ranking later
    print("Extracting per-slice backbone features for ColBERT re-ranking...")
    per_slice_q = extract_per_slice(all_query_paths, SLICE_POSITIONS_INFER)
    per_slice_t = extract_per_slice(all_target_paths, SLICE_POSITIONS_INFER)

    # ------------------------------------------------------------------ stage 3: projection heads

    print("\n=== Stage 3: Training projection heads with hard-negative mining ===")

    query_head = ProjectionHead().to(device)
    target_head = ProjectionHead().to(device)

    # Build augmented training tensor pool: all K_AUG × K_AUG (q_aug_i, t_aug_j) combos
    q_vecs_list, t_vecs_list = [], []
    for qf in train_q_aug:
        for tf in train_t_aug:
            for qid, tid in valid_pairs:
                if qid in qf and tid in tf:
                    q_vecs_list.append(qf[qid])
                    t_vecs_list.append(tf[tid])

    q_base = torch.from_numpy(np.stack(q_vecs_list))   # on CPU initially
    t_base = torch.from_numpy(np.stack(t_vecs_list))
    print(f"Augmented training pairs: {len(q_vecs_list)}")

    def make_loader(q_t: torch.Tensor, t_t: torch.Tensor, batch_size: int = TRAIN_BATCH):
        return DataLoader(TensorDataset(q_t, t_t), batch_size=batch_size, shuffle=True)

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

    # Hard negative pool (populated every HN_INTERVAL epochs)
    hn_q: torch.Tensor | None = None
    hn_t_pos: torch.Tensor | None = None
    hn_t_neg: torch.Tensor | None = None

    def mine_hard_negatives(
        qf: dict[str, np.ndarray],
        tf: dict[str, np.ndarray],
        pairs: list[tuple[str, str]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (q_vecs, t_pos_vecs, t_neg_vecs) for triplet loss."""
        qa = np.stack([qf[qid] for qid, _ in pairs if qid in qf])
        ta = np.stack([tf[tid] for _, tid in pairs if tid in tf])
        sims = qa @ ta.T                                   # (N, N)
        q_list, tp_list, tn_list = [], [], []
        for i in range(len(pairs)):
            row = sims[i].copy()
            row[i] = -1.0                                  # mask the positive
            hard_idxs = np.argsort(-row)[:HN_PER_ANCHOR]
            for j in hard_idxs:
                q_list.append(qa[i])
                tp_list.append(ta[i])
                tn_list.append(ta[j])
        return np.stack(q_list), np.stack(tp_list), np.stack(tn_list)

    loader = make_loader(q_base.to(device), t_base.to(device))

    for epoch in range(1, EPOCHS + 1):
        query_head.train(); target_head.train()

        # Refresh hard negatives
        if epoch % HN_INTERVAL == 1:
            query_head.eval(); target_head.eval()
            with torch.no_grad():
                q_embs = {qid: query_head(torch.from_numpy(train_q_aug[0][qid]).unsqueeze(0).to(device)).cpu().numpy()[0]
                          for qid in train_q_paths if qid in train_q_aug[0]}
                t_embs = {tid: target_head(torch.from_numpy(train_t_aug[0][tid]).unsqueeze(0).to(device)).cpu().numpy()[0]
                          for tid in train_t_paths if tid in train_t_aug[0]}
            hn_qa, hn_tp, hn_tn = mine_hard_negatives(q_embs, t_embs, valid_pairs)
            hn_q = torch.from_numpy(hn_qa).to(device)
            hn_t_pos = torch.from_numpy(hn_tp).to(device)
            hn_t_neg = torch.from_numpy(hn_tn).to(device)
            query_head.train(); target_head.train()

        total_loss = 0.0; n_seen = 0
        for qb, tb in loader:
            q_emb = query_head(qb)
            t_emb = target_head(tb)
            logits = q_emb @ t_emb.T / TEMPERATURE
            labels = torch.arange(len(q_emb), device=device)
            info_nce = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

            # Triplet loss on hard negatives (sample a random mini-batch)
            triplet_loss = torch.tensor(0.0, device=device)
            if hn_q is not None and len(hn_q) > 0:
                idx = torch.randperm(len(hn_q), device=device)[:min(len(qb), len(hn_q))]
                qhn = query_head(hn_q[idx])
                tpos = target_head(hn_t_pos[idx])
                tneg = target_head(hn_t_neg[idx])
                pos_sim = (qhn * tpos).sum(dim=-1)
                neg_sim = (qhn * tneg).sum(dim=-1)
                triplet_loss = F.relu(neg_sim - pos_sim + TRIPLET_MARGIN).mean()

            loss = info_nce + TRIPLET_WEIGHT * triplet_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(query_head.parameters()) + list(target_head.parameters()), 1.0
            )
            optimizer.step()
            total_loss += info_nce.item() * len(qb); n_seen += len(qb)

        scheduler.step()
        if epoch % 25 == 0 or epoch == 1:
            print(f"  epoch {epoch:03d}/{EPOCHS}  loss={total_loss / n_seen:.4f}")

    # ------------------------------------------------------------------ stage 4: project inference features

    print("\n=== Stage 4: Projecting inference features ===")
    query_head.eval(); target_head.eval()

    @torch.no_grad()
    def project(feats: dict[str, np.ndarray], head: nn.Module) -> dict[str, np.ndarray]:
        ids = sorted(feats)
        vecs = torch.from_numpy(np.stack([feats[i] for i in ids])).to(device)
        embs = head(vecs).cpu().numpy()
        return {img_id: embs[j] for j, img_id in enumerate(ids)}

    proj_q = project(infer_q_feats, query_head)
    proj_t = project(infer_t_feats, target_head)

    # Project per-slice features through heads for ColBERT re-ranking
    @torch.no_grad()
    def project_slices(
        per_slice: dict[str, np.ndarray], head: nn.Module
    ) -> dict[str, np.ndarray]:
        """Project (n_slices, backbone_dim) → (n_slices, proj_dim), per image."""
        result: dict[str, np.ndarray] = {}
        for img_id, slices in per_slice.items():
            t = torch.from_numpy(slices).to(device)   # (n_sl, backbone_dim)
            embs = head(t).cpu().numpy()               # (n_sl, proj_dim)
            result[img_id] = embs
        return result

    proj_slice_q = project_slices(per_slice_q, query_head)
    proj_slice_t = project_slices(per_slice_t, target_head)

    # ------------------------------------------------------------------ stage 5: self-training on dataset2/3

    print("\n=== Stage 5: Self-training with pseudo-pairs from dataset2/3 ===")

    pseudo_pairs_q: list[np.ndarray] = []
    pseudo_pairs_t: list[np.ndarray] = []

    for pred_set in prediction_sets:
        if pred_set["ds"] == "dataset1":
            continue
        qids = [qid for qid in sorted(pred_set["queries"]) if qid in proj_q]
        tids = [tid for tid in sorted(pred_set["targets"]) if tid in proj_t]
        if not qids or not tids:
            continue
        Q_ps = np.stack([proj_q[qid] for qid in qids])
        T_ps = np.stack([proj_t[tid] for tid in tids])
        scores_ps = Q_ps @ T_ps.T
        for i in range(len(qids)):
            best_j = int(np.argmax(scores_ps[i]))
            if scores_ps[i, best_j] >= ST_CONF_THRESH:
                pseudo_pairs_q.append(infer_q_feats[qids[i]])
                pseudo_pairs_t.append(infer_t_feats[tids[best_j]])

    print(f"Pseudo-pairs accepted: {len(pseudo_pairs_q)}")

    if pseudo_pairs_q:
        pq_tensor = torch.from_numpy(np.stack(pseudo_pairs_q)).to(device)
        pt_tensor = torch.from_numpy(np.stack(pseudo_pairs_t)).to(device)
        pseudo_loader = make_loader(pq_tensor, pt_tensor, batch_size=max(8, TRAIN_BATCH // 2))

        st_optimizer = torch.optim.AdamW(
            list(query_head.parameters()) + list(target_head.parameters()),
            lr=LR * 0.2, weight_decay=WEIGHT_DECAY,
        )
        st_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(st_optimizer, T_max=ST_EPOCHS)

        for epoch in range(1, ST_EPOCHS + 1):
            query_head.train(); target_head.train()
            total_loss = 0.0; n_seen = 0
            # Interleave real pairs and pseudo-pairs
            for (qr, tr), (qp, tp) in zip(loader, pseudo_loader):
                # Real pair loss
                q_emb_r = query_head(qr)
                t_emb_r = target_head(tr)
                logits_r = q_emb_r @ t_emb_r.T / TEMPERATURE
                labels_r = torch.arange(len(q_emb_r), device=device)
                loss_r = (F.cross_entropy(logits_r, labels_r) + F.cross_entropy(logits_r.T, labels_r)) / 2

                # Pseudo-pair loss (lower weight)
                q_emb_p = query_head(qp)
                t_emb_p = target_head(tp)
                logits_p = q_emb_p @ t_emb_p.T / TEMPERATURE
                labels_p = torch.arange(len(q_emb_p), device=device)
                loss_p = (F.cross_entropy(logits_p, labels_p) + F.cross_entropy(logits_p.T, labels_p)) / 2

                loss = loss_r + ST_WEIGHT * loss_p
                st_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(query_head.parameters()) + list(target_head.parameters()), 1.0
                )
                st_optimizer.step()
                total_loss += loss_r.item() * len(qr); n_seen += len(qr)
            st_scheduler.step()
            if epoch % 10 == 0 or epoch == 1:
                print(f"  ST epoch {epoch:03d}/{ST_EPOCHS}  loss={total_loss / n_seen:.4f}")

        # Re-project with updated heads
        proj_q = project(infer_q_feats, query_head)
        proj_t = project(infer_t_feats, target_head)
        proj_slice_q = project_slices(per_slice_q, query_head)
        proj_slice_t = project_slices(per_slice_t, target_head)

    # ------------------------------------------------------------------ stage 6: rank + re-rank

    print("\n=== Stage 6: Ranking with alpha-QE + k-reciprocal + ColBERT re-ranking ===")
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

        # Initial cosine scores
        dino_scores = Q @ T.T

        # alpha-QE expansion
        dino_scores = alpha_qe(dino_scores, Q, T)

        # k-reciprocal re-ranking
        dino_scores = k_reciprocal_rerank(dino_scores, T)

        # ColBERT MaxSim re-ranking on top-K candidates (arXiv:2507.17412)
        q_slices = [proj_slice_q[qid] for qid in query_ids if qid in proj_slice_q]
        t_slices = [proj_slice_t[tid] for tid in target_ids if tid in proj_slice_t]
        if len(q_slices) == len(query_ids) and len(t_slices) == len(target_ids):
            dino_scores = colbert_rerank(dino_scores, q_slices, t_slices)

        key = f"{ds}/{pred_set['split']}"
        has_grad = (
            key in grad_nmi_scores
            and grad_nmi_qids.get(key) == query_ids
            and grad_nmi_tids.get(key) == target_ids
        )

        if ds == "dataset1":
            q_ncc = [ncc_q_idx[qid] for qid in query_ids]
            t_ncc = [ncc_t_idx[tid] for tid in target_ids]
            ncc_sub = ncc_scores_d1[np.ix_(q_ncc, t_ncc)]
            if has_grad:
                grad_sub = grad_nmi_scores[key]
                scores = (
                    NCC_WEIGHT * minmax_rows(ncc_sub)
                    + (DINO_WEIGHT - GRAD_NMI_WEIGHT_DS1) * minmax_rows(dino_scores)
                    + GRAD_NMI_WEIGHT_DS1 * minmax_rows(grad_sub)
                )
            else:
                scores = NCC_WEIGHT * minmax_rows(ncc_sub) + DINO_WEIGHT * minmax_rows(dino_scores)
        else:
            if has_grad:
                grad_sub = grad_nmi_scores[key]
                # Gradient-NMI fills the volumetric gap: rotation-invariant,
                # modality-independent signal where NCC cannot help.
                scores = (
                    (1.0 - GRAD_NMI_WEIGHT_DS23) * minmax_rows(dino_scores)
                    + GRAD_NMI_WEIGHT_DS23 * minmax_rows(grad_sub)
                )
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
def main(out: str = "strong_submission.csv") -> None:
    print("Running strong pipeline on Modal...")
    csv_content = run_strong.remote()
    out_path = Path(out)
    out_path.write_text(csv_content, encoding="utf-8")
    print(f"Saved {len(csv_content.splitlines()) - 1} rows to {out_path}")
