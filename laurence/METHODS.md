# EHL 2026 — Retrieval Methods Summary

All methods output a ranked `submission.csv` and run on Modal GPU workers against the `ehl-2026-vol-2` volume.

---

## Quick comparison

| Method | Backbone | Params | Feature dim | Fine-tuning | Slice strategy | Re-ranking | Dataset1 extra | GPU | Est. runtime |
|---|---|---|---|---|---|---|---|---|---|
| `sift_baseline_modal.py` | SIFT (CV) | — | 128 | None | 3 axial, 192 px | None | None | CPU (4 core) | ~30 min |
| `dinov2_baseline_modal.py` | DINOv2-B | 86 M | 768 | None | 3 axial, 224 px | None | None | T4 | ~45 min |
| `dinov2_finetune_modal.py` | DINOv2-B | 86 M | 768→256 | Proj. heads (InfoNCE, 150 ep) | 3 axial (train) · 9 tri-plane (infer) | None | None | T4 | ~90 min |
| `hybrid_modal.py` | DINOv2-B | 86 M | 768→256 | Proj. heads (InfoNCE, 150 ep) | 3 axial (train) · 9 tri-plane (infer) | None | 3D NCC (w=0.7) | T4 | ~120 min |
| `rad_dino_modal.py` | RadDINO | 86 M | 768→256 | Proj. heads + aug K=4 (InfoNCE, 150 ep) | 3 axial (train) · 9 tri-plane (infer) | alpha-QE | 3D NCC (w=0.7) | T4 | ~150 min |
| `strong_modal.py` | RadDINO | 86 M | 1536→512 | Proj. heads + aug K=4 + hard-neg triplet (200 ep) + self-train (50 ep) | 3 axial (train) · 15 tri-plane (infer) + TTA ×3 | alpha-QE → k-reciprocal → ColBERT MaxSim | 3D NCC (w=0.5) | A10G | ~180 min |
| `siglip_modal.py` | SigLIP2-so400m | **400 M** | 2304→512 | Proj. heads + aug K=4 + hard-neg triplet (200 ep) + self-train (50 ep) | 3 axial (train) · 15 tri-plane @ **384 px** (infer) + TTA ×3 | alpha-QE → k-reciprocal → ColBERT MaxSim | 3D NCC (w=0.45) | A10G | ~240 min |

---

## Method details

### `sift_baseline_modal.py`
**Role:** simplest baseline, no GPU needed.

- Extracts OpenCV SIFT keypoints from 3 axial slices per volume (resized to 192 px).
- Mean-pools all SIFT descriptors → 128-dim L2-normalised feature.
- Ranks by cosine similarity. No learning.
- **Weaknesses:** SIFT is purely geometric and hand-crafted; completely ignores cross-modal intensity differences between T1 and T2.

---

### `dinov2_baseline_modal.py`
**Role:** zero-shot deep learning baseline.

- Extracts 3 axial slices at positions 35 %, 50 %, 65 % of the occupied z-range.
- Passes each through frozen **DINOv2-base** (`facebook/dinov2-base`, 86M params); takes the CLS token (768-dim).
- Averages the 3 CLS tokens → single 768-dim L2-normalised embedding per volume.
- Ranks by cosine similarity. No fine-tuning.
- **Weaknesses:** natural-image pre-training; CLS-only; axial-only; no cross-modal adaptation.

---

### `dinov2_finetune_modal.py`
**Role:** adds cross-modal alignment via learned projection heads.

- Same frozen DINOv2-B backbone.
- Training (dataset1, 350 pairs):
  - 3 axial slices; DINOv2 CLS features extracted once and cached.
  - Two small projection heads (`query_head`, `target_head`): `768 → 768 (GELU) → 256`, L2-normalised output.
  - Symmetric InfoNCE loss, temperature 0.1, AdamW, cosine LR, 150 epochs, batch 64.
- Inference: 9 slices (3 axial + 3 coronal + 3 sagittal) → average CLS → project → rank.
- **Key gain over baseline:** projection space learned to align T1 queries with T2 targets.

---

### `hybrid_modal.py`
**Role:** adds 3D volumetric registration signal for dataset1.

Everything from `dinov2_finetune_modal.py`, plus:
- **3D NCC (normalised cross-correlation):** each volume downsampled to 48³, flattened, L2-normalised → GPU matrix-multiply gives (Nq × Nt) NCC score matrix.
- **Ensemble (dataset1 only):** `0.7 × NCC_minmax + 0.3 × DINOv2_minmax`.
- dataset2/3 still use DINOv2 projected embeddings only.
- **Key gain:** NCC exploits the common registered grid in dataset1; near-perfect structural matching for aligned volumes.

---

### `rad_dino_modal.py`
**Role:** best prior method; three improvements over hybrid.

1. **RadDINO backbone** (`microsoft/rad-dino`, same ViT-B architecture as DINOv2 but pre-trained on 1.6 M radiology images with DINO objective) → much stronger MRI representations out of the box.

2. **Augmented training (K=4):** for each training image, 4 independent augmented versions are run through the frozen backbone: random h/v flip, 90°-multiple rotation, brightness/contrast jitter. All 4 × 4 = 16 (query-aug, target-aug) combinations become valid training pairs → 16 × 350 = 5 600 effective pairs. Teaches geometry-invariant embeddings.

3. **alpha-QE re-ranking:** after initial cosine ranking, each query embedding is expanded toward the mean of its top-5 matched targets, then re-normalised and re-ranked. Free accuracy gain on any embedding.

- Projection head: `768 → 768 (GELU) → LayerNorm → 256`, symmetric InfoNCE.
- 3D NCC ensemble weights unchanged (0.7 / 0.3).

---

### `strong_modal.py`
**Role:** strongest pipeline, incorporating recent literature (Jul–Sep 2025).

Builds on `rad_dino_modal.py` with six additional improvements:

#### 1. CLS + mean-patch features (1536-dim)
Uses `last_hidden_state[:, 0, :]` (CLS, 768-dim) **concatenated** with `last_hidden_state[:, 1:, :].mean(dim=1)` (patch-token mean, 768-dim) → 1536-dim backbone feature. Patch tokens carry local texture — exactly what differs between T1/T2 — at no extra inference cost.

#### 2. More slice coverage (15 slices at inference)
5 positions × 3 planes = 15 slices at inference (vs 9 in prior methods). Training still uses 3 axial slices (speed).

#### 3. Test-time augmentation (TTA × 3)
At inference, 3 independently-augmented views of every slice are embedded and averaged before L2-normalisation. Reduces variance from unlucky intensity jitter or crops.

#### 4. Hard-negative triplet mining
Every 10 epochs, the current model identifies the hardest incorrect target for each training query (highest similarity among non-matches). A triplet loss penalty `max(0, sim(q, t_hard) - sim(q, t_pos) + margin)` is added on a sampled mini-batch. Weight 0.4 relative to InfoNCE. Pushes apart look-alike-but-wrong pairs — the primary failure mode.

#### 5. Self-training on dataset2/3
After 200 main epochs on dataset1, the model ranks dataset2/3 val. Any top-1 match with cosine similarity ≥ 0.65 is used as a pseudo-pair. Projection heads are fine-tuned for 50 more epochs with real pairs + pseudo-pairs (weight 0.3). Provides free supervision for the domains without labelled training data.

#### 6. Three-stage re-ranking pipeline
Applied in sequence:
- **alpha-QE** (k=5, α=0.5) — query expansion toward top neighbours.
- **k-reciprocal encoding** (Zhong et al. 2017, adapted cross-modal) — Jaccard-weighted re-scoring using gallery-to-gallery similarity; much stronger than alpha-QE alone.
- **ColBERT MaxSim** (inspired by arXiv:2507.17412, Jul 2025) — for each query's top-50 candidates, compute the full `(n_q_slices × n_t_slices)` per-projected-slice interaction matrix; score = `mean_q(max_t(sim))`. Captures which specific slices match rather than relying on averaged embeddings.

#### Projection head
Deeper residual MLP: `1536 → 1024 (GELU+LN+Drop) → 1024 residual (GELU+LN+Drop) → 512`, output L2-normalised. 512-dim projection space.

#### Infrastructure
- GPU: **A10G** (more VRAM for 1536-dim features + TTA batches).
- LR warmup: 10-epoch linear warmup into cosine decay.
- NCC ensemble weights balanced to 0.5 / 0.5 (RadDINO + ColBERT pipeline is stronger than vanilla DINOv2, so less NCC weighting needed).

---

---

### `siglip_modal.py`
**Role:** strongest single-model candidate; different inductive bias from RadDINO makes it ideal for ensembling.

Uses **SigLIP2-so400m-patch14-384** (`google/siglip2-so400m-patch14-384`):

| Property | RadDINO (strong_modal) | SigLIP2 (siglip_modal) |
|---|---|---|
| Params | 86 M | **400 M** |
| Pre-training | DINO self-distillation on radiology images | Sigmoid image-text contrastive on web images |
| Input resolution | 224 × 224 px | **384 × 384 px** |
| Patch tokens per slice | 196 (14 × 14) | **729 (27 × 27)** |
| CLS token | ✅ (used for CLS+patch concat) | ✗ (all patch tokens) |
| Feature aggregation | CLS ⊕ mean-patch (1536-dim) | **mean-pool ⊕ max-pool** (2304-dim) |

**Key differences from `strong_modal.py`:**

1. **Mean + max pooling (2304-dim):** SigLIP2 has no CLS token. Mean-pooling all 729 patch tokens recovers a global descriptor; max-pooling captures the most activated spatial positions. Concatenating both gives complementary signals analogous to the CLS+patch strategy in strong_modal.py.

2. **384 px slice resolution:** ~2.9× more pixels than 224 px. Finer anatomical structures (vessel walls, thin cortical layers) become visible, which helps the model discriminate subtle T1/T2 correspondences.

3. **729 ColBERT tokens per slice:** more spatial coverage at re-ranking time. MaxSim has more candidates to match against, making the per-slice interaction matrix denser and more discriminative.

4. **Larger model generalisation:** 400 M params means the backbone carries richer representations even without medical-domain pre-training — the projection heads only need to rotate these into the cross-modal alignment space.

5. **NCC weight adjusted (0.45 NCC / 0.55 SigLIP2):** SigLIP2's stronger visual features reduce reliance on the 3D NCC signal compared to smaller backbones.

All other pipeline components (augmentation, hard-negative triplet mining, self-training, alpha-QE, k-reciprocal, ColBERT MaxSim) are identical to `strong_modal.py`.

---

## Ensembling `strong_modal` + `siglip_modal`

The two top methods use different backbones with orthogonal inductive biases (DINO self-distillation on radiology vs sigmoid image-text on web). Their errors are partially uncorrelated, so a simple score average is expected to outperform either alone.

```python
import csv, numpy as np

def load_submission(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["query_id"]] = r["target_id_ranking"].split()
    return rows

def rank_to_score(ranking, all_targets):
    s = np.zeros(len(all_targets))
    t2i = {t: i for i, t in enumerate(all_targets)}
    for rank, tid in enumerate(ranking):
        if tid in t2i:
            s[t2i[tid]] = 1.0 / (rank + 1)   # reciprocal rank as proxy score
    return s

# Load both submissions
strong = load_submission("strong_submission.csv")
siglip = load_submission("siglip_submission.csv")

all_qids = sorted(set(strong) | set(siglip))
# For each query, average reciprocal-rank scores and re-sort
ensemble_rows = []
for qid in all_qids:
    all_tids = list(dict.fromkeys(strong.get(qid, []) + siglip.get(qid, [])))
    s1 = rank_to_score(strong.get(qid, []), all_tids)
    s2 = rank_to_score(siglip.get(qid, []), all_tids)
    combined = 0.5 * s1 + 0.5 * s2
    ranked = [all_tids[j] for j in np.argsort(-combined)]
    ensemble_rows.append({"query_id": qid, "target_id_ranking": " ".join(ranked)})

with open("ensemble_submission.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
    w.writeheader(); w.writerows(ensemble_rows)
```

---

## Improvement lineage

```
sift_baseline
    └── dinov2_baseline          +deep features, zero-shot
            └── dinov2_finetune  +cross-modal projection heads
                    └── hybrid   +3D NCC ensemble (dataset1)
                            └── rad_dino  +radiology backbone, +augmentation, +alpha-QE
                                    └── strong        +CLS+patch, +15 slices, +TTA, +hard-neg,
                                    │                 +self-training, +k-reciprocal, +ColBERT MaxSim
                                    └── siglip        +400M backbone, +384px, +mean/max pool,
                                                      same fine-tuning + re-ranking pipeline
                                             └── ensemble(strong + siglip)   best expected score
```

---

## Key hyperparameters at a glance

| Param | `dinov2_finetune` | `hybrid` | `rad_dino` | `strong` | `siglip` |
|---|---|---|---|---|---|
| Backbone | DINOv2-B | DINOv2-B | RadDINO | RadDINO | **SigLIP2-so400m** |
| Backbone params | 86 M | 86 M | 86 M | 86 M | **400 M** |
| Input resolution | 224 px | 224 px | 224 px | 224 px | **384 px** |
| Feature dim | 768 | 768 | 768 | **1536** | **2304** |
| Proj dim | 256 | 256 | 256 | **512** | **512** |
| Temperature | 0.1 | 0.1 | 0.1 | **0.07** | **0.07** |
| Epochs | 150 | 150 | 150 | **200 + 50 ST** | **200 + 50 ST** |
| Aug views (train) | 1 | 1 | **4** | **4** | **4** |
| TTA views (infer) | 1 | 1 | 1 | **3** | **3** |
| Slices (infer) | 9 | 9 | 9 | **15** | **15** |
| Patch tokens / slice | 196 | 196 | 196 | 196 | **729** |
| Re-ranking | none | none | alpha-QE | **α-QE → k-recip → ColBERT** | **α-QE → k-recip → ColBERT** |
| Self-training | ✗ | ✗ | ✗ | **✅** | **✅** |
| Hard neg mining | ✗ | ✗ | ✗ | **✅** | **✅** |
