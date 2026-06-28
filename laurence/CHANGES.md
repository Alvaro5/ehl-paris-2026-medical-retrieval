# Experiment Log

## mi_grad_modal.py (new)
**Approach:** Gradient-magnitude NMI — pure numpy, no registration library.

**Key insight:** Raw T1/T2 NMI (score ~0.52) has two problems:
1. Intensity inversion: WM is bright in T1ce, dark in T2. Raw joint histogram is anti-correlated.
2. ds2 independent deformations: CoM corrects translation only; rotation breaks spatial NMI.

**Gradient magnitude solves both:**
- `||∇f||` is modality-independent: tissue boundaries appear at the same locations in T1 and T2 regardless of polarity.
- `||∇(Rf)|| = ||∇f||` under any rigid rotation R — gradient-NMI is inherently rotation-invariant.
- Global gradient-NMI acts as a subject-level discriminator via distribution matching (same subject → similar tissue composition → similar gradient distribution) even without spatial alignment.

**Changes over mattes_mi_modal.py:**
- `TARGET_DIM 32→48` (trilinear zoom via scipy, better coverage)
- `NMI_BINS 32→48`
- Compute gradient magnitude after CoM alignment; NMI on gradient magnitudes
- ds1 only: 50/50 ensemble with raw intensity NMI (registered grid makes spatial NMI valid there)
- ds2/ds3: gradient-NMI only (rotation-robust)

**Run:** `modal run laurence/mi_grad_modal.py`

---

## strong_modal.py (updated: +gradient-NMI ensemble for ds2/ds3)
**New stage 1b:** After 3D NCC, compute gradient-magnitude NMI for ALL datasets.
- Adds `scipy` dependency for trilinear zoom
- `GRAD_NMI_WEIGHT_DS1 = 0.15` (small additive signal on top of NCC+DINO)
- `GRAD_NMI_WEIGHT_DS23 = 0.30` (meaningful volumetric signal; 70% DINO + 30% grad-NMI)
- ds1 weights rebalanced: `0.5 NCC + 0.35 DINO + 0.15 grad-NMI`
- ds2/ds3 weights: `0.70 DINO + 0.30 grad-NMI` (was 1.0 DINO)

**Why 0.30 for ds2/ds3:** Gradient-NMI provides a completely orthogonal signal
(3D volumetric, distribution-level) versus DINO (2D slice-level, appearance).
For independent deformations and surgery, these are expected to be partially decorrelated.

---


## voxelmorph_modal.py (new)
**Approach:** Learned deformable registration (VoxelMorph) on Sobel edge maps.

**Why edge maps:** T1ce and T2 have inverted intensity contrasts (WM bright/dark),
so NCC on raw intensities produces negative correlation for the correct pair.
Sobel gradient magnitude is modality-independent — tissue boundaries look the
same in both modalities — making local NCC valid cross-modally.

**Architecture:** Custom UNet (pure PyTorch, no external registration library).
- Encoder: 4× stride-2 Conv3d (96→48→24→12→6), channels 2→16→32→32→32
- Decoder: 4× Upsample + skip concat + Conv3d
- Head: Conv3d → 3-channel displacement field
- Spatial transformer applies displacement via `F.grid_sample`

**Training:** 100 epochs on dataset1 train pairs. Loss = local NCC (9³ window)
+ 0.01 × flow gradient (smoothness). Independent random flips/90° rotations
applied to query and target separately to simulate dataset2's independently
deformed pairs.

**Inference:** All volumes → 96³ edge maps (cached). For each pair: forward
pass → score = global NCC(warped_query_edge, target_edge). Dataset1 also
blends 50% direct NCC (registered grid = direct comparison is informative).

**Model caching:** Saved to `/data/models/voxelmorph_96.pth`. Re-run reuses
saved model (~3 min inference only). Force retrain with `--retrain`.

**Run:** `modal run laurence/voxelmorph_modal.py`

---

## mattes_mi_modal.py (iterated x3)
**v1** — SimpleITK rigid Mattes MI. Single container, 28 multiprocessing
workers. Too slow (~57 min).

**v2** — Modal distributed map (1800+ containers). Batch size 10, cpu=2 per
container. Rejected: too many containers for available quota.

**v3 (current)** — Single container, pure numpy, no SimpleITK. Stride-8
downsample to ~32³, centre-of-mass translation correction, NMI from joint
histograms. Very fast (~35s) but weak on ds2/ds3: CoM only corrects
translation, not rotation, so NMI is near-noise for independently deformed
pairs. **Score: 0.52.**

---

## strong_modal.py
RAD-DINO (microsoft/rad-dino) with CLS+mean-patch features (1536-dim),
triplet hard-negative mining, TTA, k-reciprocal re-ranking, self-training
on ds2/ds3 pseudo-labels. Projection heads fine-tuned on 350 ds1 pairs.

---

## hybrid_modal.py
3D NCC (48³ downsampled) ensembled with DINOv2 cosine similarity. NCC weight
0.7 for dataset1 only; DINOv2 alone for ds2/ds3.

---

## sift_baseline_modal.py
SIFT descriptors on 3 axial slices per volume, mean-pooled to 128-dim.
Cosine similarity for ranking. Weak cross-modal baseline.
