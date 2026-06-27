# EHL Paris 2026 — Brain MRI Cross-Modal Retrieval

## Goal
Cross-modal medical image retrieval. For each query brain MRI volume (T1
post-contrast, "ceT1"), rank all gallery volumes (T2) so the same-subject T2
ranks as high as possible. This is a metric-learning / re-identification task.

## The metric (drives all strategy)
Score = mean of (dataset1_MRR, dataset2_MRR, dataset3_MRR), equal weight.
MRR = mean over queries of 1/(rank of true target). Missing/omitted target = 0.
**Two-thirds of the score comes from dataset2/3, which have NO training labels.**
This is fundamentally a GENERALIZATION problem, not a fit-dataset1 problem.

## The three datasets
- dataset1: registered common grid, intact anatomy, T1ce↔T2. 350 labeled train
  pairs + val(40) + test(100). The EASY case. Confirmed BraTS-like geometry
  (240×240×155, 1.0mm isotropic).
- dataset2: query & target deformed INDEPENDENTLY (rigid + non-linear) → no
  shared grid. No labels. val(40)+test(100). Breaks geometry.
- dataset3: pre-op → intra-op, resampled to ~same space but anatomy is
  surgically altered (tissue missing/shifted). No labels. val(20)+test(77).
  Breaks content.

## Baseline (do NOT edit slice_clip_baseline.py)
"SliceCLIP": 3 axial slices per volume → 2 tiny from-scratch 2D CNNs (separate
query/target encoders) → CLIP-style in-batch contrastive loss on dataset1 pairs
→ rank by cosine similarity. Known weaknesses: discards ~98% of the volume,
fixed slice positions assume alignment (fails on ds2/ds3), no deformation
augmentation. It's the thing to beat.

## Environment
- Use uv. Project venv at .venv (`uv venv`, `source .venv/bin/activate`).
- Do NOT use conda base. CPU-only for now; GPU TBD.
- Run scripts with the venv's python (or `uv run`). Never absolute conda paths.

## Data layout (DATA_ROOT)
datasetN/{train_pairs,val_queries,val_gallery,test_queries,test_gallery}.csv + images/
- train_pairs.csv: pair_id,query_id,target_id,query_image,target_image,query_modality,target_modality,dataset
- *_queries.csv: query_id,query_image,query_modality,dataset
- *_gallery.csv: target_id,target_image,target_modality,dataset
- submission: query_id,target_id_ranking  (space-separated, full gallery, best→worst)

## Constraints / conventions
- Kaggle: max 100 submissions/team/day. Iterate on the LOCAL eval harness; submit rarely.
- Small, verifiable functions. Explain non-obvious logic in comments.
- Don't introduce heavy deps where pandas/numpy suffice.

## Status
- [x] Repo forked, data downloaded, uv set up, data inspected (ds1 = BraTS geometry)
- [ ] Local eval harness (in progress)
- [ ] Baseline run + first three val MRR numbers
- [ ] Deformation augmentation to simulate ds2/ds3

## Local-eval noise bar
A single 50-query seed has ~±0.05 MRR of sampling noise — only trust changes LARGER than ±0.05 (re-run with a couple of seeds before believing a smaller delta).

## Proxy workflow (measure generalization without Kaggle)
Train the baseline on the CLEAN internal split (splits/internal_train_pairs.csv), then eval the trained model on the synthesized proxies (`python -m Alvaro.src.simulate --mode ds2|ds3` → splits/proxy_ds2/ & splits/proxy_ds3/, each with val_queries.csv/val_gallery.csv) and score the resulting submissions against splits/internal_val_truth.csv (IDs are preserved, so the same truth file scores all three). The ds1-clean vs proxy_ds2 vs proxy_ds3 MRR spread is the local stand-in for the real ds1/ds2/ds3 mean.