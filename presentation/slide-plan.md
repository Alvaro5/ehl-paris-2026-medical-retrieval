# Slide Plan: Cross-Modal Brain MRI Retrieval

**Source:** `SOLUTION_WRITEUP.md`
**Goal:** Present hackathon solution at EHL Paris 2026 to Inria PI + jury
**Audience:** Inria research PI + jury — technical, rewards understanding the problem over leaderboard numbers
**Takeaway:** A principled, training-free approach grounded in signal theory outperformed deep learning baselines by correctly diagnosing the problem as alignment, not representation.
**Slide count:** 10
**Narrative arc:** From the obvious failure (DINOv2), through the correct reframe (alignment), to a per-dataset diagnosis that drives a principled fix at each step — ending with a transparently reported benchmark finding.

---

## Slide Plan

| Slide | Source Section | Content | Claims |
|-------|---------------|---------|--------|
| 1. Title | Header | "Cross-Modal Brain MRI Retrieval" · EHL Paris 2026 · Inria / Paris Brain Institute · team name / presenter | — |
| 2. The Task | §1 | Cross-modal retrieval: match T1 (query) to T2 (same patient) in a gallery. MRR metric. Three datasets with fundamentally different geometry — table of dataset properties. Counterintuitive point: ds3 looks hardest (surgery) but ds2 is the real hard case. | V: dataset property table from §1 |
| 3. The Trap | §3.1 | We tried the obvious move: DINOv2 ViT-B/14, 3-slice embeddings, cosine similarity. Result: ~0.20 (barely above random). Three structural reasons it fails: global pooling discards spatial position; domain gap (natural photos → MRI); no modality bridge. | V: ~0.20 (public LB, §5); I: 3-reason analysis |
| 4. The Reframe | §0, §3.2 | Insight: this is an *alignment* problem. The identifying signal is whether voxels correspond — which deep embeddings destroy. The right question is not "do these look similar?" but "do the same voxels carry the same anatomy?" | I: framing; V: MI's voxel-level logic from §3.2 |
| 5. The Method: Mutual Information | §3.2, §6 | No neural network, no training. NMI = (H(q) + H(t)) / H(q,t). Intuition: same patient → aligned anatomy → knowing T1 voxel predicts T2 voxel → high MI. Wrong patient → anatomy misaligned → relationship washes out → low MI. Standard metric in multimodal medical registration for decades. | V: formula and mechanism from §3.2 and §6 |
| 6. First Results + Diagnosis | §3.2 | MI per-dataset real LB results table: ds1 0.98, ds2 0.18, ds3 0.98. Combined 0.713. "This table IS the diagnosis: dataset 2 is the entire gap." MI fails on ds2 because its geometry is deliberately broken — voxels no longer correspond. | V: all numbers from §3.2 table (isolated Kaggle submissions) |
| 7. The Fix: Affine Registration | §3.3 | Dataset 2 breaks MI because query and target are independently warped. Fix: register target onto query first (ANTs affine), then score with MI. Proxy: ds2 jumped 0.29→0.98. Real validation: 0.18→0.88. Combined submission: 0.951. Selective design: only apply where needed. | V: 0.18→0.88 from §3.3 (held-out ds2-val); V: 0.951 combined LB; V: proxy numbers from §3.3 |
| 8. Knowing Your Tools | §3.4 | We hypothesized deformable (SyN) would beat affine since ds2 has non-linear deformation. Tested on real ds2-val. Result: SyN 0.846 vs affine 0.884 — worse, at 2.3× the compute. Why: a dense warp can make wrong patients fit too, shrinking the gap between true match and distractors. Lesson: optimal registration flexibility for retrieval < for registration quality. Kept affine. | V: 0.846 vs 0.884 (§3.4); I: mechanistic explanation |
| 9. Benchmark Finding | §3.5 | Three teams hit exactly 1.00000. We ran a header probe: ds3 affine matrices are bit-identical between matching pairs (distance ~1e-9), 20/20 val, 77/77 test — a perfect bijection. Caused by how ds3 was built (target resampled into query's space). Ranking by nearest affine → MRR 1.00 with zero image content. We report both our content-based result (MI ~0.98) and the leak, and flag it to organizers. | V: 1e-9 distance, 20/20, 77/77 (§3.5); V: construction mechanism from README (§3.5) |
| 10. Final Results + Takeaway | §5, §0 | Journey: DINOv2 0.20 → MI 0.71 → Affine+MI 0.95 → Final 0.96. Per-dataset final table. One remaining gap: ds2 at 0.88 (top teams likely have a content-based method we don't). Takeaway sentence. | V: all numbers from §5; I: "top teams have stronger ds2 method" (§7) |

---

## Sections excluded from slides

| Section | Reason |
|---------|--------|
| §2 — How we worked (methodology) | Process story (local harness, proxies, diagnose→simulate→validate→scale); important context but breaks narrative pace for a 3–5 min talk. Can answer in Q&A. |
| §6 — Scientific background | Reference / Q&A prep. Concepts are explained inline on relevant slides. |
| §9 — Anticipated jury questions | Q&A prep only — not slide content. |
| §10 — Reproducibility | Code structure and reproduce commands; reference material, not presentation content. |

---

## Styling Options

### Option A: Academic Clean
- Palette: white (#FFFFFF), near-black text (#1C1C1E), indigo accent (#3B5BDB), soft highlight (#EEF2FF)
- Typography: Inter headings, system sans-serif body
- Density: Standard (5–7 items)
- Feel: Conference poster, readable at distance, neutral and professional

### Option B: Research Bold
- Palette: deep navy (#0D1B2A), white text (#F0F0F0), amber accent (#F59E0B), dark highlight (#1E3A5F)
- Typography: Space Grotesk headings, Inter body
- Density: Minimal (large text, 3–4 items, numbers dominate)
- Feel: High-contrast, assertive, numbers pop

### Option C: Clinical Minimal
- Palette: soft gray (#F4F6F8), dark text (#1A1A2E), teal accent (#0E9F8C), white card backgrounds
- Typography: IBM Plex Sans headings, system body
- Density: Standard with card layout
- Feel: Medical/scientific, structured, clean
