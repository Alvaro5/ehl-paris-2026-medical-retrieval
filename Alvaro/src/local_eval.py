"""Local evaluation harness for dataset1 (no Kaggle submissions burned).

Two responsibilities, both pure pandas/numpy CSV-and-metric logic:

  1. Split generation: carve dataset1/train_pairs.csv into an internal train
     pool and a held-out internal validation pool, emitting CSVs whose column
     schemas EXACTLY match the real Kaggle files (see CLAUDE.md) so the
     existing baseline can consume them unchanged.

  2. Scoring: compute MRR exactly as Kaggle does from a submission CSV plus a
     ground-truth mapping, with a macro_average helper for combining pools.

Run instantly on CPU. No torch / monai / nibabel here on purpose.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

# --- Canonical column schemas (must match CLAUDE.md / the real Kaggle CSVs) ---
TRAIN_PAIRS_COLS = [
    "pair_id", "query_id", "target_id",
    "query_image", "target_image",
    "query_modality", "target_modality", "dataset",
]
QUERIES_COLS = ["query_id", "query_image", "query_modality", "dataset"]
GALLERY_COLS = ["target_id", "target_image", "target_modality", "dataset"]
TRUTH_COLS = ["query_id", "target_id"]
SUBMISSION_COLS = ["query_id", "target_id_ranking"]


# ---------------------------------------------------------------------------
# 1) Split generation
# ---------------------------------------------------------------------------
def generate_split(
    pairs_csv: str,
    out_dir: str,
    n_val: int = 50,
    seed: int = 0,
    image_ext: str | None = None,
) -> Dict[str, str]:
    """Hold out `n_val` pairs as an internal 1:1 validation pool.

    Each row of train_pairs.csv is one same-subject (query, target) pair. We
    deterministically pick `n_val` whole rows for validation; the remaining
    rows stay as internal training pairs. Because we hold out whole pairs, the
    val gallery has exactly one true target per val query — a 1:1 retrieval
    pool, mirroring the real val/test sets.

    `image_ext` (e.g. ".nii") rewrites the image-path extension in the emitted
    CSVs. Default None keeps whatever the source CSV has (".nii.gz", matching
    the real Kaggle schema in CLAUDE.md). Use it only to point the split at a
    locally decompressed copy whose files on disk differ from the canonical
    paths — it changes path *values*, never the column schema.

    Returns a dict of the paths written.
    """
    pairs = pd.read_csv(pairs_csv)
    missing = [c for c in TRAIN_PAIRS_COLS if c not in pairs.columns]
    if missing:
        raise ValueError(f"{pairs_csv} missing expected columns: {missing}")

    if image_ext is not None:
        # Replace the canonical ".nii.gz" suffix on both image columns with the
        # requested extension. Only touch the known suffix so we never mangle
        # an unexpected path shape.
        for col in ("query_image", "target_image"):
            pairs[col] = pairs[col].str.replace(
                r"\.nii\.gz$", image_ext, regex=True
            )

    n_total = len(pairs)
    if not 0 < n_val < n_total:
        raise ValueError(
            f"n_val={n_val} must be in (0, {n_total}) for {n_total} pairs"
        )

    # Seeded permutation of row positions -> first n_val are validation.
    # RandomState(seed) makes the split reproducible across runs/machines.
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)
    val_pos = perm[:n_val]
    train_pos = perm[n_val:]

    # .iloc by integer position (perm holds positions, not index labels), then
    # restore a stable row order so the files are diff-friendly.
    val_pairs = pairs.iloc[val_pos].sort_values("pair_id").reset_index(drop=True)
    train_pairs = pairs.iloc[train_pos].sort_values("pair_id").reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "train_pairs": os.path.join(out_dir, "internal_train_pairs.csv"),
        "val_queries": os.path.join(out_dir, "internal_val_queries.csv"),
        "val_gallery": os.path.join(out_dir, "internal_val_gallery.csv"),
        "truth": os.path.join(out_dir, "internal_val_truth.csv"),
    }

    # Internal train: full train_pairs schema, untouched columns.
    train_pairs[TRAIN_PAIRS_COLS].to_csv(paths["train_pairs"], index=False)

    # Val queries: project the held-out queries to the queries schema.
    val_pairs[QUERIES_COLS].to_csv(paths["val_queries"], index=False)

    # Val gallery: project the held-out targets to the gallery schema.
    val_pairs[GALLERY_COLS].to_csv(paths["val_gallery"], index=False)

    # Ground truth: query_id -> its single true target_id.
    val_pairs[TRUTH_COLS].to_csv(paths["truth"], index=False)

    return paths


# ---------------------------------------------------------------------------
# 2) Scoring
# ---------------------------------------------------------------------------
def reciprocal_rank(ranking: Sequence[str], true_target: str) -> float:
    """1 / (1-indexed position of true_target in ranking); 0 if absent."""
    for i, tid in enumerate(ranking):
        if tid == true_target:
            return 1.0 / (i + 1)  # i is 0-indexed -> position is i+1
    return 0.0


def load_truth(truth_csv: str) -> Dict[str, str]:
    """Read internal_val_truth.csv -> {query_id: true target_id}."""
    df = pd.read_csv(truth_csv)
    return dict(zip(df["query_id"].astype(str), df["target_id"].astype(str)))


def mrr_from_submission(
    submission: pd.DataFrame,
    truth: Dict[str, str],
) -> float:
    """MRR over the GROUND-TRUTH queries (exactly as Kaggle scores).

    A query with no row in the submission, or whose true target is missing from
    its ranking, scores 0. MRR = mean of reciprocal ranks over all truth queries.
    """
    if not truth:
        return 0.0

    # query_id -> list of target_ids (best->worst), parsed from the
    # space-separated ranking string.
    sub_rankings: Dict[str, List[str]] = {}
    for qid, ranking_str in zip(
        submission["query_id"].astype(str),
        submission["target_id_ranking"].astype(str),
    ):
        sub_rankings[qid] = ranking_str.split()

    rrs = []
    for qid, true_target in truth.items():
        ranking = sub_rankings.get(qid)  # None -> query row missing -> 0
        rrs.append(reciprocal_rank(ranking, true_target) if ranking else 0.0)
    return float(np.mean(rrs))


def score_submission(submission_csv: str, truth_csv: str) -> float:
    """Convenience wrapper: load both CSVs and return MRR."""
    submission = pd.read_csv(submission_csv)
    missing = [c for c in SUBMISSION_COLS if c not in submission.columns]
    if missing:
        raise ValueError(f"{submission_csv} missing columns: {missing}")
    return mrr_from_submission(submission, load_truth(truth_csv))


def macro_average(per_pool_mrrs: Sequence[float]) -> float:
    """Average per-pool MRRs equally (the Kaggle ds1/ds2/ds3 mean).

    Feed it [ds1_mrr, ds2_mrr, ds3_mrr] once we have ds2/ds3 proxies.
    """
    if len(per_pool_mrrs) == 0:
        raise ValueError("macro_average needs at least one pool MRR")
    return float(np.mean(per_pool_mrrs))


# ---------------------------------------------------------------------------
# Sanity test (no pytest dependency; run via `selftest` subcommand)
# ---------------------------------------------------------------------------
def _run_sanity_tests() -> None:
    # Perfect ranking -> MRR 1.0 (true target first).
    truth = {"q1": "t1"}
    sub = pd.DataFrame(
        [["q1", "t1 t2 t3"]], columns=SUBMISSION_COLS
    )
    assert mrr_from_submission(sub, truth) == 1.0, "perfect rank should be 1.0"

    # True target at rank 2 -> 0.5.
    sub = pd.DataFrame([["q1", "t2 t1 t3"]], columns=SUBMISSION_COLS)
    assert mrr_from_submission(sub, truth) == 0.5, "rank 2 should be 0.5"

    # True target absent from ranking -> 0.
    sub = pd.DataFrame([["q1", "t2 t3 t4"]], columns=SUBMISSION_COLS)
    assert mrr_from_submission(sub, truth) == 0.0, "absent target should be 0"

    # Query row entirely missing from submission -> 0.
    sub = pd.DataFrame([["qX", "t1 t2"]], columns=SUBMISSION_COLS)
    assert mrr_from_submission(sub, truth) == 0.0, "missing query row should be 0"

    # Multi-query mean: rank 1 (1.0), rank 3 (1/3), absent (0) -> mean = 4/9.
    truth3 = {"q1": "t1", "q2": "t2", "q3": "t3"}
    sub = pd.DataFrame(
        [
            ["q1", "t1 t9 t8"],   # rank 1 -> 1.0
            ["q2", "t9 t8 t2"],   # rank 3 -> 1/3
            ["q3", "t9 t8 t7"],   # absent -> 0
        ],
        columns=SUBMISSION_COLS,
    )
    expected = (1.0 + 1.0 / 3.0 + 0.0) / 3.0
    got = mrr_from_submission(sub, truth3)
    assert abs(got - expected) < 1e-12, f"multi-query mean {got} != {expected}"

    # macro_average of per-pool MRRs.
    assert abs(macro_average([1.0, 0.5, 0.0]) - 0.5) < 1e-12
    assert macro_average([0.4]) == 0.4

    print("All sanity tests passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local eval harness for dataset1.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split", help="generate the internal train/val split")
    sp.add_argument("--pairs", default="dataset1/train_pairs.csv")
    sp.add_argument("--out-dir", default="splits")
    sp.add_argument("--n", type=int, default=50, help="held-out val pairs")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument(
        "--image-ext",
        default=None,
        help="rewrite image paths to this ext (e.g. .nii) for a locally "
        "decompressed copy; default keeps the canonical .nii.gz",
    )

    ss = sub.add_parser("score", help="score a submission against a truth file")
    ss.add_argument("--submission", required=True)
    ss.add_argument("--truth", default="splits/internal_val_truth.csv")

    sub.add_parser("selftest", help="run the built-in MRR sanity tests")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.cmd == "split":
        paths = generate_split(
            args.pairs, args.out_dir, args.n, args.seed, args.image_ext
        )
        print(f"Wrote split ({args.n} val pairs, seed={args.seed}):")
        for k, v in paths.items():
            print(f"  {k}: {v}")
    elif args.cmd == "score":
        mrr = score_submission(args.submission, args.truth)
        print(f"MRR: {mrr:.6f}")
    elif args.cmd == "selftest":
        _run_sanity_tests()


if __name__ == "__main__":
    main()
