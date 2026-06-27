"""Reusable evaluation engine for Phase A diagnostics.

Given a ranking function and a (queries, gallery, truth) pool, run the ranker
over every query, assemble a Kaggle-shaped submission, and score it with the
EXISTING MRR in local_eval.py (we do NOT reimplement MRR). `evaluate_across_pools`
runs a ranker over the three local pools that stand in for the real ds1/ds2/ds3:

  - ds1_internal : clean held-out dataset1 val split (the EASY case)
  - ds2_proxy    : independent rigid+elastic warp on both sides (breaks geometry)
  - ds3_proxy    : resection + scanner shift on the target (breaks content)

A ranking function has the signature:

    rank(query_id: str,
         query_path: str,
         targets: dict[target_id -> target_path]) -> list[target_id]

returning ALL target_ids best->worst. The engine handles CSV I/O and scoring so a
ranker only has to turn one query + the gallery into an ordering.

Run the smoke test:
    python -m Alvaro.src.evaluate            # random-ranker wiring check
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List

import pandas as pd

# Import the canonical MRR scorer. Works whether this file is imported as a
# package module (python -m Alvaro.src.evaluate) or run as a loose script.
try:
    from . import local_eval
except ImportError:  # pragma: no cover - fallback for direct-script execution
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    import local_eval  # type: ignore

# A ranker maps (query_id, query_path, {tid: tpath}) -> ordered list of tids.
RankFn = Callable[[str, str, Dict[str, str]], List[str]]

# Repo root = data root. This file is Alvaro/src/evaluate.py, so parents[2] is the
# repo root that the CSV image paths are relative to.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Where we write the rebuilt proxy_ds3 CSVs (the committed ones are truncated).
# Kept strictly inside Alvaro/ per the hard rule.
GENERATED_DIR = os.path.join(REPO_ROOT, "Alvaro", "generated")

TRUTH_CSV = os.path.join(REPO_ROOT, "splits", "internal_val_truth.csv")


# ---------------------------------------------------------------------------
# proxy_ds3 CSV repair (inside Alvaro/ only)
# ---------------------------------------------------------------------------
def ensure_proxy_ds3_csvs() -> Dict[str, str]:
    """Rebuild full proxy_ds3 query/gallery CSVs pointing at on-disk images.

    The repo's splits/proxy_ds3/val_*.csv were truncated to one row by a
    `--limit 1` re-run, but all 100 perturbed images (50 query + 50 target IDs)
    are present under splits/proxy_ds3/images/. We are not allowed to rewrite the
    files outside Alvaro/, so we synthesize correct CSVs INTO Alvaro/generated/.

    We take the ID lists and modality columns from the clean internal-val CSVs
    (same IDs are preserved end to end) and repoint the image path at the
    matching .nii.gz under splits/proxy_ds3/images/. Returns the written paths.
    """
    out_dir = os.path.join(GENERATED_DIR, "proxy_ds3")
    os.makedirs(out_dir, exist_ok=True)
    img_rel = os.path.join("splits", "proxy_ds3", "images")

    q = pd.read_csv(os.path.join(REPO_ROOT, "splits", "internal_val_queries.csv"))
    g = pd.read_csv(os.path.join(REPO_ROOT, "splits", "internal_val_gallery.csv"))

    # Repoint each image column at the perturbed .nii.gz copy by ID.
    q = q.copy()
    q["query_image"] = q["query_id"].astype(str).map(
        lambda i: os.path.join(img_rel, f"{i}.nii.gz")
    )
    g = g.copy()
    g["target_image"] = g["target_id"].astype(str).map(
        lambda i: os.path.join(img_rel, f"{i}.nii.gz")
    )

    # Guard: every referenced image must actually exist on disk, else scoring
    # this pool would silently be meaningless.
    for col, df in (("query_image", q), ("target_image", g)):
        for rel in df[col]:
            if not os.path.exists(os.path.join(REPO_ROOT, rel)):
                raise FileNotFoundError(
                    f"proxy_ds3 image missing on disk: {rel}. "
                    f"Re-run simulate.py --mode ds3 (outside Alvaro/) to rebuild."
                )

    qcsv = os.path.join(out_dir, "val_queries.csv")
    gcsv = os.path.join(out_dir, "val_gallery.csv")
    q[local_eval.QUERIES_COLS].to_csv(qcsv, index=False)
    g[local_eval.GALLERY_COLS].to_csv(gcsv, index=False)
    return {"queries": qcsv, "gallery": gcsv}


# ---------------------------------------------------------------------------
# Pool registry
# ---------------------------------------------------------------------------
def get_pools() -> Dict[str, Dict[str, str]]:
    """Return {pool_name: {queries, gallery, truth}} absolute CSV paths.

    All three pools share one truth file: the proxies preserve the original IDs,
    so the same query_id -> target_id mapping scores every pool.
    """
    ds3 = ensure_proxy_ds3_csvs()
    return {
        "ds1_internal": {
            "queries": os.path.join(REPO_ROOT, "splits", "internal_val_queries.csv"),
            "gallery": os.path.join(REPO_ROOT, "splits", "internal_val_gallery.csv"),
            "truth": TRUTH_CSV,
        },
        "ds2_proxy": {
            "queries": os.path.join(REPO_ROOT, "splits", "proxy_ds2", "val_queries.csv"),
            "gallery": os.path.join(REPO_ROOT, "splits", "proxy_ds2", "val_gallery.csv"),
            "truth": TRUTH_CSV,
        },
        "ds3_proxy": {
            "queries": ds3["queries"],
            "gallery": ds3["gallery"],
            "truth": TRUTH_CSV,
        },
    }


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------
def _resolve(rel: str) -> str:
    """Absolute path for a CSV image entry (relative to the repo/data root)."""
    return rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)


def evaluate_ranker(
    rank_fn: RankFn,
    queries_csv: str,
    gallery_csv: str,
    truth_csv: str,
) -> float:
    """Run `rank_fn` over one pool and return its MRR (via local_eval).

    Builds {target_id: absolute_path}, calls the ranker once per query, collects
    a (query_id, "tid tid ...") submission, and scores it exactly as Kaggle does.
    """
    queries = pd.read_csv(queries_csv)
    gallery = pd.read_csv(gallery_csv)

    # Gallery is identical for every query in the pool; build the dict once.
    targets = {
        str(row["target_id"]): _resolve(str(row["target_image"]))
        for _, row in gallery.iterrows()
    }

    rows = []
    for _, qrow in queries.iterrows():
        qid = str(qrow["query_id"])
        qpath = _resolve(str(qrow["query_image"]))
        ranking = rank_fn(qid, qpath, targets)
        rows.append({"query_id": qid, "target_id_ranking": " ".join(ranking)})

    submission = pd.DataFrame(rows, columns=local_eval.SUBMISSION_COLS)
    truth = local_eval.load_truth(truth_csv)
    return local_eval.mrr_from_submission(submission, truth)


def evaluate_across_pools(
    rank_fn: RankFn,
    pools: Dict[str, Dict[str, str]] | None = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate one ranker across all pools; return {pool_name: MRR}."""
    pools = pools or get_pools()
    results: Dict[str, float] = {}
    for name, spec in pools.items():
        if verbose:
            print(f"[evaluate] {name} ...", flush=True)
        mrr = evaluate_ranker(
            rank_fn, spec["queries"], spec["gallery"], spec["truth"]
        )
        results[name] = mrr
        if verbose:
            print(f"[evaluate] {name}: MRR={mrr:.4f}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Smoke test: a trivial random ranker should land near the random-MRR baseline
# ---------------------------------------------------------------------------
def make_random_ranker(seed: int = 0) -> RankFn:
    """A ranker that ignores the images and shuffles the gallery deterministically.

    Seeded per-query (seed + hash(qid)) so the wiring is reproducible. Expected
    MRR for a single true target uniformly placed among N is the Nth harmonic
    number over N; for N=50 that is ~0.09 — our sanity target.
    """
    import random

    def rank(query_id: str, query_path: str, targets: Dict[str, str]) -> List[str]:
        tids = list(targets.keys())
        rng = random.Random(f"{seed}:{query_id}")
        rng.shuffle(tids)
        return tids

    return rank


if __name__ == "__main__":
    print(f"REPO_ROOT = {REPO_ROOT}")
    pools = get_pools()
    print("Pools and sizes:")
    for name, spec in pools.items():
        nq = len(pd.read_csv(spec["queries"]))
        ng = len(pd.read_csv(spec["gallery"]))
        print(f"  {name}: {nq} queries, {ng} targets")
    print("\nSmoke test — random ranker (expect ~0.09 on 50-target pools):")
    results = evaluate_across_pools(make_random_ranker(seed=0))
    print("\nRandom-ranker MRR per pool:")
    for name, mrr in results.items():
        print(f"  {name:14s} {mrr:.4f}")
