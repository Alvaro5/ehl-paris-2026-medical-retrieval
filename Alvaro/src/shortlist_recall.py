"""Shortlist recall ceiling for plain MI (registration-free, runs in seconds).

Before we commit to "plain MI shortlists -> register only the top-K", we need to
know how large K must be so the TRUE target is almost never thrown away by the
shortlist. For each pool we rank every query's gallery with the EXISTING plain MI
ranker and report, per K, the fraction of queries whose true target lands in the
top-K. That fraction is the hard recall CEILING of any shortlist-then-rerank
scheme at that K -- re-ranking can only reorder what the shortlist kept.

    python -m Alvaro.src.shortlist_recall
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from . import evaluate
from .mi_ranker import MIRanker

POOLS = ["ds1_internal", "ds2_proxy", "ds3_proxy"]
KS = [5, 10, 15, 20, 25, 30]


def true_target_ranks(pool_spec: Dict[str, str]) -> List[int]:
    """Plain-MI 1-indexed rank of each query's true target within its gallery.

    One MIRanker per pool (its cache embeds each volume once). Mirrors how
    evaluate builds the gallery dict, but keeps the true target's position so we
    can compute top-K recall instead of MRR.
    """
    queries = pd.read_csv(pool_spec["queries"])
    gallery = pd.read_csv(pool_spec["gallery"])
    truth = evaluate.local_eval.load_truth(pool_spec["truth"])

    targets = {
        str(r["target_id"]): evaluate._resolve(str(r["target_image"]))
        for _, r in gallery.iterrows()
    }

    mi = MIRanker()  # default grid=64, same as plain-MI everywhere else
    ranks: List[int] = []
    for _, qrow in queries.iterrows():
        qid = str(qrow["query_id"])
        qpath = evaluate._resolve(str(qrow["query_image"]))
        ordered = [tid for tid, _ in mi.score_targets(qpath, targets)]
        true_tid = truth[qid]
        # 1-indexed position; len+1 sentinel if somehow absent (shouldn't happen).
        ranks.append(ordered.index(true_tid) + 1 if true_tid in ordered else len(ordered) + 1)
    return ranks


def recall_at_k(ranks: List[int], k: int) -> float:
    """Fraction of queries whose true target rank is within the top-k."""
    return sum(1 for r in ranks if r <= k) / len(ranks)


def main() -> None:
    pools = evaluate.get_pools()
    pool_ranks = {name: true_target_ranks(pools[name]) for name in POOLS}

    header = f"{'K':>4}" + "".join(f"{p:>16}" for p in POOLS)
    print("\nPlain-MI shortlist recall (fraction of true targets within top-K)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for k in KS:
        row = "".join(f"{recall_at_k(pool_ranks[p], k):>16.3f}" for p in POOLS)
        print(f"{k:>4}{row}")
    print("=" * len(header))
    # Gallery size context: recall must hit 1.0 by K = gallery size.
    sizes = {p: len(pool_ranks[p]) for p in POOLS}
    print("queries per pool:", sizes)


if __name__ == "__main__":
    main()
