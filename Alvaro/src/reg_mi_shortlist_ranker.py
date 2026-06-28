"""Shortlist-then-register ranking: plain MI proposes, registration re-ranks.

Real ds2 test is 100x100 = 10,000 registrations per pool -- too many for SyN. So
we only register a SHORTLIST: plain MI ranks the full gallery (cheap), we keep the
top-K, re-score ONLY those K with register-then-MI (reuse reg_mi_ranker), and
return the K re-ranked targets followed by the rest of the gallery in their
plain-MI order. That keeps every row a FULL, valid Kaggle ranking while paying for
only K registrations per query instead of the whole gallery.

Recall ceiling: a true target that misses plain MI's top-K can't be rescued -- run
shortlist_recall.py first to pick a safe K.

Same RankFn interface as evaluate.py:
    rank(query_id, query_path, {target_id: target_path}) -> [target_id ...]
"""

from __future__ import annotations

from typing import Dict, List

from .mi_ranker import MIRanker, GRID as PLAIN_GRID
from .reg_mi_ranker import RegMIRanker, REG_GRID


class ShortlistRegMIRanker:
    """Plain-MI shortlist + register-then-MI re-rank of the top-K.

    Holds one plain MIRanker (grid 64) and one RegMIRanker (grid 96); reusing the
    instance across a pool's queries reuses both caches. Registration runs at most
    K times per query.
    """

    def __init__(self, k: int = 25, transform: str = "Affine",
                 plain_grid: int = PLAIN_GRID, reg_grid: int = REG_GRID):
        self.k = k
        self.mi = MIRanker(grid=plain_grid)
        self.reg = RegMIRanker(grid=reg_grid, type_of_transform=transform)

    def rank(self, query_id: str, query_path: str, targets: Dict[str, str]) -> List[str]:
        # (a) full plain-MI ordering of the whole gallery (cheap, cached).
        plain_ids = [tid for tid, _ in self.mi.score_targets(query_path, targets)]

        # (b) shortlist = plain MI's top-K (clamp so K never exceeds the gallery).
        k = min(self.k, len(plain_ids))
        shortlist = plain_ids[:k]

        # (c) re-score ONLY the shortlist with register-then-MI, sort desc.
        reg_scored = [
            (tid, self.reg._reg_nmi(query_path, targets[tid])) for tid in shortlist
        ]
        reg_scored.sort(key=lambda x: -x[1])  # stable: ties keep shortlist order
        reranked = [tid for tid, _ in reg_scored]

        # (d) reranked top-K + the untouched plain-MI tail (== plain_ids[k:]).
        #     Disjoint by construction, so the result is a full, duplicate-free
        #     ranking of exactly len(targets) ids -- valid for Kaggle.
        return reranked + plain_ids[k:]


def make_reg_mi_shortlist_ranker(k: int = 25, transform: str = "Affine",
                                 plain_grid: int = PLAIN_GRID, reg_grid: int = REG_GRID):
    """Return an evaluate-ready RankFn backed by a fresh ShortlistRegMIRanker."""
    return ShortlistRegMIRanker(
        k=k, transform=transform, plain_grid=plain_grid, reg_grid=reg_grid
    ).rank
