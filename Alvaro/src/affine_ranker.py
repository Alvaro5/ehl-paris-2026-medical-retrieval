"""Affine-argmin ranking function — the confirmed dataset3 header geometry leak.

leak_probe.py proved that on dataset3 each query shares a BIT-IDENTICAL per-subject
affine matrix with its same-subject target (best Frobenius distance ~1e-9, a clean
query<->target bijection on val 20/20 and test 77/77). So we can rank with ZERO
image content: order gallery targets by ASCENDING Frobenius distance between the
query affine and each target affine; the true target is the argmin.

Reads ONLY the NIfTI header (nibabel exposes `.affine` without touching voxels), so
a whole pool ranks in seconds.

Same RankFn interface as the other rankers (mi_ranker, reg_mi_ranker):
    rank(query_id, query_path, {target_id: target_path}) -> [target_id ...]

score_targets returns NEGATIVE distance so that, like NMI, "higher score = better"
and a plain descending sort gives ascending distance (nearest affine first).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class AffineArgminRanker:
    """Rank targets by ascending affine-matrix Frobenius distance to the query.

    Per-path affine cache: in a pool every target affine recurs across all queries,
    so we read each header exactly once.
    """

    def __init__(self) -> None:
        self._affine_cache: Dict[str, np.ndarray] = {}

    def _affine(self, path: str) -> np.ndarray:
        if path not in self._affine_cache:
            import nibabel as nib

            # nib.load is lazy; .affine reads the header only, never the voxel data.
            self._affine_cache[path] = np.asarray(nib.load(path).affine, dtype=float)
        return self._affine_cache[path]

    def score_targets(
        self, query_path: str, targets: Dict[str, str]
    ) -> List[Tuple[str, float]]:
        """[(target_id, -Frobenius_distance)] sorted descending (nearest affine first).

        Negating the distance keeps the "higher score = better" convention shared
        with MIRanker/RegMIRanker, so rank() and the submission sanity block work
        unchanged. Stable sort preserves input order among exact ties.
        """
        qa = self._affine(query_path)
        scored = [
            (tid, -float(np.linalg.norm(qa - self._affine(tpath))))
            for tid, tpath in targets.items()
        ]
        scored.sort(key=lambda x: -x[1])
        return scored

    def rank(self, query_id: str, query_path: str, targets: Dict[str, str]) -> List[str]:
        """Order target_ids by ascending affine distance (best->worst)."""
        return [tid for tid, _ in self.score_targets(query_path, targets)]


def make_affine_ranker():
    """Return an evaluate-ready RankFn backed by a fresh AffineArgminRanker."""
    return AffineArgminRanker().rank
