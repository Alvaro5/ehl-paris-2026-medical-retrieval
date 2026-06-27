"""Mutual-information ranking function (training-free), CPU-only.

Score each query<->target pair by the NORMALIZED MUTUAL INFORMATION of their
voxel intensities on a shared downsampled grid:

    NMI = (H(A) + H(B)) / H(A, B)

which is ~1 when the two volumes are statistically independent and ~2 when one
perfectly predicts the other; higher = more similar. MI is the classic
cross-modal (T1<->T2) registration similarity precisely because it does not
assume matching intensities, only matching STRUCTURE.

The crux is the SHARED GRID: every volume is resampled to a fixed 64^3 index
grid, so corresponding indices mean corresponding anatomy ONLY when the volumes
are spatially aligned. That is true for ds1 (registered) and roughly true for ds3
(~same space), but broken for ds2 (independent deformation) -- which is exactly
the failure pattern we want to expose.

Run across the three pools:
    python -m Alvaro.src.mi_ranker
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

GRID = 64       # cube edge of the shared resample grid
BINS = 48       # intensity bins for the (joint) histogram
EPS = 1e-12


def _resample_to_grid(vol: np.ndarray, edge: int = GRID) -> np.ndarray:
    """Resample a 3D volume to a fixed (edge, edge, edge) grid with linear interp.

    Fixed OUTPUT shape (not a fixed zoom factor) so every volume lands on the same
    index grid regardless of its native dimensions -- a precondition for comparing
    intensities index-by-index.
    """
    from scipy.ndimage import zoom

    factors = [edge / s for s in vol.shape]
    return zoom(vol, factors, order=1)


def digitize_array(vol: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a float volume into (BINS-level bin indices, brain mask).

    Intensities are min/max normalized WITHIN the brain mask (nonzero voxels),
    then digitized into BINS levels. Returns int16 bin indices and a bool mask.
    Background voxels get bin 0 but are excluded via the mask, so they never enter
    the joint histogram. Shared by _prep_volume (path-based) and reg_mi_ranker
    (which digitizes an already-registered in-memory array).
    """
    mask = vol > 0  # brain = nonzero
    digit = np.zeros(vol.shape, dtype=np.int16)
    if mask.any():
        vals = vol[mask]
        mn, mx = float(vals.min()), float(vals.max())
        if mx > mn:
            # Scale to [0, BINS-1]; clip guards the exact-max voxel landing on BINS.
            scaled = (vals - mn) / (mx - mn) * (BINS - 1)
            digit[mask] = np.clip(np.round(scaled), 0, BINS - 1).astype(np.int16)
    return digit, mask


def _prep_volume(path: str, edge: int = GRID) -> Tuple[np.ndarray, np.ndarray]:
    """Load -> resample to edge^3 -> (digitized bin indices, brain mask)."""
    import nibabel as nib

    vol = nib.load(path).get_fdata(dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    vol = _resample_to_grid(vol, edge)
    return digitize_array(vol)


def _entropy_from_counts(counts: np.ndarray) -> float:
    """Shannon entropy (nats) of a histogram given raw bin counts."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def _nmi(digit_a: np.ndarray, mask_a: np.ndarray,
         digit_b: np.ndarray, mask_b: np.ndarray) -> float:
    """Normalized mutual information over the voxels where BOTH brains exist."""
    mask = mask_a & mask_b
    if mask.sum() < 2:
        # No shared support -> treat as independent (NMI floor of 1.0).
        return 1.0
    a = digit_a[mask].astype(np.int64)
    b = digit_b[mask].astype(np.int64)

    # Joint histogram via a flattened (a, b) index + bincount -- fast and exact.
    joint = np.bincount(a * BINS + b, minlength=BINS * BINS).reshape(BINS, BINS)

    h_a = _entropy_from_counts(joint.sum(axis=1))  # marginal over A
    h_b = _entropy_from_counts(joint.sum(axis=0))  # marginal over B
    h_ab = _entropy_from_counts(joint.ravel())     # joint
    if h_ab < EPS:
        return 2.0  # both constant on the overlap -> maximally "dependent"
    return (h_a + h_b) / h_ab


class MIRanker:
    """Stateful NMI ranker with a per-path (digit, mask) cache.

    `grid` sets the shared resample cube edge (default GRID=64). Each instance has
    one grid and one cache, so reusing a single instance across many query/gallery
    pools loads and downsamples each volume exactly once.
    """

    def __init__(self, grid: int = GRID):
        self.grid = grid
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def _get(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        if path not in self._cache:
            self._cache[path] = _prep_volume(path, self.grid)
        return self._cache[path]

    def score_targets(
        self, query_path: str, targets: Dict[str, str]
    ) -> List[Tuple[str, float]]:
        """[(target_id, NMI)] sorted by descending NMI with the query.

        The single source of truth for ordering: `rank` drops the scores, the
        sanity block keeps them. Stable sort on the negated score preserves input
        order among ties.
        """
        qd, qm = self._get(query_path)
        scored = [
            (tid, self._nmi_or_zero(qd, qm, *self._get(tpath)))
            for tid, tpath in targets.items()
        ]
        scored.sort(key=lambda x: -x[1])
        return scored

    def rank(self, query_id: str, query_path: str, targets: Dict[str, str]) -> List[str]:
        """Order target_ids by descending NMI with the query."""
        return [tid for tid, _ in self.score_targets(query_path, targets)]

    @staticmethod
    def _nmi_or_zero(qd, qm, td, tm) -> float:
        try:
            return _nmi(qd, qm, td, tm)
        except Exception as e:  # never let one pair abort the pool
            print(f"[mi] NMI error: {e}")
            return 0.0


def make_mi_ranker(grid: int = GRID):
    """Return an evaluate-ready RankFn backed by a fresh MIRanker cache."""
    return MIRanker(grid=grid).rank


if __name__ == "__main__":
    from . import evaluate

    results = evaluate.evaluate_across_pools(make_mi_ranker())
    print("\nMutual-information MRR per pool:")
    for name, mrr in results.items():
        print(f"  {name:14s} {mrr:.4f}")
