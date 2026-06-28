"""Register-then-MI ranking function (ANTs affine + reused NMI).

Plain MI (mi_ranker.py) assumes the query and target already share a grid, so it
collapses on ds2 where the two volumes were deformed INDEPENDENTLY. Here we first
register each target to the query, then score MI on the REGISTERED pair. If the
ds2 collapse is purely a misalignment problem, registration should recover it.

Per query<->target pair:
  1. load + downsample both volumes to grid^3 (default 96; registration needs more
     detail than the 64 plain MI uses),
  2. ants.registration(fixed=query, moving=target, type_of_transform=...) -> take
     the "warpedmovout" (the target resampled into the query's space),
  3. NMI(query, registered_target) via the EXISTING mi_ranker code (digitize_array
     + _nmi -- not reimplemented),
  4. rank gallery targets by that post-registration NMI.

type_of_transform is a parameter (default "Affine") so we can later try "SyN".

Same RankFn interface as evaluate.py expects:
    rank(query_id, query_path, {target_id: target_path}) -> [target_id ...]

Run the plain-vs-registered comparison:
    python -m Alvaro.src.reg_compare
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

# Reuse the existing harness pieces -- do NOT reimplement NMI or resampling.
from .mi_ranker import _nmi, _resample_to_grid, digitize_array

REG_GRID = 96  # default resample edge for registration (vs 64 for plain MI)


class RegMIRanker:
    """Register target->query, then score reused NMI. Pairwise, so no MI cache.

    We DO cache the loaded+downsampled array, its ANTs image, and (for the query
    side only) its digitized form per path: in a pool every target recurs across
    all queries and vice versa, so caching the downsample/convert avoids redundant
    disk reads. The registration itself depends on BOTH volumes, so it is the one
    step we must redo for every (query, target) pair.
    """

    def __init__(self, grid: int = REG_GRID, type_of_transform: str = "Affine"):
        import ants  # imported here so importing this module stays cheap

        self.ants = ants
        self.grid = grid
        self.transform = type_of_transform
        self._arr_cache: Dict[str, np.ndarray] = {}          # path -> grid^3 float
        self._ants_cache: Dict[str, "ants.ANTsImage"] = {}   # path -> ANTs image
        self._qdigit_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # query digit/mask
        self.n_reg = 0  # registrations performed (for progress/cost reporting)

    def _arr(self, path: str) -> np.ndarray:
        if path not in self._arr_cache:
            import nibabel as nib

            v = nib.load(path).get_fdata(dtype=np.float32)
            if v.ndim == 4:
                v = v[..., 0]
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            self._arr_cache[path] = _resample_to_grid(v, self.grid)
        return self._arr_cache[path]

    def _ants_img(self, path: str):
        if path not in self._ants_cache:
            self._ants_cache[path] = self.ants.from_numpy(self._arr(path))
        return self._ants_cache[path]

    def _query_digit(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        # Query is the fixed image: its digitization is identical for all targets.
        if path not in self._qdigit_cache:
            self._qdigit_cache[path] = digitize_array(self._arr(path))
        return self._qdigit_cache[path]

    def _reg_nmi(self, query_path: str, target_path: str) -> float:
        """NMI between the query and the target registered into the query's space."""
        fixed = self._ants_img(query_path)
        moving = self._ants_img(target_path)
        reg = self.ants.registration(
            fixed=fixed, moving=moving, type_of_transform=self.transform
        )
        self.n_reg += 1
        warped = reg["warpedmovout"].numpy()  # target resampled onto the query grid
        qd, qm = self._query_digit(query_path)
        td, tm = digitize_array(warped)        # registration-specific -> not cached
        return _nmi(qd, qm, td, tm)

    def score_targets(
        self, query_path: str, targets: Dict[str, str]
    ) -> List[Tuple[str, float]]:
        """[(target_id, post-registration NMI)] sorted descending.

        Mirrors MIRanker.score_targets so this ranker is drop-in for the same
        callers (rank, the submission sanity block). rank drops the scores; the
        sanity block keeps them to show top-vs-tail separation on real data.
        """
        scored: List[Tuple[str, float]] = []
        for tid, tpath in targets.items():
            try:
                score = self._reg_nmi(query_path, tpath)
            except Exception as e:  # one bad registration shouldn't abort the pool
                print(f"[reg-mi] registration/NMI error ({tid}): {e}")
                score = 0.0
            scored.append((tid, score))
        scored.sort(key=lambda x: -x[1])  # stable: ties keep input order
        return scored

    def rank(self, query_id: str, query_path: str, targets: Dict[str, str]) -> List[str]:
        """Order target_ids by descending post-registration NMI with the query."""
        return [tid for tid, _ in self.score_targets(query_path, targets)]


def make_reg_mi_ranker(grid: int = REG_GRID, type_of_transform: str = "Affine"):
    """Return an evaluate-ready RankFn backed by a fresh RegMIRanker."""
    return RegMIRanker(grid=grid, type_of_transform=type_of_transform).rank


if __name__ == "__main__":
    # Quick single-pool smoke (ds2 proxy) with progress + timing, no full sweep.
    from . import evaluate

    ranker = RegMIRanker()
    pools = evaluate.get_pools()
    spec = pools["ds2_proxy"]
    t0 = time.time()
    mrr = evaluate.evaluate_ranker(
        ranker.rank, spec["queries"], spec["gallery"], spec["truth"]
    )
    dt = time.time() - t0
    print(f"ds2_proxy reg-then-MI MRR={mrr:.4f}  "
          f"({ranker.n_reg} registrations, {dt:.1f}s, {dt/max(ranker.n_reg,1):.2f}s/reg)")
