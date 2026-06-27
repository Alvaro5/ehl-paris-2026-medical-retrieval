"""Phase A diagnostic: one comparison table, method x pool -> MRR.

Runs the DINOv2 zero-shot ranker and the mutual-information ranker across the
three local pools (ds1_internal, ds2_proxy, ds3_proxy) and prints a single table
plus the per-method macro-average (the local stand-in for the Kaggle ds1/ds2/ds3
mean). Both rankers plug into Alvaro.src.evaluate unchanged.

    python -m Alvaro.src.compare
"""

from __future__ import annotations

from . import evaluate
from .dinov2_ranker import make_dinov2_ranker
from .mi_ranker import make_mi_ranker

POOLS = ["ds1_internal", "ds2_proxy", "ds3_proxy"]


def main() -> None:
    methods = {
        "dinov2": make_dinov2_ranker,
        "mutual_information": make_mi_ranker,
    }

    rows = {}
    for name, factory in methods.items():
        print(f"\n=== {name} ===")
        rows[name] = evaluate.evaluate_across_pools(factory())

    # --- Render the table ---------------------------------------------------
    header = f"{'method':<20}" + "".join(f"{p:>14}" for p in POOLS) + f"{'macro_avg':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, res in rows.items():
        cells = "".join(f"{res[p]:>14.4f}" for p in POOLS)
        macro = evaluate.local_eval.macro_average([res[p] for p in POOLS])
        print(f"{name:<20}{cells}{macro:>12.4f}")
    print("=" * len(header))


if __name__ == "__main__":
    main()
