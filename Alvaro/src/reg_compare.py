"""Compare PLAIN MI vs REGISTER-THEN-MI across the three local pools.

The key question: does registering the target to the query before computing MI
recover ds2_proxy (where plain MI collapses to ~0.29)? And does it leave ds1/ds3
roughly intact (registering already-aligned images shouldn't hurt much)?

    python -m Alvaro.src.reg_compare                 # default Affine, grid 96
    python -m Alvaro.src.reg_compare --grid 80 --transform Affine
"""

from __future__ import annotations

import argparse
import time

from . import evaluate
from .mi_ranker import make_mi_ranker
from .reg_mi_ranker import RegMIRanker, REG_GRID

POOLS = ["ds1_internal", "ds2_proxy", "ds3_proxy"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=int, default=REG_GRID,
                    help="resample edge for registration (default 96)")
    ap.add_argument("--transform", default="Affine",
                    help="ANTs type_of_transform (Affine, SyN, ...)")
    args = ap.parse_args(argv)

    pools = evaluate.get_pools()

    # --- plain MI (fast; reuses the existing 64^3 ranker) -------------------
    print("=== plain MI ===", flush=True)
    plain = evaluate.evaluate_across_pools(make_mi_ranker(), pools=pools)

    # --- register-then-MI (one ranker instance per pool so the cache + reg ---
    #     counter are scoped to that pool; print per-pool progress and timing) -
    print(f"\n=== register-then-MI (grid={args.grid}, transform={args.transform}) ===",
          flush=True)
    reg = {}
    for name in POOLS:
        spec = pools[name]
        ranker = RegMIRanker(grid=args.grid, type_of_transform=args.transform)
        t0 = time.time()
        mrr = evaluate.evaluate_ranker(
            ranker.rank, spec["queries"], spec["gallery"], spec["truth"]
        )
        dt = time.time() - t0
        reg[name] = mrr
        print(f"  {name:14s} MRR={mrr:.4f}  "
              f"({ranker.n_reg} regs, {dt:.1f}s, {dt/max(ranker.n_reg,1):.2f}s/reg)",
              flush=True)

    # --- table -------------------------------------------------------------
    print("\n" + "=" * 46)
    print(f"{'pool':<16}{'plain_MI':>12}{'reg_then_MI':>14}")
    print("-" * 46)
    for name in POOLS:
        print(f"{name:<16}{plain[name]:>12.4f}{reg[name]:>14.4f}")
    print("=" * 46)

    # --- one-line read of the key question ---------------------------------
    d2_plain, d2_reg = plain["ds2_proxy"], reg["ds2_proxy"]
    delta = d2_reg - d2_plain
    print(f"\nds2_proxy: {d2_plain:.4f} -> {d2_reg:.4f}  (Δ={delta:+.4f})")
    if delta > 0.10:
        print("READ: affine registration RECOVERS ds2 (well above plain MI).")
    elif delta > 0.05:
        print("READ: affine registration gives a MODEST ds2 gain (above noise, "
              "but not full recovery) -- consider SyN.")
    else:
        print("READ: affine BARELY moves ds2 -- affine is not enough; try SyN next.")
    for name in ("ds1_internal", "ds3_proxy"):
        if reg[name] < plain[name] - 0.05:
            print(f"WARN: {name} REGRESSED ({plain[name]:.4f} -> {reg[name]:.4f}).")


if __name__ == "__main__":
    main()
