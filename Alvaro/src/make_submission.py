"""Build a Kaggle-format submission by running the MI ranker over the REAL data.

Mirrors the six prediction sets of laurence/dinov2_baseline_modal.py
({dataset1,2,3} x {val,test}), but ranks with our training-free
mutual-information ranker (Alvaro.src.mi_ranker) so we can submit pure-MI and read
the real per-dataset numbers. Kaggle scores omitted queries as 0 and averages
ds1/ds2/ds3 x3, so writing a single-dataset file (e.g. --datasets 2) lets us read
that one dataset's real MRR in isolation.

Pool isolation is the key invariant: every query is ranked ONLY against its own
dataset's own split gallery; we never mix pools. Each ranking is the full
space-separated gallery, best->worst.

No network, no Kaggle API -- this only writes a CSV locally.

    python -m Alvaro.src.make_submission --data-root /path/to/data
    python -m Alvaro.src.make_submission --data-root . --datasets 2 \
        --out Alvaro/generated/mi_ds2.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .mi_ranker import MIRanker

# The six prediction sets, by dataset id and split (mirrors Laurence's loop).
SPLITS = ("val", "test")
SUBMISSION_COLS = ["query_id", "target_id_ranking"]


def resolve(data_root: Path, rel: str) -> Path:
    """Absolute image path with the .nii.gz -> .nii fallback (copied from Laurence).

    Our local disk holds uncompressed .nii while the CSVs name .nii.gz, so when the
    .gz path is absent we retry the bare .nii sibling.
    """
    p = Path(rel)
    p = p if p.is_absolute() else data_root / p
    if not p.exists() and p.suffix == ".gz":
        nii = p.with_suffix("")  # strip .gz -> try the .nii sibling
        if nii.exists():
            return nii
    return p


def _load_set(
    data_root: Path, ds: int, split: str
) -> Tuple[Dict[str, Path], Dict[str, Path]] | None:
    """Load one (dataset, split) prediction set as ({qid: path}, {tid: path}).

    Returns None (with a printed warning) if either CSV is missing. Queries and
    targets whose image file is absent on disk are dropped with a warning, so a
    missing volume never aborts the whole run.
    """
    ds_dir = data_root / f"dataset{ds}"
    qcsv = ds_dir / f"{split}_queries.csv"
    gcsv = ds_dir / f"{split}_gallery.csv"
    if not qcsv.exists() or not gcsv.exists():
        print(f"  WARN dataset{ds}/{split}: missing CSV ({qcsv.name}/{gcsv.name}); skipping set")
        return None

    def collect(csv_path: Path, id_col: str, img_col: str) -> Dict[str, Path]:
        present: Dict[str, Path] = {}
        for _, row in pd.read_csv(csv_path).iterrows():
            iid = str(row[id_col])
            p = resolve(data_root, str(row[img_col]))
            if p.exists():
                present[iid] = p
            else:
                print(f"    WARN dataset{ds}/{split}: missing image for {iid}: {p}")
        return present

    queries = collect(qcsv, "query_id", "query_image")
    targets = collect(gcsv, "target_id", "target_image")
    if not queries or not targets:
        print(f"  WARN dataset{ds}/{split}: no usable queries/targets on disk; skipping set")
        return None
    print(f"  dataset{ds}/{split}: {len(queries)} queries, {len(targets)} targets")
    return queries, targets


def _sanity_block(ranker: MIRanker, queries: Dict[str, Path], targets: Dict[str, Path],
                  label: str) -> None:
    """Eyeball that MI discriminates: top-3 + worst score for the first 2 queries."""
    print(f"\n=== SANITY ({label}) — does MI separate top from tail? ===")
    str_targets = {tid: str(p) for tid, p in targets.items()}
    for qid in list(queries)[:2]:
        scored = ranker.score_targets(str(queries[qid]), str_targets)  # [(tid, nmi)] desc
        top3 = scored[:3]
        worst_tid, worst_score = scored[-1]
        spread = top3[0][1] - worst_score
        print(f"  query {qid}:")
        for rank, (tid, sc) in enumerate(top3, 1):
            print(f"    #{rank}: {tid}  NMI={sc:.4f}")
        print(f"    worst: {worst_tid}  NMI={worst_score:.4f}   (top-tail spread={spread:.4f})")
        if spread < 0.02:  # NMI lives in ~[1,2]; a real signal spreads far more
            print("    !! DEGENERATE: top barely above tail — MI is not discriminating here.")


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True,
                    help="root that the CSV image paths are relative to")
    ap.add_argument("--out", default="Alvaro/generated/mi_submission.csv")
    ap.add_argument("--datasets", default="1,2,3",
                    help="comma-separated dataset ids to include (e.g. '2')")
    ap.add_argument("--downsample", type=int, default=64,
                    help="resample cube edge passed to the MI ranker (default 64)")
    args = ap.parse_args(argv)

    data_root = Path(args.data_root).resolve()
    ds_ids = [int(x) for x in args.datasets.split(",") if x.strip()]
    print(f"data-root={data_root}  datasets={ds_ids}  downsample={args.downsample}")

    # 1) Load every requested prediction set up front (so the sanity block can run
    #    on dataset1/val before we commit to the full, slower ranking pass).
    sets: List[Tuple[int, str, Dict[str, Path], Dict[str, Path]]] = []
    for ds in ds_ids:
        for split in SPLITS:
            loaded = _load_set(data_root, ds, split)
            if loaded is not None:
                sets.append((ds, split, loaded[0], loaded[1]))
    if not sets:
        raise SystemExit("No usable prediction sets found; nothing to write.")

    n_expected = sum(len(q) for _, _, q, _ in sets)  # queries we will actually rank

    # One ranker for the whole run: its path-keyed cache makes every volume load +
    # downsample happen exactly once, even when a target recurs across all queries.
    ranker = MIRanker(grid=args.downsample)

    # 2) Sanity check BEFORE writing: prefer dataset1/val, else the first set.
    sanity = next(((q, t, f"dataset{d}/{s}") for d, s, q, t in sets
                   if d == 1 and s == "val"), None)
    if sanity is None:
        d, s, q, t = sets[0]
        sanity = (q, t, f"dataset{d}/{s}")
    _sanity_block(ranker, sanity[0], sanity[1], sanity[2])

    # 3) Rank every query against ITS OWN pool's gallery only.
    print("\n=== ranking ===")
    t0 = time.time()
    rows: List[Dict[str, str]] = []
    done = 0
    for ds, split, queries, targets in sets:
        str_targets = {tid: str(p) for tid, p in targets.items()}
        for qid, qpath in queries.items():
            ranking = ranker.rank(qid, str(qpath), str_targets)  # full gallery, best->worst
            # Ranking length == gallery size: every target appears exactly once.
            rows.append({"query_id": qid, "target_id_ranking": " ".join(ranking)})
            done += 1
            if done % 10 == 0:
                print(f"  ranked {done}/{n_expected} queries "
                      f"({time.time() - t0:.1f}s elapsed)", flush=True)

    # 4) Write + verify. Row count must equal the queries we set out to rank.
    assert len(rows) == n_expected, f"wrote {len(rows)} rows, expected {n_expected}"
    out_path = Path(args.out)
    if not out_path.is_absolute():
        # Keep relative outputs anchored at the repo root, not the CWD.
        out_path = Path(__file__).resolve().parents[2] / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=SUBMISSION_COLS).to_csv(out_path, index=False)
    print(f"\nWrote {len(rows)} rows to {out_path}  "
          f"(total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
