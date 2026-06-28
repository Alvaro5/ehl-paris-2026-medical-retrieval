"""Build a Kaggle-format submission by running the MI ranker over the REAL data.

Mirrors the six prediction sets of laurence/dinov2_baseline_modal.py
({dataset1,2,3} x {val,test}), but ranks with one of our training-free rankers so
we can submit and read the real per-dataset numbers. Kaggle scores omitted queries
as 0 and averages ds1/ds2/ds3 x3, so writing a single-dataset (or single-split)
file (e.g. --datasets 2 --splits val) lets us read that pool's real MRR in
isolation.

Rankers:
  plain_mi  -- mi_ranker.MIRanker (fast; grid 64).
  reg_mi    -- reg_mi_ranker.RegMIRanker (ANTs register-then-MI; grid 96).
               --transform {Affine,SyN} picks the ANTs transform.

--ranker sets ONE ranker for every dataset. --ranker-map overrides it PER dataset,
e.g. "1:plain_mi,2:reg_mi,3:plain_mi" -- the combined submission: ds1/ds3 use plain
MI (already ~0.98, registration only costs a tiny -0.019), only ds2 pays for
registration. Registration is O(N_queries * N_targets) per pool, so the script
prints an upfront cost estimate and projected wall-time BEFORE doing any work.

Sharding (registration is embarrassingly parallel over queries): --query-shard i/N
processes the i-th of N round-robin slices of the global query list and writes a
PARTIAL CSV (<out>.shardI-of-N.csv). --merge-shards N concatenates the N partials
into the final --out and checks full coverage. A single un-sharded run (the default,
N=1) writes --out directly. NOTE: ANTs already multithreads each registration across
all local cores, so local shards give resumability, not speedup; real fan-out needs
multiple machines (Modal).

Pool isolation is the key invariant: every query is ranked ONLY against its own
dataset's own split gallery; we never mix pools. Each ranking is the full
space-separated gallery, best->worst.

No network, no Kaggle API -- this only writes a CSV locally.

    # combined: plain MI for ds1/ds3, reg-MI affine for ds2 (377 rows)
    python -m Alvaro.src.make_submission --data-root . \
        --ranker-map "1:plain_mi,2:reg_mi,3:plain_mi" --transform Affine \
        --out Alvaro/generated/regmi_full_submission.csv
    # sharded ds2-heavy run, then merge
    python -m Alvaro.src.make_submission ... --query-shard 1/4 --out OUT.csv
    python -m Alvaro.src.make_submission ... --merge-shards 4 --out OUT.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .mi_ranker import MIRanker
from .reg_mi_ranker import RegMIRanker, REG_GRID

ALL_SPLITS = ("val", "test")
SUBMISSION_COLS = ["query_id", "target_id_ranking"]
PER_REG_SEC = 0.49  # measured affine cost at 96^3 (single-threaded ANTs)


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


def _sanity_block(ranker, queries: Dict[str, Path], targets: Dict[str, Path],
                  label: str) -> None:
    """Eyeball that the ranker discriminates: top-3 + worst score, first 2 queries.

    Works for any ranker exposing score_targets (MIRanker / RegMIRanker). For
    reg_mi the printed NMI is POST-registration, which is what we want to eyeball
    on real data.
    """
    print(f"\n=== SANITY ({label}) — does the score separate top from tail? ===")
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


def _abs_out(out: str) -> Path:
    """Anchor a relative --out at the repo root, not the CWD."""
    p = Path(out)
    return p if p.is_absolute() else Path(__file__).resolve().parents[2] / p


def _shard_path(out_path: Path, i: int, n: int) -> Path:
    """Partial-CSV path for shard i of n (1-indexed), beside the final --out."""
    return out_path.with_name(f"{out_path.stem}.shard{i}-of-{n}{out_path.suffix}")


def _parse_ranker_map(spec: str | None, default_name: str, ds_ids: List[int]) -> Dict[int, str]:
    """Per-dataset ranker name. --ranker-map 'ds:name,...' overrides --ranker default."""
    mapping = {ds: default_name for ds in ds_ids}
    if spec:
        for item in spec.split(","):
            if not item.strip():
                continue
            ds_str, name = item.split(":")
            name = name.strip()
            if name not in ("plain_mi", "reg_mi"):
                raise SystemExit(f"--ranker-map: unknown ranker '{name}'")
            mapping[int(ds_str)] = name
    return mapping


def _merge_shards(out_path: Path, n: int, n_expected: int, expected_ids: set) -> None:
    """Concatenate the n shard partials into the final --out, checking coverage."""
    frames = []
    for i in range(1, n + 1):
        sp = _shard_path(out_path, i, n)
        if not sp.exists():
            raise SystemExit(f"--merge-shards: missing shard partial {sp}")
        frames.append(pd.read_csv(sp))
    merged = pd.concat(frames, ignore_index=True)
    ids = set(merged["query_id"].astype(str))
    dups = len(merged) - merged["query_id"].nunique()
    if dups:
        raise SystemExit(f"--merge-shards: {dups} duplicate query rows across shards")
    if ids != expected_ids:
        missing, extra = expected_ids - ids, ids - expected_ids
        raise SystemExit(f"--merge-shards: coverage mismatch "
                         f"(missing {len(missing)}, extra {len(extra)})")
    assert len(merged) == n_expected, f"merged {len(merged)} rows, expected {n_expected}"
    merged[SUBMISSION_COLS].to_csv(out_path, index=False)
    print(f"Merged {n} shards -> {len(merged)} rows at {out_path}")


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True,
                    help="root that the CSV image paths are relative to")
    ap.add_argument("--out", default="Alvaro/generated/mi_submission.csv")
    ap.add_argument("--datasets", default="1,2,3",
                    help="comma-separated dataset ids to include (e.g. '2')")
    ap.add_argument("--splits", default="val,test",
                    help="comma-separated splits to include (default 'val,test')")
    ap.add_argument("--ranker", choices=["plain_mi", "reg_mi"], default="plain_mi",
                    help="ranker for ALL datasets unless overridden by --ranker-map")
    ap.add_argument("--ranker-map", default=None,
                    help="per-dataset override, e.g. '1:plain_mi,2:reg_mi,3:plain_mi'")
    ap.add_argument("--transform", default="Affine",
                    help="ANTs type_of_transform for reg_mi (Affine, SyN, ...)")
    ap.add_argument("--downsample", type=int, default=None,
                    help="reg_mi grid override (default 96); plain_mi is always 64")
    ap.add_argument("--query-shard", default="1/1",
                    help="process the i-th of N round-robin query slices (e.g. 2/4)")
    ap.add_argument("--merge-shards", type=int, default=0,
                    help="merge N shard partials into --out and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the cost estimate and exit before any ranking")
    args = ap.parse_args(argv)

    data_root = Path(args.data_root).resolve()
    ds_ids = sorted(int(x) for x in args.datasets.split(",") if x.strip())
    splits = [s for s in ALL_SPLITS if s in
              {x.strip() for x in args.splits.split(",")}]  # canonical val,test order
    ranker_of_ds = _parse_ranker_map(args.ranker_map, args.ranker, ds_ids)
    reg_grid = args.downsample if args.downsample is not None else REG_GRID
    out_path = _abs_out(args.out)
    print(f"data-root={data_root}  datasets={ds_ids}  splits={splits}\n"
          f"ranker_map={ranker_of_ds}  transform={args.transform}  "
          f"plain_grid=64 reg_grid={reg_grid}")

    # 1) Load every requested prediction set up front (deterministic order: dataset
    #    ascending, then val before test) so shard slicing is reproducible.
    sets: List[Tuple[int, str, Dict[str, Path], Dict[str, Path]]] = []
    for ds in ds_ids:
        for split in splits:
            loaded = _load_set(data_root, ds, split)
            if loaded is not None:
                sets.append((ds, split, loaded[0], loaded[1]))
    if not sets:
        raise SystemExit("No usable prediction sets found; nothing to write.")

    n_expected = sum(len(q) for _, _, q, _ in sets)  # queries across all sets
    expected_ids = {qid for _, _, q, _ in sets for qid in q}

    # Merge mode: combine existing shard partials and exit (no ranking).
    if args.merge_shards:
        _merge_shards(out_path, args.merge_shards, n_expected, expected_ids)
        return

    # 2) Flatten to a deterministic global query list, tagging each with its pool's
    #    ranker. str_targets is built once per set and shared by reference.
    flat: List[Tuple[int, str, str, str, Dict[str, str], str]] = []
    set_str_targets = []
    for ds, split, queries, targets in sets:
        st = {tid: str(p) for tid, p in targets.items()}
        set_str_targets.append(st)
        for qid in sorted(queries):  # sort -> reproducible shard membership
            flat.append((ds, split, qid, str(queries[qid]), st, ranker_of_ds[ds]))

    # 3) Round-robin shard selection (so the ~140 expensive ds2 queries spread
    #    evenly across shards rather than piling into one contiguous chunk).
    i, n = (int(x) for x in args.query_shard.split("/"))
    if not (1 <= i <= n):
        raise SystemExit(f"--query-shard {args.query_shard}: need 1 <= i <= N")
    shard = [row for idx, row in enumerate(flat) if idx % n == (i - 1)]

    # 4) Upfront cost estimate over THIS shard's queries (reg pairs dominate; plain
    #    pairs are negligible). Print BEFORE any heavy work so it can be aborted.
    reg_pairs = sum(len(st) for _, _, _, _, st, rn in shard if rn == "reg_mi")
    est_sec = reg_pairs * PER_REG_SEC
    print(f"\n[cost] shard {i}/{n}: {len(shard)} queries "
          f"({sum(rn=='reg_mi' for *_, rn in shard)} reg, "
          f"{sum(rn=='plain_mi' for *_, rn in shard)} plain)")
    print(f"[cost] {reg_pairs} registrations @ ~{PER_REG_SEC:.2f}s = "
          f"~{est_sec/60:.1f} min (~{est_sec/3600:.2f} h) projected wall-time "
          f"(+ plain MI negligible)")
    for ds, split, q, t in sets:
        rn = ranker_of_ds[ds]
        npairs = len(q) * len(t)
        tag = (f"~{npairs*PER_REG_SEC/60:.1f} min reg" if rn == "reg_mi"
               else "negligible (plain)")
        print(f"       dataset{ds}/{split} [{rn}]: {len(q)}x{len(t)} = {npairs} pairs -> {tag}")

    if args.dry_run:
        print("\n[dry-run] stopping before any ranking.")
        return

    # 5) Build only the ranker instances this shard actually needs (RegMIRanker
    #    imports ANTs, so skip it if no reg pool is in play). Caches are per-instance
    #    and reused across the shard's queries.
    need = {rn for *_, rn in shard}
    rankers: Dict[str, object] = {}
    if "plain_mi" in need:
        rankers["plain_mi"] = MIRanker(grid=64)
    if "reg_mi" in need:
        rankers["reg_mi"] = RegMIRanker(grid=reg_grid, type_of_transform=args.transform)

    # 6) Sanity block (only shard 1, to avoid every container repeating it): prefer
    #    a reg pool (that's the risky ranker), else dataset1/val, else first set.
    if i == 1:
        sanity = next(((q, t, f"dataset{d}/{s}", ranker_of_ds[d]) for d, s, q, t in sets
                       if ranker_of_ds[d] == "reg_mi"), None)
        if sanity is None:
            d, s, q, t = sets[0]
            sanity = (q, t, f"dataset{d}/{s}", ranker_of_ds[d])
        _sanity_block(rankers[sanity[3]], sanity[0], sanity[1], sanity[2])

    # 7) Rank each query against ITS OWN pool's gallery only (pool isolation).
    print("\n=== ranking ===")
    t0 = time.time()
    rows: List[Dict[str, str]] = []
    for ds, split, qid, qpath, st_targets, rn in shard:
        ranking = rankers[rn].rank(qid, qpath, st_targets)  # full gallery best->worst
        # Ranking length == gallery size: every target appears exactly once.
        rows.append({"query_id": qid, "target_id_ranking": " ".join(ranking)})
        if len(rows) % 10 == 0:
            print(f"  ranked {len(rows)}/{len(shard)} queries "
                  f"({time.time() - t0:.1f}s elapsed)", flush=True)

    # 8) Write + verify. Single run (N=1) writes --out and must total n_expected;
    #    a shard writes a partial and must total this shard's query count.
    assert len(rows) == len(shard), f"wrote {len(rows)} rows, expected {len(shard)}"
    dest = out_path if n == 1 else _shard_path(out_path, i, n)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=SUBMISSION_COLS).to_csv(dest, index=False)
    print(f"\nWrote {len(rows)} rows to {dest}  (total {time.time() - t0:.1f}s)")
    if n == 1:
        assert len(rows) == n_expected, f"total {len(rows)} != expected {n_expected}"
    else:
        print(f"Shard {i}/{n} done. Merge with: --merge-shards {n} --out {args.out}")


if __name__ == "__main__":
    main()
