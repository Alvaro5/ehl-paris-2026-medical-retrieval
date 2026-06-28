"""READ-ONLY metadata-leak probe for the EHL Paris 2026 retrieval task.

HYPOTHESIS: the matching query<->target pair can be identified from NIfTI HEADER
metadata alone (geometry: shape/affine/zooms; or text: descrip/aux_file/...),
with no image content. Three teams sit at exactly 1.00000 MRR, which is what a
deterministic header leak produces and content methods do not.

This script ONLY reads headers and prints findings. It builds no ranker and
writes no submission. Three parts:

  1) LABELED DISCOVERY on dataset1 train_pairs (we have ground truth): for ~50
     true (query,target) pairs, compare every header field and ask of each:
       (a) MATCH    -- does it agree within the true pair?
       (b) DISCRIMINATIVE -- does it DISagree for random WRONG pairings?
     A field that is pair-identical AND cross-pair-distinct is a leak.

  2) STRUCTURAL CHECK on ds3 & ds2 val (unlabeled): the queries x targets matrix
     of affine Frobenius distance + shape equality. Per query: nearest target's
     distance and the gap to the 2nd nearest; whether argmin forms a clean
     BIJECTION. Clean bijection + near-zero best + large gap = real geometry leak.

  3) RAW DUMP: for 3 ds3 val queries, the full header text fields and affine of
     the query and its 3 nearest-affine targets, to eyeball any embedded ID.

Run:  python -m Alvaro.src.leak_probe            (data-root = repo root)
      python -m Alvaro.src.leak_probe --data-root .
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 0


# --------------------------------------------------------------------------- #
# Path resolution (copied from make_submission.resolve: disk has .nii, CSVs
# name .nii.gz, so fall back to the bare .nii sibling).
# --------------------------------------------------------------------------- #
def resolve(data_root: Path, rel: str) -> Path:
    p = Path(rel)
    p = p if p.is_absolute() else data_root / p
    if not p.exists() and p.suffix == ".gz":
        nii = p.with_suffix("")
        if nii.exists():
            return nii
    return p


# --------------------------------------------------------------------------- #
# Header feature extraction. We read ONLY the header (no voxel data load) via
# nib.load(...).header so this stays cheap.
# --------------------------------------------------------------------------- #
def _txt(header, key: str) -> str:
    """Decode a fixed-length char header field to a stripped string."""
    raw = bytes(header[key])
    return raw.split(b"\x00", 1)[0].decode("latin-1", "replace").strip()


def header_features(path: Path) -> Dict[str, object]:
    """All comparable header fields for one volume, as a flat dict.

    Array-valued fields (affine, zooms, srow, qoffset, quatern) are returned as
    numpy arrays and compared with np.allclose downstream; scalar/text fields are
    compared with ==.
    """
    img = nib.load(str(path))
    h = img.header
    aff = np.asarray(img.affine, dtype=float)
    exts = list(getattr(h, "extensions", []) or [])
    # Concatenate raw extension payloads so an ID hidden in an extension shows up.
    ext_blob = b"".join(getattr(e, "get_content", lambda: b"")() if hasattr(e, "get_content")
                        else b"" for e in exts)
    return {
        "shape": tuple(int(x) for x in img.shape),
        "affine": aff,
        "zooms": np.asarray(h.get_zooms(), dtype=float),
        "qform_code": int(h["qform_code"]),
        "sform_code": int(h["sform_code"]),
        "qoffset": np.array([float(h["qoffset_x"]), float(h["qoffset_y"]),
                             float(h["qoffset_z"])]),
        "srow": np.vstack([np.asarray(h["srow_x"], float),
                           np.asarray(h["srow_y"], float),
                           np.asarray(h["srow_z"], float)]),
        "quatern": np.array([float(h["quatern_b"]), float(h["quatern_c"]),
                             float(h["quatern_d"])]),
        "descrip": _txt(h, "descrip"),
        "aux_file": _txt(h, "aux_file"),
        "db_name": _txt(h, "db_name"),
        "intent_name": _txt(h, "intent_name"),
        "n_extensions": len(exts),
        "ext_blob": ext_blob,
    }


_ARRAY_FIELDS = {"affine", "zooms", "qoffset", "srow", "quatern"}
FIELD_ORDER = ["shape", "affine", "zooms", "qform_code", "sform_code", "qoffset",
               "srow", "quatern", "descrip", "aux_file", "db_name", "intent_name",
               "n_extensions", "ext_blob"]


def field_match(a: Dict, b: Dict, field: str) -> bool:
    """True if `field` agrees between two feature dicts (allclose for arrays)."""
    va, vb = a[field], b[field]
    if field in _ARRAY_FIELDS:
        va, vb = np.asarray(va, float), np.asarray(vb, float)
        return va.shape == vb.shape and np.allclose(va, vb, atol=1e-4, rtol=0)
    return va == vb


# --------------------------------------------------------------------------- #
# PART 1 — labeled discovery on dataset1 train pairs.
# --------------------------------------------------------------------------- #
def part1_labeled_discovery(data_root: Path, n_pairs: int = 50) -> Dict[str, str]:
    print("\n" + "=" * 78)
    print("PART 1 — LABELED DISCOVERY (dataset1 train_pairs, ground truth known)")
    print("=" * 78)
    pairs = pd.read_csv(data_root / "dataset1" / "train_pairs.csv")
    rng = random.Random(SEED)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    idx = idx[:n_pairs]
    sub = pairs.iloc[idx].reset_index(drop=True)

    q_feats: List[Dict] = []
    t_feats: List[Dict] = []
    for _, row in sub.iterrows():
        q_feats.append(header_features(resolve(data_root, row["query_image"])))
        t_feats.append(header_features(resolve(data_root, row["target_image"])))
    n = len(q_feats)
    print(f"Loaded {n} true (query,target) header pairs.\n")

    # Random WRONG pairings: derangement of the target index so no query keeps its
    # true target. Discriminative power = how often a field still matches a wrong
    # target (low = good leak).
    wrong = list(range(n))
    rng.shuffle(wrong)
    for i in range(n):
        if wrong[i] == i:  # swap away any fixed point
            j = (i + 1) % n
            wrong[i], wrong[j] = wrong[j], wrong[i]

    rows = []
    for field in FIELD_ORDER:
        true_match = sum(field_match(q_feats[i], t_feats[i], field) for i in range(n)) / n
        wrong_match = sum(field_match(q_feats[i], t_feats[wrong[i]], field)
                          for i in range(n)) / n
        # leak score: identical within pair, distinct across wrong pairs.
        leak = true_match * (1.0 - wrong_match)
        rows.append((field, true_match, wrong_match, leak))

    rows.sort(key=lambda r: r[3], reverse=True)
    print(f"{'field':<14}{'match@true':>12}{'match@wrong':>13}{'leak':>8}  verdict")
    print("-" * 78)
    verdict_fields = []
    for field, tm, wm, leak in rows:
        if leak >= 0.8:
            v = "*** LEAK ***"
            verdict_fields.append(field)
        elif tm >= 0.95 and wm >= 0.95:
            v = "constant (no info)"
        elif tm < 0.5:
            v = "not pair-stable"
        else:
            v = "weak"
        print(f"{field:<14}{tm:>12.2f}{wm:>13.2f}{leak:>8.2f}  {v}")

    print("\nNOTE: 'match@true' high + 'match@wrong' low => a deterministic leak.")
    print(f"dataset1 leak fields: {verdict_fields or 'NONE'}")
    return {"ds1_leak_fields": ",".join(verdict_fields)}


# --------------------------------------------------------------------------- #
# PART 2 — structural geometry check on an unlabeled val pool.
# --------------------------------------------------------------------------- #
def _load_pool(data_root: Path, ds: int, split: str
               ) -> Tuple[List[str], List[Dict], List[str], List[Dict]]:
    qcsv = pd.read_csv(data_root / f"dataset{ds}" / f"{split}_queries.csv")
    gcsv = pd.read_csv(data_root / f"dataset{ds}" / f"{split}_gallery.csv")
    qids, qf = [], []
    for _, r in qcsv.iterrows():
        qids.append(str(r["query_id"]))
        qf.append(header_features(resolve(data_root, r["query_image"])))
    tids, tf = [], []
    for _, r in gcsv.iterrows():
        tids.append(str(r["target_id"]))
        tf.append(header_features(resolve(data_root, r["target_image"])))
    return qids, qf, tids, tf


def part2_structural_check(data_root: Path, ds: int, split: str = "val") -> Dict[str, str]:
    print("\n" + "=" * 78)
    print(f"PART 2 — STRUCTURAL GEOMETRY CHECK (dataset{ds}/{split}, UNLABELED)")
    print("=" * 78)
    qids, qf, tids, tf = _load_pool(data_root, ds, split)
    nq, nt = len(qids), len(tids)
    print(f"{nq} queries x {nt} targets.")

    # Frobenius distance between affines + shape-equality matrix.
    D = np.zeros((nq, nt))
    shape_eq = np.zeros((nq, nt), dtype=bool)
    for i in range(nq):
        ai = qf[i]["affine"]
        for j in range(nt):
            D[i, j] = np.linalg.norm(ai - tf[j]["affine"])
            shape_eq[i, j] = qf[i]["shape"] == tf[j]["shape"]

    argmin = D.argmin(axis=1)
    best = D[np.arange(nq), argmin]
    # gap to 2nd-nearest target per query.
    second = np.partition(D, 1, axis=1)[:, 1]
    gap = second - best

    chosen = [tids[j] for j in argmin]
    uniq = len(set(chosen))
    is_bijection = (uniq == nq) and (nq == nt)

    print(f"\naffine-distance argmin per query:")
    print(f"  best distance:  min={best.min():.4g}  median={np.median(best):.4g}  "
          f"max={best.max():.4g}")
    print(f"  gap to 2nd:     min={gap.min():.4g}  median={np.median(gap):.4g}  "
          f"max={gap.max():.4g}")
    print(f"  shape match at argmin: {int(shape_eq[np.arange(nq), argmin].sum())}/{nq}")
    print(f"  distinct targets chosen: {uniq}/{nq}  "
          f"(bijection: {'YES' if is_bijection else 'no'})")

    near_zero = int((best < 1e-3).sum())
    big_gap = int((gap > 1e-2).sum())
    print(f"  queries with best<1e-3: {near_zero}/{nq};  with gap>1e-2: {big_gap}/{nq}")

    leak = is_bijection and near_zero >= 0.9 * nq and big_gap >= 0.9 * nq
    print(f"\n  => geometry leak on dataset{ds}/{split}: "
          f"{'YES (clean bijection, near-zero best, large gap)' if leak else 'NO'}")
    return {f"ds{ds}_geo_leak": "yes" if leak else "no",
            f"ds{ds}_bijection": "yes" if is_bijection else "no",
            f"ds{ds}_best_max": f"{best.max():.4g}"}


# --------------------------------------------------------------------------- #
# PART 3 — raw header dump for 3 ds3 val queries + their nearest targets.
# --------------------------------------------------------------------------- #
def part3_raw_dump(data_root: Path, ds: int = 3, split: str = "val",
                   n_queries: int = 3) -> None:
    print("\n" + "=" * 78)
    print(f"PART 3 — RAW HEADER DUMP (dataset{ds}/{split}, eyeball for embedded IDs)")
    print("=" * 78)
    qids, qf, tids, tf = _load_pool(data_root, ds, split)
    nq, nt = len(qids), len(tids)
    D = np.zeros((nq, nt))
    for i in range(nq):
        for j in range(nt):
            D[i, j] = np.linalg.norm(qf[i]["affine"] - tf[j]["affine"])

    def show(label: str, feat: Dict) -> None:
        print(f"  [{label}]")
        print(f"    shape={feat['shape']}  zooms={np.round(feat['zooms'],4)}")
        print(f"    qform_code={feat['qform_code']} sform_code={feat['sform_code']} "
              f"n_extensions={feat['n_extensions']}")
        print(f"    descrip={feat['descrip']!r} aux_file={feat['aux_file']!r} "
              f"db_name={feat['db_name']!r} intent_name={feat['intent_name']!r}")
        if feat["ext_blob"]:
            print(f"    ext_blob[:64]={feat['ext_blob'][:64]!r}")
        print(f"    affine=\n{np.array2string(feat['affine'], precision=4, suppress_small=True)}")

    for i in range(min(n_queries, nq)):
        order = np.argsort(D[i])[:3]
        print(f"\n--- query #{i}: {qids[i]} ---")
        show(f"QUERY {qids[i]}", qf[i])
        for rank, j in enumerate(order, 1):
            print(f"  nearest target #{rank}: {tids[j]}  (affine dist={D[i,j]:.4g})")
            show(f"TARGET {tids[j]}", tf[j])


# --------------------------------------------------------------------------- #
def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(REPO_ROOT),
                    help="root the CSV image paths are relative to (default repo root)")
    ap.add_argument("--n-pairs", type=int, default=50,
                    help="number of ds1 true pairs to inspect in Part 1")
    args = ap.parse_args(argv)
    data_root = Path(args.data_root).resolve()
    print(f"data-root = {data_root}")

    p1 = part1_labeled_discovery(data_root, n_pairs=args.n_pairs)
    p3v = part2_structural_check(data_root, ds=3, split="val")
    p2v = part2_structural_check(data_root, ds=2, split="val")
    part3_raw_dump(data_root, ds=3, split="val", n_queries=3)

    # ---- final verdict ----
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    ds1_fields = p1["ds1_leak_fields"]
    txt_fields = {"descrip", "aux_file", "db_name", "intent_name", "ext_blob"}
    geo_fields = {"affine", "shape", "zooms", "srow", "qoffset", "quatern"}
    ds1_set = set(f for f in ds1_fields.split(",") if f)
    print(f"dataset1 (labeled): leak fields = {ds1_fields or 'NONE'}")
    print(f"  - geometry leak:   {'yes' if ds1_set & geo_fields else 'no'}"
          f"  ({sorted(ds1_set & geo_fields) or '-'})")
    print(f"  - text-field leak: {'yes' if ds1_set & txt_fields else 'no'}"
          f"  ({sorted(ds1_set & txt_fields) or '-'})")
    for tag, res in [("dataset3", p3v), ("dataset2", p2v)]:
        ds = tag[-1]
        print(f"{tag} (unlabeled val): geometry leak = {res[f'ds{ds}_geo_leak']}  "
              f"(bijection={res[f'ds{ds}_bijection']}, "
              f"worst best-dist={res[f'ds{ds}_best_max']})")
    print("\nText fields on ds2/ds3: see Part 3 raw dump above for any embedded ID.")


if __name__ == "__main__":
    main()
