from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


EXPECTED_SHA256 = "778140DF46741F9EBC7BC5EC6B4F6FB50191203FBBB43851BE39FBE2101A8700"
SUBMISSION_COLUMNS = ["query_id", "target_id_ranking"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def target_sets(data_root: Path) -> tuple[dict[str, set[str]], dict[str, int]]:
    query_to_targets: dict[str, set[str]] = {}
    expected_lengths: dict[str, int] = {}
    for dataset in ("dataset1", "dataset2", "dataset3"):
        for split in ("val", "test"):
            query_path = data_root / dataset / f"{split}_queries.csv"
            gallery_path = data_root / dataset / f"{split}_gallery.csv"
            if not query_path.exists() or not gallery_path.exists():
                continue
            queries = read_csv(query_path)
            targets = {row["target_id"] for row in read_csv(gallery_path)}
            for query in queries:
                query_id = query["query_id"]
                query_to_targets[query_id] = targets
                expected_lengths[query_id] = len(targets)
    return query_to_targets, expected_lengths


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the preserved ~0.84 submission CSV.")
    parser.add_argument("--submission", type=Path, default=Path("Kellian/submission.csv"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\kelli\Bureau\ehl-paris-medical-image-retrieval"),
    )
    parser.add_argument("--require-best-hash", action="store_true")
    args = parser.parse_args()

    rows = read_csv(args.submission)
    if not rows:
        raise SystemExit("Submission is empty")
    if list(rows[0].keys()) != SUBMISSION_COLUMNS:
        raise SystemExit(f"Bad columns: {list(rows[0].keys())}")
    if len(rows) != 377:
        raise SystemExit(f"Bad row count: {len(rows)} != 377")

    query_ids = [row["query_id"] for row in rows]
    duplicates = len(query_ids) - len(set(query_ids))
    if duplicates:
        raise SystemExit(f"Duplicate query_id rows: {duplicates}")

    query_to_targets, expected_lengths = target_sets(args.data_root)
    missing_queries = sorted(set(query_to_targets) - set(query_ids))
    extra_queries = sorted(set(query_ids) - set(query_to_targets))
    if missing_queries or extra_queries:
        raise SystemExit(f"Query coverage mismatch: missing={len(missing_queries)} extra={len(extra_queries)}")

    for row in rows:
        query_id = row["query_id"]
        ranking = row["target_id_ranking"].split()
        if len(ranking) != expected_lengths[query_id]:
            raise SystemExit(f"{query_id}: bad ranking length {len(ranking)} != {expected_lengths[query_id]}")
        if len(set(ranking)) != len(ranking):
            raise SystemExit(f"{query_id}: duplicate target IDs in ranking")
        expected = query_to_targets[query_id]
        if set(ranking) != expected:
            raise SystemExit(f"{query_id}: target set mismatch")

    digest = sha256(args.submission)
    if args.require_best_hash and digest != EXPECTED_SHA256:
        raise SystemExit(f"Best-hash check failed: {digest}")
    print(f"OK rows=377 sha256={digest}")


if __name__ == "__main__":
    main()
