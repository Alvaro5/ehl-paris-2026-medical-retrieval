from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


SUBMISSION_COLUMNS = ["query_id", "target_id_ranking"]


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def resolve_image(data_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if not path.is_absolute():
        path = data_root / path
    if not path.exists() and path.suffix == ".gz":
        uncompressed = path.with_suffix("")
        if uncompressed.exists():
            return uncompressed
    return path


def load_volume(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    array = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    if array.ndim == 4:
        array = array[..., 0]
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def write_submission(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def validate_submission_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 377:
        raise ValueError(f"Expected 377 rows, got {len(rows)}")
    query_ids = [row["query_id"] for row in rows]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query_id in submission")
    for row in rows:
        if not row["target_id_ranking"].strip():
            raise ValueError(f"Empty ranking for {row['query_id']}")
