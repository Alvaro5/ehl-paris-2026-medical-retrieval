from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, sobel, zoom
from tqdm import tqdm

from .io_utils import atomic_save_npz, file_fingerprint, load_volume, resolve_image


def robust_normalize(array: np.ndarray) -> np.ndarray:
    mask = np.abs(array) > 1e-6
    if mask.sum() < 64:
        return np.zeros_like(array, dtype=np.float32)
    values = array[mask]
    lo, hi = np.quantile(values, [0.01, 0.99])
    clipped = np.clip(array, lo, hi)
    med = np.median(clipped[mask])
    iqr = np.quantile(clipped[mask], 0.75) - np.quantile(clipped[mask], 0.25)
    iqr = max(float(iqr), 1e-6)
    out = (clipped - med) / iqr
    out[~mask] = 0.0
    return np.clip(out, -6.0, 6.0).astype(np.float32)


def crop_foreground(array: np.ndarray, margin: int = 8) -> np.ndarray:
    mask = np.abs(array) > 1e-6
    if not mask.any():
        return array
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(axis=0) - margin, 0)
    hi = np.minimum(coords.max(axis=0) + margin + 1, array.shape)
    return array[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]


def resample_to_grid(array: np.ndarray, grid: int) -> np.ndarray:
    cropped = crop_foreground(array)
    factors = [grid / max(size, 1) for size in cropped.shape[:3]]
    return zoom(cropped, factors, order=1).astype(np.float32)


def mind_descriptor(volume: np.ndarray) -> np.ndarray:
    volume = robust_normalize(volume)
    smooth = gaussian_filter(volume, sigma=1.0)
    gx = sobel(smooth, axis=0)
    gy = sobel(smooth, axis=1)
    gz = sobel(smooth, axis=2)
    grad = np.sqrt(gx * gx + gy * gy + gz * gz)
    channels = [smooth, grad]
    for offset in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        shifted = np.roll(smooth, shift=offset, axis=(0, 1, 2))
        diff = gaussian_filter((smooth - shifted) ** 2, sigma=1.0)
        channels.append(np.exp(-diff / (diff.mean() + 1e-6)))
    vector = np.concatenate([channel.reshape(-1) for channel in channels]).astype(np.float32)
    vector -= vector.mean()
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def cached_descriptor(cache_dir: Path, image_id: str, path: Path, grid: int) -> np.ndarray:
    safe_id = image_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    cache_path = cache_dir / f"{safe_id}_{file_fingerprint(path)}_g{grid}.npz"
    if cache_path.exists():
        return np.load(cache_path)["feature"].astype(np.float32)
    volume = resample_to_grid(load_volume(path), grid)
    feature = mind_descriptor(volume)
    atomic_save_npz(cache_path, feature=feature.astype(np.float32))
    return feature


def feature_matrix(
    rows: pd.DataFrame,
    id_col: str,
    image_col: str,
    data_root: Path,
    cache_dir: Path,
    grid: int,
    desc: str,
) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    features: list[np.ndarray] = []
    for row in tqdm(rows.itertuples(index=False), total=len(rows), desc=desc):
        image_id = str(getattr(row, id_col))
        image_path = resolve_image(data_root, str(getattr(row, image_col)))
        ids.append(image_id)
        features.append(cached_descriptor(cache_dir, image_id, image_path, grid))
    return ids, np.stack(features).astype(np.float32)
