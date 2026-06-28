from __future__ import annotations

from pathlib import Path

import nibabel as nib


GeometrySignature = tuple[tuple[int, ...], tuple[float, ...]]


def geometry_signature(path: Path) -> GeometrySignature:
    image = nib.load(str(path))
    shape = tuple(int(x) for x in image.shape[:3])
    affine = tuple(float(f"{x:.4f}") for x in image.affine.reshape(-1))
    return shape, affine


def geometry_distance(a: GeometrySignature, b: GeometrySignature) -> float:
    shape_a, affine_a = a
    shape_b, affine_b = b
    shape_dist = sum(abs(x - y) for x, y in zip(shape_a, shape_b))
    affine_dist = sum(abs(x - y) for x, y in zip(affine_a, affine_b))
    return float(shape_dist + affine_dist)
