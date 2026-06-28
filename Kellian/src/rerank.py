from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .features import robust_normalize


def cosine_scores(query_features: np.ndarray, target_features: np.ndarray) -> np.ndarray:
    query = query_features / np.maximum(np.linalg.norm(query_features, axis=1, keepdims=True), 1e-6)
    target = target_features / np.maximum(np.linalg.norm(target_features, axis=1, keepdims=True), 1e-6)
    return query @ target.T


def csls(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return scores.copy()
    qk = min(k, scores.shape[1])
    tk = min(k, scores.shape[0])
    query_nn = np.partition(scores, -qk, axis=1)[:, -qk:].mean(axis=1, keepdims=True)
    target_nn = np.partition(scores, -tk, axis=0)[-tk:, :].mean(axis=0, keepdims=True)
    return 2.0 * scores - query_nn - target_nn


def sinkhorn(scores: np.ndarray, tau: float, iters: int) -> np.ndarray:
    scaled = scores / max(tau, 1e-6)
    scaled -= scaled.max()
    transport = np.exp(scaled)
    for _ in range(iters):
        transport /= transport.sum(axis=1, keepdims=True) + 1e-12
        transport /= transport.sum(axis=0, keepdims=True) + 1e-12
    return transport


def rankings_from_scores(scores: np.ndarray, target_ids: list[str]) -> list[list[str]]:
    order = np.argsort(-scores, axis=1)
    return [[target_ids[int(index)] for index in row] for row in order]


def reciprocal_rank_fusion(rankings: Iterable[list[str]], weights: Iterable[float], k: float = 60.0) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, target_id in enumerate(ranking, start=1):
            scores[target_id] = scores.get(target_id, 0.0) + float(weight) / (k + rank)
    return scores


def normalized_mutual_information(fixed: np.ndarray, moving: np.ndarray, bins: int = 64) -> float:
    fixed = robust_normalize(fixed)
    moving = robust_normalize(moving)
    mask = (np.abs(fixed) > 1e-6) & (np.abs(moving) > 1e-6)
    if mask.sum() < 64:
        return 0.0
    a = fixed[mask]
    b = moving[mask]
    hist, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = hist / max(hist.sum(), 1.0)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    hxy = -np.sum(pxy[pxy > 0] * np.log(pxy[pxy > 0]))
    return float((hx + hy) / max(hxy, 1e-6))
