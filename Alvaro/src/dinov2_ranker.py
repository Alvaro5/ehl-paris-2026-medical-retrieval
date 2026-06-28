"""DINOv2 zero-shot ranking function (the baseline to beat), CPU/MPS/CUDA.

Logic ported (and copied, not imported) from laurence/dinov2_baseline_modal.py:
for each volume take 3 axial slices at 0.35/0.50/0.65 of the nonzero z-range,
normalize each to uint8 RGB at 224x224, push through a pretrained DINOv2-base ViT,
average the 3 CLS tokens, and unit-normalize. Rank gallery targets by cosine
similarity to the query feature.

We expose a ranking function compatible with Alvaro.src.evaluate. Volume features
are cached by absolute path inside the ranker, so each volume is embedded exactly
once even though evaluate passes the same gallery for all 50 queries.

Run across the three pools:
    python -m Alvaro.src.dinov2_ranker
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

MODEL_ID = "facebook/dinov2-base"
SLICE_POSITIONS = (0.35, 0.50, 0.65)
IMAGE_SIZE = 224  # DINOv2 ViT-B/14 canonical input size
BATCH_SIZE = 16


def _pick_device():
    """cuda > mps > cpu. Imported lazily so importing this module is cheap."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_slices(nii_path: str) -> list:
    """Return 3 uint8 HWC RGB PIL images at representative z positions.

    Copied from Laurence's load_slices: occupied-z range via nonzero counts, then
    per-slice min/max normalization to 0..255. Constant slices map to black.
    """
    import nibabel as nib
    from PIL import Image as PILImage

    img = nib.load(nii_path)
    vol = img.get_fdata(dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]

    # Count nonzero (finite) voxels per axial slice to find the brain's z-extent.
    nonzero = np.count_nonzero(np.isfinite(vol) & (vol != 0), axis=(0, 1))
    occ = np.where(nonzero > 0)[0]
    z_min = int(occ[0]) if len(occ) else 0
    z_max = int(occ[-1]) if len(occ) else vol.shape[2] - 1

    out = []
    for pos in SLICE_POSITIONS:
        # Linearly interpolate the slice index inside the occupied z-range.
        z = int(np.clip(round(z_min + pos * (z_max - z_min)), 0, vol.shape[2] - 1))
        sl = np.nan_to_num(vol[:, :, z], nan=0.0, posinf=0.0, neginf=0.0)
        mn, mx = float(sl.min()), float(sl.max())
        sl = (
            ((sl - mn) / (mx - mn) * 255).astype(np.uint8)
            if mx > mn
            else np.zeros_like(sl, dtype=np.uint8)
        )
        pil = PILImage.fromarray(sl).resize(
            (IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR
        ).convert("RGB")
        out.append(pil)
    return out


class DinoV2Ranker:
    """Stateful DINOv2 ranker with a per-path feature cache.

    One instance loads the model once and memoizes volume features, so reusing it
    across pools (and across the 50 queries that share a gallery) embeds each
    volume only a single time.
    """

    def __init__(self, device=None):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device or _pick_device()
        print(f"[dinov2] device = {self.device}")
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID).to(self.device).eval()
        self._cache: Dict[str, np.ndarray] = {}  # abs_path -> (768,) unit vector

    def _embed_pils(self, pils: list) -> np.ndarray:
        """Run DINOv2 on a list of PIL images; return (N, 768) float32 CLS tokens."""
        embs = []
        with self.torch.no_grad():
            for start in range(0, len(pils), BATCH_SIZE):
                batch = pils[start : start + BATCH_SIZE]
                inputs = self.processor(images=batch, return_tensors="pt").to(
                    self.device
                )
                out = self.model(**inputs)
                cls = out.last_hidden_state[:, 0, :]  # CLS token per image
                embs.append(cls.cpu().float().numpy())
        return np.concatenate(embs, axis=0)

    def feature(self, path: str) -> np.ndarray:
        """Unit-normalized average-of-3-slices CLS feature for one volume (cached)."""
        if path in self._cache:
            return self._cache[path]
        try:
            pils = _load_slices(path)
            emb = self._embed_pils(pils).mean(axis=0).astype(np.float32)
        except Exception as e:  # a bad volume should not abort the whole pool
            print(f"[dinov2] ERROR embedding {path}: {e}")
            emb = np.zeros(768, dtype=np.float32)
        norm = np.linalg.norm(emb)
        feat = emb / norm if norm > 0 else emb
        self._cache[path] = feat
        return feat

    def rank(self, query_id: str, query_path: str, targets: Dict[str, str]) -> List[str]:
        """Order target_ids by descending cosine similarity to the query."""
        q = self.feature(query_path)
        tids = list(targets.keys())
        T = np.stack([self.feature(targets[t]) for t in tids])  # (Nt, 768)
        scores = T @ q  # both unit vectors -> cosine similarity
        order = np.argsort(-scores)
        return [tids[i] for i in order]


def make_dinov2_ranker(device=None):
    """Build a DinoV2Ranker and return its .rank method (an evaluate-ready RankFn)."""
    return DinoV2Ranker(device=device).rank


if __name__ == "__main__":
    from . import evaluate

    ranker = make_dinov2_ranker()
    results = evaluate.evaluate_across_pools(ranker)
    print("\nDINOv2 zero-shot MRR per pool:")
    for name, mrr in results.items():
        print(f"  {name:14s} {mrr:.4f}")
