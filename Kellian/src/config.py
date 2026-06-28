from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    data_root: Path
    output_csv: Path
    cache_dir: Path
    grid: int = 64
    dataset2_topk: int = 30
    csls_k: int = 8
    sinkhorn_tau: float = 0.2
    sinkhorn_iters: int = 50
    skip_registration: bool = False
