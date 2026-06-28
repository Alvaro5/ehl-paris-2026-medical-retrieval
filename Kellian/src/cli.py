from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .io_utils import validate_submission_rows, write_submission
from .pipeline import build_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Kaggle submission for cross-modal MRI retrieval.")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Users\kelli\Bureau\ehl-paris-medical-image-retrieval"))
    parser.add_argument("--out", type=Path, default=Path("Kellian/submission_from_pipeline.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("Kellian/pipeline_cache"))
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--dataset2-topk", type=int, default=30)
    parser.add_argument("--csls-k", type=int, default=8)
    parser.add_argument("--sinkhorn-tau", type=float, default=0.2)
    parser.add_argument("--sinkhorn-iters", type=int, default=50)
    parser.add_argument("--skip-registration", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        data_root=args.data_root,
        output_csv=args.out,
        cache_dir=args.cache_dir,
        grid=args.grid,
        dataset2_topk=args.dataset2_topk,
        csls_k=args.csls_k,
        sinkhorn_tau=args.sinkhorn_tau,
        sinkhorn_iters=args.sinkhorn_iters,
        skip_registration=args.skip_registration,
    )
    rows = build_submission(config)
    validate_submission_rows(rows)
    write_submission(config.output_csv, rows)
    print(f"Wrote {config.output_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
