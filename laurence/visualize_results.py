"""Visualize cross-modal MRI retrieval results as PNG grids on Modal.

For each (dataset, split) present in the submission CSV, renders:
  - N_QUERIES rows, each showing the query volume slice + TOP_K ranked targets
  - Green border  = correct match  (requires a *_pairs.csv ground-truth file)
  - Orange border = correct but not rank-1
  - Red border    = wrong
  - Blue border   = query column

Slices are extracted with nibabel using memory-mapped I/O where possible
(fast for uncompressed .nii; falls back to full load for .nii.gz).

Run:
    modal run laurence/visualize_results.py --csv strong_submission.csv
    modal run laurence/visualize_results.py --csv siglip_submission.csv --out viz_siglip/ --n-queries 12 --top-k 6

Output files (saved locally):
    <out>/dataset1_val.png
    <out>/dataset2_val.png
    ...
"""

from __future__ import annotations

import csv
import io
import random
from pathlib import Path

import modal

app = modal.App("ehl-visualize")
vol = modal.Volume.from_name("ehl-2026-vol-2")

N_QUERIES = 10   # queries sampled per (dataset, split)
TOP_K = 5        # ranked targets shown per query
DPI = 130
CELL_W = 2.4     # inches per subplot column
CELL_H = 2.6     # inches per subplot row
FONT_SM = 6.5
FONT_MD = 8

# Border colours
COL_QUERY   = "#4a90d9"   # blue
COL_CORRECT = "#27ae60"   # green  — rank 1 AND correct
COL_FOUND   = "#f39c12"   # orange — correct but not rank 1
COL_WRONG   = "#e74c3c"   # red    — incorrect (GT known)
COL_UNKNOWN = "#95a5a6"   # grey   — no GT available


def _download_deps():
    import matplotlib  # noqa: F401
    import nibabel     # noqa: F401


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "nibabel>=5.3",
        "numpy>=2.0",
        "matplotlib>=3.8",
        "Pillow>=10.0",
    )
    .run_function(_download_deps)
)


# ---------------------------------------------------------------------------
# Remote function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=1800,
    memory=16384,
    cpu=4,
)
def generate_pngs(
    csv_content: str,
    n_queries: int = N_QUERIES,
    top_k: int = TOP_K,
    seed: int = 42,
) -> list[tuple[str, bytes]]:
    """
    Returns [(filename, png_bytes), ...], one entry per (dataset, split).
    """
    import csv as _csv
    import io as _io
    import random as _random

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import nibabel as nib
    import numpy as np

    _random.seed(seed)
    data_root = _find_data_root(Path("/data"))

    # ------------------------------------------------------------------ parse submission

    submission: dict[str, list[str]] = {}   # {query_id: [target_id, ...]}
    reader = _csv.DictReader(_io.StringIO(csv_content))
    for row in reader:
        submission[row["query_id"]] = row["target_id_ranking"].split()

    # ------------------------------------------------------------------ parse manifests

    def read_csv_rows(path: Path) -> list[dict[str, str]]:
        with path.open(newline="") as f:
            return list(_csv.DictReader(f))

    def resolve(rel: str) -> Path:
        p = Path(rel)
        p = p if p.is_absolute() else data_root / p
        if not p.exists() and p.suffix == ".gz":
            alt = p.with_suffix("")
            if alt.exists():
                return alt
        return p

    # Collect per-(dataset, split) manifests
    specs = [
        ("dataset1", "val"),  ("dataset1", "test"),
        ("dataset2", "val"),  ("dataset2", "test"),
        ("dataset3", "val"),  ("dataset3", "test"),
    ]

    results: list[tuple[str, bytes]] = []

    for ds, split in specs:
        qcsv = data_root / ds / f"{split}_queries.csv"
        gcsv = data_root / ds / f"{split}_gallery.csv"
        if not qcsv.exists() or not gcsv.exists():
            continue

        query_map  = {r["query_id"]:  resolve(r["query_image"])  for r in read_csv_rows(qcsv)}
        target_map = {r["target_id"]: resolve(r["target_image"]) for r in read_csv_rows(gcsv)}

        # Ground-truth pairs (optional) — look for *_pairs.csv or labels file
        gt: dict[str, str] = {}   # {query_id: correct_target_id}
        for gt_name in (f"{split}_pairs.csv", f"{split}_labels.csv", "train_pairs.csv"):
            gt_path = data_root / ds / gt_name
            if gt_path.exists():
                for r in read_csv_rows(gt_path):
                    if "query_id" in r and "target_id" in r:
                        gt[r["query_id"]] = r["target_id"]
                if gt:
                    print(f"  GT loaded from {gt_name}: {len(gt)} pairs")
                    break

        # Restrict to queries that appear in the submission
        eligible = [qid for qid in query_map if qid in submission]
        if not eligible:
            print(f"  {ds}/{split}: no matching queries in submission — skipping")
            continue

        sampled = _random.sample(eligible, min(n_queries, len(eligible)))
        print(f"  {ds}/{split}: rendering {len(sampled)} queries (top-{top_k})")

        # ---------------------------------------------------------------- slice extraction

        def mid_slice(path: Path) -> np.ndarray | None:
            """Load axial mid-slice as uint8 (H, W). Returns None on error."""
            try:
                # Memory-map when possible (fast for uncompressed .nii)
                img = nib.load(str(path), mmap=True)
                arr = np.asarray(img.dataobj, dtype=np.float32)
                if arr.ndim == 4:
                    arr = arr[..., 0]
                nz = np.count_nonzero(np.isfinite(arr) & (arr != 0), axis=(0, 1))
                occ = np.where(nz > 0)[0]
                mid = int(occ[len(occ) // 2]) if len(occ) else arr.shape[2] // 2
                sl = np.nan_to_num(arr[:, :, mid], nan=0.0, posinf=0.0, neginf=0.0)
                valid = sl[sl > 0]
                p1, p99 = (np.percentile(valid, (1, 99)) if valid.size else (0.0, 1.0))
                sl = np.clip((sl - p1) / max(p99 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)
                return sl
            except Exception as e:
                print(f"    ERROR loading {path}: {e}")
                return None

        # ---------------------------------------------------------------- render figure

        n_rows = len(sampled)
        n_cols = top_k + 1          # query col + top_k target cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * CELL_W, n_rows * CELL_H),
            squeeze=False,
        )

        fig.patch.set_facecolor("#1a1a2e")

        for row_i, qid in enumerate(sampled):
            # -- query column ------------------------------------------------
            ax_q = axes[row_i, 0]
            q_sl = mid_slice(query_map[qid]) if query_map[qid].exists() else None
            if q_sl is not None:
                ax_q.imshow(q_sl, cmap="gray", interpolation="bilinear")
            else:
                ax_q.set_facecolor("#333")
            ax_q.set_title(
                f"QUERY\n{_short(qid)}",
                fontsize=FONT_MD, color="white", pad=3, fontweight="bold",
            )
            ax_q.axis("off")
            _border(ax_q, COL_QUERY, lw=3)

            # -- target columns ----------------------------------------------
            ranked_targets = submission[qid]
            correct_tid = gt.get(qid)
            has_gt = correct_tid is not None

            for col_j in range(top_k):
                ax = axes[row_i, col_j + 1]

                if col_j < len(ranked_targets):
                    tid = ranked_targets[col_j]
                    rank = col_j + 1

                    t_path = target_map.get(tid)
                    t_sl = mid_slice(t_path) if (t_path and t_path.exists()) else None
                    if t_sl is not None:
                        ax.imshow(t_sl, cmap="gray", interpolation="bilinear")
                    else:
                        ax.set_facecolor("#333")

                    if has_gt:
                        if tid == correct_tid and rank == 1:
                            colour = COL_CORRECT    # green: rank-1 and correct
                            rank_label = f"✓ Rank {rank}"
                        elif tid == correct_tid:
                            colour = COL_FOUND      # orange: correct but not rank-1
                            rank_label = f"✓ Rank {rank}"
                        else:
                            colour = COL_WRONG
                            rank_label = f"✗ Rank {rank}"
                    else:
                        colour = COL_UNKNOWN
                        rank_label = f"Rank {rank}"

                    ax.set_title(
                        f"{rank_label}\n{_short(tid)}",
                        fontsize=FONT_SM,
                        color=colour if has_gt else "white",
                        pad=3,
                    )
                    _border(ax, colour, lw=2)
                else:
                    ax.set_facecolor("#1a1a2e")
                    ax.set_title(f"Rank {col_j + 1}\n—", fontsize=FONT_SM, color="#555", pad=3)
                    _border(ax, "#333", lw=1)

                ax.axis("off")

        # legend
        legend_patches = [
            mpatches.Patch(color=COL_QUERY,   label="Query"),
            mpatches.Patch(color=COL_CORRECT, label="Correct @ Rank 1"),
            mpatches.Patch(color=COL_FOUND,   label="Correct (not Rank 1)"),
            mpatches.Patch(color=COL_WRONG,   label="Wrong"),
            mpatches.Patch(color=COL_UNKNOWN, label="No ground truth"),
        ]
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=5,
            fontsize=FONT_SM,
            facecolor="#1a1a2e",
            edgecolor="#555",
            labelcolor="white",
            framealpha=0.9,
            bbox_to_anchor=(0.5, -0.01),
        )

        gt_status = f" | GT available: {len(gt)} pairs" if gt else " | No GT"
        fig.suptitle(
            f"{ds} / {split}  —  {len(sampled)} queries  ·  top-{top_k} targets{gt_status}",
            fontsize=FONT_MD + 1,
            color="white",
            fontweight="bold",
            y=1.002,
        )

        plt.tight_layout(pad=0.4)

        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        fname = f"{ds}_{split}.png"
        results.append((fname, buf.getvalue()))
        print(f"  → rendered {fname} ({len(buf.getvalue()) // 1024} KB)")

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short(s: str, n: int = 14) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def _border(ax, colour: str, lw: float = 2) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(colour)
        spine.set_linewidth(lw)


def _find_data_root(mount: Path) -> Path:
    for p in sorted(mount.rglob("dataset1")):
        if p.is_dir():
            found = p.parent
            print(f"Data root: {found}")
            return found
    raise RuntimeError(f"dataset1/ not found under {mount}")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    csv: str = "strong_submission.csv",
    out: str = "viz/",
    n_queries: int = N_QUERIES,
    top_k: int = TOP_K,
    seed: int = 42,
) -> None:
    csv_path = Path(csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Submission CSV not found: {csv_path}")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading submission: {csv_path}")
    csv_content = csv_path.read_text(encoding="utf-8")

    print(f"Generating PNGs on Modal (n_queries={n_queries}, top_k={top_k})...")
    pngs = generate_pngs.remote(csv_content, n_queries=n_queries, top_k=top_k, seed=seed)

    for fname, png_bytes in pngs:
        dest = out_dir / fname
        dest.write_bytes(png_bytes)
        print(f"Saved {dest}  ({len(png_bytes) // 1024} KB)")

    print(f"\nDone — {len(pngs)} PNG(s) saved to {out_dir}/")
