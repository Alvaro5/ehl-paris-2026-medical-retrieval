"""Quick data preview -- renders sample NIfTI slices from the Modal volume as PNGs.

For each dataset, picks a few volumes and shows axial / coronal / sagittal
slices at 25 %, 50 %, 75 % of the occupied range.

Run:
    modal run laurence/preview_data.py
    modal run laurence/preview_data.py --n-samples 4 --out data_preview/
"""

from __future__ import annotations

from pathlib import Path
import modal

app = modal.App("ehl-preview-data")
vol = modal.Volume.from_name("ehl-2026-vol-2")

N_SAMPLES = 3       # volumes to sample per dataset x modality (query / target)
DPI = 120


def _deps():
    import nibabel, matplotlib  # noqa: F401


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("nibabel>=5.3", "numpy>=2.0", "matplotlib>=3.8")
    .run_function(_deps)
)


@app.function(image=image, volumes={"/data": vol}, timeout=600, memory=8192, cpu=2)
def preview_data(n_samples: int = N_SAMPLES) -> list[tuple[str, bytes]]:
    import csv as _csv
    import io as _io
    import random as _random

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np

    _random.seed(0)
    data_root = _find_data_root(Path("/data"))

    def read_csv(path):
        with path.open(newline="") as f:
            return list(_csv.DictReader(f))

    def resolve(rel):
        p = Path(rel)
        p = p if p.is_absolute() else data_root / p
        if not p.exists() and p.suffix == ".gz":
            alt = p.with_suffix("")
            if alt.exists():
                return alt
        return p

    def load_arr(path: Path):
        img = nib.load(str(path))
        arr = img.get_fdata(dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def norm_slice(sl: np.ndarray) -> np.ndarray:
        """Percentile-normalise a 2D slice to uint8."""
        valid = sl[sl > 0]
        lo, hi = (np.percentile(valid, (1, 99)) if valid.size else (0.0, 1.0))
        return np.clip((sl - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

    def tri_slices(arr: np.ndarray, pos: float) -> tuple:
        """Return (axial, coronal, sagittal) slices at relative position pos."""
        def _sl(axis):
            a = np.moveaxis(arr, axis, -1)
            nz = np.count_nonzero(np.isfinite(a) & (a != 0), axis=(0, 1))
            occ = np.where(nz > 0)[0]
            lo = int(occ[0]) if len(occ) else 0
            hi = int(occ[-1]) if len(occ) else a.shape[2] - 1
            idx = int(np.clip(round(lo + pos * (hi - lo)), 0, a.shape[2] - 1))
            return norm_slice(a[:, :, idx])
        return _sl(2), _sl(1), _sl(0)   # axial, coronal, sagittal

    results = []
    POSITIONS = (0.25, 0.50, 0.75)
    PLANE_LABELS = ("axial", "coronal", "sagittal")

    for ds in ("dataset1", "dataset2", "dataset3"):
        # Collect query paths from every available split
        volume_entries: list[tuple[str, Path, str]] = []  # (id, path, modality)
        for split in ("train", "val", "test"):
            for kind, col_id, col_img in [
                ("query",  "query_id",  "query_image"),
                ("target", "target_id", "target_image"),
            ]:
                csv_name = "train_pairs.csv" if split == "train" else f"{split}_{'queries' if kind == 'query' else 'gallery'}.csv"
                csv_path = data_root / ds / csv_name
                if not csv_path.exists():
                    continue
                for row in read_csv(csv_path):
                    if col_id not in row or col_img not in row:
                        continue
                    p = resolve(row[col_img])
                    if p.exists():
                        volume_entries.append((row[col_id], p, f"{split}/{kind}"))

        if not volume_entries:
            print(f"  {ds}: no volumes found -- skipping")
            continue

        sampled = _random.sample(volume_entries, min(n_samples, len(volume_entries)))
        print(f"  {ds}: rendering {len(sampled)} sample volumes")

        # Figure: n_samples rows x (3 positions x 3 planes) columns
        n_rows = len(sampled)
        n_cols = len(POSITIONS) * len(PLANE_LABELS)  # 9 columns
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 2.0, n_rows * 2.2),
            squeeze=False,
        )
        fig.patch.set_facecolor("#12121e")

        for row_i, (vol_id, path, modality) in enumerate(sampled):
            try:
                arr = load_arr(path)
            except Exception as e:
                print(f"    ERROR {path}: {e}")
                for ax in axes[row_i]:
                    ax.set_facecolor("#333"); ax.axis("off")
                continue

            col_i = 0
            for pos_j, pos in enumerate(POSITIONS):
                axial, coronal, sagittal = tri_slices(arr, pos)
                for plane_j, (sl, label) in enumerate(
                    zip((axial, coronal, sagittal), PLANE_LABELS)
                ):
                    ax = axes[row_i, col_i]
                    ax.imshow(sl, cmap="gray", interpolation="bilinear")
                    ax.axis("off")

                    # Top-row plane label
                    if row_i == 0:
                        ax.set_title(
                            f"{label}\n@{int(pos*100)}%",
                            fontsize=6, color="#aaa", pad=2,
                        )
                    # Left-column volume label
                    if col_i == 0:
                        ax.set_ylabel(
                            f"{vol_id[:16]}\n({modality})",
                            fontsize=5.5, color="white",
                            rotation=0, labelpad=55, va="center",
                        )
                    # Thin separator between position groups
                    lw = 1.5 if plane_j == 0 else 0.3
                    col = "#4a90d9" if plane_j == 0 else "#333"
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor(col)
                        spine.set_linewidth(lw)

                    col_i += 1

        fig.suptitle(
            f"{ds}  --  {len(sampled)} sample volumes  -  3 anatomical planes  -  3 positions",
            fontsize=9, color="white", fontweight="bold", y=1.003,
        )
        plt.tight_layout(pad=0.3, w_pad=0.15, h_pad=0.4)

        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        fname = f"{ds}_samples.png"
        results.append((fname, buf.getvalue()))
        print(f"    -> {fname}  ({len(buf.getvalue()) // 1024} KB)")

    return results


def _find_data_root(mount: Path) -> Path:
    for p in sorted(mount.rglob("dataset1")):
        if p.is_dir():
            return p.parent
    raise RuntimeError(f"dataset1/ not found under {mount}")


# ---------------------------------------------------------------------------
# Axial sweep -- single paired (query, target) from dataset1
# ---------------------------------------------------------------------------

N_SWEEP = 32        # total axial slices to show
SWEEP_COLS = 8      # columns in the slice grid  ->  rows = N_SWEEP // SWEEP_COLS


@app.function(image=image, volumes={"/data": vol}, timeout=300, memory=8192, cpu=2)
def preview_sweep(n_slices: int = N_SWEEP, pair_index: int = 0) -> tuple[str, bytes]:
    """
    Load one paired (T1 query, T2 target) from dataset1/train_pairs.csv and
    render a grid of evenly-spaced axial (xy-plane) slices for both volumes.

    Layout:
      +----------------------------------------------------------+
      |  T1 QUERY  | z=5% | z=8% | z=11% | ... (SWEEP_COLS)    |
      |            | ...  |      |       |                       |  x (N_SWEEP/SWEEP_COLS) rows
      +----------------------------------------------------------+
      |  T2 TARGET | z=5% | z=8% | z=11% | ...                 |
      |            | ...  |      |       |                       |
      +----------------------------------------------------------+
    Same slice positions for both volumes -> shows T1<->T2 contrast at every depth.
    """
    import csv as _csv
    import io as _io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import nibabel as nib
    import numpy as np

    data_root = _find_data_root(Path("/data"))

    def read_csv(path):
        with path.open(newline="") as f:
            return list(_csv.DictReader(f))

    def resolve(rel):
        p = Path(rel)
        p = p if p.is_absolute() else data_root / p
        if not p.exists() and p.suffix == ".gz":
            alt = p.with_suffix("")
            if alt.exists():
                return alt
        return p

    # Load training pair
    pairs_csv = data_root / "dataset1" / "train_pairs.csv"
    if not pairs_csv.exists():
        raise RuntimeError("dataset1/train_pairs.csv not found")
    pairs = read_csv(pairs_csv)
    pair = pairs[pair_index % len(pairs)]

    q_path = resolve(pair["query_image"])
    t_path = resolve(pair["target_image"])
    q_id   = pair["query_id"]
    t_id   = pair["target_id"]
    print(f"Loading pair {pair_index}: query={q_id}  target={t_id}")

    def load_arr(path):
        img = nib.load(str(path))
        arr = img.get_fdata(dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def norm_slice(sl):
        valid = sl[sl > 0]
        lo, hi = (np.percentile(valid, (1, 99)) if valid.size else (0.0, 1.0))
        return np.clip((sl - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

    def occupied_range(arr):
        """First and last z-index that contain non-zero voxels."""
        nz = np.count_nonzero(np.isfinite(arr) & (arr != 0), axis=(0, 1))
        occ = np.where(nz > 0)[0]
        return (int(occ[0]), int(occ[-1])) if len(occ) else (0, arr.shape[2] - 1)

    q_arr = load_arr(q_path)
    t_arr = load_arr(t_path)

    # Use the query volume's occupied range to pick slice positions
    # (both volumes share the same registered grid in dataset1)
    z_lo, z_hi = occupied_range(q_arr)
    slice_indices = np.linspace(z_lo, z_hi, n_slices, dtype=int)

    rows_per_vol = n_slices // SWEEP_COLS   # e.g. 32//8 = 4
    n_cols = SWEEP_COLS

    # Figure: two stacked grids separated by a spacer row
    # Total axes rows = rows_per_vol (query) + 1 (spacer) + rows_per_vol (target)
    SPACER = 0.06   # fraction of cell height for the divider
    cell_h = 1.9    # inches per subplot row
    cell_w = 1.9    # inches per subplot column

    fig = plt.figure(
        figsize=(n_cols * cell_w, (2 * rows_per_vol) * cell_h + 0.8),
        facecolor="#12121e",
    )

    # GridSpec with height_ratios to squeeze in a thin label row between the two volumes
    heights = [1] * rows_per_vol + [SPACER] + [1] * rows_per_vol
    gs = gridspec.GridSpec(
        len(heights), n_cols,
        figure=fig,
        hspace=0.08, wspace=0.04,
        height_ratios=heights,
    )

    for vol_idx, (arr, label, label_col) in enumerate([
        (q_arr, f"T1 QUERY  -  {q_id}", "#4a90d9"),
        (t_arr, f"T2 TARGET -  {t_id}", "#e67e22"),
    ]):
        gs_row_offset = vol_idx * (rows_per_vol + 1)   # skip spacer row

        for flat_i, z_idx in enumerate(slice_indices):
            row = flat_i // n_cols
            col = flat_i % n_cols
            ax = fig.add_subplot(gs[gs_row_offset + row, col])

            sl = norm_slice(arr[:, :, z_idx])
            ax.imshow(sl, cmap="gray", interpolation="bilinear")
            ax.axis("off")

            # z-position label on first row only, every other column
            if row == 0 and col % 2 == 0:
                pct = int(round((z_idx - z_lo) / max(z_hi - z_lo, 1) * 100))
                ax.set_title(f"z={z_idx}\n({pct}%)", fontsize=5.5, color="#aaa", pad=2)

            # Thin border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#2a2a4a")
                spine.set_linewidth(0.5)

        # Section label as a text annotation on the first subplot of each volume
        first_ax = fig.add_subplot(gs[gs_row_offset, 0])
        first_ax.set_visible(False)   # invisible overlay just for the label
        fig.text(
            0.01,
            1.0 - (vol_idx * (rows_per_vol + 1) + 0.5 * rows_per_vol) / (2 * rows_per_vol + 1),
            label,
            ha="left", va="center",
            fontsize=8, color=label_col, fontweight="bold",
            transform=fig.transFigure,
            rotation=90,
        )

    fig.suptitle(
        f"Dataset1  --  axial sweep  -  {n_slices} slices  -  pair {pair_index}",
        fontsize=9.5, color="white", fontweight="bold", y=1.002,
    )

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    fname = f"dataset1_sweep_pair{pair_index}.png"
    print(f"  -> {fname}  ({len(buf.getvalue()) // 1024} KB)")
    return fname, buf.getvalue()


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    out: str = "data_preview/",
    n_samples: int = N_SAMPLES,
    sweep: bool = False,
    sweep_pair: int = 0,
    n_slices: int = N_SWEEP,
) -> None:
    """
    --sweep          also render the axial slice sweep for dataset1
    --sweep-pair N   which training pair to use (default 0)
    --n-slices N     how many axial slices in the sweep (default 32)
    """
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Multi-dataset sample grid
    pngs = preview_data.remote(n_samples=n_samples)
    for fname, png_bytes in pngs:
        dest = out_dir / fname
        dest.write_bytes(png_bytes)
        print(f"Saved {dest}  ({len(png_bytes) // 1024} KB)")

    # Optional axial sweep
    if sweep:
        fname, png_bytes = preview_sweep.remote(n_slices=n_slices, pair_index=sweep_pair)
        dest = out_dir / fname
        dest.write_bytes(png_bytes)
        print(f"Saved {dest}  ({len(png_bytes) // 1024} KB)")

    print(f"\nDone -- {len(pngs) + (1 if sweep else 0)} PNG(s) in {out_dir}/")
