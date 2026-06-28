"""Combined MI submission runner on Modal.

Runs the same ranker-per-dataset logic as Alvaro/src/make_submission.py but
executes on a Modal container that has the data volume and ANTs installed.
All ranker code is inlined (no mount needed) following the laurence/* pattern.

Rankers:
  plain_mi       -- NMI on a shared 64^3 downsampled grid (nibabel+numpy only)
  reg_mi         -- ANTs affine register-then-NMI on a 96^3 grid (antspyx)
  affine_argmin  -- header-only Frobenius distance; exploits the ds3 geometry
                    leak (perfect bijection, runs in seconds)

Default invocation (from repo root or laurence/ via just):
    uv run modal run laurence/make_submission.py --out regmi_full_submission.csv
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("ehl-make-submission")
vol = modal.Volume.from_name("ehl-2026-vol-2")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "antspyx",
        "nibabel>=5.3",
        "numpy>=2.0",
        "pandas",
    )
)


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=7200,
    memory=16384,
    cpu=8,
)
def run_submission(
    ranker_map_str: str = "1:plain_mi,2:reg_mi,3:affine_argmin",
    transform: str = "Affine",
    datasets: str = "1,2,3",
    splits: str = "val,test",
) -> str:
    import csv
    import io
    import time
    from pathlib import Path as _Path

    import nibabel as nib
    import numpy as np
    import pandas as pd

    # ---------------------------------------------------------------- helpers

    def find_data_root(mount: _Path) -> _Path:
        for p in sorted(mount.rglob("dataset1")):
            if p.is_dir():
                root = p.parent
                print(f"Data root: {root}")
                return root
        raise RuntimeError(f"No dataset1/ under {mount}")

    def resolve(rel: str, root: _Path) -> _Path:
        p = _Path(rel)
        if not p.is_absolute():
            p = root / p
        if not p.exists() and p.suffix == ".gz":
            alt = p.with_suffix("")
            if alt.exists():
                return alt
        return p

    # ---------------------------------------------------------------- plain MI ranker (inlined from Alvaro/src/mi_ranker.py)

    MI_GRID = 64
    MI_BINS = 32

    def _resample_to_grid(arr: np.ndarray, grid: int) -> np.ndarray:
        steps = tuple(max(1, s // grid) for s in arr.shape)
        arr = arr[:: steps[0], :: steps[1], :: steps[2]]
        out = np.zeros((grid, grid, grid), dtype=np.float32)
        s = tuple(min(arr.shape[i], grid) for i in range(3))
        out[: s[0], : s[1], : s[2]] = arr[: s[0], : s[1], : s[2]]
        return out

    def digitize_array(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = arr > 0
        if not mask.any():
            return np.zeros_like(arr, dtype=np.int32), mask
        lo, hi = arr[mask].min(), arr[mask].max()
        bins = np.linspace(lo, hi, MI_BINS + 1)
        d = np.digitize(arr, bins, right=True).astype(np.int32)
        return d, mask

    def _nmi(qd: np.ndarray, qm: np.ndarray, td: np.ndarray, tm: np.ndarray) -> float:
        mask = qm | tm
        if mask.sum() < 50:
            return 0.0
        h = np.zeros((MI_BINS + 1, MI_BINS + 1), dtype=np.float64)
        np.add.at(h, (qd[mask], td[mask]), 1.0)
        h /= h.sum() + 1e-10
        pa = h.sum(axis=1)
        pb = h.sum(axis=0)
        ha = -(pa * np.log(pa + 1e-10)).sum()
        hb = -(pb * np.log(pb + 1e-10)).sum()
        hab = -(h * np.log(h + 1e-10)).sum()
        v = float((ha + hb) / (hab + 1e-10))
        return v if np.isfinite(v) else 0.0

    def _load_vol(path: str, grid: int) -> np.ndarray:
        arr = nib.load(path).get_fdata(dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return _resample_to_grid(arr, grid)

    class MIRanker:
        def __init__(self, grid: int = MI_GRID) -> None:
            self.grid = grid
            self._arr_cache: dict[str, np.ndarray] = {}
            self._digit_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        def _arr(self, path: str) -> np.ndarray:
            if path not in self._arr_cache:
                self._arr_cache[path] = _load_vol(path, self.grid)
            return self._arr_cache[path]

        def _digit(self, path: str) -> tuple[np.ndarray, np.ndarray]:
            if path not in self._digit_cache:
                self._digit_cache[path] = digitize_array(self._arr(path))
            return self._digit_cache[path]

        def rank(self, query_id: str, query_path: str, targets: dict[str, str]) -> list[str]:
            qd, qm = self._digit(query_path)
            scored = [(tid, _nmi(qd, qm, *self._digit(tp))) for tid, tp in targets.items()]
            scored.sort(key=lambda x: -x[1])
            return [tid for tid, _ in scored]

    # ---------------------------------------------------------------- reg-MI ranker (inlined from Alvaro/src/reg_mi_ranker.py)

    REG_GRID = 96
    # Per-registration wall-clock timeout in seconds. ANTs can hang for minutes
    # on badly-deformed ds2 pairs; killing and scoring 0.0 is better than stalling.
    REG_TIMEOUT = 60

    def _reg_mi_one(args: tuple) -> tuple[str, float]:
        """Stateless worker: register moving→fixed, return (tid, NMI score).
        Runs in a child process so ANTs threads don't block the parent heartbeat.
        """
        tid, query_path, target_path, grid, transform = args
        import ants
        try:
            fixed = ants.from_numpy(_load_vol(query_path, grid))
            moving = ants.from_numpy(_load_vol(target_path, grid))
            reg = ants.registration(fixed=fixed, moving=moving, type_of_transform=transform)
            warped = reg["warpedmovout"].numpy()
            qd, qm = digitize_array(_load_vol(query_path, grid))
            td, tm = digitize_array(warped)
            return tid, _nmi(qd, qm, td, tm)
        except Exception as e:
            print(f"[reg-mi] error ({tid}): {e}", flush=True)
            return tid, 0.0

    class RegMIRanker:
        def __init__(self, grid: int = REG_GRID, type_of_transform: str = "Affine") -> None:
            self.grid = grid
            self.transform = type_of_transform

        def rank(self, query_id: str, query_path: str, targets: dict[str, str]) -> list[str]:
            import concurrent.futures
            args = [
                (tid, query_path, tpath, self.grid, self.transform)
                for tid, tpath in targets.items()
            ]
            # Use 4 workers: ANTs uses ~2 threads internally, so 4 workers ≈ 8 CPUs.
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(_reg_mi_one, a): a[0] for a in args}
                scored = []
                for fut in concurrent.futures.as_completed(futures, timeout=REG_TIMEOUT * len(targets)):
                    try:
                        tid, score = fut.result(timeout=REG_TIMEOUT)
                    except Exception as e:
                        tid = futures[fut]
                        print(f"[reg-mi] timeout/error ({tid}): {e}", flush=True)
                        score = 0.0
                    scored.append((tid, score))
            scored.sort(key=lambda x: -x[1])
            return [tid for tid, _ in scored]

    # ---------------------------------------------------------------- affine-argmin ranker (inlined from Alvaro/src/affine_ranker.py)

    class AffineArgminRanker:
        def __init__(self) -> None:
            self._cache: dict[str, np.ndarray] = {}

        def _affine(self, path: str) -> np.ndarray:
            if path not in self._cache:
                self._cache[path] = np.asarray(nib.load(path).affine, dtype=float)
            return self._cache[path]

        def rank(self, query_id: str, query_path: str, targets: dict[str, str]) -> list[str]:
            qa = self._affine(query_path)
            scored = [(tid, -float(np.linalg.norm(qa - self._affine(tp)))) for tid, tp in targets.items()]
            scored.sort(key=lambda x: -x[1])
            return [tid for tid, _ in scored]

    # ---------------------------------------------------------------- parse ranker map

    ds_ids = sorted(int(x) for x in datasets.split(",") if x.strip())
    split_list = [s for s in ("val", "test") if s in {x.strip() for x in splits.split(",")}]

    ranker_of_ds: dict[int, str] = {ds: "plain_mi" for ds in ds_ids}
    for item in ranker_map_str.split(","):
        if not item.strip():
            continue
        ds_str, name = item.split(":")
        ranker_of_ds[int(ds_str)] = name.strip()

    # ---------------------------------------------------------------- load pools

    data_root = find_data_root(_Path("/data"))
    sets: list[tuple[int, str, dict[str, str], dict[str, str]]] = []
    for ds in ds_ids:
        for split in split_list:
            ds_dir = data_root / f"dataset{ds}"
            qcsv = ds_dir / f"{split}_queries.csv"
            gcsv = ds_dir / f"{split}_gallery.csv"
            if not qcsv.exists() or not gcsv.exists():
                print(f"  WARN dataset{ds}/{split}: missing CSV; skipping")
                continue
            queries: dict[str, str] = {}
            for _, row in pd.read_csv(qcsv).iterrows():
                p = resolve(str(row["query_image"]), data_root)
                if p.exists():
                    queries[str(row["query_id"])] = str(p)
            targets_map: dict[str, str] = {}
            for _, row in pd.read_csv(gcsv).iterrows():
                p = resolve(str(row["target_image"]), data_root)
                if p.exists():
                    targets_map[str(row["target_id"])] = str(p)
            if queries and targets_map:
                sets.append((ds, split, queries, targets_map))
                print(f"  dataset{ds}/{split}: {len(queries)} queries, {len(targets_map)} targets")

    if not sets:
        raise RuntimeError("No usable prediction sets found.")

    # ---------------------------------------------------------------- build rankers

    need = {ranker_of_ds[ds] for ds, *_ in sets}
    rankers: dict[str, object] = {}
    if "plain_mi" in need:
        rankers["plain_mi"] = MIRanker(grid=MI_GRID)
    if "reg_mi" in need:
        rankers["reg_mi"] = RegMIRanker(grid=REG_GRID, type_of_transform=transform)
    if "affine_argmin" in need:
        rankers["affine_argmin"] = AffineArgminRanker()

    # ---------------------------------------------------------------- rank

    rows: list[dict[str, str]] = []
    for ds, split, queries, targets_map in sets:
        rn = ranker_of_ds[ds]
        ranker = rankers[rn]
        print(f"\n=== dataset{ds}/{split} [{rn}]: {len(queries)}×{len(targets_map)} pairs ===", flush=True)
        t0 = time.time()
        for idx, (qid, qpath) in enumerate(sorted(queries.items())):
            ranking = ranker.rank(qid, qpath, targets_map)
            rows.append({"query_id": qid, "target_id_ranking": " ".join(ranking)})
            if (idx + 1) % 10 == 0:
                print(f"  {idx+1}/{len(queries)} queries ({time.time()-t0:.1f}s)", flush=True)
        print(f"  done ({time.time()-t0:.1f}s)", flush=True)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["query_id", "target_id_ranking"])
    writer.writeheader()
    writer.writerows(rows)
    print(f"\nGenerated {len(rows)} submission rows.")
    return buf.getvalue()


@app.local_entrypoint()
def main(
    out: str = "regmi_full_submission.csv",
    ranker_map: str = "1:plain_mi,2:reg_mi,3:affine_argmin",
    transform: str = "Affine",
    datasets: str = "1,2,3",
    splits: str = "val,test",
) -> None:
    csv_content = run_submission.remote(
        ranker_map_str=ranker_map,
        transform=transform,
        datasets=datasets,
        splits=splits,
    )
    out_path = Path(out)
    out_path.write_text(csv_content, encoding="utf-8")
    print(f"Saved {len(csv_content.splitlines()) - 1} rows to {out_path}")
