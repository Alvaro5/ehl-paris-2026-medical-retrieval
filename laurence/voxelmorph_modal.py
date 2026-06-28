"""VoxelMorph cross-modal MRI retrieval on Modal.

Cross-modal challenge: T1ce and T2 have inverted intensity contrasts (WM is
bright in T1, dark in T2), so NCC on raw intensities is unreliable.  Instead
we use Sobel edge maps as the image representation — edges at tissue boundaries
look similar in both modalities, making local NCC valid for cross-modal use.

Pipeline
--------
1. Pre-load all unique volumes, resample to 96³, compute Sobel edge maps.
2. Train VoxelMorph (UNet → displacement field → spatial transformer) on
   dataset1 T1ce/T2 pairs.  Loss = local NCC(warped_edge_query, edge_target)
   + 0.01 × flow gradient (smoothness).
   Training uses random 64³ patches for speed; inference uses full 96³ volumes.
   Augmentation: independent random flips/90° rotations on query and target
   simulate dataset2's independently deformed pairs.
3. Save trained model to /data/models/voxelmorph_96.pth (skipped on rerun).
4. Inference: for every retrieval pair run a forward pass → score by global
   NCC(warped_query_edge, target_edge).  Dataset1 also adds a direct NCC score
   (no registration) weighted 50/50 with the registered score.
5. Write Kaggle submission CSV.

Run:
    modal run laurence/voxelmorph_modal.py
    modal run laurence/voxelmorph_modal.py --retrain   # force retrain
    modal run laurence/voxelmorph_modal.py --out my_submission.csv
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import modal

app = modal.App("ehl-voxelmorph")
vol = modal.Volume.from_name("ehl-2026-vol-2")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.3",
        "nibabel>=5.3",
        "numpy>=2.0",
        "scipy>=1.11",
    )
)

VOL_SIZE    = (96, 96, 96)   # resample target (kept for inference)
PATCH_SIZE  = (64, 64, 64)   # random crop size used during training
N_EPOCHS    = 50             # reduced from 100; early stopping handles the rest
EARLY_STOP_PATIENCE = 10     # stop if no NCC improvement for this many epochs
LR          = 1e-4
TRAIN_BATCH = 4              # increased from 2; patches are smaller than full vols
INFER_BATCH = 4
REG_WEIGHT  = 0.01
NCC_WIN     = 9
MODEL_PATH  = "/data/models/voxelmorph_96.pth"


# ---------------------------------------------------------------------------
# Model components (pure PyTorch, no external registration library)
# ---------------------------------------------------------------------------

def _build_model_and_losses():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SpatialTransformer(nn.Module):
        """Warp src by a voxel-space displacement field.
        Dynamic grid — works at any spatial resolution, so the same model
        trained on 64³ patches runs correctly at 96³ inference time.
        """
        def forward(self, src, flow):
            H, W, D = src.shape[2:]
            device = src.device
            vecs = [torch.arange(s, dtype=torch.float32, device=device) for s in (H, W, D)]
            grid = torch.stack(torch.meshgrid(vecs, indexing='ij'), dim=-1).unsqueeze(0)
            new_locs = grid + flow.permute(0, 2, 3, 4, 1)
            new_locs[..., 0] = 2 * new_locs[..., 0] / (H - 1) - 1
            new_locs[..., 1] = 2 * new_locs[..., 1] / (W - 1) - 1
            new_locs[..., 2] = 2 * new_locs[..., 2] / (D - 1) - 1
            new_locs = new_locs[..., [2, 1, 0]]  # grid_sample wants (x,y,z)
            return F.grid_sample(src, new_locs, align_corners=True,
                                 mode='bilinear', padding_mode='border')

    def _conv(ic, oc, stride=1):
        return nn.Sequential(
            nn.Conv3d(ic, oc, 3, stride=stride, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    class VoxelMorphNet(nn.Module):
        """
        Lighter encoder-decoder UNet: ~4× fewer params than original.

        Encoder:  2 → 8 → 16 → 16 → 16   (stride-2 convs)
        Decoder (with skips):
          d3: cat(up(e4)=16, e3=16) = 32 → 16
          d2: cat(up(d3)=16, e2=16) = 32 → 16
          d1: cat(up(d2)=16, e1= 8) = 24 →  8
          d0: cat(up(d1)= 8, inp=2) = 10 →  8
        Flow:  8 → 3
        """
        def __init__(self):
            super().__init__()
            self.e1 = _conv(2,  8,  stride=2)
            self.e2 = _conv(8,  16, stride=2)
            self.e3 = _conv(16, 16, stride=2)
            self.e4 = _conv(16, 16, stride=2)

            self.d3 = _conv(32, 16)
            self.d2 = _conv(32, 16)
            self.d1 = _conv(24,  8)
            self.d0 = _conv(10,  8)

            self.flow = nn.Conv3d(8, 3, 3, padding=1)
            nn.init.zeros_(self.flow.weight)
            nn.init.zeros_(self.flow.bias)

            self.up  = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
            self.stn = SpatialTransformer()

        def forward(self, moving, fixed):
            inp = torch.cat([moving, fixed], 1)
            e1  = self.e1(inp)
            e2  = self.e2(e1)
            e3  = self.e3(e2)
            e4  = self.e4(e3)
            x   = self.d3(torch.cat([self.up(e4), e3], 1))
            x   = self.d2(torch.cat([self.up(x),  e2], 1))
            x   = self.d1(torch.cat([self.up(x),  e1], 1))
            x   = self.d0(torch.cat([self.up(x),  inp], 1))
            flow   = self.flow(x)
            moved  = self.stn(moving, flow)
            return moved, flow

    def local_ncc_loss(pred, target, win=NCC_WIN):
        """Negative local NCC (minimise to maximise alignment)."""
        k = torch.ones([1, 1] + [win]*3, device=pred.device) / win**3
        p  = lambda x: F.conv3d(x, k, padding=win//2)
        I  = pred;  J  = target
        I2 = p(I*I); J2 = p(J*J); IJ = p(I*J)
        Im = p(I);   Jm = p(J)
        cross = IJ - Im*Jm
        Iv    = I2 - Im*Im
        Jv    = J2 - Jm*Jm
        cc    = cross*cross / (Iv*Jv + 1e-5)
        return -cc.mean()

    def grad_loss(flow):
        """L2 penalty on flow field gradients (encourages smooth warps)."""
        dy = (flow[:, :, 1:] - flow[:, :, :-1]).pow(2).mean()
        dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).pow(2).mean()
        dz = (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).pow(2).mean()
        return (dy + dx + dz) / 3

    def global_ncc_per_sample(moved, fixed):
        """Global NCC for each pair in the batch. Returns (B,) in [-1,1]."""
        B   = moved.shape[0]
        a   = moved.reshape(B, -1);  b = fixed.reshape(B, -1)
        a  -= a.mean(1, keepdim=True);  b -= b.mean(1, keepdim=True)
        num = (a * b).sum(1)
        den = (a.pow(2).sum(1) * b.pow(2).sum(1)).sqrt() + 1e-6
        return num / den

    return VoxelMorphNet, local_ncc_loss, grad_loss, global_ncc_per_sample


# ---------------------------------------------------------------------------
# Main remote function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="A100",
    timeout=7200,
    memory=32768,
    cpu=8,
)
def run_voxelmorph(retrain: bool = False) -> str:
    import csv as _csv
    import io as _io
    import random
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path as _Path

    import nibabel as nib
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.ndimage import sobel, zoom

    t_start = time.time()

    def elapsed():
        return f"[+{time.time()-t_start:5.0f}s]"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"  VoxelMorph cross-modal MRI retrieval")
    print(f"{'='*60}")
    print(f"  Device  : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU     : {props.name}  ({props.total_memory // 1024**2} MB)")
    print(f"  Config  : vol={VOL_SIZE}, patch={PATCH_SIZE}, epochs={N_EPOCHS}, lr={LR}")
    print(f"  Batch   : train={TRAIN_BATCH}, infer={INFER_BATCH}")
    print(f"  Retrain : {retrain}")
    print(f"{'='*60}\n")

    VoxelMorphNet, local_ncc_loss, grad_loss, global_ncc = _build_model_and_losses()
    print(f"{elapsed()} Model classes loaded.")

    # ---------------------------------------------------------------- helpers

    def find_data_root(mount: _Path) -> _Path:
        for p in sorted(mount.rglob("dataset1")):
            if p.is_dir():
                found = p.parent
                print(f"Data root: {found}")
                return found
        raise RuntimeError(f"No dataset1/ under {mount}")

    def read_csv(path: _Path) -> list[dict]:
        with path.open(newline="") as f:
            return list(_csv.DictReader(f))

    def resolve(rel: str, root: _Path) -> _Path:
        p = _Path(rel)
        if not p.is_absolute():
            p = root / p
        if not p.exists() and p.suffix == ".gz":
            alt = p.with_suffix("")
            if alt.exists():
                return alt
        return p

    data_root = find_data_root(_Path("/data"))
    print(f"{elapsed()} Data root located: {data_root}\n")

    # ---------------------------------------------------------------- manifests

    SPECS = [
        ("dataset1", "val"),
        ("dataset1", "test"),
        ("dataset2", "val"),
        ("dataset2", "test"),
        ("dataset3", "val"),
        ("dataset3", "test"),
    ]

    print("--- Manifests ---")
    prediction_sets: list[dict] = []
    for ds, split in SPECS:
        qcsv = data_root / ds / f"{split}_queries.csv"
        gcsv = data_root / ds / f"{split}_gallery.csv"
        if not qcsv.exists() or not gcsv.exists():
            print(f"  SKIP {ds}/{split}: CSV missing")
            continue
        queries = {r["query_id"]: resolve(r["query_image"], data_root) for r in read_csv(qcsv)}
        targets = {r["target_id"]: resolve(r["target_image"], data_root) for r in read_csv(gcsv)}
        queries = {k: v for k, v in queries.items() if v.exists()}
        targets = {k: v for k, v in targets.items() if v.exists()}
        if queries and targets:
            prediction_sets.append({"ds": ds, "split": split, "queries": queries, "targets": targets})
            print(f"  {ds}/{split}: {len(queries)} queries × {len(targets)} targets")

    # ----------------------------------------------------------------
    # Pre-load all unique volumes → 96³ edge maps
    # ----------------------------------------------------------------
    # Sobel edge maps are modality-independent: tissue boundaries look
    # similar in T1ce and T2, so local NCC on edge maps is valid
    # cross-modally.

    all_paths: dict[str, _Path] = {}
    for ps in prediction_sets:
        all_paths.update(ps["queries"])
        all_paths.update(ps["targets"])

    train_csv_path = data_root / "dataset1" / "train_pairs.csv"
    if train_csv_path.exists():
        for r in read_csv(train_csv_path):
            for key, col in [("query_id", "query_image"), ("target_id", "target_image")]:
                p = resolve(r[col], data_root)
                if p.exists():
                    all_paths[r[key]] = p

    print(f"\n--- Volume pre-loading ---")
    print(f"{elapsed()} {len(all_paths)} unique volumes → resample to {VOL_SIZE} → Sobel edge map")
    print(f"  Using 8 threads. Estimated ~2 min …")

    def _load_edge(item):
        img_id, path = item
        arr = nib.load(str(path)).get_fdata(dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        arr = np.nan_to_num(arr)

        factors = tuple(VOL_SIZE[i] / arr.shape[i] for i in range(3))
        arr = zoom(arr, factors, order=1)

        ex = sobel(arr, axis=0); ey = sobel(arr, axis=1); ez = sobel(arr, axis=2)
        edges = np.sqrt(ex**2 + ey**2 + ez**2).astype(np.float32)
        fg = edges[edges > 0]
        if len(fg):
            p1, p99 = np.percentile(fg, (1, 99))
            edges = np.clip((edges - p1) / (p99 - p1 + 1e-6), 0, 1)

        return img_id, edges.astype(np.float32)

    edge_cache: dict[str, np.ndarray] = {}
    n_total = len(all_paths)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for n_done, (img_id, arr) in enumerate(
            ex.map(_load_edge, sorted(all_paths.items())), start=1
        ):
            edge_cache[img_id] = arr
            if n_done % 100 == 0 or n_done == n_total:
                print(f"  {elapsed()} loaded {n_done}/{n_total} volumes")

    example = next(iter(edge_cache.values()))
    mb = sum(a.nbytes for a in edge_cache.values()) / 1024**2
    print(f"{elapsed()} Done — {len(edge_cache)} edge maps, shape {example.shape}, "
          f"total cache {mb:.0f} MB\n")

    def to_tensor(arr):
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W,D)

    def random_patch(arr: np.ndarray) -> np.ndarray:
        """Crop a random PATCH_SIZE sub-volume from a 96³ array."""
        ph, pw, pd = PATCH_SIZE
        oh = random.randint(0, arr.shape[0] - ph)
        ow = random.randint(0, arr.shape[1] - pw)
        od = random.randint(0, arr.shape[2] - pd)
        return arr[oh:oh+ph, ow:ow+pw, od:od+pd]

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------

    model = VoxelMorphNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"--- Model ---")
    print(f"  Parameters : {n_params:,}")
    model_path = _Path(MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists() and not retrain:
        print(f"{elapsed()} Loading saved model from {model_path}")
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        print(f"{elapsed()} Model loaded.\n")
    else:
        print(f"\n{'='*60}")
        print(f"  Training VoxelMorph  (FP16 + patch={PATCH_SIZE} + early_stop={EARLY_STOP_PATIENCE})")
        print(f"{'='*60}")

        train_csv = data_root / "dataset1" / "train_pairs.csv"
        train_pairs = [
            (r["query_id"], r["target_id"])
            for r in read_csv(train_csv)
            if r["query_id"] in edge_cache and r["target_id"] in edge_cache
        ]
        steps_per_epoch = len(train_pairs) // TRAIN_BATCH
        print(f"  Pairs    : {len(train_pairs)}")
        print(f"  Steps/ep : {steps_per_epoch}")
        print(f"  Epochs   : {N_EPOCHS} (max)")
        print(f"  Est. time: ~{steps_per_epoch * N_EPOCHS * 0.015 / 60:.0f} min on A100")
        print()

        def augment(arr):
            """Random flips and 90° rotations applied independently to each
            image in a pair to simulate dataset2's independent deformations."""
            for axis in range(3):
                if random.random() < 0.5:
                    arr = np.flip(arr, axis=axis).copy()
            k = random.randint(0, 3)
            if k:
                ax = random.choice([(0,1),(0,2),(1,2)])
                arr = np.rot90(arr, k, ax).copy()
            return arr

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCHS)
        scaler = torch.cuda.amp.GradScaler()  # FP16 loss scaler

        best_ncc = float("inf")
        epochs_no_improve = 0

        t_train_start = time.time()
        for epoch in range(1, N_EPOCHS + 1):
            random.shuffle(train_pairs)
            total_ncc  = 0.0
            total_grad = 0.0
            steps = 0

            model.train()
            for start in range(0, len(train_pairs) - TRAIN_BATCH + 1, TRAIN_BATCH):
                batch = train_pairs[start : start + TRAIN_BATCH]

                # Patch crop then augment independently for query and target
                moving = torch.cat([
                    to_tensor(augment(random_patch(edge_cache[qid]))) for qid, _ in batch
                ])  # (B,1,64,64,64)
                fixed = torch.cat([
                    to_tensor(augment(random_patch(edge_cache[tid]))) for _, tid in batch
                ])

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    moved, flow = model(moving, fixed)
                    l_ncc  = local_ncc_loss(moved, fixed)
                    l_grad = grad_loss(flow)
                    loss   = l_ncc + REG_WEIGHT * l_grad

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_ncc  += l_ncc.item()
                total_grad += l_grad.item()
                steps += 1

            scheduler.step()

            if steps == 0:
                print(f"  {elapsed()} epoch {epoch:03d}/{N_EPOCHS}  WARNING: no steps")
                continue

            epoch_ncc = total_ncc / steps
            if epoch % 10 == 0 or epoch == 1:
                ep_sec = time.time() - t_train_start
                eta    = ep_sec / epoch * (N_EPOCHS - epoch)
                print(f"  {elapsed()} epoch {epoch:03d}/{N_EPOCHS}  "
                      f"ncc={epoch_ncc:.4f}  grad={total_grad/steps:.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"ETA {eta/60:.1f} min")

            # Early stopping
            if epoch_ncc < best_ncc - 1e-4:
                best_ncc = epoch_ncc
                epochs_no_improve = 0
                torch.save(model.state_dict(), str(model_path))  # save best
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    print(f"  {elapsed()} Early stop at epoch {epoch} "
                          f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
                    break

        # Load best checkpoint
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        vol.commit()
        print(f"\n{elapsed()} Best model saved → {model_path}")

    # ----------------------------------------------------------------
    # Inference: score all retrieval pairs (full 96³ volumes)
    # ----------------------------------------------------------------

    print(f"\n{'='*60}")
    print(f"  Inference")
    print(f"{'='*60}")
    model.eval()

    rows: list[dict] = []

    for ps in prediction_sets:
        ds     = ps["ds"]
        split  = ps["split"]
        q_ids  = sorted(k for k in ps["queries"] if k in edge_cache)
        t_ids  = sorted(k for k in ps["targets"] if k in edge_cache)
        nq, nt = len(q_ids), len(t_ids)
        n_fwd  = (nq * nt + INFER_BATCH - 1) // INFER_BATCH
        print(f"\n{elapsed()} {ds}/{split}: {nq} queries × {nt} targets "
              f"= {nq*nt} pairs  ({n_fwd} forward passes)")

        reg_scores  = np.zeros((nq, nt), dtype=np.float32)
        dir_scores  = np.zeros((nq, nt), dtype=np.float32)

        t_tensors = torch.cat([to_tensor(edge_cache[tid]) for tid in t_ids])
        print(f"  Gallery tensors on {device}: {t_tensors.shape}  "
              f"({t_tensors.element_size() * t_tensors.nelement() / 1024**2:.0f} MB)")

        t_pool_start = time.time()
        with torch.no_grad():
            for i, qid in enumerate(q_ids):
                if i % 10 == 0:
                    rate = i / (time.time() - t_pool_start + 1e-6)
                    eta  = (nq - i) / (rate + 1e-6)
                    print(f"  {elapsed()} query {i:3d}/{nq}  "
                          f"({rate:.1f} q/s,  ETA {eta:.0f}s)")
                q_t = to_tensor(edge_cache[qid])

                for j0 in range(0, nt, INFER_BATCH):
                    j1       = min(j0 + INFER_BATCH, nt)
                    B        = j1 - j0
                    fixed_b  = t_tensors[j0:j1]
                    moving_b = q_t.expand(B, -1, -1, -1, -1)

                    with torch.cuda.amp.autocast():
                        moved_b, _ = model(moving_b, fixed_b)

                    reg_scores[i, j0:j1] = global_ncc(moved_b, fixed_b).cpu().numpy()
                    dir_scores[i, j0:j1] = global_ncc(moving_b, fixed_b).cpu().numpy()

        pool_time = time.time() - t_pool_start
        print(f"  {elapsed()} Pool done in {pool_time:.0f}s  "
              f"({nq*nt/pool_time:.0f} pairs/s)")

        direct_weight = 0.5 if ds == "dataset1" else 0.0
        scores = (1 - direct_weight) * reg_scores + direct_weight * dir_scores
        print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]  "
              f"(direct_weight={direct_weight})")

        for i, qid in enumerate(q_ids):
            ranked = [t_ids[j] for j in np.argsort(-scores[i])]
            rows.append({"query_id": qid, "target_id_ranking": " ".join(ranked)})

    buf = _io.StringIO()
    writer = _csv.DictWriter(buf, fieldnames=["query_id", "target_id_ranking"])
    writer.writeheader()
    writer.writerows(rows)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Done!  {len(rows)} submission rows  |  total time {total_time/60:.1f} min")
    print(f"{'='*60}")
    return buf.getvalue()


@app.local_entrypoint()
def main(out: str = "voxelmorph_submission.csv", retrain: bool = False) -> None:
    print(f"Running VoxelMorph retrieval on Modal (retrain={retrain}) …")
    csv_content = run_voxelmorph.remote(retrain=retrain)
    out_path = Path(out)
    out_path.write_text(csv_content, encoding="utf-8")
    print(f"Saved {len(csv_content.splitlines()) - 1} rows to {out_path}")
