"""DINOv2 + MLP pairwise similarity scorer for dataset1.

Pipeline:
  1. Load up to N_SCANS training pairs from dataset1.
  2. Extract a single axial (xy-plane, z=50%) slice per volume.
  3. Embed each slice with frozen DINOv2-base (768-dim CLS token).
  4. Train a small fully-connected MLP:
       [emb_q || emb_t]  (1536-dim)
       → Linear(1536, 512) + ReLU
       → Linear(512, 128) + ReLU
       → Linear(128, 1) + Sigmoid
     Positive pairs (label=1) vs. in-batch negatives (label=0).
     Loss: binary cross-entropy.
  5. Rank dataset1 val/test gallery by MLP score.

Run with:
    modal run laurence/mlp_similarity_modal.py
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import modal

app = modal.App("ehl-mlp-similarity")
vol = modal.Volume.from_name("ehl-2026-vol-2")

MODEL_ID = "facebook/dinov2-base"
BACKBONE_DIM = 768
IMAGE_SIZE = 224
N_SCANS = 100       # max training pairs to use
MID_SLICE = 0.5     # axial position (fraction of occupied z-range)
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 32


def _download_model():
    from transformers import AutoImageProcessor, AutoModel
    AutoImageProcessor.from_pretrained(MODEL_ID)
    AutoModel.from_pretrained(MODEL_ID)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.3",
        "torchvision>=0.18",
        "transformers>=4.40",
        "nibabel>=5.3",
        "numpy>=2.0",
        "Pillow>=10.0",
    )
    .run_function(_download_model)
)


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="T4",
    timeout=3600,
    memory=16384,
)
def run_mlp_similarity() -> str:
    import csv as _csv
    import io as _io
    from pathlib import Path as _Path

    import nibabel as nib
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image as PILImage
    from transformers import AutoImageProcessor, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = _find_data_root(_Path("/data"))

    def read_csv(path: _Path) -> list[dict[str, str]]:
        with path.open(newline="") as f:
            return list(_csv.DictReader(f))

    def resolve(rel: str) -> _Path:
        p = _Path(rel)
        p = p if p.is_absolute() else data_root / p
        if not p.exists() and p.suffix == ".gz":
            alt = p.with_suffix("")
            if alt.exists():
                return alt
        return p

    # ------------------------------------------------------------------
    # Slice extraction: single axial (xy-plane) slice at MID_SLICE
    # ------------------------------------------------------------------
    def load_axial_slice(nii_path: _Path) -> PILImage.Image:
        img = nib.load(str(nii_path))
        vol = img.get_fdata(dtype=np.float32)
        if vol.ndim == 4:
            vol = vol[..., 0]

        nz = np.count_nonzero(np.isfinite(vol) & (vol != 0), axis=(0, 1))
        occ = np.where(nz > 0)[0]
        z_min = int(occ[0]) if len(occ) else 0
        z_max = int(occ[-1]) if len(occ) else vol.shape[2] - 1
        z = int(np.clip(round(z_min + MID_SLICE * (z_max - z_min)), 0, vol.shape[2] - 1))

        sl = np.nan_to_num(vol[:, :, z], nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = (np.percentile(sl[sl > 0], (1, 99)) if sl.any() else (0.0, 1.0))
        sl = np.clip((sl - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        return PILImage.fromarray(sl).resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR).convert("RGB")

    # ------------------------------------------------------------------
    # DINOv2 backbone (frozen)
    # ------------------------------------------------------------------
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    backbone = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def embed(pil_images: list[PILImage.Image]) -> np.ndarray:
        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        out = backbone(**inputs)
        cls = out.last_hidden_state[:, 0, :].cpu().float().numpy()
        return cls  # (N, 768)

    def embed_paths(paths: dict[str, _Path]) -> dict[str, np.ndarray]:
        ids = sorted(paths)
        pils, order = [], []
        for img_id in ids:
            try:
                pils.append(load_axial_slice(paths[img_id]))
            except Exception as e:
                print(f"  ERROR {paths[img_id]}: {e}")
                pils.append(PILImage.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)))
            order.append(img_id)
        batch_size = 32
        all_cls = []
        for start in range(0, len(pils), batch_size):
            all_cls.append(embed(pils[start : start + batch_size]))
        embs = np.concatenate(all_cls, axis=0)  # (N, 768)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.where(norms > 0, norms, 1.0)
        return {img_id: embs[i] for i, img_id in enumerate(order)}

    # ------------------------------------------------------------------
    # MLP similarity model
    # ------------------------------------------------------------------
    class SimilarityMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(BACKBONE_DIM * 2, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

        def forward(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.net(torch.cat([q, t], dim=-1)).squeeze(-1)

    mlp = SimilarityMLP().to(device)

    # ------------------------------------------------------------------
    # Load training pairs (capped at N_SCANS)
    # ------------------------------------------------------------------
    train_csv = data_root / "dataset1" / "train_pairs.csv"
    all_pairs = read_csv(train_csv) if train_csv.exists() else []
    all_pairs = all_pairs[:N_SCANS]
    print(f"Using {len(all_pairs)} training pairs (cap={N_SCANS})")

    train_q_paths, train_t_paths, valid_pairs = {}, {}, []
    for row in all_pairs:
        qp, tp = resolve(row["query_image"]), resolve(row["target_image"])
        if qp.exists() and tp.exists():
            train_q_paths[row["query_id"]] = qp
            train_t_paths[row["target_id"]] = tp
            valid_pairs.append((row["query_id"], row["target_id"]))

    print(f"Valid pairs on disk: {len(valid_pairs)}")

    print("Embedding training queries...")
    q_feats = embed_paths(train_q_paths)
    print("Embedding training targets...")
    t_feats = embed_paths(train_t_paths)

    q_ids = [p[0] for p in valid_pairs]
    t_ids = [p[1] for p in valid_pairs]
    Q = torch.tensor(np.stack([q_feats[i] for i in q_ids]), dtype=torch.float32)
    T = torch.tensor(np.stack([t_feats[i] for i in t_ids]), dtype=torch.float32)
    N = len(valid_pairs)

    # ------------------------------------------------------------------
    # Training: positive pairs (diagonal) + in-batch negatives
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)
    criterion = nn.BCELoss()

    mlp.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(N)
        total_loss = 0.0
        steps = 0
        for start in range(0, N, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            bq = Q[idx].to(device)   # (B, 768)
            bt = T[idx].to(device)   # (B, 768)
            B = bq.shape[0]

            # All B*B pairs; diagonal = positive, off-diagonal = negative
            bq_exp = bq.unsqueeze(1).expand(B, B, BACKBONE_DIM).reshape(B * B, BACKBONE_DIM)
            bt_exp = bt.unsqueeze(0).expand(B, B, BACKBONE_DIM).reshape(B * B, BACKBONE_DIM)
            labels = torch.eye(B, device=device).reshape(B * B)

            scores = mlp(bq_exp, bt_exp)
            loss = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch+1}/{EPOCHS}  loss={total_loss/steps:.4f}")

    mlp.eval()

    # ------------------------------------------------------------------
    # Inference: embed val/test queries & gallery, score with MLP
    # ------------------------------------------------------------------
    submission_rows: list[dict[str, str]] = []

    for split in ("val", "test"):
        qcsv = data_root / "dataset1" / f"{split}_queries.csv"
        gcsv = data_root / "dataset1" / f"{split}_gallery.csv"
        if not qcsv.exists() or not gcsv.exists():
            print(f"Skipping dataset1/{split}: CSVs not found")
            continue

        q_paths = {r["query_id"]: resolve(r["query_image"]) for r in read_csv(qcsv) if resolve(r["query_image"]).exists()}
        g_paths = {r["target_id"]: resolve(r["target_image"]) for r in read_csv(gcsv) if resolve(r["target_image"]).exists()}
        print(f"dataset1/{split}: {len(q_paths)} queries, {len(g_paths)} gallery")

        inf_q = embed_paths(q_paths)
        inf_g = embed_paths(g_paths)

        q_list = sorted(inf_q)
        g_list = sorted(inf_g)
        Q_inf = torch.tensor(np.stack([inf_q[i] for i in q_list]), dtype=torch.float32).to(device)
        G_inf = torch.tensor(np.stack([inf_g[i] for i in g_list]), dtype=torch.float32).to(device)

        Nq, Ng = Q_inf.shape[0], G_inf.shape[0]
        scores = np.zeros((Nq, Ng), dtype=np.float32)

        with torch.no_grad():
            row_bs = 16
            for qi in range(0, Nq, row_bs):
                bq = Q_inf[qi : qi + row_bs]             # (rb, 768)
                rb = bq.shape[0]
                bq_exp = bq.unsqueeze(1).expand(rb, Ng, BACKBONE_DIM).reshape(rb * Ng, BACKBONE_DIM)
                bg_exp = G_inf.unsqueeze(0).expand(rb, Ng, BACKBONE_DIM).reshape(rb * Ng, BACKBONE_DIM)
                s = mlp(bq_exp, bg_exp).cpu().numpy().reshape(rb, Ng)
                scores[qi : qi + rb] = s

        for i, qid in enumerate(q_list):
            ranked = [g_list[j] for j in np.argsort(-scores[i])]
            submission_rows.append({"query_id": qid, "target_id_ranking": " ".join(ranked)})

    buf = _io.StringIO()
    writer = _csv.DictWriter(buf, fieldnames=["query_id", "target_id_ranking"])
    writer.writeheader()
    writer.writerows(submission_rows)
    print(f"Generated {len(submission_rows)} submission rows.")
    return buf.getvalue()


def _find_data_root(mount: Path) -> Path:
    for p in sorted(mount.rglob("dataset1")):
        if p.is_dir():
            found = p.parent
            print(f"Data root: {found}")
            return found
    raise RuntimeError(f"Could not find dataset1/ under {mount}")


@app.local_entrypoint()
def main(out: str = "mlp_similarity_submission.csv") -> None:
    print("Running MLP similarity on Modal...")
    csv_content = run_mlp_similarity.remote()
    out_path = Path(out)
    out_path.write_text(csv_content, encoding="utf-8")
    print(f"Saved {len(csv_content.splitlines()) - 1} rows to {out_path}")
