"""Download only *_dinov2.f32 sidecars from Modal Volume to local data directory."""
from __future__ import annotations
from pathlib import Path
import modal

LOCAL_DATA = Path(
    r"C:\Users\laure\Projects\ehl2026\ehl-paris-2026-medical-retrieval"
    r"\data\ehl-paris-medical-image-retrieval"
)
VOL_PREFIX = "ehl-paris-medical-image-retrieval"

vol = modal.Volume.from_name("ehl-2026-vol-2")

print("Listing volume files…")
all_entries = list(vol.listdir(VOL_PREFIX, recursive=True))
f32_entries = [e for e in all_entries if e.path.endswith("_dinov2.f32")]
print(f"Found {len(f32_entries)} .f32 files on volume ({len(all_entries)} total entries)")

downloaded = skipped = 0
for i, entry in enumerate(f32_entries, 1):
    p = entry.path
    rel = p[len(VOL_PREFIX):].lstrip("/\\")
    local = LOCAL_DATA / Path(rel)
    if local.exists():
        skipped += 1
        if skipped % 100 == 0:
            print(f"  [{i}/{len(f32_entries)}] skipped {skipped} (already local)")
        continue
    local.parent.mkdir(parents=True, exist_ok=True)
    with open(local, "wb") as f:
        for chunk in vol.read_file(p):
            f.write(chunk)
    downloaded += 1
    print(f"  [{i}/{len(f32_entries)}] downloaded {local.name}")

print(f"Done. downloaded={downloaded} skipped={skipped}")
