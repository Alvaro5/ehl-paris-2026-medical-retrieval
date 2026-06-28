"""Modal web endpoint that serves _dinov2.f32 files directly from the Volume.

Deploy once:
    modal deploy laurence/serve_embeddings_modal.py

Then set MODAL_EMBEDDINGS_URL in nii-viewer/.env.local to the printed URL.
"""
from __future__ import annotations
from pathlib import Path
import modal

app = modal.App("ehl-dinov2-serve")
vol = modal.Volume.from_name("ehl-2026-vol-2")

image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]")

VOL_PREFIX = "ehl-paris-medical-image-retrieval"


@app.function(image=image, volumes={"/data": vol}, max_containers=3)
@modal.fastapi_endpoint(method="GET")
def get_embedding(path: str) -> modal.Response:
    """Serve a _dinov2.f32 sidecar for the given relative image path.

    path: relative image path as in the CSV, e.g.
          dataset1/images/train/gallery/g_xxx.nii
    """
    import re

    # Strip extension(s) to get stem, append _dinov2.f32
    stem = re.sub(r"\.nii(\.gz)?$", "", Path(path).name)
    sidecar_name = f"{stem}_dinov2.f32"
    # path is relative to DATA_ROOT (ehl-paris-medical-image-retrieval/), volume is at /data
    sidecar_path = Path("/data") / VOL_PREFIX / Path(path).parent / sidecar_name

    if not sidecar_path.exists():
        return modal.Response(
            content=f"Not found: {sidecar_path}",
            status_code=404,
            media_type="text/plain",
        )

    data = sidecar_path.read_bytes()
    return modal.Response(
        content=data,
        status_code=200,
        media_type="application/octet-stream",
        headers={"Content-Length": str(len(data))},
    )
