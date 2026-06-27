import nibabel as nib, pandas as pd
from pathlib import Path

pairs = pd.read_csv("dataset1/train_pairs.csv")
print(pairs.head())                      # confirm the column layout
vol = nib.load("dataset1/images/train/gallery/g_001a480c194d.nii")
print(vol.shape, vol.header.get_zooms()) # shape + voxel spacing

ROOT = Path("/path/to/your/DATA_ROOT")

for ds in ["dataset1", "dataset2", "dataset3"]:
    q = pd.read_csv(ROOT/ds/"val_queries.csv").iloc[0]["query_image"]
    g = pd.read_csv(ROOT/ds/"val_gallery.csv").iloc[0]["target_image"]
    print(ds, "query", nib.load(ROOT/q).shape, "| gallery", nib.load(ROOT/g).shape)