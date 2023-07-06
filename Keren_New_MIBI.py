import os
from pathlib import Path
import numpy as np
import yaml

import utils
from raw_to_dcl import dcl_zip

data_path = Path("/data/data-registry/data/labels/static/2d/Tissue-Breast/Keren_New_MIBI")
fname = "FOV410.npz"
outpath = Path("/home/administrator/data-registry_dcl_outputs/") / "/".join(data_path.parts[-2:])
out_fname = os.path.splitext(fname)[0] + "_mpm_project.zip"

# Load data
data = np.load(data_path / fname)
# Corresponding metadata
with open(data_path / (fname + ".dvc")) as fh:
    metadata = yaml.load(fh, yaml.Loader)

# Load index-to-marker mapping from metadata
idx_to_ch = {
    ch["index"]: ch["target"] for ch in
    metadata["meta"]["sample"]["channels"]
}
channels = list(idx_to_ch.values())


# Load image and cast to float
img = data["X"].astype(float)
mask = data["y"]
print(f"Image shape: {img.shape}")
print(f"Number of cells: {mask.max()}")

# Filter out empty channels
empty_ch_mask = img.sum(axis=(0, 1, 2)) == 0
if np.any(empty_ch_mask):
    channels = np.asarray(channels)
    print(
        (
            f"The following channels are empty and will be dropped:\n"
            f"\t{channels[empty_ch_mask]}"
        )
    )
    channels = channels[~empty_ch_mask].tolist()
    img = img[..., ~empty_ch_mask]

# Normalize and convert to uint8
chmax = img.max(axis=(1, 2), keepdims=True)
chmin = img.min(axis=(1, 2), keepdims=True)
X = ((img - chmin) / (chmax - chmin) * 255).astype(np.uint8)
# ometiff is very picky about dimensions
# TODO: figure out why
X = X.transpose((3, 0, 1, 2))
y = mask.transpose((0, 3, 1, 2))

# Marker-positivity "cell types"
cell_types = utils.make_empty_marker_positivity(channels)

dcl_zip(X, y, cell_types, channels, fname=outpath / out_fname)
