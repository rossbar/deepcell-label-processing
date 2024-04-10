import pandas as pd
import napari
import tifffile as tff
from pathlib import Path
import numpy as np

from raw_to_dcl import dcl_zip
import constants

datapath = Path("/data2/lung-data-hra-pub")
db_path = datapath / "D265-LLL-7A7-12/D265-LLL-7A7-12_cell_segm.csv"
img_path = datapath / "D265-LLL-7A7-12_Scan1-002.qptiff"
output_dir = Path.home() / f"dryad_dcl_outputs/hra/lung_healthy"
output_dir.mkdir(exist_ok=True, parents=True)

#NOTE: image_mpp = 0.5056 - from Yash

# Load marker data
df = pd.read_csv(db_path)
markers = [m.split(":")[0] for m in df.columns if ":" in m]

# Extract markers from csv in channel order
ordered_unique_markers = []
seen = set()
for m in markers:
    if (m not in seen) and (m != "Cell") and (m != "Nucleus"):
        ordered_unique_markers.append(m)
        seen.add(m)
chmapping = {chname.upper(): idx for idx, chname in enumerate(ordered_unique_markers)}

# Load image
img = tff.imread(img_path, selection=(slice(None), slice(0, 19080), slice(0, 17760)))
mask = tff.imread(datapath / "DAPI_CD298_Q1_mask.tif")
y = mask.transpose(3, 0, 1, 2).astype(np.int32)

# Visualize
#import napari
#nim = napari.view_image(img, name=chmapping, channel_axis=0)
#nim.add_labels(mask, name="segmentation")

# Reformat image for dcl
chmax = img.max(axis=(1, 2), keepdims=True)
chmin = img.min(axis=(1, 2), keepdims=True)
X = ((img - chmin) / (chmax - chmin) * 255).astype(np.uint8)
del img
# Sort channels alphabetically
data_ch_to_idx = dict(sorted(chmapping.items()))
X = X[np.array(list(data_ch_to_idx.values())), ...]
# Add a dim for dcl
X = X[:, np.newaxis, :, :]

# Get cell types
ct_config_file = Path.home() / "Downloads/core_tree.yaml"
with open(ct_config_file) as fh:
    ctdata = yaml.load(fh, yaml.Loader)
ctlist = ravel_dict(ctdata)
cell_types = make_empty_cell_types(ctlist)

# Channel names
channels = list(data_ch_to_idx)

# Tile
for (xmin, xmax) in itertools.pairwise(np.linspace(0, X.shape[2], 14).astype(int)):
    for (ymin, ymax) in itertools.pairwise(np.linspace(0, X.shape[3], 14).astype(int)):
        print(f"{xmin}:{xmax}, {ymin}:{ymax}")
        dcl_zip(
            X[..., xmin:xmax, ymin:ymax],
            y[..., xmin:xmax, ymin:ymax],
            cell_types,
            channels,
            fname=output_dir / f"tile_{xmin}-{xmax}_{ymin}-{ymax}.zip"
        )
