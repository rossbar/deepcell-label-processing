import matplotlib.pyplot as plt
from tqdm import tqdm
plt.ion()
import numpy as np
import pandas as pd
from deepcell.applications import Mesmer
from pathlib import Path
import yaml
import itertools
import os
import tifffile as tff

from raw_to_dcl import dcl_zip
import constants

def ravel_dict(d):
    leafs = []
    for k, v in d.items():
        if v == {}:
            leafs.append(k)
        else:
            leafs.extend(ravel_dict(v))
    return leafs


def make_empty_cell_types(ctlist):
    """ Return cellTypes.json with master cell types list but no labels """
    cell_types_json = []
    colors = itertools.cycle(constants.COLOR_MAP)
    for i, ct in enumerate(ctlist):
        cell_types_json.append({'id': i+1, 'cells': [], 'color': next(colors),
                               'name': ct, 'feature': 0})
    return cell_types_json

data_dir = Path("/data2/omap-reference-datasets/omap-pancreas-6")
output_dir = Path.home() / f"dryad_dcl_outputs/omap6-pancreas/"
output_dir.mkdir(exist_ok=True, parents=True)

# Stack image from individual tiffs
img_dict = {
    fpath.name.split(".")[0].split("_")[-1]: tff.imread(fpath)
    for fpath in tqdm(data_dir.iterdir())
    if fpath.name.endswith(".tiff")
}
img = np.array(list(img_dict.values()))
data_chnames = [ch.upper() for ch in img_dict]

# Map marker names to channel indices
data_ch_to_idx = {name: idx for idx, name in enumerate(data_chnames)}

# Prepare segmentation input
segmentation_input = np.stack(
    [
        img[data_ch_to_idx["HOECHST"]],
        img[data_ch_to_idx["E-CADHERIN"]],
    ],
    axis=-1,
)[np.newaxis, ...]

# Segment
app = Mesmer()
mpp = 0.5  # TODO: Find this value
mask = app.predict(segmentation_input, image_mpp=mpp)
y = mask.transpose(3, 0, 1, 2).astype(np.int32)

# Reformat image for dcl
chmax = img.max(axis=(1, 2), keepdims=True)
chmin = img.min(axis=(1, 2), keepdims=True)
X = ((img - chmin) / (chmax - chmin) * 255).astype(np.uint8)
# Sort channels alphabetically
data_ch_to_idx = dict(sorted(data_ch_to_idx.items()))
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
for (xmin, xmax) in itertools.pairwise(np.linspace(0, X.shape[2], 12).astype(int)):
    for (ymin, ymax) in itertools.pairwise(np.linspace(0, X.shape[3], 14).astype(int)):
        print(f"{xmin}:{xmax}, {ymin}:{ymax}")
        dcl_zip(
            X[..., xmin:xmax, ymin:ymax],
            y[..., xmin:xmax, ymin:ymax],
            cell_types,
            channels,
            fname=output_dir / f"tile_{xmin}-{xmax}_{ymin}-{ymax}.zip"
        )
