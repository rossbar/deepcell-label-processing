import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import tifffile as tff
from ome_types import from_tiff
from deepcell.applications import Mesmer
from pathlib import Path
import yaml
import itertools
import os

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

data_dir = Path("/data/cytokit_spleen/")
dataset = "HBM279.RTXC.523-586b77e11e6183de4363fe7a9385282f"
fname = data_dir / dataset / "ometiff-pyramids/stitched/expressions/reg1_stitched_expressions.ome.tif"
output_dir = Path.home() / f"dryad_dcl_outputs/cytokit_spleen/{dataset}"
output_dir.mkdir(exist_ok=True, parents=True)

# Multiplex img and metadata
img = tff.imread(fname)
metadata = from_tiff(fname)

# Clip images back to original size
xpad = (img.shape[1] - 9072) // 2
ypad = (img.shape[2] - 9408) // 2
img = img[..., xpad:-xpad, ypad:-ypad]
assert img.shape[1] == 9072, img.shape[2] == 9408

# Channel names capitalized
data_chnames = [ch.name.upper() for ch in metadata.images[0].pixels.channels]

# Sort channels alphabetically
data_ch_to_idx = {name: idx for idx, name in enumerate(data_chnames)}
data_ch_to_idx = dict(sorted(data_ch_to_idx.items()))

ch_to_idx = {k: v for v, k in enumerate(data_ch_to_idx)}

# Prepare segmentation input
segmentation_input = np.stack(
    [
        img[ch_to_idx["DAPI-02"]],
        img[ch_to_idx["E-CAD"]],
    ],
    axis=-1,
)[np.newaxis, ...]

# Segment
mpp = 0.377
app = Mesmer()
mask = app.predict(segmentation_input, image_mpp=mpp)
y = mask.transpose(3, 0, 1, 2).astype(np.int32)

# Reformat image for dcl
chmax = img.max(axis=(1, 2), keepdims=True)
chmin = img.min(axis=(1, 2), keepdims=True)
X = ((img - chmin) / (chmax - chmin) * 255).astype(np.uint8)
X = X[:, np.newaxis, :, :]

# Get cell types
ct_config_file = Path.home() / "Downloads/core_tree.yaml"
with open(ct_config_file) as fh:
    ctdata = yaml.load(fh, yaml.Loader)
ctlist = ravel_dict(ctdata)
cell_types = make_empty_cell_types(ctlist)

# Channel names
channels = list(data_ch_to_idx)


# Original tiling scheme: 7x9 grid - 64 tiles total
xstep = X.shape[2] // 9
ystep = X.shape[3] // 7
for i in range(0, X.shape[2], xstep):
    for j in range(0, X.shape[3], ystep):
        print(f"{i}:{i+xstep}, {j}:{j+ystep}")
        dcl_zip(
            X[..., i:i+xstep, j:j+ystep],
            y[..., i:i+xstep, j:j+ystep],
            cell_types,
            channels,
            fname=output_dir / f"reg001_{i}-{i+xstep}_{j}-{j+ystep}.zip"
        )
