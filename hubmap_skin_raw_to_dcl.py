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

data_dir = Path("/data/skin/")
dataset = "HBM975.FVCG.922-fa238ec2a83302fb4af92442c3683a23"
output_dir = Path.home() / f"dryad_dcl_outputs/skin/{dataset}"
output_dir.mkdir(exist_ok=True, parents=True)

# Multiplex img in memmap mode
fpath = "data_directory/HuBMAP_OME/region_011/S20030105_region_011.ome.tif"
img = tff.imread(data_dir / dataset / fpath)
metadata = from_tiff(data_dir / dataset / fpath)

# Channel names capitalized
data_chnames = [ch.name.upper() for ch in metadata.images[0].pixels.channels]

assert len(data_chnames) == img.shape[0]

data_ch_to_idx = {name: idx for idx, name in enumerate(data_chnames)}
ch_to_idx = {k: v for v, k in enumerate(data_ch_to_idx)}

# Prepare segmentation input
# TODO: double-check these channel choices - chatgpt inspired
segmentation_input = np.stack(
    [
        img[ch_to_idx["DAPI_INIT"]],
        img[ch_to_idx["PCAD"]],
    ],
    axis=-1,
)[np.newaxis, ...]

# Segment
mpp = 0.325
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

# NOTE: not sure what the original tiling is. Arbitrarily select a
# roughly 2k x 2k tiling scheme
xsteps = np.linspace(0, X.shape[2], 5).astype(int)  # ~1877 pixels
ysteps = np.linspace(0, X.shape[3], 10).astype(int)  # ~2475 pixels
for xstart, xstop in itertools.pairwise(xsteps):
    for ystart, ystop in itertools.pairwise(ysteps):
        print(f"{xstart}:{xstop}, {ystart}:{ystop}")
        dcl_zip(
            X[..., xstart:xstop, ystart:ystop],
            y[..., xstart:xstop, ystart:ystop],
            cell_types,
            channels,
            fname=output_dir / f"reg001_{xstart}-{xstop}_{ystart}-{ystop}.zip"
        )
