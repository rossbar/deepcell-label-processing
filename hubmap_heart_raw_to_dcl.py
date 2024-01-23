import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import tifffile as tff
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

data_dir = Path("/data/heart/")
dataset = "HBM296.PBWW.776-650dc93e8463bfdb103287be56dab1c0"
output_dir = Path.home() / f"dryad_dcl_outputs/heart/{dataset}"
output_dir.mkdir(exist_ok=True, parents=True)

# Multiplex img in memmap mode
im = tff.memmap(data_dir / f"{dataset}/drv_Heart01_septum/processed_2021-12-14/stitched/reg001/Experiment_211110_144849_reg001.tif")
im = im.reshape((im.shape[0] * im.shape[1], im.shape[2], im.shape[3]))
print(im.shape)

# Channel names capitalized
with open(data_dir / dataset / "drv_Heart01_septum/channelNames.txt") as fh:
    data_chnames = [l.rstrip().upper() for l in fh.readlines()]

assert len(data_chnames) == im.shape[0]

# Handle duplicate names in channels
names, cts = np.unique(data_chnames, return_counts=True)
dupl_names = names[cts != 1]
for name in dupl_names:
    counter = 1
    for i, item in enumerate(data_chnames):
        if item == name:
            data_chnames[i] = f"{item}{counter}"
            counter += 1

# Filter channels
empties = [ch for ch in data_chnames if ch.startswith("EMPTY")]
blanks = [ch for ch in data_chnames if ch.startswith("BLANK")]
handes = [ch for ch in data_chnames if ch.startswith("HANDE")]
# These are the nuclear channels at the start of each cycle, c.f. Hoechst in Hickey data
cyclestarters = [ch for ch in data_chnames if ch.startswith("CH1CY")]
# NOTE: The segmentation notes say to use CH1CY2 as the nuclear channel
cyclestarters.remove("CH1CY2")

# Drop unused channels and sort alphabetically
data_ch_to_idx = {name: idx for idx, name in enumerate(data_chnames)}
data_ch_to_idx = dict(sorted(data_ch_to_idx.items()))
for k in empties + blanks + cyclestarters + handes:
    data_ch_to_idx.pop(k)

ch_to_idx = {k: v for v, k in enumerate(data_ch_to_idx)}

# Only load necessary channels into memory
indices = list(data_ch_to_idx.values())
img = im[indices, ...]

# Prepare segmentation input
segmentation_input = np.stack(
    [
        img[ch_to_idx["CH1CY2"]],
        img[ch_to_idx["CD45"]],
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

# Original tiling scheme: 7x9 grid - 63 tiles total
xstep = X.shape[2] // 7
ystep = X.shape[3] // 9
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
