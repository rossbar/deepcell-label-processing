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

data_dir = Path("/data/dryad_annotated_data/")
dataset = "tonsil"
output_dir = Path.home() / f"dryad_dcl_outputs/{dataset}"
output_dir.mkdir(exist_ok=True)

# Multiplex img in memmap mode
im = tff.memmap(data_dir / f"{dataset}/reg001_montage.tif")
im = im.reshape((im.shape[0] * im.shape[1], im.shape[2], im.shape[3]))

# Channel names capitalized
with open(data_dir / f"{dataset}/channels.txt") as fh:
    data_chnames = [l.rstrip().upper() for l in fh.readlines()]

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
hoechsts = [ch for ch in data_chnames if ch.startswith("HOECHST") or ch.startswith("HOESCHT")]
hoechsts_to_drop = hoechsts[1:]

# Drop unused channels and sort alphabetically
data_ch_to_idx = {name: idx for idx, name in enumerate(data_chnames)}
data_ch_to_idx = dict(sorted(data_ch_to_idx.items()))
for k in empties + blanks + hoechsts_to_drop + handes:
    data_ch_to_idx.pop(k)

ch_to_idx = {k: v for v, k in enumerate(data_ch_to_idx)}

# Only load necessary channels into memory
indices = list(data_ch_to_idx.values())
img = im[indices, ...]

# Prepare segmentation input
segmentation_input = np.stack(
    [
        img[ch_to_idx["HOECHST1"]],
        img[ch_to_idx["CYTOKERATIN"]],
    ],
    axis=-1,
)[np.newaxis, ...]

# Segment
mpp = 0.377 * 2
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

#dcl_zip(X[..., :1008, :1344], y[..., :1008, :1344], cell_types, channels, fname="/home/administrator/foo.zip")

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
