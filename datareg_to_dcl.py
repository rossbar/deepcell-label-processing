import numpy as np
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

# Load image and mask
datapath = Path.home() / "data-registry/data/labels/static/2d/Tissue-GI/Jackson_BreastCancer_IMC/BaselTMA_SP43_29-3kx15-5ky_7000x8000_3_20170928_Liver_Liver_X1Y1_3_a0.npz"
fname = datapath.name
data = np.load(datapath)
img = data["X"].squeeze()
#w, h, _ = img.shape  # NOTE: Take upper-quadrant for large datasets
#img = img[:w // 2, :h // 2, ...]
mask = data["y"].squeeze()
#mask = mask[:w // 2, :h // 2]
y = mask[..., 0] if len(mask.shape) == 3 else mask
y = y[np.newaxis, np.newaxis, ...].astype(np.int32)

# Load channel names
with open(str(datapath) + ".dvc" , "r") as fh:
    metadata = yaml.load(fh, yaml.Loader)
channels = [rec["target"] for rec in metadata["meta"]["sample"]["channels"]]


# Reformat image for dcl
img = img.transpose((2, 0, 1))
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

dcl_zip(X, y, cell_types, channels, fname=str(datapath)[:-4] + "_dcl.zip")
