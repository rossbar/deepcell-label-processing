import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import pandas as pd
from deepcell.applications import Mesmer
from pathlib import Path
import yaml
import itertools
import os
from imaris_ims_file_reader.ims import ims

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

data_dir = Path("/data2/omap-reference-datasets/omap-retina-12")
dataset = "HuRET_021_IBEX_OMAP.ims"
output_dir = Path.home() / f"dryad_dcl_outputs/omap12-retina"
output_dir.mkdir(exist_ok=True, parents=True)

# Multiplex img
ims_data = ims(data_dir / dataset)
# Check square pixels
assert ims_data.resolution[-2] == ims_data.resolution[-1]
mpp = ims_data.resolution[-1]  # Pixel size, in microns (hopefully)
# TODO: Arbitrarily select a center channel - better way to combine this?
img = ims_data[0, :, 7, :, :]  # res, ch, time, x, y

# Channel names capitalized
# TODO: Need to know channel mapping
#df = pd.read_excel(data_dir / dataset.replace(".ims", ".xlsx"))
#data_chnames = np.array(
#    [name.upper() for name in df["Target Name"] if not name.startswith("(")]
#)
#assert len(data_chnames) == img.shape[0]
data_chnames = [f"Channel {n}" for n in range(img.shape[0])]

# Map marker names to channel indices
data_ch_to_idx = {name: idx for idx, name in enumerate(data_chnames)}

# Prepare segmentation input
# TODO: Just guesses based on viz - CH. 4 looks nuclear and Ch0 looks mem
segmentation_input = np.stack(
    [
        img[4],  # nuclear
        img[0],  # pan-membrane
    ],
    axis=-1,
)[np.newaxis, ...]

# Segment
app = Mesmer()
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

# Original tiling scheme: 7x9 grid - 63 tiles total
for (xmin, xmax) in itertools.pairwise(np.linspace(0, X.shape[2], 2).astype(int)):
    for (ymin, ymax) in itertools.pairwise(np.linspace(0, X.shape[3], 2).astype(int)):
        print(f"{xmin}:{xmax}, {ymin}:{ymax}")
        dcl_zip(
            X[..., xmin:xmax, ymin:ymax],
            y[..., xmin:xmax, ymin:ymax],
            cell_types,
            channels,
            fname=output_dir / f"tile_{xmin}-{xmax}_{ymin}-{ymax}.zip"
        )
