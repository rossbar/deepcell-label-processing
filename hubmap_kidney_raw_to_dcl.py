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

data_dir = Path("/data/kidney/")
dataset = "HBM482.MVFZ.845-6be2f666384a91ddf600067a65abbc6d"
output_dir = Path.home() / f"dryad_dcl_outputs/kidney/{dataset}"
output_dir.mkdir(exist_ok=True, parents=True)

# Multiplex img in memmap mode
# NOTE: Had to pull out a single tile to run segmentation on a normal machine
im = tff.imread(data_dir / f"{dataset}/HuBMAP_OME/region_006/central_tile.tif")

# Channel names capitalized
data_chnames = np.loadtxt(
    data_dir / dataset / "channel_list.txt",
    dtype=str,
    delimiter=",",
    usecols=-1,
    converters=lambda s: s.strip().upper(),
)

assert len(data_chnames) == im.shape[0]

# Drop unused channels and sort alphabetically
data_ch_to_idx = {name: idx for idx, name in enumerate(data_chnames)}
ch_to_idx = {k: v for v, k in enumerate(data_ch_to_idx)}

img = im

# Prepare segmentation input
# TODO: Is PCAD a good membrane marker? Chose based on visual distribution + chatgpt
# NOTE: Interpreting PCAD as "Pan-cadherin"
segmentation_input = np.stack(
    [
        img[ch_to_idx["DAPI_S009"]],
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

# Original tiling scheme: 7x9 grid - 63 tiles total
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
