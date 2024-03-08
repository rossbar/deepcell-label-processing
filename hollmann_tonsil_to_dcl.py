import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import pandas as pd
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

data_dir = Path("/data/hollmann_raw")
dataset = "hollmann_tonsil"
output_dir = Path.home() / f"dryad_dcl_outputs/{dataset}"
output_dir.mkdir(exist_ok=True)

# Multiplex img in memmap mode
im = tff.memmap(data_dir / "CODEX_Tnsl.tif")
im = im.reshape((im.shape[0] * im.shape[1], im.shape[2], im.shape[3]))

# Load channel data
df_path = data_dir / "annotation_panel_table.xlsx"
fh = pd.ExcelFile(df_path)
# Raw channel data in dataframe
df = pd.read_excel(fh, "Codex", skiprows=41, usecols=(1, 2))
# Extract to mapping
chmapping = dict(zip(df["Unnamed: 1"], df["Unnamed: 2"]))
# Filter channels
other_dapis = {f"DAPI{i}" for i in range(2, 9)}
chname_to_idx = {
    cn: idx for idx, cn in chmapping.items()
    if (cn.upper() != "BLANK") and (cn.upper() != "EMPTY") and (cn not in other_dapis)
}
# Sort channels alphabetically
data_ch_to_idx = dict(sorted(chname_to_idx.items()))

# Only load necessary channels into memory
indices = list(data_ch_to_idx.values())
img = im[indices, ...]
ch_to_idx = {k: v for v, k in enumerate(data_ch_to_idx)}

# Prepare segmentation input
segmentation_input = np.stack(
    [
        img[ch_to_idx["DAPI1"]],
        img[ch_to_idx["CD45RO"]],
    ],
    axis=-1,
)[np.newaxis, ...]

# Segment
mpp = 0.3776
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

for (xmin, xmax) in itertools.pairwise(np.linspace(0, X.shape[2], 4).astype(int)):
    for (ymin, ymax) in itertools.pairwise(np.linspace(0, X.shape[3], 6).astype(int)):
        print(f"{xmin}:{xmax}, {ymin}:{ymax}")
        dcl_zip(
            X[..., xmin:xmax, ymin:ymax],
            y[..., xmin:xmax, ymin:ymax],
            cell_types,
            channels,
            fname=output_dir / f"tile_{xmin}-{xmax}_{ymin}-{ymax}.zip"
        )
