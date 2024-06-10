import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import tifffile as tff
from deepcell.applications import Mesmer
from pathlib import Path
import yaml
import itertools
import os
import pandas as pd

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

data_dir = Path("/data2/hubmap-imc/lymph_node")
dataset = "HBM342.QHMH.584-3f1657d4f4d883ae17e9fffe6723a5b5"
output_dir = Path.home() / f"dryad_dcl_outputs/hubmap_imc/lymph_node/{dataset}"
output_dir.mkdir(exist_ok=True, parents=True)

# Multiplex img in memmap mode
img = tff.memmap(data_dir / dataset / "ometiff/20191203_HuBMAP_LN/20191203_HuBMAP_LN_s0_p13_r3_a3_ac.ome.tiff")
print(img.shape)

# Channel names capitalized
df = pd.read_csv(data_dir / dataset / "ometiff/20191203_HuBMAP_LN/20191203_HuBMAP_LN_AcquisitionChannel_meta.csv")
# NOTE: Ch 6 chosen somewhat arbitrarily:
# - matches number of channels in image (after dropping channels named X, Y, and Z)
# - img filename has `r6_a6` in it
chnames = df[df["AcquisitionID"] == 6]["ChannelLabel"].to_numpy()[3:]
# Further process channelnames to drop ions
chnames = [n.split("(")[0].upper() for n in chnames]
assert len(chnames) == img.shape[0]
chmapping = {chname: idx for idx, chname in enumerate(chnames)}

# Drop isotope channels (keeping Iridium out of curiosity)
chmapping.pop("80ArAr".upper())
chmapping.pop("131Xe".upper())
chmapping.pop("134Xe".upper())
chmapping.pop("136Ba".upper())

# Sort alphabetically
chmapping = dict(sorted(chmapping.items()))
# Reorder image
img = img[list(chmapping.values()), ...]

ch_to_idx = {k: v for v, k in enumerate(chmapping)}

# Prepare segmentation input
segmentation_input = np.stack(
    [
        img[ch_to_idx["Histone_126".upper()]],
        img[ch_to_idx["HLA-ABC_1960".upper()]],
    ],
    axis=-1,
)[np.newaxis, ...]

# Segment
# NOTE: mpp inferred from the schema.xml by subtracting ROIEnd{X,Y}PosUm from
# ROIStart{X,Y}PosUm (dividing starts by 1000 due to bug) which gives ~1000um.
# Since the img shape is 1000x1000, seems like 1 um/pix
mpp = 1.0
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
channels = list(ch_to_idx)
dcl_zip(X, y, cell_types, channels, fname=output_dir / "whole_slide.zip")
