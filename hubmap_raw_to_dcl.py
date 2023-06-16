import yaml
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tff
plt.ion()

# Get the channel names for the dataset
with open("channelnames.txt", "r") as fh:
    data_chnames = [l.rstrip() for l in fh.readlines()]
    
# Create some mappings from channel names to img index
data_ch_to_idx = {name: idx for idx, name in enumerate(data_chnames)}
data_ch_to_idx_upper = {k.upper(): v for k, v in data_ch_to_idx.items()}
data_chset = set(data_ch_to_idx_upper)

# Channels to drop from consideration from the raw data
empties = [ch for ch in data_chnames if ch.upper().startswith("EMPTY")]
blanks = [ch for ch in data_chnames if ch.upper().startswith("BLANK")]
hoechsts = [ch for ch in data_chnames if ch.upper().startswith("HOECHST")]
hoechsts_to_drop = hoechsts[1:]
# Drop em
for k in empties + blanks + hoechsts_to_drop:
    data_ch_to_idx.pop(k)
    
with open("/home/administrator/repos/deepcelltypes-hubmap/model/config.yaml") as fh:
    model_config = yaml.load(fh, yaml.Loader)
    
# Get model channel names
model_chnames = model_config["channels"]
model_chset = set(model_chnames)

# Manually inspect the data and model channel sets
model_chset & data_chset
model_chset - data_chset
data_chset - model_chset

# Create a mapping from model_channel to raw data image index
# NOTE: the m_ch.upper() is necessary for CD49a -> CD49A
model_ch_to_idx = {
    m_ch: data_ch_to_idx_upper[m_ch.upper()]
    for m_ch in model_chnames
    if m_ch.upper() in data_ch_to_idx_upper
}
# A few manual additions: Hoechst1 for segmentation sanity check and
# cytokeratin
model_ch_to_idx["PANCK"] = data_ch_to_idx_upper["CYTOKERATIN"]
model_ch_to_idx["Hoechst1"] = data_ch_to_idx["Hoechst1"]

# Load a sample image
image_fname = "processed/input_tiles/reg001_X02_Y07.tif"
img = tff.imread(image_fname)

# Reshape from (cyc, ch) to (cyc * ch)
img = img.reshape((img.shape[0] * img.shape[1], img.shape[2], img.shape[3]))
# Visually inspect the first (best?) nuclear channel
plt.imshow(img[0, ...])

# Generate mask with mesmer
from deepcell.applications import Mesmer
app = Mesmer()

# Create input image for mesmer using Hoechst and Cytokeratin
# as nuclear and whole cell channels, respectively
segmentation_input = np.stack(
    [
        img[model_ch_to_idx["Hoechst1"]],
        img[model_ch_to_idx["PANCK"]],
    ],
    axis=-1
)[np.newaxis, ...]

# Note that the pixel size appears to be 0.377 for all the Hickey data
mask = app.predict(segmentation_input, image_mpp=0.377)

# Mask should be int32 and have shape (1, 1, y, x) to match
# the multiplexed image
y = mask.transpose(3, 0, 1, 2).astype(np.int32)

# Create the multplexed image from the model/channel mapping
multiplexed_img = img[list(model_ch_to_idx.values())]
channels = list(model_ch_to_idx)

# Normalize to range 0-255 and convert to uint8
chmax = multiplexed_img.max(axis=(1, 2), keepdims=True)
chmin = multiplexed_img.min(axis=(1, 2), keepdims=True)
X = ((multiplexed_img - chmin) / (chmax - chmin) * 255).astype(np.uint8)
# Add dummy dimension to get X into CTYX format
X = X[:, np.newaxis, :, :]

# Get cell types from model config
import utils
model_config["cell_types"]
# Ignore BACKGROUND category
# cell_types = model_config["cell_types"][1:]
cell_types = utils.make_empty_cell_types()
# TODO: Add other categories for annotation (e.g. UNSURE)

from raw_to_dcl import dcl_zip
dcl_zip(X, y, cell_types, channels)
