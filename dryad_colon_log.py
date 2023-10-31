# IPython log file

import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import tifffile as tff
from deepcell.applications import Mesmer
from pathlib import Path
chnames = np.loadtxt("channelNames.txt", dtype=str, delimiter=" ")
get_ipython().run_line_magic('ls', '')
with open("channelNames.txt") as fh:
    chnames = np.array([ch.rstrip() for ch in fh.readlines()])
    
chnames
len(s.startswith("empty") for s in chnames)
len([s.startswith("empty") for s in chnames])
s
chnames
sum([s.startswith("empty") for s in chnames])
import tifffile as tff
im = tff.memmap("reg001_X01_Y01_Z01.tif")
im.shape
im = im.reshape((24*4, 9072, 9408))
im.shape
with open("channelNames.txt") as fh:
    data_chnames = [l.rstrip().upper() for l in fh.readlines()]
    
data_chnames
empties = [ch for ch in data_chnames if ch.startswith("EMPTY")]
blanks = [ch for ch in data_chnames if ch.startswith("BLANK")]
hoechsts = [ch for ch in data_chnames if ch.startswith("HOECHST")]
hoechsts_to_drop = hoechsts[1:]
handes = [ch for ch in data_chnames if ch.startswith("HANDE")]
data_ch_to_idx = {name: idx for idx, name in enumerate(data_chnames)}
for k in empties + blanks + hoechsts_to_drop + handes:
    data_ch_to_idx.pop(k)
    
data_ch_to_idx
hoechsts = [ch for ch in data_chnames if ch.startswith("HOECHST") or ch.startswith("HOESCHT")]
hoechsts
hoechsts_to_drop = hoechsts[1:]
for k in empties + blanks + hoechsts_to_drop + handes:
    data_ch_to_idx.pop(k)
    
data_ch_to_idx = {name: idx for idx, name in enumerate(data_chnames)}
for k in empties + blanks + hoechsts_to_drop + handes:
    data_ch_to_idx.pop(k)
    
len(data_ch_to_idx)
data_ch_to_idx
indices = list(data_ch_to_idx.values())
indices
img = im[indices, ...]
img.shape
img.nbytes / 1e9
img.dtype
im.shape
np.prod(im.shape)
np.prod(im.shape) * 2 / 1e9
sorted(data_ch_to_idx)
get_ipython().run_line_magic('ls', '')
vim segmentation.json
get_ipython().system('cat segmentation.json')
chnames
chnames[60:64]
chnames[56:64]
data_ch_to_idx
ch_to_idx = {k: v for v, k in enumerate(data_ch_to_idx)}
ch_to_idx
plt.imshow(img[47, ...])
pgp = np.clip(img[47, ...], 0, np.quantile(img[47, ...].ravel(), 0.99))
plt.imshow(pgp)
get_ipython().system('cat')
get_ipython().run_line_magic('ls', '')
im = tff.memmap("reg001_X01_Y01_Z01.tif")
im.shape
data_ch_to_idx
chnames
chnames.reshape((24, 4))
chnames.reshape((24, 4))[15, 3]
ch_to_idx
cyto = np.clip(img[41, ...], 0, np.quantile(img[41, ...].ravel(), 0.99))
plt.imshow(cyto)
vim = np.clip(img[16, ...], 0, np.quantile(img[16, ...].ravel(), 0.99))
plt.imshow(vim)
segmentation_input = np.stack(
    [
        img[ch_to_idx["HOECHST1"]],
        img[ch_to_idx["CYTOKERATIN"]],
    ],
    axis=-1,
)[np.newaxis, ...]
segmentation_input.shape
app = Mesmer()
mask = app.predict(segmentation_input, image_mpp=0.377)
mask.shape
plt.imshow(mask.squeeze())
mask.max()
plt.imshow(mask.squeeze())
y = mask.transpose(3, 0, 1, 2).astype(np.int32)
y.shape
import utils
import sys
sys.path.append("/home/administrator/repos/deepcell-label-processing/")
import utils
get_ipython().run_line_magic('pinfo', 'utils.make_empty_cell_types')
cell_types = utils.make_empty_marker_positivity(list(ch_to_idx))
cell_types
indices = list(sorted(data_ch_to_idx.values()))
indices
indices = list(sorted(data_ch_to_idx).values())
data_ch_to_idx
sorted(data_ch_to_idx)
data_ch_to_idx
sorted(data_ch_to_idx.items())
dict(sorted(data_ch_to_idx.items()))
indices = list(dict(sorted(data_ch_to_idx.items())))
indices
indices = list(dict(sorted(data_ch_to_idx.items())).values())
indices
img = im[indices, ...]
im = tff.memmap("reg001_X01_Y01_Z01.tif")
img = im[indices, ...]
im.shape
im = tff.memmap("reg001_X01_Y01_Z01.tif")
im = im.reshape((24*4, 9072, 9408))
img = im[indices, ...]
img.shape
data_ch_to_idx = {name: idx for idx, name in enumerate(data_chnames)}
data_ch_to_idx = dict(sorted(data_ch_to_idx.items()))
data_ch_to_idx
for k in empties + blanks + hoechsts_to_drop + handes:
    data_ch_to_idx.pop(k)
    
data_ch_to_idx
ch_to_idx = {k: v for v, k in enumerate(data_ch_to_idx)}
ch_to_idx
plt.imshow(img[ch_to_idx["HOECHST1"]])
plt.imshow(mask.squeeze())
cell_types = utils.make_empty_marker_positivity(list(ch_to_idx))
cell_types
img.shape
chmax = multiplexed_img.max(axis=(1, 2), keepdims=True)
img.dtype
chmax = img.max(axis=(1, 2), keepdims=True)
chmax
chq = np.quantile(img, 0.99, axis=(1, 2), keepdims=True)
chq
X = ((img - chmin) / (chmax - chmin) * 255).astype(np.uint8)
chmin = img.min(axis=(1, 2), keepdims=True)
X = ((img - chmin) / (chmax - chmin) * 255).astype(np.uint8)
X.shape
9072 / 7
9408 / 9
9408 / 7
9072 / 9
from raw_to_dcl import dcl_zip
y.shape
channels = list(data_ch_to_idx)
channels
9408 / 7
9072 / 9
img.shape
dcl_zip(X[:, :1008, :1344], y[:, :1008, :1344], cell_types, channels, fname="/home/administrator/foo.zip")
X.shape
X = X[:, np.newaxis, :, :]
X.shape
y.shape
dcl_zip(X[..., :1008, :1344], y[..., :1008, :1344], cell_types, channels, fname="/home/administrator/foo.zip")
get_ipython().run_line_magic('pinfo', 'utils.make_empty_cell_types')
cell_types = utils.make_empty_cell_types(list(data_ch_to_idx))
def make_empty_cell_types(ctlist):
    """ Return cellTypes.json with master cell types list but no labels """
    cell_types_json = []
    for i in range(1, len(ctlist) + 1):
        cell_types_json.append({'id': i, 'cells': [], 'color': constants.COLOR_MAP[i - 1],
                               'name': constants.MASTER_TYPES[i - 1], 'feature': 0})
    return cell_types_json
    
import yaml
def ravel_dict(d):
    leafs = []
    for k, v in d.items():
        if v == {}:
            leafs.append(k)
        else:
            leafs.extend(ravel_dict(v))
    return leafs
    
ct_config_file = Path.home() / "Downloads/core_tree.yaml")
ct_config_file = Path.home() / "Downloads/core_tree.yaml"
with open(ct_config_file) as fh:
    ctdata = yaml.load(fh, yaml.Loader)
    
ctdata
ctlist = ravel_dict(d)
ctlist = ravel_dict(ctdata)
ctlist
cell_types = utils.make_empty_cell_types(ctlist)
cell_types = make_empty_cell_types(ctlist)
def make_empty_cell_types(ctlist):
    """ Return cellTypes.json with master cell types list but no labels """
    cell_types_json = []
    for i, ct in enumerate(ctlist):
        cell_types_json.append({'id': i+1, 'cells': [], 'color': constants.COLOR_MAP[i - 1],
                               'name': ct, 'feature': 0})
    return cell_types_json
    
import constants
constants.COLOR_MAP
import itertools
def make_empty_cell_types(ctlist):
    """ Return cellTypes.json with master cell types list but no labels """
    cell_types_json = []
    for i, ct in enumerate(ctlist):
        cell_types_json.append({'id': i+1, 'cells': [], 'color': itertools.repeat(constants.COLOR_MAP)[i],
                               'name': ct, 'feature': 0})
    return cell_types_json
    
cell_types = make_empty_cell_types(ctlist)
def make_empty_cell_types(ctlist):
    """ Return cellTypes.json with master cell types list but no labels """
    cell_types_json = []
    for i, ct in enumerate(ctlist):
        cell_types_json.append({'id': i+1, 'cells': [], 'color': next(itertools.repeat(constants.COLOR_MAP)),
                               'name': ct, 'feature': 0})
    return cell_types_json
    
cell_types = make_empty_cell_types(ctlist)
cell_types
def make_empty_cell_types(ctlist):
    """ Return cellTypes.json with master cell types list but no labels """
    cell_types_json = []
    for i, ct in enumerate(ctlist):
        cell_types_json.append({'id': i+1, 'cells': [], 'color': next(itertools.repeat(*constants.COLOR_MAP)),
                               'name': ct, 'feature': 0})
    return cell_types_json
    
cell_types
cell_types = make_empty_cell_types(ctlist)
get_ipython().run_line_magic('pinfo', 'itertools.cycle')
def make_empty_cell_types(ctlist):
    """ Return cellTypes.json with master cell types list but no labels """
    cell_types_json = []
    for i, ct in enumerate(ctlist):
        cell_types_json.append({'id': i+1, 'cells': [], 'color': next(itertools.cycle(constants.COLOR_MAP)),
                               'name': ct, 'feature': 0})
    return cell_types_json
    
cell_types = make_empty_cell_types(ctlist)
cell_types
dcl_zip(X[..., :1008, :1344], y[..., :1008, :1344], cell_types, channels, fname="/home/administrator/foo.zip")
constants.COLOR_MAP
def make_empty_cell_types(ctlist):
    """ Return cellTypes.json with master cell types list but no labels """
    cell_types_json = []
    for i, ct in enumerate(ctlist):
        cell_types_json.append({'id': i+1, 'cells': [], 'color': next(itertools.cycle(*constants.COLOR_MAP)),
                               'name': ct, 'feature': 0})
    return cell_types_json
    
cell_types = make_empty_cell_types(ctlist)
def make_empty_cell_types(ctlist):
    """ Return cellTypes.json with master cell types list but no labels """
    cell_types_json = []
    colors = itertools.cycle(constants.COLOR_MAP)
    for i, ct in enumerate(ctlist):
        cell_types_json.append({'id': i+1, 'cells': [], 'color': next(colors),
                               'name': ct, 'feature': 0})
    return cell_types_json
    
cell_types = make_empty_cell_types(ctlist)
cell_types
dcl_zip(X[..., :1008, :1344], y[..., :1008, :1344], cell_types, channels, fname="/home/administrator/foo.zip")
ctlist
ctdata
plt.imshow(pgp)
dcl_zip(X[..., :1008, :1344], y[..., :1008, :1344], cell_types, channels, fname="/home/administrator/foo.zip")
X.shape
for x in range(0, X.shape[2], 1008):
    for y in range(0, X.shape[3], 1344):
        print(x, y)
        
for x in range(0, X.shape[2], 1008):
    for y in range(0, X.shape[3], 1344):
        print(f"{x}:{x+1008}, {y}:{y+1344}")
        
X[..., 8064:9072, ...]
X[..., 8064:9072, :]
dcl_zip(X[..., :1008, :1344], y[..., :1008, :1344], cell_types, channels, fname="/home/administrator/foo.zip")
y = mask.transpose(3, 0, 1, 2).astype(np.int32)
dcl_zip(X[..., :1008, :1344], y[..., :1008, :1344], cell_types, channels, fname="/home/administrator/foo.zip")
for i in range(0, X.shape[2], 1008):
    for j in range(0, X.shape[3], 1344):
        print(f"{x}:{x+1008}, {y}:{y+1344}")
        dcl_zip(X[..., i:i+1008, j:j+1344], y[..., i:i+1008, j:j+1344], cell_types, channels, fname=f"/home/administrator/dryad_dcl_outputs/B004_reg001_{i}-{i+1008}_{j}-{j+1344}")
        
for i in range(0, X.shape[2], 1008):
    for j in range(0, X.shape[3], 1344):
        print(f"{i}:{i+1008}, {j}:{j+1344}")
        dcl_zip(X[..., i:i+1008, j:j+1344], y[..., i:i+1008, j:j+1344], cell_types, channels, fname=f"/home/administrator/dryad_dcl_outputs/B004_reg001_{i}-{i+1008}_{j}-{j+1344}")
        
get_ipython().run_line_magic('clear', '')
for i in range(0, X.shape[2], 1008):
    for j in range(0, X.shape[3], 1344):
        print(f"{i}:{i+1008}, {j}:{j+1344}")
        dcl_zip(X[..., i:i+1008, j:j+1344], y[..., i:i+1008, j:j+1344], cell_types, channels, fname=f"/home/administrator/dryad_dcl_outputs/B004_reg001_{i}-{i+1008}_{j}-{j+1344}.zip")
        
get_ipython().run_line_magic('logstart', '/home/administrator/repos/deepcell-label-processing/dryad_colon_log.py')
exit()
