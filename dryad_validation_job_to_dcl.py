import pandas as pd
from pathlib import Path
import json
import zipfile
import tifffile as tff
from collections import defaultdict
import matplotlib.pyplot as plt

# Params
dryad_path = "/data/dryad_annotated_data/raw/BE_Tonsil_l3_dryad.csv"
unpacked_from_dcl = Path.home() / "dcl_caitlin_evals/RuAVfHPzY_Rh"
original_zip = unpacked_from_dcl / "RuAVfHPzY_Rh.zip"
donor = "B004"
tissue = "tonsil"
region = 1
ymin = 3024
ymax = 3528
xmin = 4032
xmax = 4704
if tissue in {"tonsil", "Barretts Esophagus"}:
    ymin *= 2
    ymax *= 2
    xmin *= 2
    xmax *= 2

# Celltypes from annotator
with open(unpacked_from_dcl / "cellTypes.json") as fh:
    ct_caitlin = json.load(fh)
    
mask = tff.imread(unpacked_from_dcl / "y.ome.tiff")
ctidx_to_type = {idx: row["name"] for row in ct_caitlin for idx in row["cells"]}

# Celltypes from dryad database
df = pd.read_csv(dryad_path)

# Filter dryad db
# All dryad databases include a tissue key, though it's named differently
tissue_key = "tissue" if tissue in {"SB", "CL"} else "sample_name"
df = df[df[tissue_key] == tissue]

# Only GI databases have donors and regions
if tissue in {"SB", "CL"}:
    donor_mask = df["donor"] == donor
    region_mask = df["region"] == region
    df = df[donor_mask & region_mask]
print(f"Number of cells with donor: {donor}, reg: {region}, tissue: {tissue} = {df.shape}")

# Filter by tile
tile_mask = (df["y"] > ymin) & (df["y"] < ymax) & (df["x"] > xmin) & (df["x"] < xmax)
df = df[tile_mask]
print(f"Number of cells in tile mask: {df.shape}")

xs, ys = df["x"], df["y"]
xs -= xmin
ys -= ymin
if tissue in {"tonsil", "Barretts Esophagus"}:
    xs /= 2
    ys /= 2

# Visually verify
plt.imshow(mask)
plt.plot(xs, ys, '.', color='r')
plt.show()

# Handle variations in GI vs. Colon data
celltype_key = "Cell Type" if tissue in {"SB", "CL"} else "cell_type"

# Crude centroid->celltype lookup, ignoring 0's
dryad_ct_to_idx = defaultdict(list)
for _, row in df.iterrows():
    x, y = int(row["x"]), int(row["y"])
    if mask[y, x] == 0: continue
    ct = row[celltype_key]
    dryad_ct_to_idx[ct].append(mask[y, x])
    
# Create the list-of-dict struct to be exported in DCL format
colors = [row["color"] for row in ct_caitlin]
output = []
for i, (k, v) in enumerate(dryad_ct_to_idx.items()):
    output.append(
        {"id": i, "cells": [int(j) for j in v], "color": colors[i], "name": k, "feature": 0}
    )

# Export to zipfile
with zipfile.ZipFile(original_zip, "r") as zipin:
    with zipfile.ZipFile(unpacked_from_dcl / "out.zip", "w") as zipout:
        for item in zipin.infolist():
            if item.filename == "cellTypes.json":
                zipout.writestr(item, json.dumps(output, indent=2))
            else:
                zipout.writestr(item, zipin.read(item.filename))
