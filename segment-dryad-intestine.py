import matplotlib.pyplot as plt
from tqdm import tqdm
plt.ion()
import numpy as np
import tifffile as tff
from deepcell.applications import Mesmer
from pathlib import Path
import yaml
import itertools
import os

data_dir = Path("/data/dryad_annotated_data/raw/intestine")
datasets = ["B008_SB"]
for dataset in tqdm(datasets):
    fnames = [f.name for f in (data_dir / dataset).iterdir() if f.name.endswith("Z01.tif")]
    for fname in fnames:
        # Multiplex img in memmap mode
        im = tff.memmap(data_dir / dataset / fname)
        im = im.reshape((im.shape[0] * im.shape[1], im.shape[2], im.shape[3]))
        
        # Channel names capitalized
        with open(data_dir / f"{dataset}/channelNames.txt") as fh:
            data_chnames = [l.rstrip().upper() for l in fh.readlines()]
        
        # Filter channels
        empties = [ch for ch in data_chnames if ch.startswith("EMPTY") or ch.startswith ("EMTPY")]
        blanks = [ch for ch in data_chnames if ch.startswith("BLANK")]
        handes = [ch for ch in data_chnames if ch.startswith("HANDE")]
        hoechsts = [ch for ch in data_chnames if ch.startswith("HOECHST") or ch.startswith("HOESCHT")]
        hoechsts_to_drop = hoechsts[1:]
        
        # Drop unused channels and sort alphabetically
        data_ch_to_idx = {name: idx for idx, name in enumerate(data_chnames)}
        data_ch_to_idx = dict(sorted(data_ch_to_idx.items()))
        for k in empties + blanks + hoechsts_to_drop + handes:
            try:
                data_ch_to_idx.pop(k)
            except KeyError:
                continue
        
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
        app = Mesmer()
        mask = app.predict(segmentation_input, image_mpp=0.377)
        y = mask.transpose(3, 0, 1, 2).astype(np.int32)
        
        # Channel names
        channels = list(data_ch_to_idx)
        
        reg = fname.split("_")[0]
        
        print("Writing tif...")
        with tff.TiffWriter(data_dir / dataset / f"{reg}_processed.tif", ome=True, bigtiff=True) as tif:
            metadata = {"axes": "CYX", "Channel": {"Name": channels}}
            tif.write(img, metadata=metadata)
        
        print("Writing mask...")
        with tff.TiffWriter(data_dir / dataset / f"{reg}_mask.tif", ome=True) as tif:
            tif.write(y, metadata={"axes": "CZYX"})
