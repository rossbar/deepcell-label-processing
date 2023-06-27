import os
import re
from pathlib import Path
from tqdm import tqdm

from hubmap_raw_to_dcl import hubmap_hickey_to_dcl

data_dir = Path("/data/small_intestine/HBM394.VSKR.883-5afac98d9f449f2169b505179dd96f0e")
img_dir = Path("processed/bestFocus")
outpath = Path("/home/administrator/hubmap_dcl_outputs/small_intestine/HBM394.VSKR.883")

def hickey_raw_dir_to_dcl(data_dir, img_dir, outpath):
    img_fnames = [
        fname for fname in os.listdir(data_dir / img_dir)
        if re.match("reg00._X.._Y...*\.tif", fname)
    ]
    assert len(img_fnames) == 63  # at least for Hickey
    
    # Create the deepcell-label projects and store the number of cells
    # per image
    num_cells = {
        img: hubmap_hickey_to_dcl(data_dir, img_dir / img, outpath)
        for img in tqdm(img_fnames)
    }
    
    # Save number of cells per image, sorted from most to least
    lines = [
        f"{k} {num_cells[k]}\n" for k in
        sorted(num_cells, key=num_cells.get, reverse=True)
    ]
    with open(outpath / "cells_per_img.txt", "w") as fh:
        fh.writelines(lines)

if __name__ == "__main__":
    hickey_raw_dir_to_dcl(data_dir, img_dir, outpath)
