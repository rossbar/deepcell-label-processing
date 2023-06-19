import os
import yaml
from pathlib import Path
import numpy as np
import tifffile as tff
import click

@click.command()
@click.option("--data-dir", default=None, help="Path to hubmap data")
@click.option(
    "--image-path", default=None, help="Path to multiplex image, relative to data_dir"
)
@click.option(
    "--output-path",
    default=None,
    help="Location to save the deepcell label project. Defaults to data-dir"
)
def hubmap_hickey_to_dcl(data_dir, image_path, output_path):
    # Inpute validation
    if data_dir is None:
        raise ValueError("Specify path to raw hubmap dataset with --data-dir")
    if image_path is None:
        raise ValueError(
            "Specify relative path to an image, e.g. processed/reg001_X01_Y01.tif"
        )
    # Path to data, e.g. /data/large_intestine/HBM...
    data_dir = Path(data_dir)
    # Rel path from data_dir to img, e.g. /processed/bestFocus/reg001...
    image_path = Path(image_path)
    image_fname = image_path.name
    out_fname = os.path.splitext(image_fname)[0] + "_project.zip"
    # Determine output path given data-dir and img-path
    if output_path is None:
        output_path = data_dir / "/".join(image_path.parts[:-1])
    else:
        output_path = Path(output_path)
    # Validate output path
    if not output_path.exists():
        raise ValueError(
            (
                f"\nOutput path:\n\t {output_path}\ndoes not exist. "
                "Create it and try again."
            )
        )
    # Prepare output filename
    out_path = output_path / out_fname

    # Get the channel names for the dataset
    with open(data_dir / "channelnames.txt", "r") as fh:
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
    # A few manual additions: Hoechst1 and cytokeratin for segmentation
    # sanity check
    model_ch_to_idx["PANCK"] = data_ch_to_idx_upper["CYTOKERATIN"]
    model_ch_to_idx["Hoechst1"] = data_ch_to_idx["Hoechst1"]

    # Load multiplexed image
    img = tff.imread(data_dir / image_path)

    # Reshape from (cyc, ch) to (cyc * ch)
    # NOTE: Assumes data has shape (cycle, ch, Y, X) where
    # cycle is usually 24ish and ch is always 4.
    # NOTE: if `z-plane` is included in the shape, this will bork,
    # so for now make sure to only use input images where the "best"
    # z-plane has already been selected
    img = img.reshape((img.shape[0] * img.shape[1], img.shape[2], img.shape[3]))

    # Generate mask with mesmer
    from deepcell.applications import Mesmer
    app = Mesmer()

    # Create input image for mesmer using Hoechst and Cytokeratin
    # as nuclear and whole cell channels, respectively
    segmentation_input = np.stack(
        [
            img[model_ch_to_idx["Hoechst1"]],
            img[model_ch_to_idx["PANCK"]] + img[model_ch_to_idx["VIMENTIN"]],
        ],
        axis=-1
    )[np.newaxis, ...]

    # Note that the pixel size appears to be 0.377 for all the Hickey data
    mask = app.predict(segmentation_input, image_mpp=0.377)

    # Mask should be int32 and have shape (1, 1, y, x) to match
    # the multiplexed image
    y = mask.transpose(3, 0, 1, 2).astype(np.int32)
    print(f"Number of cells: {y.max()}")

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
    # TODO: Modify make_empty_cell_types to read from model configuration
    # instead of `constants` module
    cell_types = utils.make_empty_cell_types()
    # TODO: Add other categories for annotation (e.g. UNSURE)


    from raw_to_dcl import dcl_zip
    dcl_zip(X, y, cell_types, channels, fname=out_path)

if __name__ == "__main__":
    hubmap_hickey_to_dcl()
