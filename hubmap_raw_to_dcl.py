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
def hubmap_hickey_cli(data_dir, image_path, output_path):
    # Inpute validation
    if data_dir is None:
        raise ValueError("Specify path to raw hubmap dataset with --data-dir")
    if image_path is None:
        raise ValueError(
            "Specify relative path to an image, e.g. processed/reg001_X01_Y01.tif"
        )
    hubmap_hickey_to_dcl(data_dir, image_path, output_path)


def hubmap_hickey_to_dcl(data_dir, image_path, output_path=None):
    """Create a deepcell label job from raw HubMAP CODEX data.

    Parameters
    ----------
    data_dir : pathlib.Path
        Path to top-level directory for a hubmap dataset which
        must contain:
        1. A ``processed/`` folder containing the .tifs and
        2. A ``channelnames.txt`` which maps the image indices to
           the corresponding protein marker.

        For example: ``/data/large_intestine/HBM???-<hash>``

    image_path : pathlib.Path
        Relative path from `data_dir` to the .tif file which will
        be used to create the DCL project.

        For exmaple: ``processed/bestFocus/reg00?_X0?_Y0?.tif``

    output_path : pathlib.Path, optional
        Location to store the ``.zip`` file containing the DCL project.
        If not specified, the project will be saved in the same directory
        as the input image used to create it.
        This location must exist; if it does not, an exception is raised
        prompting you to create it.

        For example: ``$HOME/hubmap_to_dcl/``

    """
    # Path to data, e.g. /data/large_intestine/HBM...
    data_dir = Path(data_dir)
    # Rel path from data_dir to img, e.g. /processed/bestFocus/reg001...
    image_path = Path(image_path)
    image_fname = image_path.name
    out_fname = os.path.splitext(image_fname)[0] + "_mpm_project.zip"
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
        data_chnames = [l.rstrip().upper() for l in fh.readlines()]

    # Create some mappings from channel names to img index
    data_ch_to_idx = {
        name: idx for idx, name in enumerate(data_chnames)
    }

    # Channels to drop from consideration from the raw data
    empties = [ch for ch in data_chnames if ch.startswith("EMPTY")]
    blanks = [ch for ch in data_chnames if ch.startswith("BLANK")]
    hoechsts = [ch for ch in data_chnames if ch.startswith("HOECHST")]
    hoechsts_to_drop = hoechsts[1:]
    # Drop em
    for k in empties + blanks + hoechsts_to_drop:
        data_ch_to_idx.pop(k)

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
            img[data_ch_to_idx["HOECHST1"]],
            img[data_ch_to_idx["CYTOKERATIN"]] + img[data_ch_to_idx["VIMENTIN"]],
        ],
        axis=-1
    )[np.newaxis, ...]

    # Note that the pixel size appears to be 0.377 for all the Hickey data
    mask = app.predict(segmentation_input, image_mpp=0.377)

    # Mask should be int32 and have shape (1, 1, y, x) to match
    # the multiplexed image
    y = mask.transpose(3, 0, 1, 2).astype(np.int32)
    number_of_cells = y.max()
    print(f"Number of cells in {image_fname}: {number_of_cells}")

    # Create the multplexed image from the model/channel mapping
    multiplexed_img = img[list(data_ch_to_idx.values())]
    channels = list(data_ch_to_idx)

    # Normalize to range 0-255 and convert to uint8
    chmax = multiplexed_img.max(axis=(1, 2), keepdims=True)
    chmin = multiplexed_img.min(axis=(1, 2), keepdims=True)
    X = ((multiplexed_img - chmin) / (chmax - chmin) * 255).astype(np.uint8)
    # Add dummy dimension to get X into CTYX format
    X = X[:, np.newaxis, :, :]

    # Get cell types from model config
    import utils
    cell_types = utils.make_empty_marker_positivity(list(data_ch_to_idx))
    # TODO: Add other categories for annotation (e.g. UNSURE)


    from raw_to_dcl import dcl_zip
    dcl_zip(X, y, cell_types, channels, fname=out_path)

    return number_of_cells

if __name__ == "__main__":
    hubmap_hickey_cli()
