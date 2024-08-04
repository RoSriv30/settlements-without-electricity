import pyprojroot
import sys
import os
root = pyprojroot.here()
sys.path.append(str(root))
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
import os
from typing import List
from dataclasses import dataclass
from pathlib import Path

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary
)

from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.preprocessing.subtile_esd_hw02 import Subtile
from src.visualization.restitch_plot import (
    restitch_eval,
    restitch_and_plot
)
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import tifffile
@dataclass
class EvalConfig:
    processed_dir: str | os.PathLike = root / 'data/processed/4x4'
    raw_dir: str | os.PathLike = root / 'data/raw/Train'
    results_dir: str | os.PathLike = root / 'data/prediction' / "UNet++"
    selected_bands: None = None
    tile_size_gt: int = 4
    batch_size: int = 8
    seed: int = 12378921
    num_workers: int = 11
    model_path: str | os.PathLike = root / "models" / "UNet++" / "last.ckpt"



def main(options):
    """
    Prepares datamodule and loads model, then runs the evaluation loop

    Inputs:
        options: EvalConfig
            options for the experiment
    """
    # Complete this function using the code snippets below. Do not forget to remove this line.
    # Load datamodule
    datamodule = ESDDataModule(
        processed_dir=options.processed_dir,
        raw_dir=options.raw_dir,
        batch_size=options.batch_size,
        seed=options.seed
    )
    datamodule.setup('fit', augment = False)

    # load model from checkpoint at options.model_path
    model = ESDSegmentation.load_from_checkpoint(options.model_path)

    # set the model to evaluation mode (model.eval())
    # this is important because if you don't do this, some layers
    # will not evaluate properly
    model.eval()

    # instantiate pytorch lightning trainer
    trainer = pl.Trainer()

    # run the validation loop with trainer.validate
    results = trainer.validate(model, dataloaders=datamodule.val_dataloader())





    # run restitch_and_plot
        
    restitch_and_plot(options, datamodule, model, "Tile9", "sentinel2", [3, 2, 1], options.results_dir)




    # # for every subtile in options.processed_dir/Val/subtiles
    # # run restitch_eval on that tile followed by picking the best scoring class
    # # save the file as a tiff using tifffile
    # # save the file as a png using matplotlib
    # tiles = os.path.join(options.processed_dir, "Val", "subtiles")


    val_dir = os.path.join(options.processed_dir, "Val", "subtiles")
    unique_parent_tiles = set()
    for filename in os.listdir(val_dir):
        parent_tile = filename.split("_")[0] 
        unique_parent_tiles.add(parent_tile)

    # Convert the set to a list if needed
    unique_parent_tiles_list = list(unique_parent_tiles)
    tiles = options.processed_dir
    for parent_tile_id in unique_parent_tiles_list:
        _, gt, y_pred = restitch_eval(dir=tiles, satellite_type="sentinel2", tile_id=parent_tile_id, range_x=(0,4), range_y=(0,4), datamodule=datamodule, model=model)
        
        # freebie: plots the predicted image as a jpeg with the correct colors
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Settlements", np.array(['#ff0000', '#0000ff', '#ffff00', '#b266ff']), N=4)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(y_pred, vmin=-0.5, vmax=3.5,cmap=cmap)
        plt.savefig(options.results_dir / f"{parent_tile_id}_prediction.png")








    

if __name__ == '__main__':
    config = EvalConfig()
    parser = ArgumentParser()

    parser.add_argument("--model_path", type=str, help="Model path.", default=config.model_path)
    parser.add_argument("--raw_dir", type=str, default=config.raw_dir, help='Path to raw directory')
    parser.add_argument("-p", "--processed_dir", type=str, default=config.processed_dir,
                        help=".")
    parser.add_argument("--results_dir", type=str, default=config.results_dir, help="Results dir")
    main(EvalConfig(**parser.parse_args().__dict__))
