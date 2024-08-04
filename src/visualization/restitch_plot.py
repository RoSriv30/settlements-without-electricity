import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from src.preprocessing.subtile_esd_hw02 import TileMetadata, Subtile
import torch


def restitch_and_plot(options, datamodule, model, parent_tile_id, satellite_type="sentinel2", rgb_bands=[3,2,1], image_dir: None | str | os.PathLike = None,):
    """
    Plots the 1) rgb satellite image 2) ground truth 3) model prediction in one row.

    Args:
        options: EvalConfig
        datamodule: ESDDataModule
        model: ESDSegmentation
        parent_tile_id: str
        satellite_type: str
        rgb_bands: List[int]
    """

    # Custom colormap for settlements visualization
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Settlements", np.array(['#ff0000', '#0000ff', '#ffff00', '#b266ff']), N=4)


    
    

    fig, axs = plt.subplots(nrows=1, ncols=3)

    # make sure to use cmap=cmap, vmin=-0.5 and vmax=3.5 when running
    # axs[i].imshow on the 1d images in order to have the correct 
    # colormap for the images.
    # On one of the 1d images' axs[i].imshow, make sure to save its output as 
    # `im`, i.e, im = axs[i].imshow
    # Assuming that the stitched images are correctly returned in the expected format
    # Convert RGB bands to correct format for display

    # Retrieve and process the stitched images for visualization
    stitched_rgb, stitched_gt, stitched_pred = restitch_eval(dir=options.processed_dir,satellite_type=satellite_type,tile_id=parent_tile_id,range_x=(0,4),
        range_y= (0,4),
        datamodule=datamodule,
        model=model
    )

    

    rgb_image_stitched_modified = np.stack((stitched_rgb[0, 3, :, :], stitched_rgb[0, 2, :, :], stitched_rgb[0, 1, :, :]), axis=-1)

    # Display stitched RGB image
    im_rgb = axs[0].imshow(rgb_image_stitched_modified)
    axs[0].set_title('RGB Image')
    axs[0].axis('off')

    # Display ground truth
    im_gt = axs[1].imshow(stitched_gt, cmap=cmap, vmin=-0.5, vmax=3.5)
    axs[1].set_title('Ground Truth')
    axs[1].axis('off')

    # Display model prediction
    im_pred = axs[2].imshow(stitched_pred, cmap=cmap, vmin=-0.5, vmax=3.5)
    axs[2].set_title('Model Prediction')
    axs[2].axis('off')



    # The following lines sets up the colorbar to the right of the images    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im_gt, cax=cbar_ax)
    cbar.set_ticks([0,1,2,3])
    cbar.set_ticklabels(['Sttlmnts Wo Elec', 'No Sttlmnts Wo Elec', 'Sttlmnts W Elec', 'No Sttlmnts W Elec'])
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / f"{parent_tile_id}_restitched.png")
        plt.close()


    
    
    
    


def restitch_eval(dir: str | os.PathLike, satellite_type: str, tile_id: str, range_x: Tuple[int, int], range_y: Tuple[int, int], datamodule, model) -> np.ndarray:
    """
    Given a directory of processed subtiles, a tile_id and a satellite_type, 
    this function will retrieve the tiles between (range_x[0],range_y[0])
    and (range_x[1],range_y[1]) in order to stitch them together to their
    original image. It will also get the tiles from the datamodule, evaluate
    it with model, and stitch the ground truth and predictions together.

    Input:
        dir: str | os.PathLike
            Directory where the subtiles are saved
        satellite_type: str
            Satellite type that will be stitched
        tile_id: str
            Tile id that will be stitched
        range_x: Tuple[int, int]
            Range of tiles that will be stitched on width dimension [0,5)]
        range_y: Tuple[int, int]
            Range of tiles that will be stitched on height dimension
        datamodule: pytorch_lightning.LightningDataModule
            Datamodule with the dataset
        model: pytorch_lightning.LightningModule
            LightningModule that will be evaluated
    
    Output:
        stitched_image: np.ndarray
            Stitched image, of shape (time, bands, width, height)
        stitched_ground_truth: np.ndarray
            Stitched ground truth of shape (width, height)
        stitched_prediction_subtile: np.ndarray
            Stitched predictions of shape (width, height)
    """

    dir = Path(dir)
    satellite_subtile = []
    ground_truth_subtile = []
    predictions_subtile = []
    satellite_metadata_from_subtile = []
    for i in range(*range_x):
        satellite_subtile_row = []
        ground_truth_subtile_row = []
        predictions_subtile_row = []
        satellite_metadata_from_subtile_row = []
        for j in range(*range_y):
            subtile = None

            # Create two file paths
            train_path = os.path.join(dir, 'Train', 'subtiles', f"{tile_id}_{i}_{j}.npz")
            val_path = os.path.join(dir, 'Val', 'subtiles', f"{tile_id}_{i}_{j}.npz")

            
            # Check if tile in Train or Val
            
            X, y = None, None
            if os.path.exists(train_path):
                subtile = Subtile().load(dir / 'Train' / 'subtiles' / f"{tile_id}_{i}_{j}.npz")
                dataset = datamodule.train_dataset
                for idx in range(len(dataset)):
                    X, y, tile_md = dataset[idx]
                    if tile_id == tile_md.parent_tile_id and i == tile_md.x_gt and j == tile_md.y_gt:
                        break
            elif os.path.exists(val_path):
                subtile = Subtile().load(dir / 'Val' / 'subtiles' / f"{tile_id}_{i}_{j}.npz")
                dataset = datamodule.val_dataset
                for idx in range(len(dataset)):
                    X, y, tile_md = dataset[idx]
                    if tile_id == tile_md.parent_tile_id and i == tile_md.x_gt and j == tile_md.y_gt:
                        break


                      


            # prediction
            X = X.float()
            prediction = model(X.unsqueeze(0))
            predictions = torch.argmax(prediction, dim=1)



            ground_truth_subtile_row.append(y)
            predictions_subtile_row.append(predictions)
            satellite_subtile_row.append(subtile.satellite_stack[satellite_type])
            satellite_metadata_from_subtile_row.append(subtile.tile_metadata)
        ground_truth_subtile.append(np.concatenate(ground_truth_subtile_row, axis=-1))
        predictions_subtile.append(np.concatenate(predictions_subtile_row, axis=-1))
        satellite_subtile.append(np.concatenate(satellite_subtile_row, axis=-1))
        satellite_metadata_from_subtile.append(satellite_metadata_from_subtile_row)
    return np.concatenate(satellite_subtile, axis=-2), np.concatenate(ground_truth_subtile, axis=-2).reshape(16, 16), np.concatenate(predictions_subtile, axis=-2).reshape(16, 16)
