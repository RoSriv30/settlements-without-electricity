""" Exploratory Data Analysis Script """
import os
import sys
import pyprojroot
root = pyprojroot.here()
sys.path.append(str(root))
from src.preprocessing.file_utils import (
    load_satellite,
    load_satellite_dir,
)
from src.preprocessing.preprocess_sat import (
    maxprojection_viirs,
)
from src.visualization.plot_utils import (
    plot_satellite_by_bands,
    plot_viirs_by_date,
    plot_viirs_histogram,
    plot_ground_truth,
    plot_gt_counts,
    plot_sentinel2_histogram,
    plot_landsat_histogram,
    plot_sentinel1_histogram,
    plot_viirs
)

if __name__ == "__main__":
    tile1_dir = os.path.join(root, 'data', 'raw', 'Train', 'Tile1')
    train_dir = os.path.join(root, 'data', 'raw', 'Train')
    save_plots_dir = os.path.join(root, 'plots')

    viirs_stack, metadata = load_satellite(tile1_dir, "viirs")
    max_proj_viirs = maxprojection_viirs(viirs_stack).squeeze(0)
    # Plot the VIIRS max projection image
    plot_viirs(max_proj_viirs, "VIIRS Max Projection", image_dir=save_plots_dir)
    viirs_stack_h, metadata_h = load_satellite_dir(train_dir, "viirs")
    plot_viirs_by_date(viirs_stack, metadata, image_dir=save_plots_dir)
    plot_viirs_histogram(viirs_stack_h, image_dir=save_plots_dir)

    gt_stack, gt_metadata = load_satellite(tile1_dir, "gt")
    plot_ground_truth(gt_stack, "Ground Truth", image_dir=save_plots_dir)
    gt_dataset_stack, gt_metadata_stack = load_satellite_dir(train_dir, "gt")
    plot_gt_counts(gt_dataset_stack, image_dir=save_plots_dir)

    sentinel2_stack, s2_metadata = load_satellite(tile1_dir, "sentinel2")
    s2_stack_h, metadata_s2_h = load_satellite_dir(train_dir, "sentinel2")
    plot_satellite_by_bands(
        sentinel2_stack,
        s2_metadata,
        [['04', '03', '02'], ['08', '04', '03'], ['12', '8A', '04']],
        "sentinel2",
        image_dir=save_plots_dir
        )
    plot_sentinel2_histogram(s2_stack_h, metadata_s2_h, image_dir=save_plots_dir)

    sentinel1_stack, s1_metadata = load_satellite(tile1_dir, "sentinel1")
    s1_stack_h, metadata_s1_h = load_satellite_dir(train_dir, "sentinel1")
    plot_satellite_by_bands(
        sentinel1_stack,
        s1_metadata,
        [['VV', 'VH', 'VV-VH']],
        "sentinel1",
        image_dir=save_plots_dir
        )
    plot_sentinel1_histogram(s1_stack_h, metadata_s1_h, image_dir=save_plots_dir)

    landsat_stack, file_names = load_satellite(tile1_dir, "landsat")
    ls_stack_h, metadata_ls_h = load_satellite_dir(train_dir, "landsat")
    plot_landsat_histogram(ls_stack_h, metadata_ls_h, image_dir=save_plots_dir)

    # # # 4, 3, 2 RGB -> 5, 4, 3
    # # # 5, 4, 3 color infrared ->
    plot_satellite_by_bands(
        landsat_stack, file_names,
        [['4', '3', '2'], ['7', '6', '4'], ['7', '5', '3']],
        "landsat", image_dir=save_plots_dir)
