""" This module contains functions for plotting satellite images. """
import os
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from ..preprocessing.file_utils import Metadata
from ..preprocessing.preprocess_sat import minmax_scale
from ..preprocessing.preprocess_sat import (
    preprocess_sentinel1,
    preprocess_sentinel2,
    preprocess_landsat,
    preprocess_viirs
)


def plot_viirs_histogram(
        viirs_stack: np.ndarray,
        image_dir: None | str | os.PathLike = None,
        n_bins=100
        ) -> None:
    """
    This function plots the histogram over all VIIRS values.
    note: viirs_stack is a 4D array of shape (time, band, height, width)

    Parameters
    ----------
    viirs_stack : np.ndarray
        The VIIRS image stack volume.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """

    plt.hist(viirs_stack.flatten(), bins=n_bins, color='blue', alpha=0.7, log=True)
    

    # Adjust labels
    plt.xlabel('VIIRS Values')
    plt.ylabel('Frequency (log scale)')

    # Save the single PNG file
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "VIIRS_histogram.png")
        plt.close()




def plot_sentinel1_histogram(
        sentinel1_stack: np.ndarray,
        metadata: List[Metadata],
        image_dir: None | str | os.PathLike = None,
        n_bins=20
        ) -> None:
    """
    This function plots the Sentinel-1 histogram over all Sentinel-1 values.
    note: sentinel1_stack is a 4D array of shape (time, band, height, width)

    Parameters
    ----------
    sentinel1_stack : np.ndarray
        The Sentinel-1 image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-1 image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    tile, time, band, height, width = sentinel1_stack.shape
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))


    for b in range(band):
        ax = axs[b]
        
        ax.hist(sentinel1_stack[:, :, b, :, :].flatten(), bins=n_bins, color='blue', alpha=0.7, log=True)
        ax.set_title(f'Band {metadata[0][0].bands[b]}')

    plt.subplots_adjust(hspace=0.5)

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "sentinel1_histogram.png")
        plt.close()
   


def plot_sentinel2_histogram(
        sentinel2_stack: np.ndarray,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None,
        n_bins=20) -> None:
    """
    This function plots the Sentinel-2 histogram over all Sentinel-2 values.

    Parameters
    ----------
    sentinel2_stack : np.ndarray
        The Sentinel-2 image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-2 image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    tile, time, band, height, width = sentinel2_stack.shape
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 10))

    row = 0
    for b in range(band):
        if b % 3 == 0 and b!=0:
            row +=1
        ax = axs[row, b%3]
        
        ax.hist(sentinel2_stack[:, :, b, :, :].flatten(), bins=n_bins, color='blue', alpha=0.7, log=True)
        ax.set_title(f'Band {metadata[0][0].bands[b]}')

    plt.subplots_adjust(hspace=0.5)
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "sentinel2_histogram.png")
        plt.close()
    


def plot_landsat_histogram(
        landsat_stack: np.ndarray,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None,
        n_bins=20
        ) -> None:
    """
    This function plots the landsat histogram over all landsat values over all
    tiles present in the landsat_stack.

    Parameters
    ----------
    landsat_stack : np.ndarray
        The landsat image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the landsat image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    tile, time, band, height, width = landsat_stack.shape
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 10))

    row = 0
    for b in range(band):
        if b % 3 == 0 and b!=0:
            row +=1
        ax = axs[row, b%3]
        
        ax.hist(landsat_stack[:, :, b, :, :].flatten(), bins=n_bins, color='blue', alpha=0.7, log=True)
        ax.set_title(f'Band {metadata[0][0].bands[b]}')

    plt.subplots_adjust(hspace=0.5)
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "landsat_histogram.png")
        plt.close()
  


def plot_gt_counts(ground_truth: np.ndarray,
                   image_dir: None | str | os.PathLike = None
                   ) -> None:
    """
    This function plots the ground truth histogram over all ground truth
    values over all tiles present in the groundTruth_stack.

    Parameters
    ----------
    groundTruth : np.ndarray
        The ground truth image stack volume.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    plt.hist(ground_truth.flatten(), color='blue', alpha=0.7, log=True)
    

    # Adjust labels
    plt.xlabel('Ground Truth Values')
    plt.ylabel('Frequency (log scale)')
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "ground_truth_histogram.png")
        plt.close()
 


def plot_viirs(
        viirs: np.ndarray, plot_title: str = '',
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """ This function plots the VIIRS image.

    Parameters
    ----------
    viirs : np.ndarray
        The VIIRS image.
    plot_title : str
        The title of the plot.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    plt.imshow(viirs, cmap='viridis')
    plt.title(plot_title)
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "viirs_max_projection.png")
        plt.close()
    


def plot_viirs_by_date(
        viirs_stack: np.array,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None) -> None:
    """
    This function plots the VIIRS image by band in subplots.

    Parameters
    ----------
    viirs_stack : np.ndarray
        The VIIRS image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the VIIRS image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    time, bands, height, width = viirs_stack.shape
    fig, axs = plt.subplots(nrows=time, ncols=bands, figsize=(15, 10))

    for t in range(time):
        for b in range(bands):
            ax = axs[t]
            ax.imshow(viirs_stack[t][b], cmap='viridis')  # You can change the colormap if needed
            ax.set_title(f'Time {metadata[t].time}, Band {metadata[t].bands[b]}')
            ax.axis('off')
    
    plt.tight_layout()
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "viirs_plot_by_date.png")
        plt.close()
 


def preprocess_data(
        satellite_stack: np.ndarray,
        satellite_type: str
        ) -> np.ndarray:
    """
    This function preprocesses the satellite data based on the satellite type.

    Parameters
    ----------
    satellite_stack : np.ndarray
        The satellite image stack volume.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    np.ndarray
    """
    preprocess_functions = {"sentinel2":preprocess_sentinel2, "sentinel1":preprocess_sentinel1, "landsat":preprocess_landsat, "viirs":preprocess_viirs}

    return preprocess_functions[satellite_type](satellite_stack)


def create_rgb_composite_s1(
        processed_stack: np.ndarray,
        bands_to_plot: List[List[str]],
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """
    This function creates an RGB composite for Sentinel-1.
    This function needs to extract the band identifiers from the metadata
    and then create the RGB composite. For the VV-VH composite, after
    the subtraction, you need to minmax scale the image.

    Minmax VV, minmax VH, and Minmax VV-VH

    Parameters
    ----------
    processed_stack : np.ndarray
        The Sentinel-1 image stack volume.
    bands_to_plot : List[List[str]]
        The bands to plot. Cannot accept more than 3 bands.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-1 image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    if len(bands_to_plot) > 3:
        raise ValueError("Cannot plot more than 3 bands.")

    time, bands, height, width = processed_stack.shape
    fig, axs = plt.subplots(nrows=time, ncols=1, figsize=(15, 10))

    for t in range(time):
        rgb = np.stack([minmax_scale(processed_stack[t][1], True), minmax_scale(processed_stack[t][0], True), minmax_scale(minmax_scale(processed_stack[t][1], True) - minmax_scale(processed_stack[t][0], True), True)], axis=-1)
        # rgb = np.stack([processed_stack[t][1], processed_stack[t][0], minmax_scale(processed_stack[t][1] - processed_stack[t][0], False)], axis=-1)
        ax = axs[t]
        ax.imshow(rgb)  # You can change the colormap if needed
        ax.set_title(f'Time {metadata[t].time}')
        ax.axis('off')

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "plot_sentinel1.png")
        plt.close()
 



def validate_band_identifiers(
          bands_to_plot: List[List[str]],
          band_mapping: dict) -> None:
    """
    This function validates the band identifiers.

    Parameters
    ----------
    bands_to_plot : List[List[str]]
        The bands to plot.
    band_mapping : dict
        The band mapping.

    Returns
    -------
    None
    """
    for time in bands_to_plot:
        for band in time:
            assert band in band_mapping


def plot_images(
        processed_stack: np.ndarray,
        bands_to_plot: List[List[str]],
        band_mapping: dict,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None
        ):
    """
    This function plots the satellite images.

    Parameters
    ----------
    processed_stack : np.ndarray
        The satellite image stack volume.
    bands_to_plot : List[List[str]]
        The bands to plot.
    band_mapping : dict
        The band mapping.
    metadata : List[List[Metadata]]
        The metadata for the satellite image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    time, bands, height, width = processed_stack.shape
    fig, axs = plt.subplots(nrows=time, ncols=len(bands_to_plot), figsize=(15, 10))
  

    
    for t in range(time):
        for b in range(len(bands_to_plot)):
            stacked = np.stack([processed_stack[t][band_mapping[bands_to_plot[b][0]]], processed_stack[t][band_mapping[bands_to_plot[b][1]]], processed_stack[t][band_mapping[bands_to_plot[b][2]]]], axis=-1)
            ax = axs[t, b]
            ax.imshow(stacked)  # You can change the colormap if needed
            ax.set_title(f'Time {metadata[t].time}, Band {bands_to_plot[b]}')
            ax.axis('off')

    plt.subplots_adjust(hspace=0.5)
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(
            Path(image_dir) / f"plot_{metadata[0].satellite_type}.png"
            )
        plt.close()



def extract_band_ids(metadata: List[Metadata]) -> List[List[str]]:
    """
    Extract the band identifiers from file names for each timestamp based on
    satellite type.

    Parameters
    ----------
    file_names : List[List[str]]
        A list of file names.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    List[List[str]]
        A list of band identifiers.
    """
    band_list = []
    for m in metadata:
        temp = []
        for b in m.bands:
            temp.append(b)
        band_list.append(temp)
    return band_list


def plot_satellite_by_bands(
        satellite_stack: np.ndarray,
        metadata: List[Metadata],
        bands_to_plot: List[List[str]],
        satellite_type: str,
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """
    This function plots the satellite image by band in subplots.

    Parameters
    ----------
    satellite_stack : np.ndarray
        The satellite image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the satellite image stack.
    bands_to_plot : List[List[str]]
        The bands to plot.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    None
    """
    processed_stack = preprocess_data(satellite_stack, satellite_type)

    if satellite_type == "sentinel1":
        create_rgb_composite_s1(processed_stack, bands_to_plot, metadata, image_dir=image_dir)
    else:
        band_ids_per_timestamp = extract_band_ids(metadata)
        all_band_ids = [band_id for timestamp in band_ids_per_timestamp for
                        band_id in timestamp]
        unique_band_ids = sorted(list(set(all_band_ids)))
        band_mapping = {band_id: idx for
                        idx, band_id in enumerate(unique_band_ids)}
        validate_band_identifiers(bands_to_plot, band_mapping)
        plot_images(
            processed_stack,
            bands_to_plot,
            band_mapping,
            metadata,
            image_dir
            )





def plot_ground_truth(
        ground_truth: np.array,
        plot_title: str = '',
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """
    This function plots the groundTruth image.

    Parameters
    ----------
    tile_dir : str
        The directory containing the VIIRS tiles.
    """
    plt.imshow(ground_truth[0][0])
    plt.title(plot_title)
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "ground_truth.png")
        plt.close()
 
