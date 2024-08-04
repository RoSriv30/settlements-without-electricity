""" This code applies preprocessing functions on the IEEE GRSS ESD satellite data."""
import numpy as np
from scipy.ndimage import gaussian_filter


def per_band_gaussian_filter(img: np.ndarray, sigma: float = 1):
    """
    For each band in the image, apply a gaussian filter with the given sigma.

    Parameters
    ----------
    img : np.ndarray
        The image to be filtered.
    sigma : float
        The sigma of the gaussian filter.

    Returns
    -------
    np.ndarray
        The filtered image.
    """

    for i in range(img.shape[0]):
        img[i] = gaussian_filter(img[i], sigma)
    return img



def quantile_clip(img_stack: np.ndarray,
                  clip_quantile: float,
                  group_by_time=True
                  ) -> np.ndarray:
    """
    This function clips the outliers of the image stack by the given quantile.

    Parameters
    ----------
    img_stack : np.ndarray
        The image stack to be clipped.
    clip_quantile : float
        The quantile to clip the outliers by.

    Returns
    -------
    np.ndarray
        The clipped image stack.
    """
    if group_by_time:
        axis = (-2, -1)
    else:
        axis = (0, -2, -1)
    data_lower_bound = np.quantile(
        img_stack,
        clip_quantile,
        axis=axis,
        keepdims=True
        )
    data_upper_bound = np.quantile(
        img_stack,
        1-clip_quantile,
        axis=axis,
        keepdims=True
        )
    img_stack = np.clip(img_stack, data_lower_bound, data_upper_bound)

    return img_stack



def minmax_scale(img: np.ndarray, group_by_time=True):
    """
    This function minmax scales the image stack.

    Parameters
    ----------
    img : np.ndarray
        The image stack to be minmax scaled.
    group_by_time : bool
        Whether to group by time or not.

    Returns
    -------
    np.ndarray
        The minmax scaled image stack.
    """
    if group_by_time:
        axis = (-2, -1)
    else:
        axis = (0, -2, -1)
    img = img.astype(np.float32)
    min_val = img.min(axis=axis, keepdims=True)
    max_val = img.max(axis=axis, keepdims=True)
    normalized_img = (img - min_val) / (max_val - min_val)
    return normalized_img



def brighten(img, alpha=0.13, beta=0):
    """
    Function to brighten the image.

    Parameters
    ----------
    img : np.ndarray
        The image to be brightened.
    alpha : float
        The alpha parameter of the brightening.
    beta : float
        The beta parameter of the brightening.

    Returns
    -------
    np.ndarray
        The brightened image.
    """
    return np.clip(alpha * img + beta, 0.0, 1.0)



def gammacorr(band, gamma=2):
    """
    This function applies a gamma correction to the image.

    Parameters
    ----------
    band : np.ndarray
        The image to be gamma corrected.
    gamma : float
        The gamma parameter of the gamma correction.

    Returns
    -------
    np.ndarray
        The gamma corrected image.
    """
    return np.power(band, 1/gamma)



def maxprojection_viirs(
        viirs_stack: np.ndarray,
        clip_quantile: float = 0.01
        ) -> np.ndarray:
    """
    This function takes a directory of VIIRS tiles and returns a single
    image that is the max projection of the tiles.

    Parameters
    ----------
    tile_dir : str
        The directory containing the VIIRS tiles.

    Returns
    -------
    np.ndarray
    """
    for i in range(viirs_stack.shape[0]):
        viirs_data_lower_bound = np.quantile(
            viirs_stack[i, :, :, :],
            clip_quantile
            )
        viirs_data_upper_bound = np.quantile(
            viirs_stack[i, :, :, :],
            1-clip_quantile
            )
        viirs_stack[i, :, :, :] = np.clip(
            viirs_stack[i, :, :, :],
            viirs_data_lower_bound,
            viirs_data_upper_bound
            )

    # Calculate the max projection of the viirs_data_stack along the third axis
    # and assign it to the blank_array
    viirs_stack = np.max(viirs_stack, axis=0)
    viirs_stack = minmax_scale(viirs_stack)

    return viirs_stack


def preprocess_sentinel1(
        sentinel1_stack: np.ndarray,
        clip_quantile: float = 0.01,
        sigma=1
        ) -> np.ndarray:
    """
    In this function we will preprocess sentinel1. The steps for preprocessing
    are the following:
        - Convert data to dB (log scale)
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gaussian filter
        - Minmax scale
    """

    # convert data to dB
    sentinel1_stack = np.log10(sentinel1_stack)

    # clip outliers
    sentinel1_stack = quantile_clip(
        sentinel1_stack,
        clip_quantile=clip_quantile
        )
    sentinel1_stack = per_band_gaussian_filter(sentinel1_stack, sigma=sigma)
    sentinel1_stack = minmax_scale(sentinel1_stack)

    return sentinel1_stack


def preprocess_sentinel2(sentinel2_stack: np.ndarray,
                         clip_quantile: float = 0.1,
                         gamma: float = 2.2
                         ) -> np.ndarray:
    """
    In this function we will preprocess sentinel-2. The steps for
    preprocessing are the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gamma correction
        - Minmax scale
    """
    sentinel2_stack = quantile_clip(
        sentinel2_stack,
        clip_quantile=clip_quantile,
        group_by_time=False
        )
    sentinel2_stack = gammacorr(sentinel2_stack, gamma=gamma)
    sentinel2_stack = minmax_scale(sentinel2_stack, group_by_time=False)

    return sentinel2_stack



def preprocess_landsat(
        landsat_stack: np.ndarray,
        clip_quantile: float = 0.05,
        gamma: float = 2.2
        ) -> np.ndarray:
    """
    In this function we will preprocess landsat. The steps for preprocessing
    are the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gamma correction
        - Minmax scale
    """
    landsat_stack = quantile_clip(
        landsat_stack,
        clip_quantile=clip_quantile,
        group_by_time=False
        )
    landsat_stack = gammacorr(landsat_stack, gamma=gamma)
    landsat_stack = minmax_scale(landsat_stack, group_by_time=False)

    return landsat_stack


def preprocess_viirs(viirs_stack, clip_quantile=0.05) -> np.ndarray:
    """
    In this function we will preprocess viirs. The steps for preprocessing are
    the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Minmax scale
    """
    viirs_stack = quantile_clip(
        viirs_stack,
        clip_quantile=clip_quantile,
        group_by_time=True
        )
    viirs_stack = minmax_scale(viirs_stack, group_by_time=True)
    return viirs_stack


def process_viirs_filename(filename: str) -> tuple[str, str]:
    """
    This function takes in the filename of a VIIRS file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: DNB_VNP46A1_A2020221.tif
    Example output: ("2020221", "0")

    Parameters
    ----------
    filename : str
        The filename of the VIIRS file.

    Returns
    -------
    tuple[str, str]
        A tuple containing the date and band.
    """
    date = filename.split('_')[2][1:8]
    band = "0"

    return (date, band)


def process_s1_filename(filename: str) -> tuple[str, str]:
    """
    This function takes in the filename of a Sentinel-1 file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: S1A_IW_GRDH_20200804_VV.tif
    Example output: ("20200804", "VV")

    Parameters
    ----------
    filename : str
        The filename of the Sentinel-1 file.

    Returns
    -------
    tuple[str, str]
        A tuple containing the date and band.
    """
    parts = filename.split('_')
    date = parts[3]
    band = parts[4][:2]

    return (date, band)


def process_s2_filename(filename: str) -> tuple[str, str]:
    """
    This function takes in the filename of a Sentinel-2 file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: L2A_20200816_B01.tif
    Example output: ("20200804", "B01")

    Parameters
    ----------
    filename : str
        The filename of the Sentinel-2 file.

    Returns
    -------
    tuple[str, str]
    """
    parts = filename.split('_')
    date = parts[1]
    band = parts[2][1:3]

    return (date, band)


def process_landsat_filename(filename: str) -> tuple[str, str]:
    """
    This function takes in the filename of a Landsat file and outputs
    a tuple containing two strings, in the format (date, band)

    Example input: LC08_L1TP_2020-08-30_B9.tif
    Example output: ("2020-08-30", "B9")

    Parameters
    ----------
    filename : str
        The filename of the Landsat file.

    Returns
    -------
    tuple[str, str]
        A tuple containing the date and band.
    """
    parts = filename.split('_')
    date = parts[2]
    band = parts[3].split('.')[0][1:]

    return (date, band)


def process_ground_truth_filename(filename: str) -> tuple[str, str]:
    """
    This function takes in the filename of the ground truth file and returns
    ("0", "0"), as there is only one ground truth file.

    Example input: groundTruth.tif
    Example output: ("0", "0")

    Parameters
    ----------
    filename: str
        The filename of the ground truth file though we will ignore it.

    Returns
    -------
    tuple[str, str]
        A tuple containing the date and band.
    """
    return ("0", "0")
