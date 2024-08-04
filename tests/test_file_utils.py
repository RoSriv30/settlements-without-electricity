""" Unit tests for file_utils.py """
import unittest
import os
from pathlib import Path
import sys
import numpy as np
import pyprojroot
root = pyprojroot.here()
sys.path.append(str(root))

from src.preprocessing.file_utils import (
    process_viirs_filename,
    process_s1_filename,
    process_s2_filename,
    process_ground_truth_filename,
    process_landsat_filename,
    get_satellite_files,
    get_filename_pattern,
    read_satellite_files,
    stack_satellite_data,
    get_unique_dates_and_bands,
    load_satellite,
    load_satellite_dir,
)

CURRENT_POINTS_TEST = 0
MAX_POINTS_TEST = 20


class TestProcessSatelliteFunctions(unittest.TestCase):
    """ Unit tests for the process satellite functions in file_utils.py"""
    def setUp(self):
        self.tile_dir = Path(
            os.path.join(root, 'data', 'raw', 'unit_test_data', 'Tile1')
            )
        self.train_dir = Path(
            os.path.join(root, 'data', 'raw', 'unit_test_data')
            )

    def tearDown(self):
        pass

    def test_process_viirs_filename(self):
        """
        Test that the process_viirs_filename function returns the correct
        tuple.
        """
        filename = "DNB_VNP46A1_A2020221.tif"
        result = process_viirs_filename(filename)
        self.assertEqual(
            result,
            ("2020221", "0"),
            "test_process_viirs_filename does not return correct tuple."
            )

        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_process_s1_filename(self):
        """
        Test that the process_s1_filename function returns the correct
        tuple.
        """
        filename = "S1A_IW_GRDH_20200804_VV.tif"
        result = process_s1_filename(filename)
        self.assertEqual(
            result,
            ("20200804", "VV"),
            "test_process_s1_filename does not return correct tuple."
            )

        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_process_s2_filename(self):
        """
        Test that the process_s2_filename function returns the correct
        """
        filename = "L2A_20200811_B01.tif"
        result = process_s2_filename(filename)
        self.assertEqual(
            result,
            ("20200811", "01"),
            "test_process_s2_filename does not return correct tuple."
            )

        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_process_landsat_filename(self):
        """
        Test that the process_landsat_filename function returns the correct
        tuple.
        """
        filename = "LC08_L1TP_2020-07-29_B5.tif"
        result = process_landsat_filename(filename)
        self.assertEqual(result,
                         ("2020-07-29", "5"),
                         "test_process_landsat_filename does not "
                         "return correct tuple."
                         )

        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_process_ground_truth_filename(self):
        """
        Test that the process_ground_truth_filename function returns
        the correct tuple.
        """
        filename = "groundTruth.tif"
        result = process_ground_truth_filename(filename)
        self.assertEqual(
            len(result),
            2,
            "test_process_ground_truth_filename does not return correct tuple."
            )
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")


class TestFileRetrieval(unittest.TestCase):
    """ Unit tests for the file retrieval functions in file_utils.py """
    def setUp(self):
        self.tile_dir = Path(
            os.path.join(root, 'data', 'raw', 'unit_test_data', 'Tile1')
            )
        self.train_dir = Path(
            os.path.join(root, 'data', 'raw', 'unit_test_data')
            )

    def test_get_sentinel1_files(self):
        """
        Test that the get_sentinel1_files function returns the correct
        number of files.
        """
        satellite_type = "sentinel1"
        result = get_satellite_files(self.tile_dir, satellite_type)
        self.assertEqual(
            len(result),
            8,
            "test_get_sentinel1_files does not return correct number of files."
            )
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_get_sentinel2_files(self):
        """
        Test that the get_sentinel2_files function returns the correct
        number of files.
        """
        satellite_type = "sentinel2"
        result = get_satellite_files(self.tile_dir, satellite_type)
        self.assertEqual(
            len(result),
            48,
            "test_get_sentinel2_files does not return correct number of files."
            )
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_get_landsat_files(self):
        """
        Test that the get_landsat_files function returns the correct
        number of files.
        """
        satellite_type = "landsat"
        result = get_satellite_files(self.tile_dir, satellite_type)
        self.assertEqual(
            len(result),
            33,
            "test_get_landsat_files does not return correct number of files."
            )
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_get_viirs_files(self):
        """
        Test that the get_viirs_files function returns the correct
        number of files.
        """
        satellite_type = "viirs"
        result = get_satellite_files(self.tile_dir, satellite_type)
        self.assertEqual(
            len(result),
            9,
            "test_get_viirs_files does not return correct number of files."
            )
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_get_ground_truth_files(self):
        """
        Test that the get_groundTruth_files function returns the correct
        number of files.
        """
        satellite_type = "gt"
        result = get_satellite_files(self.tile_dir, satellite_type)
        self.assertEqual(
            len(result),
            1,
            "test_get_groundTruth_files does not return correct number "
            "of files."
        )
        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_filename_pattern(self):
        """
        Test that the get_filename_pattern function returns the correct
        pattern for each satellite type.
        """
        patterns = {
            "viirs": 'DNB_VNP46A1_*',
            "sentinel1": 'S1A_IW_GRDH_*',
            "sentinel2": 'L2A_*',
            "landsat": 'LC08_L1TP_*',
            "gt": "groundTruth.tif"
        }

        for key, value in patterns.items():
            self.assertEqual(
                value,
                get_filename_pattern(key),
                f"Pattern does not work for {key}"
                )

        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_read_satellite_file(self):
        """
        Test that the read_satellite_file function returns the correct
        number of files and dimensions.
        """
        list_of_satellite_paths = [
            self.tile_dir / "DNB_VNP46A1_A2020221.tif",
            self.tile_dir / "DNB_VNP46A1_A2020224.tif"
            ]
        result = read_satellite_files(list_of_satellite_paths)
        self.assertEqual(
            len(result),
            2,
            "test_read_satellite_file does not return correct number of files."
            )
        self.assertEqual(result[0].shape, (800, 800))
        self.assertEqual(
            result[1].shape,
            (800, 800),
            "test_read_satellite_file does not return correct dimensions."
            )

        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_stack_satellite_data(self):
        """
        Test that the stack_satellite_data function returns the correct
        dimensions and metadata.
        """
        list_of_satellite_paths = [
            self.tile_dir / "S1A_IW_GRDH_20200723_VH.tif",
            self.tile_dir / "S1A_IW_GRDH_20200723_VV.tif",
            self.tile_dir / "S1A_IW_GRDH_20200804_VH.tif",
            self.tile_dir / "S1A_IW_GRDH_20200804_VV.tif",
            self.tile_dir / "S1A_IW_GRDH_20200816_VH.tif",
            self.tile_dir / "S1A_IW_GRDH_20200816_VV.tif",
            self.tile_dir / "S1A_IW_GRDH_20200828_VH.tif",
            self.tile_dir / "S1A_IW_GRDH_20200828_VV.tif"
        ]
        satellite_type = "sentinel1"
        sat_data = [np.random.rand(800, 800)]*8
        stacked_data, metadata = stack_satellite_data(
            sat_data,
            list_of_satellite_paths,
            satellite_type
        )
        self.assertEqual(stacked_data.shape,
                         (4, 2, 800, 800),
                         "test_stack_satellite_data does not return correct "
                         "dimensions."
                         )
        self.assertEqual(
            len(metadata),
            4,
            "test_stack_satellite_data does not return the correct length of "
            "metadata."
            )
        self.assertEqual(metadata[0].bands, ["VH", "VV"],
                         "test_stack_satellite_data does not return correct "
                         "band metadata."
                         )
        self.assertEqual(
            metadata[1].bands,
            ["VH", "VV"],
            "test_stack_satellite_data does not return correct band metadata."
            )

        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 3
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_get_unique_dates_and_bands(self):
        """
        Test that the get_unique_dates_and_bands function returns the correct
        number of dates and bands.
        """
        metadata_keys = set([
            ("20200723", "VH"),
            ("20200723", "VV"),
            ("20200804", "VH"),
            ("20200804", "VV")
            ])
        result_dates, result_bands = get_unique_dates_and_bands(metadata_keys)
        self.assertEqual(
            len(result_dates),
            2,
            "test_get_unique_dates_and_bands does "
            "not return correct number of dates."
             )
        self.assertEqual(
            len(result_bands),
            2,
            "test_get_unique_dates_and_bands does "
            "not return correct number of bands."
            )

        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_load_satellite(self):
        """
        Test that the load_satellite function returns the correct
        dimensions and metadata.

        """
        satellite_type = "sentinel1"
        result_data_stack, result_filenames = load_satellite(
            self.tile_dir,
            satellite_type
            )
        self.assertEqual(
            result_data_stack.shape,
            (4, 2, 800, 800),
            "test_load_satellite does not return correct dimensions."
            )
        self.assertEqual(
            len(result_filenames),
            4,
            "test_load_satellite does not return correct number of filenames."
            )

        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 1
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")

    def test_load_satellite_dir(self):
        """
        Test that the load_satellite_dir function returns the correct
        dimensions and metadata.
        """
        satellite_type = "sentinel1"
        result_data_stack, result_filenames_list = load_satellite_dir(
            self.train_dir,
            satellite_type
            )
        self.assertEqual(
            result_data_stack.shape,
            (1, 4, 2, 800, 800),
            "test_load_satellite_dir does not return correct dimensions for "
            "the resulting np.ndarray."
            )
        self.assertEqual(
            len(result_filenames_list),
            1,
            "test_load_satellite_dir does not return "
            "correct number of filenames.")

        if self._outcome.success:
            global CURRENT_POINTS_TEST
            CURRENT_POINTS_TEST += 3
        print(f"\nCurrent points: {CURRENT_POINTS_TEST}/{MAX_POINTS_TEST}")


if __name__ == '__main__':
    unittest.main()
