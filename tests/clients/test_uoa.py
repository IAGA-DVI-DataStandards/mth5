# -*- coding: utf-8 -*-
"""
Test UoA Client (supports PR6-24 and Orange Box instruments)

Created on Thu Nov 7 15:45:00 2025

@author: bkay
"""

# =============================================================================
# Imports
# =============================================================================
import unittest

from pathlib import Path

from mth5.clients.uoa import UoAClient

# =============================================================================


class TestUoAClientBase(unittest.TestCase):
    """Test basic UoAClient functionality"""

    def setUp(self):
        self.file_path = Path(__file__).parent.joinpath("test.h5")
        self.base_pr624 = UoAClient(
            self.file_path.parent,
            instrument_type='pr624',
            **{"h5_mode": "w", "h5_driver": "sec2"}
        )
        self.base_orange = UoAClient(
            self.file_path.parent,
            instrument_type='orange',
            **{"h5_mode": "w", "h5_driver": "sec2"}
        )

    def test_h5_kwargs(self):
        keys = [
            "compression",
            "compression_opts",
            "data_level",
            "driver",
            "file_version",
            "fletcher32",
            "mode",
            "shuffle",
        ]
        self.assertListEqual(keys, sorted(self.base_pr624.h5_kwargs.keys()))

    def test_instrument_type_pr624(self):
        """Test PR6-24 instrument type is set correctly"""
        self.assertEqual(self.base_pr624.instrument_type, 'pr624')

    def test_instrument_type_orange(self):
        """Test Orange Box instrument type is set correctly"""
        self.assertEqual(self.base_orange.instrument_type, 'orange')

    def test_invalid_instrument_type(self):
        """Test that invalid instrument type raises ValueError"""
        self.assertRaises(
            ValueError,
            UoAClient,
            self.file_path.parent,
            instrument_type='invalid'
        )

    def test_set_save_path(self):
        self.base_pr624.save_path = self.file_path
        with self.subTest("_save_path"):
            self.assertEqual(self.base_pr624._save_path, self.file_path.parent)
        with self.subTest("filename"):
            self.assertEqual(self.base_pr624.mth5_filename, self.file_path.name)
        with self.subTest("save_path"):
            self.assertEqual(self.base_pr624.save_path, self.file_path)

    def test_initial_fail_None(self):
        self.assertRaises(ValueError, UoAClient, None, instrument_type='pr624')

    def test_initial_fail_bad_directory(self):
        self.assertRaises(IOError, UoAClient, r"a:\\", instrument_type='pr624')


class TestPR624Client(unittest.TestCase):
    """Test PR6-24 (Earth Data Logger) specific functionality"""

    def setUp(self):
        self.file_path = Path(__file__).parent
        self.client = UoAClient(
            self.file_path,
            instrument_type='pr624'
        )

    def test_pr624_instrument_type(self):
        """PR6-24 instrument type should be set"""
        self.assertEqual(self.client.instrument_type, 'pr624')

    def test_default_mth5_filename(self):
        """Default MTH5 filename should be 'from_uoa.h5'"""
        self.assertEqual(self.client.mth5_filename, 'from_uoa.h5')


class TestOrangeBoxClient(unittest.TestCase):
    """Test Orange Box specific functionality"""

    def setUp(self):
        self.file_path = Path(__file__).parent
        self.client = UoAClient(
            self.file_path,
            instrument_type='orange'
        )

    def test_orange_instrument_type(self):
        """Orange Box instrument type should be set"""
        self.assertEqual(self.client.instrument_type, 'orange')

    def test_default_mth5_filename(self):
        """Default MTH5 filename should be 'from_uoa.h5'"""
        self.assertEqual(self.client.mth5_filename, 'from_uoa.h5')


class TestUoAClientPaths(unittest.TestCase):
    """Test UoA client path handling"""

    def setUp(self):
        self.file_path = Path(__file__).parent

    def test_pr624_with_custom_filename(self):
        """Test PR6-24 with custom MTH5 filename"""
        client = UoAClient(
            self.file_path,
            instrument_type='pr624',
            mth5_filename='custom_pr624.h5'
        )
        self.assertEqual(client.mth5_filename, 'custom_pr624.h5')

    def test_orange_with_custom_filename(self):
        """Test Orange Box with custom MTH5 filename"""
        client = UoAClient(
            self.file_path,
            instrument_type='orange',
            mth5_filename='custom_orange.h5'
        )
        self.assertEqual(client.mth5_filename, 'custom_orange.h5')

    def test_pr624_with_save_path(self):
        """Test PR6-24 with custom save path"""
        save_path = self.file_path / 'output'
        client = UoAClient(
            self.file_path,
            instrument_type='pr624',
            save_path=save_path
        )
        self.assertEqual(client._save_path, save_path)

    def test_orange_with_save_path(self):
        """Test Orange Box with custom save path"""
        save_path = self.file_path / 'output'
        client = UoAClient(
            self.file_path,
            instrument_type='orange',
            save_path=save_path
        )
        self.assertEqual(client._save_path, save_path)


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
