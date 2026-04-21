# -*- coding: utf-8 -*-
"""
Test LEMI Client (supports both LEMI-424 and LEMI-423 instruments)

Created on Fri Oct 11 11:48:24 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path

from mth5.clients.lemi import LEMIClient


# =============================================================================


class TestLEMIClientBase(unittest.TestCase):
    """Test basic LEMIClient functionality (both LEMI-424 and LEMI-423)"""

    def setUp(self):
        self.file_path = Path(__file__).parent.joinpath("test.h5")
        self.base = LEMIClient(
            self.file_path.parent, **{"h5_mode": "w", "h5_driver": "sec2"}
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
        self.assertListEqual(keys, sorted(self.base.h5_kwargs.keys()))

    def test_set_sample_rates_float(self):
        self.base.sample_rates = 10.5
        self.assertListEqual([10.5], self.base.sample_rates)

    def test_set_sample_rates_str(self):
        self.base.sample_rates = "10, 42, 1200"
        self.assertListEqual([10.0, 42.0, 1200.0], self.base.sample_rates)

    def test_set_sample_rate_list(self):
        self.base.sample_rates = [10, 42, 1200]
        self.assertListEqual([10.0, 42.0, 1200.0], self.base.sample_rates)

    def test_set_sample_rate_fail(self):
        def set_sample_rates(value):
            self.base.sample_rates = value

        self.assertRaises(TypeError, set_sample_rates, None)

    def test_set_save_path(self):
        self.base.save_path = self.file_path
        with self.subTest("_save_path"):
            self.assertEqual(self.base._save_path, self.file_path.parent)
        with self.subTest("filename"):
            self.assertEqual(self.base.mth5_filename, self.file_path.name)
        with self.subTest("save_path"):
            self.assertEqual(self.base.save_path, self.file_path)

    def test_initial_fail_None(self):
        self.assertRaises(ValueError, LEMIClient, None)

    def test_initial_fail_bad_directory(self):
        self.assertRaises(IOError, LEMIClient, r"a:\\")


class TestLEMI424SampleRates(unittest.TestCase):
    """Test LEMI-424 specific sample rate (1 Hz)"""

    def setUp(self):
        self.file_path = Path(__file__).parent
        self.client = LEMIClient(self.file_path, sample_rates=[1])

    def test_lemi424_sample_rate(self):
        """LEMI-424 operates at 1 Hz"""
        self.assertEqual(self.client.sample_rates, [1.0])


class TestLEMI423SampleRates(unittest.TestCase):
    """Test LEMI-423 specific sample rates (broadband)"""

    def setUp(self):
        self.file_path = Path(__file__).parent

    def test_lemi423_single_sample_rate(self):
        """LEMI-423 can operate at a single specified rate"""
        client = LEMIClient(self.file_path, sample_rates=[1000])
        self.assertEqual(client.sample_rates, [1000.0])

    def test_lemi423_multiple_sample_rates(self):
        """LEMI-423 can search for multiple sample rates"""
        client = LEMIClient(self.file_path, sample_rates=[4000, 2000, 1000, 500, 250])
        self.assertEqual(client.sample_rates, [4000.0, 2000.0, 1000.0, 500.0, 250.0])

    def test_default_sample_rates(self):
        """Default sample rates should include both LEMI-424 and LEMI-423 rates"""
        # Note: Default is [1] for backward compatibility
        client = LEMIClient(self.file_path)
        self.assertEqual(client.sample_rates, [1.0])


class TestBackwardCompatibility(unittest.TestCase):
    """Test deprecated LEMI424Client alias still works"""

    def test_lemi424client_alias(self):
        """LEMI424Client should still work as an alias"""
        from mth5.clients.lemi import LEMI424Client

        file_path = Path(__file__).parent

        # LEMI424Client is just an alias, so it works without warning
        client = LEMI424Client(file_path)

        # Should be an instance of LEMIClient (since it's an alias)
        self.assertIsInstance(client, LEMIClient)


class TestLEMI120CoilResponse(unittest.TestCase):
    """Test LEMI-120 coil response calibration data"""

    def test_coil_response_creation(self):
        """Test that LEMI120CoilResponse class can be instantiated"""
        from mth5.io.lemi.lemi423 import LEMI120CoilResponse

        lemi120 = LEMI120CoilResponse()
        self.assertIsNotNone(lemi120)

    def test_coil_response_data(self):
        """Test that coil response has correct calibration data"""
        from mth5.io.lemi.lemi423 import LEMI120CoilResponse

        lemi120 = LEMI120CoilResponse()

        # Should have 57 calibration points (from the spec sheet)
        self.assertEqual(len(lemi120._calibration_data), 57)

        # Check frequency range
        frequencies = lemi120._calibration_data[:, 0]
        self.assertAlmostEqual(frequencies.min(), 0.0001, places=4)
        self.assertAlmostEqual(frequencies.max(), 1001.8, places=1)

    def test_get_coil_response_filter(self):
        """Test that get_coil_response returns a proper filter object"""
        from mth5.io.lemi.lemi423 import LEMI120CoilResponse

        lemi120 = LEMI120CoilResponse()
        filter_obj = lemi120.get_coil_response()

        # Check filter properties
        self.assertEqual(filter_obj.units_in, "nanoTesla")
        self.assertEqual(filter_obj.units_out, "milliVolt")
        self.assertEqual(filter_obj.type, "fap")
        self.assertTrue(filter_obj.name.startswith("lemi_120"))

        # Check that frequencies, amplitudes, and phases are numpy arrays
        import numpy as np

        self.assertIsInstance(filter_obj.frequencies, np.ndarray)
        self.assertIsInstance(filter_obj.amplitudes, np.ndarray)
        self.assertIsInstance(filter_obj.phases, np.ndarray)

        # Check that they have the same length
        self.assertEqual(len(filter_obj.frequencies), len(filter_obj.amplitudes))
        self.assertEqual(len(filter_obj.frequencies), len(filter_obj.phases))

    def test_coil_response_with_serial_number(self):
        """Test that coil response can use serial number in name"""
        from mth5.io.lemi.lemi423 import LEMI120CoilResponse

        lemi120 = LEMI120CoilResponse()
        filter_obj = lemi120.get_coil_response(coil_number="12345")

        self.assertEqual(filter_obj.name, "lemi_120_12345_response")


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
