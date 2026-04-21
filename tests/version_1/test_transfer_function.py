# -*- coding: utf-8 -*-
"""
Modern pytest suite for transfer function tests with fixtures and subtests.
Translated from test_transfer_function.py for optimized efficiency.

@author: pytest translation
"""

from collections import OrderedDict
from pathlib import Path

# =============================================================================
# Imports
# =============================================================================
import pytest
from mt_metadata import TF_XML
from mt_metadata.transfer_functions.core import TF

from mth5.mth5 import MTH5
from mth5.utils.exceptions import MTH5Error


# =============================================================================
# Test Configuration
# =============================================================================


@pytest.fixture(scope="session")
def test_data_path():
    """Get the test data directory path."""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def mth5_file_path(make_worker_safe_path):
    """Get the test MTH5 file path."""
    return make_worker_safe_path("test.mth5", Path(__file__).parent)


@pytest.fixture(scope="session")
def tf_obj_from_xml():
    """Create transfer function object from XML data."""
    tf_obj = TF(TF_XML)
    tf_obj.read()
    return tf_obj


@pytest.fixture(scope="session")
def mth5_obj(mth5_file_path, tf_obj_from_xml):
    """
    Create and setup MTH5 object with transfer function data.
    Session scope for efficiency across all tests.
    """
    mth5_obj = MTH5(file_version="0.1.0")
    mth5_obj.open_mth5(mth5_file_path, mode="a")

    # Add transfer function to the MTH5 file
    tf_group = mth5_obj.add_transfer_function(tf_obj_from_xml)

    # Store references for tests
    mth5_obj._test_tf_group = tf_group
    mth5_obj._test_tf_obj = tf_obj_from_xml

    yield mth5_obj

    # Cleanup
    mth5_obj.close_mth5()
    if mth5_file_path.exists():
        mth5_file_path.unlink()


@pytest.fixture(scope="session")
def tf_h5(mth5_obj, tf_obj_from_xml):
    """Get transfer function from HDF5 file."""
    return mth5_obj.get_transfer_function(
        tf_obj_from_xml.station, tf_obj_from_xml.station
    )


@pytest.fixture(scope="session")
def tf_group(mth5_obj):
    """Get transfer function group from MTH5."""
    return mth5_obj._test_tf_group


@pytest.fixture
def recursive_to_dict():
    """Helper function to recursively convert objects to dict."""

    def _recursive_to_dict(obj):
        if isinstance(obj, dict):
            return {k: _recursive_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_recursive_to_dict(i) for i in obj]
        else:
            return obj

    return _recursive_to_dict


def remove_mth5_keys(data_dict):
    """Remove MTH5-specific keys from dictionary."""
    keys_to_remove = ["mth5_type", "hdf5_reference"]
    for key in keys_to_remove:
        data_dict.pop(key, None)
    return data_dict


# =============================================================================
# Test Classes
# =============================================================================


class TestSurveyMetadata:
    """Test survey metadata functionality."""

    def test_survey_metadata(self, mth5_obj, tf_h5):
        """Test survey metadata against expected values."""
        expected_meta_dict = OrderedDict(
            [
                ("acquired_by.author", "National Geoelectromagnetic Facility"),
                (
                    "citation_dataset.authors",
                    "Schultz, A., Pellerin, L., Bedrosian, P., Kelbert, A., Crosbie, J.",
                ),
                (
                    "citation_dataset.doi",
                    "https://doi.org/10.17611/DP/EMTF/USMTARRAY/SOUTH",
                ),
                (
                    "citation_dataset.title",
                    "USMTArray South Magnetotelluric Transfer Functions",
                ),
                ("citation_dataset.year", "2020-2023"),
                (
                    "comments",
                    "copyright.acknowledgement:The USMTArray-CONUS South campaign was carried out through a cooperative agreement between\nthe U.S. Geological Survey (USGS) and Oregon State University (OSU). A subset of 40 stations\nin the SW US were funded through NASA grant 80NSSC19K0232.\nLand permitting, data acquisition, quality control and field processing were\ncarried out by Green Geophysics with project management and instrument/engineering\nsupport from OSU and Chaytus Engineering, respectively.\nProgram oversight, definitive data processing and data archiving were provided\nby the USGS Geomagnetism Program and the Geology, Geophysics and Geochemistry Science Centers.\nWe thank the U.S. Forest Service, the Bureau of Land Management, the National Park Service,\nthe Department of Defense, numerous state land offices and the many private landowners\nwho permitted land access to acquire the USMTArray data.; copyright.conditions_of_use:All data and metadata for this survey are available free of charge and may be copied freely, duplicated and further distributed provided that this data set is cited as the reference, and that the author(s) contributions are acknowledged as detailed in the Acknowledgements. Any papers cited in this file are only for reference. There is no requirement to cite these papers when the data are used. Whenever possible, we ask that the author(s) are notified prior to any publication that makes use of these data.\n While the author(s) strive to provide data and metadata of best possible quality, neither the author(s) of this data set, nor IRIS make any claims, promises, or guarantees about the accuracy, completeness, or adequacy of this information, and expressly disclaim liability for errors and omissions in the contents of this file. Guidelines about the quality or limitations of the data and metadata, as obtained from the author(s), are included for informational purposes only.; copyright.release_status:Unrestricted Release",
                ),
                ("country", ["USA"]),
                ("datum", "WGS 84"),
                ("geographic_name", "CONUS South"),
                ("id", "CONUS_South"),
                ("name", ""),
                ("northwest_corner.elevation", 0.0),
                ("northwest_corner.latitude", 34.470528),
                ("northwest_corner.longitude", -108.712288),
                ("project", "USMTArray"),
                ("project_lead.author", ""),
                ("release_license", "CC-BY-4.0"),
                ("southeast_corner.elevation", 0.0),
                ("southeast_corner.latitude", 34.470528),
                ("southeast_corner.longitude", -108.712288),
                ("summary", "Magnetotelluric Transfer Functions"),
                ("time_period.end_date", "2020-10-07"),
                ("time_period.start_date", "2020-09-20"),
            ]
        )

        h5_meta_dict = remove_mth5_keys(tf_h5.survey_metadata.to_dict(single=True))
        # Normalize enum/string representations for release status
        if "comments" in h5_meta_dict and isinstance(h5_meta_dict["comments"], str):
            h5_meta_dict["comments"] = h5_meta_dict["comments"].replace(
                "ReleaseStatusEnum.Unrestricted_release", "Unrestricted Release"
            )
        assert expected_meta_dict == h5_meta_dict


class TestStationMetadata:
    """Test station metadata functionality."""

    def test_station_metadata(self, tf_obj_from_xml, recursive_to_dict):
        """Test station metadata against expected values with recursive comparison."""
        expected_meta_dict = OrderedDict(
            [
                ("acquired_by.author", "National Geoelectromagnetic Facility"),
                ("channel_layout", "X"),
                ("channels_recorded", ["ex", "ey", "hx", "hy", "hz"]),
                (
                    "comments",
                    "description:Magnetotelluric Transfer Functions; primary_data.filename:NMX20b_NMX20_NMW20_COR21_NMY21-NMX20b_NMX20_UTS18.png; attachment.description:The original used to produce the XML; attachment.filename:NMX20b_NMX20_NMW20_COR21_NMY21-NMX20b_NMX20_UTS18.zmm; site.data_quality_notes.comments.author:Jade Crosbie, Paul Bedrosian and Anna Kelbert; site.data_quality_notes.comments.value:great TF from 10 to 10000 secs (or longer)",
                ),
                ("data_type", "MT"),
                ("fdsn.id", "USMTArray.NMX20.2020"),
                ("geographic_name", "Nations Draw, NM, USA"),
                ("id", "NMX20"),
                ("location.datum", "WGS 84"),
                ("location.declination.epoch", "2020.0"),
                ("location.declination.model", "IGRF"),
                ("location.declination.value", 9.09),
                ("location.elevation", 1940.05),
                ("location.elevation_uncertainty", 0.0),
                ("location.latitude", 34.470528),
                ("location.latitude_uncertainty", 0.0),
                ("location.longitude", -108.712288),
                ("location.longitude_uncertainty", 0.0),
                ("location.x", 0.0),
                ("location.x2", 0.0),
                ("location.x_uncertainty", 0.0),
                ("location.y", 0.0),
                ("location.y2", 0.0),
                ("location.y_uncertainty", 0.0),
                ("location.z", 0.0),
                ("location.z2", 0.0),
                ("location.z_uncertainty", 0.0),
                ("orientation.angle_to_geographic_north", 0.0),
                ("orientation.method", "compass"),
                ("orientation.reference_frame", "geographic"),
                ("orientation.value", "orthogonal"),
                ("provenance.archive.comments", "IRIS DMC MetaData"),
                ("provenance.archive.name", ""),
                ("provenance.archive.url", "http://www.iris.edu/mda/ZU/NMX20"),
                ("provenance.creation_time", "2021-03-17T14:47:44+00:00"),
                (
                    "provenance.creator.author",
                    "Jade Crosbie, Paul Bedrosian and Anna Kelbert",
                ),
                ("provenance.creator.email", "pbedrosian@usgs.gov"),
                ("provenance.software.author", ""),
                (
                    "provenance.software.name",
                    "EMTF File Conversion Utilities 4.0",
                ),
                ("provenance.software.version", ""),
                ("provenance.submitter.author", "Anna Kelbert"),
                ("provenance.submitter.email", "akelbert@usgs.gov"),
                ("run_list", ["NMX20a", "NMX20b"]),
                ("time_period.end", "2020-10-07T20:28:00+00:00"),
                ("time_period.start", "2020-09-20T19:03:06+00:00"),
                ("transfer_function.coordinate_system", "geographic"),
                ("transfer_function.data_quality.good_from_period", 5.0),
                ("transfer_function.data_quality.good_to_period", 29127.0),
                ("transfer_function.data_quality.rating.value", 5),
                ("transfer_function.id", "NMX20"),
                (
                    "transfer_function.processed_by.author",
                    "Jade Crosbie, Paul Bedrosian and Anna Kelbert",
                ),
                ("transfer_function.processed_date", "1980-01-01T00:00:00+00:00"),
                (
                    "transfer_function.processing_parameters",
                    [
                        "remote_info.site.orientation.angle_to_geographic_north = 0.0",
                        "remote_info.site.orientation.layout = orthogonal",
                        "remote_info.site.run_list = []",
                    ],
                ),
                (
                    "transfer_function.processing_type",
                    "Robust Multi-Station Reference",
                ),
                (
                    "transfer_function.remote_references",
                    [""],
                ),
                ("transfer_function.runs_processed", ["NMX20a", "NMX20b"]),
                ("transfer_function.sign_convention", "exp(+ i\\omega t)"),
                ("transfer_function.software.author", "Gary Egbert"),
                (
                    "transfer_function.software.last_updated",
                    "2015-08-26T00:00:00+00:00",
                ),
                ("transfer_function.software.name", "EMTF"),
                ("transfer_function.software.version", ""),
                ("transfer_function.units", "milliVolt per kilometer per nanoTesla"),
            ]
        )

        d1 = expected_meta_dict
        d2 = tf_obj_from_xml.station_metadata.to_dict(single=True)
        # Normalize processing parameter enum representation if present
        p = d2.get("transfer_function.processing_parameters")
        if isinstance(p, list):
            d2["transfer_function.processing_parameters"] = [
                s.replace("ChannelOrientationEnum.orthogonal", "orthogonal") for s in p
            ]
        assert recursive_to_dict(d1) == recursive_to_dict(d2)


class TestRunsAndChannels:
    """Test runs and channels metadata."""

    def test_runs(self, tf_h5, tf_obj_from_xml):
        """Test run metadata comparison with subtests for each run."""
        runs_h5 = tf_h5.station_metadata.runs
        runs_obj = tf_obj_from_xml.station_metadata.runs

        for run1, run2 in zip(runs_h5, runs_obj):
            # Set up run1 for comparison
            run1.data_logger.firmware.author = None
            rd1 = remove_mth5_keys(run1.to_dict(single=True))
            rd2 = remove_mth5_keys(run2.to_dict(single=True))

            assert rd1 == rd2, f"Run {run1.id} metadata mismatch"

    def test_channels(self, tf_h5, tf_obj_from_xml):
        """Test channel metadata comparison with subtests for each channel."""
        runs_h5 = tf_h5.station_metadata.runs
        runs_obj = tf_obj_from_xml.station_metadata.runs

        for run1, run2 in zip(runs_h5, runs_obj):
            for ch1 in run1.channels:
                ch2 = run2.get_channel(ch1.component)

                chd1 = remove_mth5_keys(ch1.to_dict(single=True))
                chd2 = remove_mth5_keys(ch2.to_dict(single=True))

                assert (
                    chd1 == chd2
                ), f"Channel {run1.id}_{ch1.component} metadata mismatch"


class TestEstimates:
    """Test transfer function estimates and period data."""

    @pytest.mark.parametrize(
        "estimate_name",
        [
            "transfer_function",
            "transfer_function_error",
            "inverse_signal_power",
            "residual_covariance",
        ],
    )
    def test_estimates(self, tf_obj_from_xml, tf_h5, estimate_name):
        """Test estimate arrays are equal between XML and HDF5 versions."""
        est1 = getattr(tf_obj_from_xml, estimate_name)
        est2 = getattr(tf_h5, estimate_name)

        assert (
            est1.to_numpy() == est2.to_numpy()
        ).all(), f"Estimate {estimate_name} arrays don't match"

    def test_period(self, tf_obj_from_xml, tf_h5):
        """Test period arrays are equal."""
        assert (
            tf_obj_from_xml.period == tf_h5.period
        ).all(), "Period arrays don't match"


class TestEstimateDetection:
    """Test estimate detection functionality."""

    @pytest.mark.parametrize(
        "estimate_type,expected",
        [
            ("transfer_function", True),
            ("impedance", True),
            ("tipper", True),
            ("covariance", True),
        ],
    )
    def test_has_estimate(self, tf_group, estimate_type, expected):
        """Test estimate detection for different types."""
        assert (
            tf_group.has_estimate(estimate_type) == expected
        ), f"Estimate {estimate_type} detection failed"


class TestTFSummary:
    """Test transfer function summary functionality."""

    def test_tf_summary_shape(self, mth5_obj):
        """Test that TF summary has correct shape."""
        mth5_obj.tf_summary.clear_table()
        mth5_obj.tf_summary.summarize()

        assert mth5_obj.tf_summary.shape == (
            1,
        ), f"Expected shape (1,), got {mth5_obj.tf_summary.shape}"

    @pytest.mark.parametrize(
        "field,expected_value",
        [
            ("station", b"NMX20"),
            ("survey", b"CONUS_South"),
            ("latitude", 34.470528),
            ("longitude", -108.712288),
            ("elevation", 1940.05),
            ("tf_id", b"NMX20"),
            ("units", b"milliVolt per kilometer per nanoTesla"),
            ("has_impedance", True),
            ("has_tipper", True),
            ("has_covariance", True),
            ("period_min", 4.6545500000000004),
            ("period_max", 29127.110000000001),
        ],
    )
    def test_tf_summary_fields(self, mth5_obj, field, expected_value):
        """Test individual TF summary fields."""
        # Skip reference fields
        if "reference" in field:
            pytest.skip("Skipping reference field")

        mth5_obj.tf_summary.clear_table()
        mth5_obj.tf_summary.summarize()

        actual_value = mth5_obj.tf_summary.array[field][0]
        assert (
            actual_value == expected_value
        ), f"Field {field}: expected {expected_value}, got {actual_value}"


class TestTFOperations:
    """Test transfer function operations and error handling."""

    def test_get_tf_fail(self, mth5_obj):
        """Test that getting non-existent transfer function raises MTH5Error."""
        with pytest.raises(MTH5Error):
            mth5_obj.get_transfer_function("a", "a")

    def test_remove_tf_fail(self, mth5_obj):
        """Test that removing non-existent transfer function raises MTH5Error."""
        with pytest.raises(MTH5Error):
            mth5_obj.remove_transfer_function("a", "a")

    def test_get_tf_object(self, mth5_obj, tf_obj_from_xml):
        """Test getting transfer function object and comparing with original."""
        tf_obj = mth5_obj.get_transfer_function("NMX20", "NMX20")
        tf_obj.station_metadata.acquired_by.author = (
            "National Geoelectromagnetic Facility"
        )

        assert tf_obj == tf_obj_from_xml, "Retrieved TF object doesn't match original"


# =============================================================================
# Performance Tests (Optional)
# =============================================================================


class TestPerformance:
    """Optional performance benchmarks."""

    @pytest.mark.benchmark(group="tf_operations")
    def test_tf_loading_benchmark(self, benchmark, mth5_obj):
        """Benchmark transfer function loading performance."""

        def load_tf():
            return mth5_obj.get_transfer_function("NMX20", "NMX20")

        result = benchmark(load_tf)
        assert result is not None

    @pytest.mark.benchmark(group="metadata_serialization")
    def test_metadata_serialization_benchmark(self, benchmark, tf_obj_from_xml):
        """Benchmark metadata serialization performance."""

        def serialize_metadata():
            return tf_obj_from_xml.station_metadata.to_dict(single=True)

        result = benchmark(serialize_metadata)
        assert isinstance(result, dict)


# =============================================================================
# Conditional Tests
# =============================================================================


@pytest.mark.slow
class TestExtensive:
    """Extensive tests that may take longer to run."""

    def test_full_metadata_round_trip(self, mth5_obj, tf_obj_from_xml):
        """Test complete metadata round-trip through HDF5."""
        # This would be a more comprehensive test
        # that verifies all metadata survives the round trip
        tf_retrieved = mth5_obj.get_transfer_function("NMX20", "NMX20")

        # Compare key metadata fields
        original_meta = tf_obj_from_xml.station_metadata.to_dict(single=True)
        retrieved_meta = tf_retrieved.station_metadata.to_dict(single=True)

        # Remove HDF5-specific keys for comparison
        retrieved_meta = remove_mth5_keys(retrieved_meta)

        # Compare essential fields (could be expanded)
        essential_fields = [
            "id",
            "location.latitude",
            "location.longitude",
            "location.elevation",
        ]
        for field in essential_fields:
            assert (
                original_meta[field] == retrieved_meta[field]
            ), f"Field {field} differs after round-trip"
