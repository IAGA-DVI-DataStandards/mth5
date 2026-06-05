import h5py
import numpy as np

from mth5.groups.base import BaseGroup
from mth5.groups.channel_dataset import ChannelDataset, ElectricDataset
from mth5.groups.estimate_dataset import EstimateDataset
from mth5.groups.fc_dataset import FCChannelDataset
from mth5.groups.feature_dataset import FeatureChannelDataset


def _fail_write(*args, **kwargs):
    raise AssertionError("Unexpected metadata write for unchanged metadata")


def test_estimate_dataset_skips_unchanged_metadata_write(tmp_path, monkeypatch):
    """Regression: unchanged Estimate metadata should not be rewritten."""
    h5_path = tmp_path / "estimate_wrapper.h5"

    with h5py.File(h5_path, "w") as h5:
        dataset = h5.create_dataset(
            "estimate", data=np.zeros((2, 2, 2), dtype=np.complex128)
        )
        EstimateDataset(dataset)

    monkeypatch.setattr(EstimateDataset, "write_metadata", _fail_write)

    with h5py.File(h5_path, "a") as h5:
        dataset = h5["estimate"]
        EstimateDataset(dataset, write_metadata=True)


def test_channel_dataset_skips_unchanged_metadata_write(tmp_path, monkeypatch):
    """Regression: unchanged Channel metadata should not be rewritten."""
    h5_path = tmp_path / "channel_wrapper.h5"

    with h5py.File(h5_path, "w") as h5:
        dataset = h5.create_dataset("ex", data=np.zeros(16, dtype=np.float32))
        ElectricDataset(dataset)

    monkeypatch.setattr(ChannelDataset, "write_metadata", _fail_write)

    with h5py.File(h5_path, "a") as h5:
        dataset = h5["ex"]
        ElectricDataset(dataset, write_metadata=True)


def test_feature_dataset_skips_unchanged_metadata_write(tmp_path, monkeypatch):
    """Regression: unchanged Feature metadata should not be rewritten."""
    h5_path = tmp_path / "feature_wrapper.h5"

    with h5py.File(h5_path, "w") as h5:
        dataset = h5.create_dataset(
            "feature_ex", data=np.zeros((4, 8), dtype=np.float64)
        )
        FeatureChannelDataset(dataset)

    monkeypatch.setattr(FeatureChannelDataset, "write_metadata", _fail_write)

    with h5py.File(h5_path, "a") as h5:
        dataset = h5["feature_ex"]
        FeatureChannelDataset(dataset, write_metadata=True)


def test_fc_dataset_skips_unchanged_metadata_write(tmp_path, monkeypatch):
    """Regression: unchanged FC metadata should not be rewritten."""
    h5_path = tmp_path / "fc_wrapper.h5"

    with h5py.File(h5_path, "w") as h5:
        dataset = h5.create_dataset("fc_ex", data=np.zeros((4, 8), dtype=np.complex128))
        FCChannelDataset(dataset)

    monkeypatch.setattr(FCChannelDataset, "write_metadata", _fail_write)

    with h5py.File(h5_path, "a") as h5:
        dataset = h5["fc_ex"]
        FCChannelDataset(dataset, write_metadata=True)


def test_base_group_skips_unchanged_metadata_attr_updates(tmp_path, monkeypatch):
    """Regression: BaseGroup.write_metadata should skip unchanged attributes."""
    h5_path = tmp_path / "base_group.h5"

    with h5py.File(h5_path, "w") as h5:
        group = h5.create_group("Base")
        wrapper = BaseGroup(group)
        wrapper.metadata.id = "base"
        wrapper.write_metadata()

        attrs_type = type(wrapper.hdf5_group.attrs)
        monkeypatch.setattr(attrs_type, "create", _fail_write)
        monkeypatch.setattr(attrs_type, "modify", _fail_write)

        wrapper.write_metadata()
