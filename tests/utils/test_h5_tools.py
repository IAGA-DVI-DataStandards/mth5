import h5py
import numpy as np
import pytest

from mth5.utils.h5_tools import repack_hdf5


def test_repack_hdf5_copies_structure_data_and_attributes(tmp_path):
    """Repacking should copy groups, datasets, and attrs into a new file."""
    source = tmp_path / "source.h5"
    target = tmp_path / "repacked.h5"

    with h5py.File(source, "w") as h5:
        h5.attrs["file_attr"] = "root"

        group = h5.create_group("level1")
        group.attrs["group_attr"] = 7

        data = np.arange(10, dtype=np.float64)
        dataset = group.create_dataset("values", data=data)
        dataset.attrs["unit"] = "counts"

    returned_path = repack_hdf5(source, target)
    assert returned_path == target
    assert target.exists()

    with h5py.File(target, "r") as h5:
        assert h5.attrs["file_attr"] == "root"
        assert "level1" in h5

        group = h5["level1"]
        assert group.attrs["group_attr"] == 7

        dataset = group["values"]
        np.testing.assert_array_equal(dataset[()], np.arange(10, dtype=np.float64))
        assert dataset.attrs["unit"] == "counts"


def test_repack_hdf5_raises_if_source_is_missing(tmp_path):
    """Missing source file should raise FileNotFoundError."""
    missing_source = tmp_path / "does_not_exist.h5"
    target = tmp_path / "output.h5"

    with pytest.raises(FileNotFoundError):
        repack_hdf5(missing_source, target)


def test_repack_hdf5_respects_overwrite_flag(tmp_path):
    """Output file should not be overwritten unless explicitly requested."""
    source = tmp_path / "source.h5"
    target = tmp_path / "output.h5"

    with h5py.File(source, "w") as h5:
        h5.create_dataset("x", data=np.array([1, 2, 3], dtype=np.int64))

    # First write creates target.
    repack_hdf5(source, target)

    # Second write should fail without overwrite.
    with pytest.raises(FileExistsError):
        repack_hdf5(source, target)

    # Overwrite should succeed.
    returned_path = repack_hdf5(source, target, overwrite=True)
    assert returned_path == target
