"""Utilities for HDF5 maintenance tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py


PathLike = Union[str, Path]


def repack_hdf5(
    input_file: PathLike,
    output_file: PathLike,
    *,
    overwrite: bool = False,
) -> Path:
    """Repack an HDF5 file by copying all objects into a new file.

    Repacking can reduce file size when metadata or datasets were repeatedly
    modified and the source file contains unused internal space.

    This mirrors the core idea of ``h5repack``: rewrite all objects into a new
    HDF5 container so only live content remains. It does *not* expose the full
    ``h5repack`` feature set (for example: filter/layout transformation,
    chunk-size retuning, and advanced option flags).

    Parameters
    ----------
    input_file : str | pathlib.Path
        Source HDF5 file to repack.
    output_file : str | pathlib.Path
        Destination path for the repacked HDF5 file.
    overwrite : bool, default=False
        If ``True``, overwrite ``output_file`` when it exists.

    Returns
    -------
    pathlib.Path
        The output path.

    Raises
    ------
    FileNotFoundError
        If ``input_file`` does not exist.
    FileExistsError
        If ``output_file`` exists and ``overwrite`` is ``False``.

    Examples
    --------
    >>> from mth5.utils.h5_tools import repack_hdf5
    >>> repacked = repack_hdf5("survey_original.h5", "survey_repacked.h5", overwrite=True)
    >>> print(repacked)
    survey_repacked.h5
    """

    source_path = Path(input_file)
    target_path = Path(output_file)

    if not source_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {source_path}")

    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {target_path}. "
            "Set overwrite=True to replace it."
        )

    if target_path.exists() and overwrite:
        target_path.unlink()

    with h5py.File(source_path, "r") as source_h5, h5py.File(
        target_path, "w"
    ) as target_h5:
        # Copy file-level attributes.
        for key, value in source_h5.attrs.items():
            target_h5.attrs[key] = value

        # Copy root objects recursively (groups, datasets, attributes).
        for name in source_h5.keys():
            source_h5.copy(name, target_h5, name=name)

    return target_path
