# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:36:26 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from loguru import logger


# =============================================================================


class ClientBase:
    def __init__(
        self,
        data_path: Union[str, Path],
        sample_rates: Sequence[float] = [1],
        save_path: Optional[Union[str, Path]] = None,
        mth5_filename: str = "from_client.h5",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ClientBase.

        Any h5 parameters should be in kwargs as `h5_parameter_name`.

        Parameters
        ----------
        data_path : str or Path
            Directory where data files are located.
        sample_rates : Sequence[float], optional
            List of sample rates to look for. Default is [1].
        save_path : str or Path, optional
            Directory to save the mth5 file. If None, uses data_path.
        mth5_filename : str, optional
            Name of the mth5 file to create. Default is 'from_client.h5'.
        **kwargs : Any
            Additional keyword arguments for h5 parameters.

        Examples
        --------
        >>> client = ClientBase(data_path="./data", sample_rates=[1, 8, 256])
        >>> client.save_path
        PosixPath('data/from_client.h5')
        """
        self.logger = logger

        self.data_path = data_path
        self.mth5_filename = mth5_filename
        self.sample_rates = sample_rates
        self.save_path = save_path

        self.interact = False

        self.mth5_version = "0.2.0"
        self.h5_compression = "gzip"
        self.h5_compression_opts = 4
        self.h5_shuffle = True
        self.h5_fletcher32 = True
        self.h5_data_level = 1
        self.mth5_file_mode = "a"

        self.collection: Optional[Any] = None

        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

    @property
    def h5_kwargs(self) -> dict[str, Any]:
        """
        Dictionary of HDF5 keyword arguments for file creation.

        Returns
        -------
        dict
            Dictionary of HDF5 file creation parameters.

        Examples
        --------
        >>> client = ClientBase(data_path="./data")
        >>> client.h5_kwargs["compression"]
        'gzip'
        """
        h5_params = dict(
            file_version=self.mth5_version,
            compression=self.h5_compression,
            compression_opts=self.h5_compression_opts,
            shuffle=self.h5_shuffle,
            fletcher32=self.h5_fletcher32,
            data_level=self.h5_data_level,
        )

        for key, value in self.__dict__.items():
            if key.startswith("h5"):
                h5_params[key[3:]] = value

        return h5_params

    @property
    def data_path(self) -> Path:
        """
        Path to data directory.

        Returns
        -------
        Path
            Path to the data directory.

        Examples
        --------
        >>> client = ClientBase(data_path="./data")
        >>> client.data_path
        PosixPath('data')
        """
        return self._data_path

    @data_path.setter
    def data_path(self, value: Union[str, Path]) -> None:
        """
        Set the data path.

        Parameters
        ----------
        value : str or Path
            Directory where files are located.

        Raises
        ------
        IOError
            If the path does not exist.
        ValueError
            If value is None.

        Examples
        --------
        >>> client = ClientBase(data_path="./data")
        >>> client.data_path
        PosixPath('data')
        """
        if value is not None:
            self._data_path = Path(value)
            if not self._data_path.exists():
                raise IOError(f"Could not find {self._data_path}")
        else:
            raise ValueError("data_path cannot be None")

    @property
    def sample_rates(self) -> list[float]:
        """
        List of sample rates to look for.

        Returns
        -------
        list of float
            Sample rates.

        Examples
        --------
        >>> client = ClientBase(data_path="./data", sample_rates=[1, 8, 256])
        >>> client.sample_rates
        [1.0, 8.0, 256.0]
        """
        return self._sample_rates

    @sample_rates.setter
    def sample_rates(self, value: Union[int, float, str, Sequence[float]]) -> None:
        """
        Set the sample rates as a list of floats.

        Parameters
        ----------
        value : int, float, str, or Sequence[float]
            Sample rates to set. If str, should be comma-separated.

        Raises
        ------
        TypeError
            If value cannot be parsed.

        Examples
        --------
        >>> client = ClientBase(data_path="./data", sample_rates="1,8,256")
        >>> client.sample_rates
        [1.0, 8.0, 256.0]
        """
        if isinstance(value, (int, float)):
            self._sample_rates = [float(value)]
        elif isinstance(value, str):
            self._sample_rates = [float(v) for v in value.split(",")]
        elif isinstance(value, (tuple, list)):
            self._sample_rates = [float(v) for v in value]
        else:
            raise TypeError(f"Cannot parse {type(value)}")

    @property
    def save_path(self) -> Path:
        """
        Path to save the mth5 file.

        Returns
        -------
        Path
            Full path to the mth5 file.

        Examples
        --------
        >>> client = ClientBase(data_path="./data", save_path="./output", mth5_filename="test.h5")
        >>> client.save_path
        PosixPath('output/test.h5')
        """
        return self._save_path.joinpath(self.mth5_filename)

    @save_path.setter
    def save_path(self, value: Optional[Union[str, Path]]) -> None:
        """
        Set the path to save the mth5 file.

        Parameters
        ----------
        value : str or Path, optional
            Directory or file path to save the mth5 file. If None, uses data_path.

        Examples
        --------
        >>> client = ClientBase(data_path="./data", save_path="./output/test.h5")
        >>> client.save_path
        PosixPath('output/test.h5')
        """
        if value is not None:
            value = Path(value)
            if value.exists():
                if value.is_dir():
                    self._save_path = value
                elif value.is_file():
                    self._save_path = value.parent
                    self.mth5_filename = value.name
            else:
                if "." in value.name:
                    self._save_path = value.parent
                    self.mth5_filename = value.name
                else:
                    self._save_path = value
                self._save_path.mkdir(exist_ok=True)
        else:
            self._save_path = self.data_path

    def get_run_dict(self) -> Any:
        """
        Get run information from the collection.

        Returns
        -------
        Any
            Run information as returned by the collection's get_runs method.

        Examples
        --------
        >>> client = ClientBase(data_path="./data")
        >>> client.collection = ...  # assign a collection with get_runs method
        >>> client.get_run_dict()
        ...
        """

        if self.collection is None:
            raise ValueError("Collection is not set.")
        return self.collection.get_runs(sample_rates=self.sample_rates)
