# -*- coding: utf-8 -*-
"""
Base Group Class
====================

Contains all the base functions that will be used by group classes.

Created on Fri May 29 15:09:48 2020

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license:
    MIT
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import inspect
import weakref
from typing import Any, Type

import h5py
import numpy as np
from loguru import logger
from mt_metadata import timeseries as metadata
from mt_metadata.base import MetadataBase
from mt_metadata.features import (
    Feature,
    FeatureDecimationChannel,
    FeatureFCRun,
    FeatureTSRun,
)
from mt_metadata.processing.fourier_coefficients import Decimation, FC, FCChannel
from mt_metadata.transfer_functions.tf import TransferFunction

from mth5.helpers import (
    add_attributes_to_metadata_class_pydantic,
    get_tree,
    read_attrs_to_dict,
    to_numpy_type,
    validate_name,
)
from mth5.utils.exceptions import MTH5Error


# make a dictionary of available metadata classes
meta_classes = dict(inspect.getmembers(metadata, inspect.isclass))
meta_classes["TransferFunction"] = TransferFunction
meta_classes["FCDecimation"] = Decimation
meta_classes["FCChannel"] = FCChannel
meta_classes["FC"] = FC
meta_classes["Feature"] = Feature
meta_classes["FeatureTSRun"] = FeatureTSRun
meta_classes["FeatureFCRun"] = FeatureFCRun
meta_classes["FeatureDecimation"] = Decimation
meta_classes["FeatureDecimationChannel"] = FeatureDecimationChannel


# =============================================================================
#
# =============================================================================
class BaseGroup:
    """
    Base class for HDF5 group management with metadata handling.

    Provides core functionality for reading, writing, and managing HDF5 groups
    with integrated metadata validation using mt_metadata standards.

    Parameters
    ----------
    group : h5py.Group or h5py.Dataset
        HDF5 group or dataset object to wrap.
    group_metadata : MetadataBase, optional
        Metadata container with validated attributes. Default is None.
    **kwargs : dict
        Additional keyword arguments to set as instance attributes.

    Attributes
    ----------
    hdf5_group : h5py.Group or h5py.Dataset
        Weak reference to the underlying HDF5 group.
    metadata : MetadataBase
        Metadata object with validation and standards compliance.
    logger : loguru.Logger
        Logger instance for tracking operations.
    compression : str, optional
        HDF5 compression method (e.g., 'gzip').
    compression_opts : int, optional
        Compression options/level.
    shuffle : bool
        Enable HDF5 shuffle filter. Default is False.
    fletcher32 : bool
        Enable HDF5 Fletcher32 checksum. Default is False.

    Notes
    -----
    - All HDF5 group references are weak references to prevent lingering
      file references after the group is closed.
    - Metadata changes should be written using `write_metadata()` method.
    - This is a base class inherited by more specific group types like
      SurveyGroup, StationGroup, RunGroup, etc.

    Examples
    --------
    Create and manage a group with metadata

    >>> import h5py
    >>> with h5py.File('data.h5', 'r+') as f:
    ...     group = f.create_group('MyGroup')
    ...     base_obj = BaseGroup(group)
    ...     print(base_obj)
    ...     # Set and write metadata
    ...     base_obj.metadata.id = 'MyGroup'
    ...     base_obj.write_metadata()

    Access metadata and group structure

    >>> print(base_obj.metadata.id)
    'MyGroup'
    >>> print(base_obj.groups_list)
    ['subgroup1', 'subgroup2']
    >>> print(base_obj.hdf5_group.ref)  # Get HDF5 reference
    <HDF5 Group Reference>

    """

    def __init__(
        self,
        group: h5py.Group | h5py.Dataset,
        group_metadata: MetadataBase | None = None,
        **kwargs: Any,
    ) -> None:
        self.compression = None
        self.compression_opts = None
        self.shuffle = False
        self.fletcher32 = False
        self._has_read_metadata = False

        self.logger = logger

        # make sure the reference to the group is weak so there are no lingering
        # references to a closed HDF5 file.
        if group is not None and isinstance(group, (h5py.Group, h5py.Dataset)):
            self.hdf5_group = weakref.ref(group)()
        # initialize metadata
        self._initialize_metadata()

        # if metadata, make sure that its the same class type
        if group_metadata is not None:
            self.metadata = group_metadata

            # write out metadata to make sure that its in the file.
            self.write_metadata()

        # if any other keywords
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        """
        Generate a string representation of the group hierarchy.

        Returns
        -------
        str
            Tree structure of the HDF5 group and its contents, or error message
            if file is closed.

        Examples
        --------
        >>> print(base_obj)
        /MyGroup
            /subgroup1
                /dataset1
            /subgroup2
        """
        try:
            self.hdf5_group.ref

            return get_tree(self.hdf5_group)
        except ValueError:
            msg = "MTH5 file is closed and cannot be accessed."
            self.logger.warning(msg)
            return msg

    def __repr__(self) -> str:
        """
        Return the string representation of the group.

        Returns
        -------
        str
            String representation identical to __str__.
        """
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        """
        Check equality with another group.

        Parameters
        ----------
        other : object
            Another BaseGroup instance to compare with.

        Returns
        -------
        bool
            True if groups are equal, False otherwise.

        Raises
        ------
        MTH5Error
            Equality comparison is not yet implemented.

        Examples
        --------
        >>> group1 == group2
        MTH5Error: Cannot test equals yet
        """
        raise MTH5Error("Cannot test equals yet")

    # Iterate over key, value pairs
    def __iter__(self):
        """
        Iterate over key-value pairs in the HDF5 group.

        Yields
        ------
        tuple
            (name, object) pairs for each item in the group.

        Examples
        --------
        >>> for name, obj in base_obj:
        ...     print(f"{name}: {type(obj)}")
        subgroup1: <class 'h5py._hl.group.Group'>
        dataset1: <class 'h5py._hl.dataset.Dataset'>
        """
        return self.hdf5_group.items().__iter__()

    @property
    def _class_name(self) -> str:
        """
        Extract the base class name without 'Group' suffix.

        Returns
        -------
        str
            Class name (e.g., 'Survey', 'Station', 'Run').

        Examples
        --------
        >>> print(survey_obj._class_name)
        'Survey'
        """
        return self.__class__.__name__.split("Group")[0]

    def _initialize_metadata(self) -> None:
        """
        Initialize metadata object with custom attributes.

        Creates a metadata object of the appropriate type based on the class name
        and adds MTH5-specific attributes (mth5_type, hdf5_reference) for tracking.

        Notes
        -----
        This is called automatically during __init__. The metadata class is determined
        by matching self._class_name to the meta_classes dictionary. Falls back to
        MetadataBase if no specific class is found.

        Examples
        --------
        >>> # Called automatically during initialization
        >>> obj = SurveyGroup(hdf5_group)
        >>> print(type(obj._metadata))
        <class 'mt_metadata.timeseries.survey.Survey'>
        >>> print(obj._metadata.mth5_type)
        'Survey'
        """
        metadata_obj = MetadataBase
        if self._class_name not in ["Standards"]:
            try:
                metadata_obj = meta_classes[self._class_name]
            except KeyError:
                metadata_obj = MetadataBase
        # add 2 attributes that will help with querying using the new Pydantic approach
        self._metadata = add_attributes_to_metadata_class_pydantic(metadata_obj)

        # set mth5 specific parameters
        self._metadata.mth5_type = self._class_name
        self._metadata.hdf5_reference = self.hdf5_group.ref

    @property
    def metadata(self) -> MetadataBase:
        """
        Get metadata object with lazy loading from HDF5 attributes.

        Returns
        -------
        MetadataBase
            Metadata container with all attributes and validation.

        Notes
        -----
        Metadata is loaded on first access and cached for subsequent accesses.

        Examples
        --------
        >>> meta = base_obj.metadata
        >>> print(meta.id)
        'MyGroup'
        >>> print(meta.mth5_type)
        'Survey'
        """
        if not self._has_read_metadata:
            self.read_metadata()
        return self._metadata

    @metadata.setter
    def metadata(self, metadata_object: MetadataBase) -> None:
        """
        Set metadata with type validation.

        Parameters
        ----------
        metadata_object : MetadataBase
            Metadata container to set. Must be compatible with current class type.

        Raises
        ------
        MTH5Error
            If metadata_object is not compatible with the current class.

        Notes
        -----
        Direct field assignment is used to preserve complex objects like Provenance
        that may lose information during to_dict/from_dict conversion.

        Examples
        --------
        >>> from mt_metadata.timeseries import Survey
        >>> survey_meta = Survey()
        >>> survey_meta.id = 'NewSurvey'
        >>> survey_obj.metadata = survey_meta
        """
        if not isinstance(metadata_object, (type(self._metadata), MetadataBase)):
            msg = (
                f"Metadata must be of type {meta_classes[self._class_name]} "
                f"not {type(metadata_object)}"
            )
            self.logger.error(msg)
            raise MTH5Error(msg)

        # Instead of round-trip conversion, directly copy the metadata fields
        # to preserve complex objects like Provenance that may lose information
        # during to_dict/from_dict conversion
        if hasattr(metadata_object, "__dict__"):
            # For pydantic models, copy field values directly
            for field_name, field_value in metadata_object.__dict__.items():
                if hasattr(self._metadata, field_name):
                    setattr(self._metadata, field_name, field_value)
        else:
            # Fallback to the original conversion method
            self._metadata.from_dict(metadata_object.to_dict())

        # Note: mth5_type and hdf5_reference are set during field creation
        # They can be updated later if needed through the model's normal field assignment

    @property
    def groups_list(self) -> list[str]:
        """
        Get list of all subgroup names in the HDF5 group.

        Returns
        -------
        list of str
            Names of all subgroups and datasets.

        Examples
        --------
        >>> print(base_obj.groups_list)
        ['Station_001', 'Station_002', 'metadata']
        """
        return list(self.hdf5_group.keys())

    @property
    def dataset_options(self) -> dict[str, Any]:
        """
        Get the HDF5 dataset creation options.

        Returns
        -------
        dict
            Dictionary containing compression, shuffle, and checksum settings.

        Examples
        --------
        >>> options = base_obj.dataset_options
        >>> print(options)
        {'compression': 'gzip', 'compression_opts': 4,
         'shuffle': True, 'fletcher32': False}
        """
        return {
            "compression": self.compression,
            "compression_opts": self.compression_opts,
            "shuffle": self.shuffle,
            "fletcher32": self.fletcher32,
        }

    def read_metadata(self) -> None:
        """
        Read metadata from HDF5 group attributes into metadata object.

        Loads all HDF5 attributes and converts them to appropriate Python types
        before populating the metadata object with validation.

        Notes
        -----
        This method is called automatically on first metadata access if metadata
        has not been read yet. Empty attributes are skipped with a debug message.

        Examples
        --------
        Manually read metadata after file changes

        >>> base_obj.read_metadata()
        >>> print(base_obj.metadata.id)
        'MyGroup'

        Check what attributes were read

        >>> base_obj.read_metadata()
        >>> attrs = list(base_obj.metadata.to_dict().keys())
        >>> print(f"Attributes: {attrs}")
        Attributes: ['id', 'comments', 'provenance']
        """
        meta_dict = read_attrs_to_dict(dict(self.hdf5_group.attrs), self._metadata)
        # Defensive check: skip if meta_dict is empty
        if not meta_dict:
            self.logger.debug(
                f"No metadata found for {self._class_name}, skipping from_dict."
            )
            return
        self._metadata.from_dict({self._class_name: meta_dict})
        self._has_read_metadata = True

    def write_metadata(self) -> None:
        """
        Write metadata from object to HDF5 group attributes.

        Converts metadata values to numpy-compatible types before writing to
        HDF5 attributes. Handles read-only mode gracefully with warnings.

        Raises
        ------
        KeyError
            If HDF5 write fails for reasons other than read-only mode.
        ValueError
            If synchronous group creation fails for reasons other than read-only mode.

        Notes
        -----
        - Keys that already exist are overwritten.
        - Read-only files will log a warning instead of raising an error.
        - This method should be called after any metadata changes.

        Examples
        --------
        Update metadata and write to file

        >>> base_obj.metadata.id = 'UpdatedGroup'
        >>> base_obj.metadata.comments = 'New comments'
        >>> base_obj.write_metadata()

        Verify write by reloading

        >>> base_obj._has_read_metadata = False
        >>> base_obj.read_metadata()
        >>> print(base_obj.metadata.id)
        'UpdatedGroup'
        """
        try:
            for key, value in self.metadata.to_dict(single=True).items():
                value = to_numpy_type(value)
                if key in self.hdf5_group.attrs:
                    if self._values_equal(self.hdf5_group.attrs[key], value):
                        continue
                    self.hdf5_group.attrs.modify(key, value)
                else:
                    self.hdf5_group.attrs.create(key, value)
                self.logger.debug(f"wrote metadata {key} = {value}")
        except KeyError as key_error:
            if "no write intent" in str(key_error):
                self.logger.warning("File is in read-only mode, cannot write metadata.")
            else:
                raise KeyError(key_error)
        except ValueError as value_error:
            if "Unable to synchronously create group" in str(value_error):
                self.logger.warning("File is in read-only mode, cannot write metadata.")
            else:
                raise ValueError(value_error)

    @staticmethod
    def _values_equal(existing_value: Any, expected_value: Any) -> bool:
        """Compare HDF5 attribute values, including ndarray attributes."""
        if isinstance(existing_value, np.ndarray) or isinstance(
            expected_value, np.ndarray
        ):
            return np.array_equal(
                np.asarray(existing_value), np.asarray(expected_value)
            )

        if isinstance(existing_value, np.generic):
            existing_value = existing_value.item()
        if isinstance(expected_value, np.generic):
            expected_value = expected_value.item()

        return existing_value == expected_value

    def initialize_group(self, **kwargs: Any) -> None:
        """
        Initialize group by setting attributes and writing metadata.

        Convenience method that sets keyword arguments as instance attributes
        and writes all metadata to the HDF5 file.

        Parameters
        ----------
        **kwargs : dict
            Key-value pairs to set as instance attributes.

        Examples
        --------
        Initialize with compression settings

        >>> base_obj.initialize_group(
        ...     compression='gzip',
        ...     compression_opts=4,
        ...     shuffle=True
        ... )
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.write_metadata()

    def _add_group(
        self,
        name: str,
        group_class: Type,
        group_metadata: MetadataBase | None = None,
        match: str = "id",
    ) -> BaseGroup | None:
        """
        Add a new group to the HDF5 file.

        Creates a new subgroup with optional metadata validation. If the group
        already exists, returns the existing group without modification.

        Parameters
        ----------
        name : str
            Name of the group to create. Will be validated and normalized.
        group_class : type
            Group class to instantiate for the new group.
        group_metadata : MetadataBase, optional
            Metadata container with validated attributes. Default is None.
        match : str, optional
            Metadata field to match with group name. Default is 'id'.

        Returns
        -------
        BaseGroup or None
            Instance of group_class for the new/existing group, or None if
            file is in read-only mode.

        Raises
        ------
        MTH5Error
            If group name doesn't match group_metadata.id.

        Notes
        -----
        - Group name is validated and normalized via validate_name().
        - Weak HDF5 references are set automatically for tracking.
        - If group exists, log message indicates this and returns existing group.

        Examples
        --------
        Add a new group with metadata

        >>> from mt_metadata.timeseries import Station
        >>> station_meta = Station(id='MT_001')
        >>> station = survey_obj._add_group(
        ...     'MT_001',
        ...     StationGroup,
        ...     group_metadata=station_meta
        ... )
        >>> print(station.metadata.id)
        'MT_001'

        Add group without metadata

        >>> run_obj = station_obj._add_group(
        ...     'MT_001a',
        ...     RunGroup
        ... )

        Handle existing group

        >>> # If group exists, it returns the existing one
        >>> run1 = station_obj._add_group('MT_001a', RunGroup)
        >>> run2 = station_obj._add_group('MT_001a', RunGroup)  # Returns same group
        >>> run1 is run2
        True
        """
        name = validate_name(name)

        try:
            if group_metadata is not None:
                if validate_name(group_metadata.id) != name:
                    msg = (
                        f"{group_class.__name__} name {name} must be "
                        f"the same as group_metadata.{match} "
                        f"{group_metadata.id}"
                    )
                    self.logger.error(msg)
                    raise MTH5Error(msg)
            new_group = self.hdf5_group.create_group(name)
            return_obj = group_class(new_group, **self.dataset_options)
            if group_metadata is None:
                return_obj._metadata.update_attribute(match, name)
            else:
                return_obj.metadata = group_metadata
            # need to add the hdf5 reference to the metadata
            return_obj.metadata.hdf5_reference = new_group.ref
            return_obj.write_metadata()
            if hasattr(return_obj, "initialize_group"):
                return_obj.initialize_group()
        except ValueError as error:
            if "no write intent" in str(error):
                self.logger.warning(
                    f"File is in read-only mode, cannot create group {name}"
                )
                return
            elif "name already exists" in str(error):
                msg = (
                    f"{group_class.__name__} {name} already exists, "
                    "returning existing group."
                )
                self.logger.info(msg)
                return_obj = self._get_group(name, group_class)
        return return_obj

    def _get_group(self, name: str, group_class: Type) -> BaseGroup:
        """
        Get an existing group from the HDF5 file.

        Retrieves a subgroup by name, automatically reading its metadata.

        Parameters
        ----------
        name : str
            Name of the group to retrieve. Will be validated and normalized.
        group_class : type
            Group class to instantiate for the retrieved group.

        Returns
        -------
        BaseGroup
            Instance of group_class for the retrieved group.

        Raises
        ------
        MTH5Error
            If the group does not exist.

        Examples
        --------
        Get an existing station

        >>> station = survey_obj._get_group('MT_001', StationGroup)
        >>> print(station.metadata.id)
        'MT_001'

        Handle non-existent group

        >>> try:
        ...     station = survey_obj._get_group('NonExistent', StationGroup)
        ... except MTH5Error as e:
        ...     print(f"Group not found: {e}")
        Group not found: Error: NonExistent does not exist...
        """
        name = validate_name(name)
        try:
            # get the group and be sure to read the metadata
            group = group_class(self.hdf5_group[name], **self.dataset_options)
            group.read_metadata()
            return group
        except KeyError:
            msg = (
                f"Error: {name} does not exist, check groups_list for " "existing names"
            )
            self.logger.debug(msg)
            raise MTH5Error(msg)

    def _remove_group(self, name: str) -> None:
        """
        Remove a group from the HDF5 file.

        Deletes a subgroup by name. Note that this removes the reference in the
        HDF5 file but does not free the disk space (a limitation of HDF5 format).

        Parameters
        ----------
        name : str
            Name of the group to remove. Will be validated and normalized.

        Raises
        ------
        MTH5Error
            If the group does not exist or cannot be deleted.

        Warnings
        --------
        Removing a group does not reduce the HDF5 file size, it only removes
        the reference. To reclaim disk space, create a new file and copy
        the desired groups into it.

        Examples
        --------
        Remove a group

        >>> survey_obj._remove_group('MT_001')
        >>> print('MT_001' in survey_obj.groups_list)
        False

        Handle errors when group doesn't exist

        >>> try:
        ...     survey_obj._remove_group('NonExistent')
        ... except MTH5Error as e:
        ...     print(f"Cannot remove: {e}")
        Cannot remove: Error: NonExistent does not exist...
        """
        name = validate_name(name)
        try:
            del self.hdf5_group[name]
            self.logger.info(
                "Deleting a station does not reduce the HDF5"
                "file size it simply remove the reference. If "
                "file size reduction is your goal, simply copy"
                " what you want into another file."
            )
        except KeyError as key_error:
            if "Couldn't delete link" in str(key_error):
                self.logger.warning(f"File is in read-only mode, cannot delete {name}")
            else:
                msg = f"{name} does not exist. Check station_list for existing names"
                self.logger.debug(msg)
                raise MTH5Error(msg)

    def rename_group(self, new_name: str) -> None:
        """
        Rename the current group in the HDF5 file.

        Parameters
        ----------
        new_name : str
            New name for the group. Will be validated and normalized.

        Raises
        ------
        MTH5Error
            If renaming fails due to read-only mode or other issues.

        Examples
        --------
        Rename a group

        >>> print(survey_obj.hdf5_group.name)
        '/OldSurveyName'
        >>> survey_obj.rename_group('NewSurveyName')
        >>> print(survey_obj.hdf5_group.name)
        '/NewSurveyName'
        """
        new_name = validate_name(new_name)
        try:
            parent_group = self.hdf5_group.parent
            parent_group.move(self.hdf5_group.name, new_name)
            self.logger.info(f"Renamed group to {new_name}")
            # Update hdf5 reference in metadata
            self.metadata.hdf5_reference = self.hdf5_group.ref
        except ValueError as error:
            if "no write intent" in str(error):
                self.logger.warning("File is in read-only mode, cannot rename group.")
            else:
                msg = f"Failed to rename group to {new_name}: {error}"
                self.logger.error(msg)
                raise MTH5Error(msg)
