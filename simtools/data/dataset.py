import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path


import sys


from astropy.utils import iers

import h5py


import logging

import warnings

iers.conf.iers_degraded_accuracy = "warn"

torch._logging.set_logs(
    dynamo=logging.CRITICAL, aot=logging.CRITICAL, inductor=logging.CRITICAL
)

VALID_UNIT_PREFIXES = {"K": 3, "M": 6, "G": 9, "T": 12, "P": 15}
CACHE_CLEANING_POLICIES = ["oldest", "youngest", "largest"]


class DataCache:
    """
    Creates a new data cache for files in the dataset.

    The cache records are structured as follows:
    records = { UNIQUE_ID: { data: DATA, size: MEMORY_SIZE }, ... }

    The data has to have the following form:
    DATA = (torch.tensor, list[dict], dict)

    Parameters
    ----------

    max_size: str or int
        The maximum memory size of the file cache. Can be given as bytes (int)
        or a string (e.g. `"10G"` or `"1M"`).

    cleaning_policy: str
        The way the program will remove files from the cache if it has reached its
        memory limit.
        You can choose from:
            1. `oldest` -> the oldest file in the cache will be removed
            2. `youngest` -> the youngest file in the cache will be removed
            3. `largest` -> the largest file in the cache will be removed

    """

    def __init__(self, max_size, cleaning_policy):
        self._records = dict()
        self._memsize = 0
        self._timeline = np.array([], dtype=int)

        match max_size:
            case str():
                if max_size[-1] not in VALID_UNIT_PREFIXES:
                    raise ValueError(
                        f"The valid unit prefixes are: {VALID_UNIT_PREFIXES.keys()}"
                    )

                self._max_size = (
                    int(max_size[:-1]) * 10 ** (VALID_UNIT_PREFIXES[max_size[-1]])
                )
            case int():
                self._max_size = max_size
            case _:
                raise TypeError(
                    "Only str or float are valid types for the maximum size!"
                )

        if cleaning_policy not in CACHE_CLEANING_POLICIES:
            raise ValueError(
                f"The given cleaning policy does not exist! Valid values are {CACHE_CLEANING_POLICIES}"
            )

        self.cleaning_policy = cleaning_policy

    def __getitem__(self, i):
        return self._records[str(i)] if str(i) in self._records else None

    def __len__(self):
        return len(self._records)

    def add(self, uid, data):
        record = dict(
            data=data,
            size=int(
                data[0].nelement() * data[0].element_size()
                + sys.getsizeof(data[0])
                + np.sum([sys.getsizeof(d) for d in data[1]])
                + sys.getsizeof(data[1])
                + sys.getsizeof(data[2])
            ),
        )

        MAX_ITER = 10
        while self._memsize + record["size"] > self._max_size:
            if MAX_ITER <= 0:
                warnings.warn(
                    f"The record with the uid {uid} could not be cached because it is too large! "
                    f"Current cache size: {self._memsize}"
                )

            self.clean()
            MAX_ITER -= 1

        self._timeline = np.append(self._timeline, uid)
        self._records[str(uid)] = record
        self._memsize += record["size"]

    def clean(self):
        uid = 0
        match self.cleaning_policy:
            case "oldest":
                uid = self._timeline[0]
            case "youngest":
                uid = self._timeline[-1]
            case "largest":
                largest = [0, 0]
                for id, record in self._records.items():
                    if record["size"] > largest[1]:
                        largest[0] = int(id)
                        largest[1] = record["size"]
                uid = largest[0]

        record = self._records[str(uid)]

        self._timeline = np.delete(self._timeline, np.where(self._timeline == uid))
        self._memsize -= record["size"]
        self._records.pop(str(uid), None)

    def clear(self):
        del self._records
        del self._timeline
        del self._memsize

        return self(self.max_size, self.cleaning_policy)


class Dataset:
    """

    Initialize a dataset containing true models from one or multiple h5-files.

    Parameters
    ----------

    data_path : str
        The path to the HDF5-file or a folder with HDF5-files containing the data in three datasets:
            1. "y" -> containing the stokes i images
            2. "metadata" -> containing metadata of the observations (e.g. pointings etc.)
            3. "params" -> containing the simulation parameters of the models
        IMPORTANT: If the path is a directory with HDF5-files, all the selected files have to
        be the same length! Otherwise this can lead to unexpected behavior!

    required_pattern : str or None, optional
        String pattern to include (in case a directory with h5 files is given.
        (e.g. `"*train*"` -> only files containing the phrase train somewhere in
        their names will be loaded). If set to `None`, the pattern will be "*".

    cache_loaded: bool, optional
        Whether to save HDF5 files if they are loaded to reduce time consumption
        of loading and unloading the same file multiple times.
        Default is `True`.

        WARNING: This will increase the memory consumption of the software.
        Additionally this means, that modifying files in runtime of the code
        is likely not reflected in the results since they are not loaded from disk
        but from memory.

    max_cache_size: str or int, optional
        The maximum memory size of the file cache. Can be given as bytes (int)
        or a string (e.g. `"10G"` or `"1M"`). The smaller this number, the less
        impact the caching will have on the runtime of the code.

    cache_cleaning_policy: str, optional
        The way the program will remove files from the cache if it has reached its
        memory limit.
        You can choose from:
            1. `oldest` -> the oldest file in the cache will be removed
            2. `youngest` -> the youngest file in the cache will be removed
            3. `largest` -> the largest file in the cache will be removed

    """

    def __init__(
        self,
        data_path: str,
        required_pattern: str or None = None,
        cache_loaded: bool = True,
        max_cache_size: str = "12G",
        cache_cleaning_policy: str = "oldest",
    ):
        self.data_path = Path(data_path)
        if not (self.data_path.is_dir() or self.data_path.is_file()):
            raise IOError("The provided path neither contains a director nor a file!")

        pattern = required_pattern if required_pattern is not None else "*"

        if pattern[-3:] != ".h5":
            pattern += ".h5"

        if self.data_path.is_dir():
            p = self.data_path.glob(pattern)
            self._file_paths = [x for x in p if x.is_file()]

            if len(self._file_paths) == 0:
                raise FileNotFoundError(
                    "The provided directory does not "
                    "contain files matching the pattern!"
                )

        else:
            if self.data_path.name[-3:] != ".h5":
                raise IOError("The given file is not a HDF5 file!")

            self._file_paths = [self.data_path]

        with h5py.File(self._file_paths[0], "r") as hf:
            self._batch_size = len(hf["y"])

        self._batch_num = len(self._file_paths)

        if cache_loaded:
            self._cache = DataCache(max_cache_size, cache_cleaning_policy)
        else:
            self._cache = None

    """
    Gets the indices of a specific image of the dataset as a tuple
    of (index of batch, index in batch).

    Parameters
    ----------

    i : int
        The index of the image

    Returns
    -------

    tuple[int, int]:
        The index of the batch of the image and the index of the image in the batch

    """

    def _get_batch_indices(self, i: int):
        return i // self._batch_size, i % self._batch_size

    def _get_index(self, batch_idx: int, in_batch_idx: int):
        return batch_idx * (self._batch_size - 1) + in_batch_idx

    """
    Gets the length of the dataset.

    Returns
    -------
    int:
        The length of the dataset.
    """

    def __len__(self):
        return self._batch_size * self._batch_num

    """
    Gets the image data, metadata and parameters associated with the index.

    Parameters
    ----------

    i : int
        The index of the image

    Returns
    -------
    tuple[`torch.Tensor`, list(dict), dict]
        The image data, metadata and simulation parameters for this index.

    """

    def __getitem__(self, i: int):
        idx = self._get_batch_indices(i)
        return self.get_image(idx[0], idx[1])

    """
    Gets the image data, metadata and parameters associated with the index.

    Parameters
    ----------

    batch_idx : int
        The index of the batch

    in_batch_idx : int
        The index of the image inside the batch

    Returns
    -------
    tuple[`torch.Tensor`, list(dict), dict]
        The image data, metadata and simulation parameters for these indices.

    """

    def get_image(self, batch_idx: int, in_batch_idx: int):
        idx = self._get_index(batch_idx, in_batch_idx)

        if self._cache is not None:
            cached_data = self._cache[idx]
            if cached_data is not None:
                return cached_data["data"]

        with h5py.File(self._file_paths[batch_idx], "r") as hf:
            data = (
                torch.from_numpy(hf["y"][()])[in_batch_idx],
                list(eval(hf["metadata"].asstr()[()]))[in_batch_idx],
                eval(hf["params"].asstr()[()]),
            )

            if self._cache is not None:
                self._cache.add(idx, data)

            return data

    """
    Plots the images contained in the dataset.

    Parameters
    ----------

    idx : tuple or array_like, optional
        The slice of indices of images in the dataset to be plotted
        (e.g. [0, -1] will plot every image except the last one).
        An index with value `None` denotes the last possible index
        (e.g. [None, None] will only plot the last image in the set).

    norm : norm from :class:`matplotlib.colors`, optional
        The norm to be applied to the images.

    fig : matplotlib.figure.Figure, optional
        A figure to put the plot into

    ax : matplotlib.axes._axes.Axes, optional
        A axis to put the plot into

    """

    def plot_models(self, idx: tuple[int, int] = [0, 6], norm=None, fig=None, ax=None):
        if None in (fig, ax) and not all(x is None for x in (fig, ax)):
            raise KeyError(
                "The parameters ax and fig have to be both None or not None!"
            )

        if idx[0] is None:
            idx[0] = self.__len__() - 1

        if idx[1] is None:
            idx[1] = self.__len__()

        if idx[0] < 0:
            idx[0] = self.__len__() + idx[0]

        if idx[1] < 0:
            idx[1] = self.__len__() + idx[1]

        min_idx = idx[0]
        max_idx = np.min([idx[1], self.__len__()])

        if min_idx > max_idx:
            raise ValueError("The lower index may not be larger than the upper index!")

        length = max_idx - min_idx

        if length == 0:
            raise ValueError("Cannot plot list of models with length 0!")

        if ax is None:
            fig, ax = plt.subplots(
                int(np.ceil(length / 3)),
                3 if length >= 3 else length,
                sharex=True,
                sharey=True,
                layout="constrained",
            )

        ax = np.ravel(ax)

        if length % 3 != 0 and length > 3:
            off_axes = ax[-(3 - length % 3) :]
            for off_ax in off_axes:
                off_ax.set_axis_off()

        models = []
        for i in range(min_idx, max_idx):
            models.append(self[i][0])

        for i in range(0, length):
            axi = ax[i]
            model = models[i]

            axi.imshow(model, cmap="inferno", norm=norm, origin="lower")

            if i % 3 == 0:
                axi.set_ylabel("pixels")

            if (
                i >= length - length % 3
                or (i >= length - 3 and length % 3 == 0)
                or length < 3
            ):
                axi.set_xlabel("pixels")
