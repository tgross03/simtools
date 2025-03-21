import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path


from astropy.utils import iers

import h5py


import logging

iers.conf.iers_degraded_accuracy = "warn"

torch._logging.set_logs(
    dynamo=logging.CRITICAL, aot=logging.CRITICAL, inductor=logging.CRITICAL
)


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

    auto_load : bool, optional
        If set to `True`, the dataset is loaded in the constructor. Default is `True`.

    """

    def __init__(self, data_path: str, required_pattern: str or None = None):
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

    def _get_index(self, i: int):
        return i // self._batch_size, i % self._batch_size

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
        idx = self._get_index(i)
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
        with h5py.File(self._file_paths[batch_idx], "r") as hf:
            return (
                torch.from_numpy(hf["y"][()])[in_batch_idx],
                list(eval(hf["metadata"].asstr()[()]))[in_batch_idx],
                eval(hf["params"].asstr()[()]),
            )

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
