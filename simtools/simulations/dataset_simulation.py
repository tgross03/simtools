import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from pyvisgen.simulation.observation import Observation
from pyvisgen.simulation.visibility import vis_loop
from pyvisgen.fits.writer import create_hdu_list

import astropy

import h5py
import toml

from datetime import datetime

from tqdm.notebook import tqdm

import logging

astropy.utils.iers.conf.iers_degraded_accuracy = "warn"

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

    def __init__(
        self, data_path: str, required_pattern: str or None = None, image_key: str = "y"
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
                raise IOError(
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

    def _get_index(self, i):
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

    def __getitem__(self, i):
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

    def get_image(self, batch_idx, in_batch_idx):
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

    def plot_models(self, idx=[0, 6], norm=None, fig=None, ax=None):
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


class DatasetSimulation:
    def __init__(self, dataset, config):
        self._ds = dataset
        self.models = dataset.models
        self.metadata = dataset.metadata
        self.config = toml.load(config)["sampling_options"]

    def _random_date(self, time_range, rng):
        if len(time_range) == 1 or isinstance(time_range, str):
            return time_range

        start_time_l = datetime.strptime(time_range[0], "%d-%m-%Y %H:%M:%S")
        start_time_h = datetime.strptime(time_range[1], "%d-%m-%Y %H:%M:%S")
        start_times = pd.date_range(start_time_l, start_time_h, freq="1h").strftime(
            "%d-%m-%Y %H:%M:%S"
        )
        return rng.choice(
            [datetime.strptime(time, "%d-%m-%Y %H:%M:%S") for time in start_times]
        )

    def simulate_dataset(
        self,
        out,
        out_prefix="vis",
        batch_size="auto",
        start_index=0,
        end_index=None,
        fov_multiplier=1,
        show_individual_progress=False,
        generate_config=True,
        overwrite=True,
        verbose=False,
        obs_only=False,
    ):
        self.out = Path(out)
        self.observations = []

        for i in tqdm(
            np.arange(
                start_index, self.models.shape[0] if end_index is None else end_index
            ),
            desc="Dataset Simulation",
        ):
            model = self.models[i]
            mdata = self.metadata[i]

            rng = np.random.default_rng(self.config["seed"])

            obs_data = dict(
                image_size=model.shape[1],
                fov=mdata["cell_size"] * model.shape[1] * fov_multiplier,
                src_ra=mdata["src_ra"],
                src_dec=mdata["src_dec"],
                start_time=self._random_date(self.config["scan_start"], rng),
                scan_duration=rng.integers(
                    self.config["scan_duration"][0], self.config["scan_duration"][1]
                ),
                num_scans=self.config["num_scans"],
                scan_separation=self.config["scan_separation"],
                integration_time=self.config["corr_int_time"],
                ref_frequency=self.config["ref_frequency"],
                frequency_offsets=self.config["frequency_offsets"],
                bandwidths=self.config["bandwidths"],
                array_layout=self.config["layout"],
                corrupted=self.config["corrupted"],
                device=self.config["device"],
                dense=self.config["mode"] == "dense",
                sensitivity_cut=self.config["sensitivity_cut"],
            )

            if verbose:
                print(
                    "------------------------"
                    + f"\nid={mdata['index']}\n\n  METADATA -> {obs_data}\n\n  PARAMETERS -> {self._ds.parameters}\n"
                )
            obs = Observation(**obs_data)
            self.observations.append(obs)

            if obs_only:
                return

            vis_data = vis_loop(
                obs,
                torch.from_numpy(model)[None],
                noisy=self.config["noisy"],
                mode=self.config["mode"],
                batch_size=batch_size,
                show_progress=show_individual_progress,
                normalize=True,
            )

            hdu_list = create_hdu_list(vis_data, obs)

            self.out.mkdir(parents=True, exist_ok=True)

            hdu_list.writeto(
                self.out / f"{out_prefix}{mdata['index']}.fits", overwrite=overwrite
            )

            if generate_config:
                with open(
                    self.out / f"{out_prefix}_config{mdata['index']}.toml", "w"
                ) as f:
                    toml.dump(dict(sampling_options=obs_data), f)

            torch.cuda.empty_cache()
