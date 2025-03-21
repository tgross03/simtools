import numpy as np
import torch
import pandas as pd

from pathlib import Path

from pyvisgen.simulation.observation import Observation
from pyvisgen.simulation.visibility import vis_loop
from pyvisgen.fits.writer import create_hdu_list

from simtools.data import Dataset

from astropy import iers

import toml

from datetime import datetime

from tqdm.notebook import tqdm

import logging

iers.conf.iers_degraded_accuracy = "warn"

torch._logging.set_logs(
    dynamo=logging.CRITICAL, aot=logging.CRITICAL, inductor=logging.CRITICAL
)


class DatasetSimulation:
    """

    Initialize a pyvisgen radio interferometer simulation of a Dataset.

    Parameters
    ----------
    dataset: simtools.data.Dataset
        The Dataset to simulate

    config: str
        The TOML-configuration file containing the simulation parameters

    """

    def __init__(self, dataset: Dataset, config: str):
        self.dataset = dataset
        self.config = toml.load(config)["sampling_options"]

    """
    Returns a randomized date between a range of dates.

    Parameters
    ----------

    time_range: tuple[str, str]
        The range of datetimes as strings in the format `%d-%m-%Y %H:%M:%S`.

    rng: numpy.random.Generator
        The Random Generator to use for choosing the date.

    Returns
    -------

    datetime:
        The random datetime between the given datetime range

    """

    def _random_date(
        self,
        time_range: list[str, str] or tuple[str, str],
        rng: np.random.Generator,
    ):
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

    """
    Simulate (a selected subset of) the dataset.

    Parameters
    ----------

    out: str
        The directory to put the generated FITS files into

    out_prefix: str, optional
        The string to prepend to the output files' names

    batch_size: str or int, optional
        The amount of calculations to run in parallel. Use `"auto"` to
        let toma determine the best size automatically.

    start_index: int or None, optional
        The index of images to start at. None indicates that the last possible
        index is chosen.

    end_index: int or None, optional
        The index of images to end at. None indicates that the last possible
        index is chosen.

    fov_multiplier: float, optional
        A constant factor to multiply the simulated Field of View with.

    show_individual_progress: bool, optional
        Whether pyvisgen should show a progress bar for each visibility calculation.

    verbose: bool, optional
        Whether the programm should output additional information for each calculation.

    return_obs: bool, optional
        Whether to return a list of the created
        `pyvisgen.simulation.Observation.Observation` objects.

    obs_only: bool, optional
        Whether to only generate the `pyvisgen.simulation.Observation.Observation` objects
        and skip the visiblity simulations.

    Returns
    -------

    observations: list[pyvisgen.simulation.Observation.Observation]
        If the `obs_only` parameter is `True`, the method will return a list
        of the created observations.

    """

    def simulate_dataset(
        self,
        out: str,
        out_prefix: str = "vis",
        batch_size: str or int = "auto",
        start_index: int or None = 0,
        end_index: int or None = None,
        fov_multiplier: float = 1,
        show_individual_progress: bool = False,
        generate_config: bool = True,
        overwrite: bool = True,
        verbose: bool = False,
        return_obs: bool = False,
        obs_only: bool = False,
    ):
        out = Path(out)

        if not out.is_dir():
            raise NotADirectoryError("The provided out path is no directory!")

        observations = []

        for i in tqdm(
            np.arange(
                start_index, len(self.dataset) if end_index is None else end_index
            ),
            desc="Dataset Simulation",
        ):
            model, mdata, params = self.dataset[i]

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
                    f"\nid={mdata['index']}\n\n  "
                    f"METADATA -> {obs_data}\n\n  "
                    f"PARAMETERS -> {params}\n"
                )
            obs = Observation(**obs_data)

            if return_obs:
                observations.append(obs)

            if obs_only:
                return observations if return_obs else None

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

            out.mkdir(parents=True, exist_ok=True)

            batch_idx, img_idx = self.dataset._get_index(i)
            name = f"{out_prefix}_{self.dataset._file_paths[batch_idx].stem}"

            hdu_list.writeto(out / f"{name}_{img_idx}.fits", overwrite=overwrite)

            if generate_config:
                with open(out / f"{name}_config_{img_idx}.toml", "w") as f:
                    toml.dump(dict(sampling_options=obs_data), f)

            torch.cuda.empty_cache()

            return observations if return_obs else None
