import numpy as np
import torch
import pandas as pd

from pathlib import Path

from pyvisgen.simulation.observation import Observation
from pyvisgen.simulation.visibility import vis_loop
from pyvisgen.fits.writer import create_hdu_list

import astropy

import toml

from datetime import datetime

from tqdm.notebook import tqdm

import logging

astropy.utils.iers.conf.iers_degraded_accuracy = "warn"

torch._logging.set_logs(
    dynamo=logging.CRITICAL, aot=logging.CRITICAL, inductor=logging.CRITICAL
)


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
