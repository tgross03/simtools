# IMPORTS

# python libraries
import subprocess

import os
import shutil
import warnings
import numpy as np
import datetime
import torch
import pandas as pd

import logging

torch._logging.set_logs(
    dynamo=logging.CRITICAL, aot=logging.CRITICAL, inductor=logging.CRITICAL
)

# PyYAML
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

yaml.add_representer(
    np.float64, lambda dumper, scalar: dumper.represent_float(float(scalar))
)
yaml.add_representer(
    pd._libs.tslibs.timestamps.Timestamp,
    lambda dumper, timestamp: dumper.represent_str(timestamp.isoformat()),
)

# matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm

# pathlib
from pathlib import Path

# astropy
from astropy.io import fits
from astropy.constants import c
import astropy.units as units

# pyvisgen
from pyvisgen.simulation.visibility import vis_loop
from pyvisgen.simulation.observation import Observation
import pyvisgen.fits.writer as writer


# radiotools
from radiotools.layouts import Layout
from radiotools.visibility import SourceVisibility

from radiotools.measurements import Measurement
from radiotools.gridding import Gridder

# radio_stats
from radio_stats.cuts.dyn_range import rms_cut
from radio_stats.cuts.dbscan_clean import dbscan_clean

# casatasks
from casatasks import simobserve, simanalyze


class Simulation:
    def __init__(
        self,
        project_name,
        skymodel,
        scan_duration,
        int_time,
        observatory,
        obs_date=None,
    ):
        self.project_name = project_name
        self._dirs = {}

        self._parent_dir = os.getcwd()
        self._project_dir = f"{self._parent_dir}/{self.project_name}"

        proj_path = Path(f"{self._parent_dir}/{self.project_name}")

        if not proj_path.is_dir():
            proj_path.mkdir()

        self.skymodel = skymodel

        self.config_casa = None
        self.config_pyvisgen = None

        self.metadata = skymodel.get_metadata()

        if obs_date is None:
            obs_date = self.metadata["obs_date"]

        self._src_vis = SourceVisibility(
            (self.metadata["src_ra"], self.metadata["src_dec"]),
            obs_date,
            observatory,
            scan_duration / 3600,
        )
        self.obs_time = self._src_vis.get_optimal_date()[0]
        self.scan_duration = scan_duration
        self.int_time = int_time
        self.fov_multiplier = 1

    def _ch_parentdir(self):
        os.chdir(self._parent_dir)

    def _ch_projectdir(self):
        os.chdir(self._project_dir)

    def save_configs(self):
        config = {
            "project_name": self.project_name,
            "Skymodel": {
                "source": self.skymodel.name,
                "path": self.skymodel.cleaned_path,
                "metadata": self.skymodel.get_metadata(),
            },
            "observation_params": {
                "obs_time": self._src_vis.get_optimal_date()[0],
                "fov_multiplier": self.fov_multiplier,
                "scan_duration": self.scan_duration,
                "integration_time": self.int_time,
            },
        }

        config_dir = Path(f"{self._project_dir}/configs")

        if not config_dir.is_dir():
            config_dir.mkdir()

        with open(
            f"{config_dir}/{self.project_name}_pyvisgen_conf.yml", "w"
        ) as outfile:
            pyvisgen_conf = self.config_pyvisgen

            # if pyvisgen_conf is not None:
            #     pyvisgen_conf = pyvisgen_conf.copy()
            #     pyvisgen_conf.pop("device", None)
            #     pyvisgen_conf.pop("overwrite", None)
            #     pyvisgen_conf.pop("show_progress", None)

            yaml.dump(
                config
                | {
                    "pyvisgen_params": pyvisgen_conf
                    if pyvisgen_conf is not None
                    else {}
                },
                outfile,
                default_flow_style=False,
                sort_keys=False,
            )

        with open(f"{config_dir}/{self.project_name}_casa_conf.yml", "w") as outfile:
            casa_conf = self.config_casa

            # if casa_conf is not None:
            #     casa_conf = casa_conf.copy()
            #     casa_conf.pop("overwrite", None)

            yaml.dump(
                config | {"casa_params": casa_conf if casa_conf is not None else {}},
                outfile,
                default_flow_style=False,
                sort_keys=False,
            )

    def init_pyvisgen(
        self,
        array_layout,
        device,
        sensitivity_cut,
        mode,
        batch_size=1000,
        noisy=0,
        show_progress=True,
        corrupted=False,
        overwrite=False,
    ):
        self.config_pyvisgen = {
            "array_layout": array_layout,
            "device": device,
            "sensitivity_cut": sensitivity_cut,
            "dense": mode == "dense",
            "corrupted": corrupted,
            "noisy": noisy,
            "mode": mode,
            "batch_size": batch_size,
            "show_progress": show_progress,
            "overwrite": overwrite,
        }

        return self

    def init_casa(self, array_layout, thermalnoise="", overwrite=False):
        self.config_casa = {
            "array_layout": array_layout,
            "thermalnoise": thermalnoise,
            "overwrite": overwrite,
        }

        return self

    def simulate(self, software, fov_multiplier=1):
        if software not in ("casa", "pyvisgen"):
            raise KeyError(
                f"The software {software} does not exist! Use (casa or pyvisgen)!"
            )

        print(f" |--- Simulation of {self.skymodel.name} with {software} ---|")

        self.fov_multiplier = fov_multiplier

        path = Path(f"{self._project_dir}/{software}")

        if not path.is_dir():
            path.mkdir()

        match software:
            case "pyvisgen":
                self._simulate_pyvisgen()

            case "casa":
                self._simulate_casa()

    def wsclean(self, software, niter=[50000, 50000], verbose=False):
        if software not in ("casa", "pyvisgen"):
            raise KeyError(
                f"The software {software} does not exist! Use (casa or pyvisgen)!"
            )

        if software == "pyvisgen":
            measurement = Measurement.from_fits(
                f"{self._project_dir}/pyvisgen/{self.project_name}.fits"
            )
            measurement.save_as_ms(
                f"{self._project_dir}/pyvisgen/pyvisgen.ms", overwrite=True
            )

        ms_name = (
            "pyvisgen.ms"
            if software == "pyvisgen"
            else f"casa.{self.casa_conf['array_layout']}.ms"
        )
        print(f"|--- WSClean Run 1 of {self.skymodel.name} for {software} ---|")

        cmd = f"""
        wsclean -multiscale -mgain 0.8 \
        -name {f"{self._project_dir}/{software}/{software}.fits"} \
        -mem 30 \
        -weight natural \
        -no-mf-weighting \
        -size {self.metadata["img_size"]} {self.metadata["img_size"]} \
        -scale {self.metadata["cell_size"] * self.fov_multiplier}asec \
        -pol I \
        -data-column DATA \
        -niter {niter[0]} \
        -auto-threshold 3 \
        -auto-mask 5 \
        -gain 0.1 \
        -padding 1 \
        {'-quiet' if not verbose else ""} \
        {self._project_dir}/{software}/{ms_name}
        """

        subprocess.run(cmd, shell=True)

        print(f"|--- WSClean Run 2 of {self.skymodel.name} for {software} ---|")

        cmd = f"""
        wsclean -multiscale -mgain 0.8 \
        -name {f"{self._project_dir}/{software}/{software}.fits"} \
        -mem 30 \
        -weight natural \
        -no-mf-weighting \
        -size {self.metadata["img_size"]} {self.metadata["img_size"]} \
        -scale {self.metadata["cell_size"] * self.fov_multiplier}asec \
        -pol I \
        -data-column MODEL_DATA \
        -niter {niter[1]} \
        -auto-threshold 1 \
        -auto-mask 3 \
        -gain 0.1 \
        -padding 1 \
        {'-quiet' if not verbose else ""} \
        {self._project_dir}/{software}/{ms_name}
        """

        subprocess.run(cmd, shell=True)

    def tclean(self, software, niter=10000, overwrite=False):
        self._ch_projectdir()

        if software not in ("casa", "pyvisgen"):
            raise KeyError(
                f"The software {software} does not exist! Use (casa or pyvisgen)!"
            )

        if software == "pyvisgen":
            measurement = Measurement.from_fits(
                f"{self._project_dir}/pyvisgen/{self.project_name}.fits"
            )
            measurement.save_as_ms(
                f"{self._project_dir}/pyvisgen/pyvisgen.ms", overwrite=True
            )

            if not Path(f"{self._project_dir}/pyvisgen/pyvisgen.skymodel").is_dir():
                p = Path(f"{self._project_dir}/casa/").glob("*.skymodel")
                casa_skymodel = [x for x in p if x.is_dir()][0]
                shutil.copytree(
                    str(casa_skymodel),
                    f"{self._project_dir}/pyvisgen/pyvisgen.skymodel",
                )

        print(f"|--- tclean of {self.skymodel.name} for {software} ---|")
        simanalyze(
            project=software,
            niter=niter,
            imsize=[int(self.metadata["img_size"]), int(self.metadata["img_size"])],
            cell=f"{self.metadata['cell_size'] * self.fov_multiplier}arcsec",
            graphics="none",
            overwrite=overwrite,
        )

        self._ch_parentdir()

    def _simulate_casa(self):
        self._ch_projectdir()

        conf = self.config_casa

        if conf is None:
            raise ValueError("The configuration for the CASA simulation is not set!")

        obs_time = self.obs_time + datetime.timedelta(days=1)

        simobserve(
            project="casa",
            skymodel=f"../{self.skymodel.cleaned_path}",
            incell=f'{self.metadata["cell_size"] * self.fov_multiplier}arcsec',
            incenter=f'{self.metadata["frequency"]}Hz',
            inwidth=f'{self.metadata["bandwidth"]}Hz',
            antennalist=f'{conf["array_layout"]}.cfg',
            obsmode="int",
            setpointings=True,
            refdate=obs_time.strftime("%Y/%m/%d/%H:%M:%S"),
            thermalnoise=conf["thermalnoise"],
            totaltime=f"{self.scan_duration}s",
            graphics="none",
            overwrite=conf["overwrite"],
        )

        self._ch_parentdir()

    def _simulate_pyvisgen(self):
        conf = self.config_pyvisgen

        if conf is None:
            raise ValueError(
                "The configuration for the pyvisgen simulation is not set!"
            )

        obs = Observation(
            array_layout=conf["array_layout"],
            src_ra=self.metadata["src_ra"],
            src_dec=self.metadata["src_dec"],
            start_time=self.obs_time,
            scan_duration=self.scan_duration,
            scan_separation=0,
            num_scans=1,
            integration_time=self.int_time,
            ref_frequency=self.metadata["frequency"],
            frequency_offsets=[0],
            bandwidths=[self.metadata["bandwidth"]],
            fov=self.metadata["fov"] * self.fov_multiplier,
            image_size=self.metadata["img_size"],
            corrupted=conf["corrupted"],
            device=conf["device"],
            dense=conf["dense"],
            sensitivity_cut=conf["sensitivity_cut"],
        )

        vis_data = vis_loop(
            obs,
            torch.from_numpy(self.skymodel.get_cleaned_model().astype("float")),
            noisy=conf["noisy"],
            mode=conf["mode"],
            batch_size=conf["batch_size"],
            show_progress=conf["show_progress"],
        )

        hdu_list = writer.create_hdu_list(vis_data, obs)
        hdu_list.writeto(
            f"{self._project_dir}/pyvisgen/{self.project_name}.fits",
            overwrite=conf["overwrite"],
        )

    def get_gridder(self, software):
        match software:
            case "pyvisgen":
                return Gridder.from_fits(
                    f"{self._project_dir}/pyvisgen/{self.project_name}.fits",
                    img_size=self.metadata["img_size"],
                    fov=self.metadata["fov"] * self.fov_multiplier,
                )

            case "casa":
                return Gridder.from_ms(
                    f"{self._project_dir}/casa/casa.{self.config_casa['array_layout']}.ms",
                    img_size=self.metadata["img_size"],
                    fov=self.metadata["fov"] * self.fov_multiplier,
                )

    @classmethod
    def load_from_cfg(cls, project_path):
        config_dir = Path(f"{project_path}/configs")

        if not config_dir.is_dir():
            raise FileNotFoundError("No configs found to load!")

        with open(f"{config_dir}/{cls.project_name}_casa_conf.yml", "w") as file:
            casa_config = yaml.safe_load(file)

        with open(f"{config_dir}/{cls.project_name}_pyvisgen_conf.yml", "w") as file:
            pyvisgen_config = yaml.safe_load(file)

        skymodel = Skymodel(
            casa_config["skymodel"]["path"].replace("_cleaned", ""),
            casa_config["skymodel"]["source"],
            casa_config["skymodel"]["path"],
        )

        cls = cls(
            project_path.split("/")[-1],
            skymodel,
            casa_config["observation_params"]["scan_duration"],
            casa_config["observation_params"]["integration_time"],
            casa_config["observation_params"]["observatory"],
        )

        cls.config_casa = casa_config["casa_params"]
        cls.config_pyvisgen = pyvisgen_config["pyvisgen_params"]

        return cls


class Skymodel:
    def __init__(self, path, source_name, cleaned_path=None):
        self.name = source_name
        self.original_path = path
        self.cleaned_path = cleaned_path

    def get_metadata(self):
        f = fits.open(self.cleaned_path)[0]
        header = f.header

        img_size = f.data[0, 0].shape[0]
        cell_size = np.abs(header["cdelt1"] * 3600)

        return {
            "img_size": img_size,
            "cell_size": cell_size,
            "fov": cell_size * img_size,
            "frequency": header["CRVAL3"],
            "bandwidth": header["CDELT3"],
            "wavelength": c.value / header["CRVAL3"],
            "src_ra": header["CRVAL1"],
            "src_dec": f.header["CRVAL2"],
            "obs_date": f.header["DATE-OBS"].split("T")[0],
        }

    def get_info(self):
        return {"source_name": self.name} | self.get_metadata()

    def get_cleaned_model(self):
        return fits.open(self.cleaned_path)[0].data[0]

    def _get_filename(self):
        return self.original_path.split(".fits")[0]

    def clean(
        self,
        crop=([None, None], [None, None]),
        intensity_cut=0,
        rms_cut_args={"sigma": 2.9},
        dbscan_args={"min_brightness": 1e-4},
        output_path=None,
        overwrite=False,
    ):
        if output_path is None:
            cleaned_path = f"{self._get_filename()}_cleaned.fits"
        else:
            cleaned_path = output_path

        self.cleaned_path = cleaned_path
        cleaned_file = Path(cleaned_path)

        if cleaned_file.is_file() and not overwrite:
            warnings.warn(
                "The file already exists and is not supposed to be overwritten. Skipping cleaning."
            )
            return self

        f = fits.open(self.original_path)
        skymodel = f[0].data[0, 0]

        skymodel_cleaned = rms_cut(skymodel, **rms_cut_args)
        skymodel_cleaned = dbscan_clean(skymodel_cleaned, **dbscan_args)

        skymodel_cleaned[skymodel_cleaned < intensity_cut] = intensity_cut
        skymodel_cleaned = skymodel_cleaned[
            crop[0][0] : crop[0][1], crop[1][0] : crop[1][1]
        ]

        f[0].data = skymodel_cleaned[None, None]
        f[0].writeto(cleaned_path, overwrite=overwrite)

        return self

    def plot_clean(
        self,
        exp=1,
        crop=([None, None], [None, None]),
        plot_args={"cmap": "inferno"},
        colorbar_shrink=1,
        save_to=None,
        save_args={},
        fig=None,
        ax=None,
    ):
        skymodel = fits.open(self.cleaned_path)[0].data[0, 0]

        if None in (fig, ax) and not all(x is None for x in (fig, ax)):
            raise KeyError(
                "The parameters ax and fig have to be both None or not None!"
            )

        if ax is None:
            fig, ax = plt.subplots()

        im = ax.imshow(skymodel, norm=PowerNorm(gamma=exp), **plot_args, origin="lower")
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")

        ax.set_xlim(crop[0][0], crop[0][1])
        ax.set_ylim(crop[1][0], crop[1][1])

        fig.colorbar(im, ax=ax, shrink=colorbar_shrink, label="Flussdichte in Jy/beam")

        if save_to is not None:
            fig.savefig(save_to, **save_args)

        return fig, ax

    def plot_original(
        self,
        exp=1,
        crop=([None, None], [None, None]),
        plot_args={"cmap": "inferno"},
        colorbar_shrink=1,
        save_to=None,
        save_args={},
        fig=None,
        ax=None,
    ):
        skymodel = fits.open(self.original_path)[0].data[0, 0]

        if None in (fig, ax) and not all(x is None for x in (fig, ax)):
            raise KeyError(
                "The parameters ax and fig have to be both None or not None!"
            )

        if ax is None:
            fig, ax = plt.subplots()

        im = ax.imshow(skymodel, norm=PowerNorm(gamma=exp), **plot_args, origin="lower")
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")

        ax.set_xlim(crop[0][0], crop[0][1])
        ax.set_ylim(crop[1][0], crop[1][1])

        fig.colorbar(im, ax=ax, shrink=colorbar_shrink, label="Flussdichte in Jy/beam")

        if save_to is not None:
            fig.savefig(save_to, **save_args)

        return fig, ax

    def plot_comp(
        self,
        original_args={},
        clean_args={},
        save_to=None,
        save_args={},
        figsize=[10, 10],
    ):
        fig, ax = plt.subplots(1, 2, layout="constrained", figsize=figsize)

        ax[0].set_title("Originales Modell")
        ax[1].set_title("Bereinigtes Modell")

        self.plot_original(fig=fig, ax=ax[0], colorbar_shrink=0.3, **original_args)
        self.plot_clean(fig=fig, ax=ax[1], colorbar_shrink=0.3, **clean_args)

        if save_to is not None:
            fig.savefig(save_to, **save_args)

        return fig, ax
