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

# PyYAML
import yaml

# matplotlib
import matplotlib.pyplot as plt

from matplotlib.colors import PowerNorm

from matplotlib.patches import Ellipse

# pathlib
from pathlib import Path

# astropy
from astropy.io import fits
from astropy.constants import c


# pyvisgen
from pyvisgen.simulation.visibility import vis_loop
from pyvisgen.simulation.observation import Observation
import pyvisgen.fits.writer as writer


# radiotools
from radiotools.visibility import SourceVisibility

from radiotools.measurements import Measurement
from radiotools.gridding import Gridder

# radio_stats
from radio_stats.cuts.dyn_range import rms_cut
from radio_stats.cuts.dbscan_clean import dbscan_clean

# casa
from casatasks import simobserve, simanalyze
from casatools.table import table

# torch dynamo logging
torch._logging.set_logs(
    dynamo=logging.CRITICAL, aot=logging.CRITICAL, inductor=logging.CRITICAL
)

# yaml representers for timestamps and numpy floats
yaml.add_representer(
    np.float64, lambda dumper, scalar: dumper.represent_float(float(scalar))
)
yaml.add_representer(
    pd._libs.tslibs.timestamps.Timestamp,
    lambda dumper, timestamp: dumper.represent_str(timestamp.isoformat()),
)


def calculate_beam_correction(bmin, bmaj, model_incell):
    """

    Calculates the correcting factor to convert from Jy/beam to Jy/px

    Parameters
    ----------
    bmin : float
        The minor axis of the beam FWHM in asec

    bmaj : float
        The major axis of the beam FWHM in asec

    model_incell : float
        The physical size of one pixel in asec

    """

    return 2 * np.pi * bmaj * bmin / ((model_incell) ** 2 * 8 * np.log(2))


def plot_text(
    text,
    ax,
    pos=(0, 1),
    text_options=dict(fontsize=12, fontfamily="monospace"),
    bbox=dict(facecolor="lightgray", edgecolor="black", boxstyle="round"),
):
    """

    Plot a boxed text onto a matplotlib plot

    Parameters
    ----------
    text : str
        The text to write in the box

    ax : matplotlib.axes._axes.Axes
        A axis to put the text into

    pos : tuple, optional
        The relative position of the text box

    text_options : dict, optional
        The options for the annotation

    bbox : dict, optional
        The bbox argument of the annotation parameters

    """

    textanchor = ax.get_window_extent()
    ax.annotate(
        text,
        pos,
        xycoords=textanchor,
        va="top",
        bbox=bbox,
        **text_options,
    )


def _plot_label(text, ax):
    """

    Plot a boxed label in the upper left corner of the plot

    Parameters
    ----------
    text : str
        The text to write in the box

    ax : matplotlib.axes._axes.Axes
        A axis to put the text into

    """

    plot_text(
        text,
        ax,
        pos=(0.05, 0.95),
        text_options=dict(fontsize=15),
        bbox=dict(
            facecolor="lightgray", alpha=0.8, edgecolor="black", boxstyle="round"
        ),
    )


class Simulation:
    """
    The main class for simulating and cleaning radio interferometry observations
    """

    def __init__(
        self,
        project_name,
        skymodel,
        scan_duration,
        int_time,
        observatory,
        fov_multiplier=1,
        obs_date=None,
    ):
        """
        Create a new simulation project

        Parameters
        ----------
        project_name : str
            Name of the project

        skymodel : simulation_chain.Skymodel
            The skymodel to use for the simulation

        scan_duration : int
            The duration of the observation in seconds

        int_time : int
            The integration time of the correlator

        observatory : str
            The observing observatory (e.g. ALMA)

        fov_multiplier : float, optional
            The factor to multiply the fov and cell_size by
            for the simulations and cleaning

        obs_date : str, optional
            The date of the observation, set None for use of date saved
            in Skymodel header

        """

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

        self.observatory = observatory
        self._src_vis = SourceVisibility(
            (self.metadata["src_ra"], self.metadata["src_dec"]),
            obs_date,
            observatory,
            scan_duration / 3600,
        )
        self.obs_time = self._src_vis.get_optimal_date()[0]
        self.scan_duration = scan_duration
        self.int_time = int_time
        self.fov_multiplier = fov_multiplier

    def _ch_parentdir(self):
        """
        Internal method to change the current working directory
        to the parent directory of the project
        """
        os.chdir(self._parent_dir)

    def _ch_projectdir(self):
        """
        Internal method to change the current working directory
        to the project repository
        """
        os.chdir(self._project_dir)

    def get_config(self):
        """
        Get the current state of the parameters
        """

        config = {
            "project_name": self.project_name,
            "skymodel": {
                "source": self.skymodel.name,
                "path": self.skymodel.cleaned_path,
                "metadata": self.metadata,
            },
            "observation_params": {
                "observatory": self.observatory,
                "obs_time": self._src_vis.get_optimal_date()[0],
                "fov_multiplier": self.fov_multiplier,
                "scan_duration": self.scan_duration,
                "integration_time": self.int_time,
            },
        }

        pyvisgen_conf = self.config_pyvisgen
        casa_conf = self.config_casa

        return (
            config
            | {"pyvisgen_params": pyvisgen_conf if pyvisgen_conf is not None else {}}
            | {"casa_params": casa_conf if casa_conf is not None else {}}
        )

    def save_configs(self):
        """
        Save the current state of the simulation parameters
        in a yaml config file. Creates a new folder configs in the
        project directory
        """

        config_dir = Path(f"{self._project_dir}/configs")

        if not config_dir.is_dir():
            config_dir.mkdir()

        with open(f"{config_dir}/{self.project_name}_conf.yml", "w") as outfile:
            yaml.dump(
                self.get_config(),
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
        """
        Initialize the simulation parameters for the simulation
        with the pyvisgen package

        Parameters
        ----------
        array_layout : str
            Name of the pyvisgen array layout configuration file
            without the file extension

        device : str
            The device to use for the calculations with torch
            (e.g. cpu, cuda:0 or cuda:1)

        sensitivity_cut : float
            The intensity of the source distribution blow which
            a pixel is not included in the calculation and assumed as zero

        mode : str
            The mode of the pyvisgen simulation_chain
            (available: full, grid, dense (GPU only))

        noisy : int, optional
            The SEFD to use for the noise generation of pyvisgen

        show_progress : bool , optional
            Whether to show the progress of the visibility calculation
            as a tqdm progress bar

        corrupted : bool, optional
            Whether to take DDEs (direction-dependent-effects) into
            account for the simulations (RIME E-matrix)

        overwrite : bool, optional
            Whether to overwrite existing simulation results
        """

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
        """
        Initialize the simulation parameters for the simulation
        with the simobserve command of NRAO CASA

        Parameters
        ----------

        array_layout : str
            Name of the pyvisgen array layout configuration file
            without the file extension

        thermalnoise : str, optional
            The type of thermal noise to use for the simulation
            (available: tsys-atm, tsys-manual or empty string)

        overwrite : bool, optional
            Whether to overwrite existing simulation results

        """
        self.config_casa = {
            "array_layout": array_layout,
            "thermalnoise": thermalnoise,
            "overwrite": overwrite,
        }

        return self

    def simulate(self, software):
        """
        Initialize the simulation parameters for the simulation
        with the simobserve command of NRAO CASA

        Parameters
        ----------

        software : str
            The software to use for the simulation
            (available: casa or pyvisgen)
        """

        if software not in ("casa", "pyvisgen"):
            raise KeyError(
                f"The software {software} does not exist! Use (casa or pyvisgen)!"
            )

        print(f" |--- Simulation of {self.skymodel.name} with {software} ---|")

        path = Path(f"{self._project_dir}/{software}")

        if not path.is_dir():
            path.mkdir()

        match software:
            case "pyvisgen":
                self._simulate_pyvisgen()

            case "casa":
                self._simulate_casa()

    def wsclean(
        self,
        software,
        niter=[50000, 50000],
        data_column="MODEL_DATA",
        save_config=False,
        verbose=False,
    ):
        """
        Apply CLEAN algorithm using WSCLEAN to the image
        of a simulation

        Parameters
        ----------

        software : str
            The software whose results are supposed to be
            cleaned (available: casa or pyvisgen)

        niter : tuple or array_like, optional
            The amount of maximal iterations to perform on the
            two runs of WSCLEAN

        data_column : str, optional
            The column to save the results of the first run of
            WSCLEAN in the measurement set

        save_config : bool, optional
            Whether to save to configuration of the simulations
            and cleanings after the execution

        verbose : bool, optional
            Whether to perform the WSCLEAN with the verbose flag

        """

        if software not in ("casa", "pyvisgen"):
            raise KeyError(
                f"The software {software} does not exist! Use (casa or pyvisgen)!"
            )

        niter_dict = {"run1": niter[0], "run2": niter[1]}

        if software == "pyvisgen":
            measurement = Measurement.from_fits(
                f"{self._project_dir}/pyvisgen/{self.project_name}.fits"
            )
            measurement.save_as_ms(
                f"{self._project_dir}/pyvisgen/pyvisgen.ms", overwrite=True
            )
            self.config_pyvisgen["wsclean_niter"] = niter_dict
        else:
            self.config_casa["wsclean_niter"] = niter_dict

        ms_name = (
            "pyvisgen.ms"
            if software == "pyvisgen"
            else f"casa.{self.config_casa['array_layout']}.ms"
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
        -data-column {data_column} \
        -niter {niter[1]} \
        -auto-threshold 1 \
        -auto-mask 3 \
        -gain 0.1 \
        -padding 1 \
        {'-quiet' if not verbose else ""} \
        {self._project_dir}/{software}/{ms_name}
        """

        subprocess.run(cmd, shell=True)

        if save_config:
            self.save_configs()

    def tclean(
        self,
        software,
        niter=10000,
        threshold=1e-8,
        save_config=False,
        overwrite=False,
        verbose=False,
    ):
        """
        Apply CLEAN algorithm using CASA simanalyze to the image
        of a simulation

        Parameters
        ----------

        software : str
            The software whose results are supposed to be
            cleaned (available: casa or pyvisgen)

        niter : int, optional
            The amount of maximal iterations to perform on the
            execution of simanalyze

        threshold : float, optional
            The fluxdensity threshold at which the cleaning is ended

        save_config : bool, optional
            Whether to save to configuration of the simulations
            and cleanings after the execution

        overwrite : bool, optional
            Whether to overwrite existing cleaning results (deletes
            results before (!) execution of simanalyze

        verbose : bool, optional
            Whether to perform the simanalyze execution with verbose output

        """

        if software not in ("casa", "pyvisgen"):
            raise KeyError(
                f"The software {software} does not exist! Use (casa or pyvisgen)!"
            )

        self._ch_projectdir()

        if overwrite:
            excluded_dir_suffices = ["ms", "skymodel"]
            dirs = [
                x
                for x in Path(f"./{software}").glob("*.*")
                if x.is_dir()
                and str(x).split(".")[-1] not in excluded_dir_suffices
                and (str(x).split("/")[-1].split(".")[0] == software)
            ]

            for dirx in dirs:
                shutil.rmtree(dirx)

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
            self.config_pyvisgen["tclean_niter"] = niter
            self.config_pyvisgen["tclean_threshold"] = threshold
        else:
            self.config_casa["tclean_niter"] = niter
            self.config_casa["tclean_threshold"] = threshold

        print(f"|--- tclean of {self.skymodel.name} for {software} ---|")
        simanalyze(
            project=software,
            niter=niter,
            imsize=[int(self.metadata["img_size"]), int(self.metadata["img_size"])],
            cell=f"{self.metadata['cell_size'] * self.fov_multiplier}arcsec",
            graphics="none",
            threshold=f"{threshold}Jy",
            overwrite=overwrite,
            verbose=verbose,
        )

        self._ch_parentdir()

        if save_config:
            self.save_configs()

    def plot_tclean_result(
        self,
        software,
        exp=1,
        rot90=0,
        invert_x=False,
        invert_y=False,
        cut_negative_flux=False,
        show_beam=True,
        flux_per_beam=True,
        save_to=None,
        save_args={},
        plot_args={"cmap": "inferno"},
        colorbar_shrink=1,
        annotation=None,
        return_image=False,
        fig=None,
        ax=None,
    ):
        """
        Plots the result of the simanalyze cleaning of a simulation

        Parameters
        ----------

        software : str
            The software whose cleaning result is supposed
            to be plotted

        exp : float, optional
            The exponent for the matplotlib.colors.PowerNorm norm

        rot90 : int, optional
            The amount of times the image is supposed to be rotated
            90 degrees counterclockwise

        invert_x : bool, optional
            Whether the image should be inverted on the x-axis

        invert_y : bool, optional
            Whether the image should be inverted on the y-axis

        cut_negative_flux : bool, optional
            Whether to set all negative fluxdensities to zero

        show_beam : bool optional
            Whether to show the beam size and position angle
            as a white ellipse in the corner

        flux_per_beam : bool, optional
            Whether to use Jy/beam as a unit. If set to False
            Jy/px is used

        save_to : str, optional
            Path to save the figure to

        save_args : str, optional
            The arguments for the savefig function

        plot_args : dict, optional
            The arguments for the pyplot imshow plot of the
            cleaned image

        colorbar_shrink : float, optional
            The shrink parameter for the colorbar

        annotation : str, optional
            The label the plot is supposed to have, None for no label

        return_image : bool, optional
            Whether to return the image matrix additionally
            to the figure and axis

        fig : matplotlib.figure.Figure, optional
            A figure to put the plot into

        ax : matplotlib.axes._axes.Axes, optional
            A axis to put the plot into

        """

        if None in (fig, ax) and not all(x is None for x in (fig, ax)):
            raise KeyError(
                "The parameters ax and fig have to be both None or not None!"
            )

        if ax is None:
            fig, ax = plt.subplots(layout="constrained")

        img_path = f"{self._project_dir}/{software}/"
        beam_path = img_path

        match software:
            case "pyvisgen":
                img_path += "pyvisgen.image/"
                beam_path += "pyvisgen.psf/"
            case "casa":
                img_path += f"casa.{self.config_casa['array_layout']}.image/"
                beam_path += f"casa.{self.config_casa['array_layout']}.psf/"

        img = table(img_path).getcol("map")[:, :, 0, 0, 0]

        beam = table(beam_path)
        beam_desc = beam.getdesc()["_keywords_"]["imageinfo"]["restoringbeam"]
        beam_info = {
            "bmin": beam_desc["minor"]["value"],
            "bmaj": beam_desc["major"]["value"],
            "bpa": beam_desc["positionangle"]["value"],
        }

        cell_size = self.skymodel.get_metadata()["cell_size"] * self.fov_multiplier

        if show_beam:
            img_size = self.skymodel.get_metadata()["img_size"]

            ax.add_patch(
                Ellipse(
                    (int(img_size / 10), int(img_size / 10)),
                    width=beam_info["bmin"] / cell_size,
                    height=beam_info["bmaj"] / cell_size,
                    angle=beam_info["bpa"] / cell_size,
                    facecolor="white",
                )
            )

        if cut_negative_flux:
            img[img < 0] = 0

        if invert_x:
            img = np.fliplr(img)

        if invert_y:
            img = np.flipud(img)

        if not flux_per_beam:
            img /= calculate_beam_correction(
                bmin=beam_info["bmin"], bmaj=beam_info["bmaj"], model_incell=cell_size
            )

        multiplier = 1 if software != "pyvisgen" else 2

        norm = None if exp == 1 else PowerNorm(gamma=exp)

        im = ax.imshow(
            np.rot90(img, rot90) * multiplier,
            interpolation="none",
            origin="lower",
            norm=norm,
            **plot_args,
        )

        if annotation is not None:
            _plot_label(annotation, ax)

        ax.set_ylabel("Pixel")
        ax.set_xlabel("Pixel")
        fig.colorbar(
            im,
            ax=ax,
            label=f"Flussdichte in Jy/{'beam' if flux_per_beam else 'px'}",
            shrink=colorbar_shrink,
        )

        if save_to is not None:
            fig.savefig(save_to, **save_args)

        if return_image:
            return fig, ax, np.rot90(img, rot90) * multiplier
        else:
            return fig, ax

    def plot_wsclean_result(
        self,
        software,
        exp=1,
        rot90=0,
        invert_x=False,
        invert_y=False,
        cut_negative_flux=False,
        show_beam=True,
        flux_per_beam=True,
        save_to=None,
        save_args={},
        plot_args={"cmap": "inferno"},
        colorbar_shrink=1,
        annotation=None,
        return_image=False,
        fig=None,
        ax=None,
    ):
        """
        Plots the result of the WSCLEAN cleaning of a simulation

        Parameters
        ----------

        software : str
            The software whose cleaning result is supposed
            to be plotted

        exp : float, optional
            The exponent for the matplotlib.colors.PowerNorm norm

        rot90 : int, optional
            The amount of times the image is supposed to be rotated
            90 degrees counterclockwise

        invert_x : bool, optional
            Whether the image should be inverted on the x-axis

        invert_y : bool, optional
            Whether the image should be inverted on the y-axis

        cut_negative_flux : bool, optional
            Whether to set all negative fluxdensities to zero

        show_beam : bool optional
            Whether to show the beam size and position angle
            as a white ellipse in the corner

        flux_per_beam : bool, optional
            Whether to use Jy/beam as a unit. If set to False
            Jy/px is used

        save_to : str, optional
            Path to save the figure to

        save_args : str, optional
            The arguments for the savefig function

        plot_args : dict, optional
            The arguments for the pyplot imshow plot of the
            cleaned image

        colorbar_shrink : float, optional
            The shrink parameter for the colorbar

        annotation : str, optional
            The label the plot is supposed to have, None for no label

        return_image : bool, optional
            Whether to return the image matrix additionally
            to the figure and axis

        fig : matplotlib.figure.Figure, optional
            A figure to put the plot into

        ax : matplotlib.axes._axes.Axes, optional
            A axis to put the plot into

        """

        if None in (fig, ax) and not all(x is None for x in (fig, ax)):
            raise KeyError(
                "The parameters ax and fig have to be both None or not None!"
            )

        if ax is None:
            fig, ax = plt.subplots(layout="constrained")

        img_path = f"{self._project_dir}/{software}/{software}.fits-dirty.fits"
        beam_path = f"{self._project_dir}/{software}/{software}.fits-psf.fits"

        img = fits.open(img_path)[0].data[0, 0]

        cell_size = self.skymodel.get_metadata()["cell_size"] * self.fov_multiplier

        beam = fits.open(beam_path)[0]
        header = beam.header
        beam_info = {
            "bmin": header["BMIN"] * 3600,
            "bmaj": header["BMAJ"] * 3600,
            "bpa": header["BPA"],
        }

        if show_beam:
            img_size = self.skymodel.get_metadata()["img_size"]

            ax.add_patch(
                Ellipse(
                    (int(img_size / 10), int(img_size / 10)),
                    width=beam_info["bmin"] / cell_size,
                    height=beam_info["bmaj"] / cell_size,
                    angle=beam_info["bpa"],
                    facecolor="white",
                )
            )

        if cut_negative_flux:
            img[img < 0] = 0

        if invert_x:
            img = np.fliplr(img)

        if invert_y:
            img = np.flipud(img)

        if not flux_per_beam:
            img /= calculate_beam_correction(
                bmin=beam_info["bmin"], bmaj=["bmaj"], model_incell=cell_size
            )

        multiplier = 1 if software != "pyvisgen" else 2

        norm = None if exp == 1 else PowerNorm(gamma=exp)

        im = ax.imshow(
            np.rot90(img, rot90) * multiplier,
            origin="lower",
            norm=norm,
            **plot_args,
        )

        if annotation is not None:
            _plot_label(annotation, ax)

        ax.set_ylabel("Pixel")
        ax.set_xlabel("Pixel")
        fig.colorbar(
            im,
            ax=ax,
            label=f"Flussdichte in Jy/{'beam' if flux_per_beam else 'px'}",
            interpolation="none",
            shrink=colorbar_shrink,
        )

        if save_to is not None:
            fig.savefig(save_to, **save_args)

        if return_image:
            return fig, ax, np.rot90(img, rot90)

    def _simulate_casa(self):
        """
        Internal function to perform the simobserve simulation
        """

        self._ch_projectdir()

        conf = self.config_casa

        if conf is None:
            raise ValueError("The configuration for the CASA simulation is not set!")

        obs_time = self.obs_time + datetime.timedelta(days=1)

        simobserve(
            project="casa",
            skymodel=self.skymodel.cleaned_path,
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
        """
        Internal function to perform the pyvisgen simulation
        """
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
        """
        Get the radiotools.gridding.Gridder for a simulation

        Parameters
        ----------

        software : str
            The software whose results to get the gridder for

        """
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

    def export_results(
        self,
        archive=False,
        compress=False,
        keep_directory=False,
        save_args={},
        softwares=["pyvisgen", "casa"],
        skymodel_plot_args={},
        mask_plot_args={},
        mask_abs_args={},
        dirty_img_args={},
        wsclean_img_args={},
        tclean_img_args={},
    ):
        """
        Creates a comprehensive image of the results
        and exports the results if wanted

        Parameters
        ----------

        archive : bool, optional
            Whether to create a directory to save the result pdfs
            and the current configuration file to

        compress : bool, optional
            Whether to create a zip-archive of the created directory,
            requires archive parameter to be True

        keep_directory : bool, optional
            Whether to keep the directory after creating the zip archive,
            requires archive and compress parameters to be True

        save_args : dict, optional
            The arguments for the savefig function

        softwares : tuple or array_like, optional
            The softwares whose results are supposed to be plotted

        skymodel_plot_args : dict, optional
            The arguments for the call of simulation_chain.Skymodel.plot_clean

        mask_plot_args : dict, optional
            The arguments for the call of Gridder.plot_mask

        mask_abs_args : dict, optional
            The arguments for the call of Gridder.plot_mask_absolute

        dirty_img_args : dict, optional
            The arguments for the call of Gridder.plot_dirty_image

        wsclean_img_args : dict, optional
            The arguments for the call of plot_wsclean_result

        tclean_img_args : dict, optional
            The arguments for the call of plot_tclean_result

        """

        if archive:
            archive_path = Path(self._project_dir) / "archive"

            if not archive_path.is_dir():
                archive_path.mkdir()

            archive_path /= f"sim-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

            if not archive_path.is_dir():
                archive_path.mkdir()

            shutil.copytree(
                f"{self._project_dir}/configs", archive_path, dirs_exist_ok=True
            )

        config = self.get_config()

        for software in softwares:
            labels = {
                "SKY": "Skymodel",
                "CFG": "Konfiguration",
                "ABS": "Amplitude der Visibilities",
                "DIRTY": "Dirty Image",
                "WS": "WSCLEAN",
                "TC": "tclean",
                "MASK": "Gerasterte $(u,v)$-Abdeckung",
            }

            fig, ax = plt.subplot_mosaic(
                [["SKY", "MASK"], ["ABS", "DIRTY"], ["WS", "TC"], ["CFG", "_"]],
                layout="constrained",
                figsize=(12, 20),
            )

            skymodel_metadata = config["skymodel"]["metadata"]
            obs_config = config["observation_params"]

            if software == "pyvisgen":
                fig.suptitle(
                    f"Beobachtung von {self.skymodel.name}\n[ {software} ({self.config_pyvisgen['mode']})]"
                )
            else:
                fig.suptitle(f"Beobachtung von {self.skymodel.name}\n[ {software} ]")

            tclean_exec = "tclean_niter" in config[f"{software}_params"]
            wsclean_exec = "wsclean_niter" in config[f"{software}_params"]

            labels["WS"] += (
                f" (niter={config[f'{software}_params']['wsclean_niter']})"
                if wsclean_exec
                else ""
            )
            labels["TC"] += (
                f" (niter={config[f'{software}_params']['tclean_niter']})"
                if tclean_exec
                else ""
            )

            for label, axis in ax.items():
                if label != "CFG" and label != "_":
                    axis.set_title(labels[label])

                gridder = self.get_gridder(software=software)

                match label:
                    case "SKY":
                        self.skymodel.plot_clean(
                            fig=fig, ax=axis, colorbar_shrink=0.9, **skymodel_plot_args
                        )
                    case "CFG":
                        axis.axis("off")

                        config_text = (
                            f'img_size:\t{skymodel_metadata["img_size"]} px^2\n'
                        )
                        config_text += f'cell_size:\t{skymodel_metadata["cell_size"]} * {obs_config["fov_multiplier"]} asec/px\n'
                        config_text += f'fov:\t\t{skymodel_metadata["fov"]} * {obs_config["fov_multiplier"]} asec\n'
                        config_text += (
                            f'scan_duration:\t{obs_config["scan_duration"]} s\n'
                        )
                        config_text += f'int_time:\t{obs_config["integration_time"]} s'

                        plot_text(config_text.expandtabs(), ax=axis)
                    case "MASK":
                        gridder.plot_mask(
                            fig=fig, ax=axis, colorbar_shrink=1.0002, **mask_plot_args
                        )
                    case "ABS":
                        gridder.plot_mask_absolute(
                            fig=fig, ax=axis, colorbar_shrink=0.92, **mask_abs_args
                        )
                    case "DIRTY":
                        gridder.plot_dirty_image(
                            fig=fig, ax=axis, colorbar_shrink=0.9, **dirty_img_args
                        )
                    case "WS":
                        if wsclean_exec:
                            self.plot_wsclean_result(
                                software,
                                fig=fig,
                                ax=axis,
                                colorbar_shrink=0.9,
                                **wsclean_img_args,
                            )
                        else:
                            axis.axis("off")
                    case "TC":
                        if tclean_exec:
                            self.plot_tclean_result(
                                software,
                                fig=fig,
                                ax=axis,
                                colorbar_shrink=0.9,
                                **tclean_img_args,
                            )
                        else:
                            axis.axis("off")
                    case "_":
                        axis.axis("off")

            if archive:
                fig.savefig(
                    f"{str(archive_path)}/{self.skymodel.name}_{software}_results.pdf",
                    **save_args,
                )
                print(f"Created archive directory {str(archive_path)}")

        if archive and compress:
            if (archive_path / ".ipynb_checkpoints").is_dir():
                shutil.rmtree(archive_path / ".ipynb_checkpoints")
            zip_path = shutil.make_archive(
                f"{str(archive_path).split('/')[-1]}", "zip", root_dir=archive_path
            )
            shutil.move(zip_path, archive_path.parents[0])
            print(f"Created zip-archive {str(archive_path).split('/')[-1]}")

            if not keep_directory:
                shutil.rmtree(archive_path)

    @classmethod
    def load_from_cfg(cls, project_path):
        """
        Loads a Simulation from an existing project

        Parameters
        ----------
        project_path : str, path to the project to load

        """

        config_dir = Path(f"{project_path}/configs")

        if not config_dir.is_dir():
            raise FileNotFoundError("No configs found to load!")

        project_name = project_path.split("/")[-1]

        with open(f"{config_dir}/{project_name}_conf.yml", "r") as file:
            config = yaml.safe_load(file)

        skymodel = Skymodel(
            config["skymodel"]["path"].replace("_cleaned", ""),
            config["skymodel"]["source"],
            config["skymodel"]["path"],
        )

        cls = cls(
            project_name=project_name,
            skymodel=skymodel,
            scan_duration=config["observation_params"]["scan_duration"],
            int_time=config["observation_params"]["integration_time"],
            observatory=config["observation_params"]["observatory"],
            fov_multiplier=config["observation_params"]["fov_multiplier"],
        )

        cls._parent_dir = str(Path(project_path).resolve().parents[0])
        cls._project_dir = f"{cls._parent_dir}/{project_name}"

        cls.config_casa = config["casa_params"]
        cls.config_pyvisgen = config["pyvisgen_params"]

        return cls


class Skymodel:
    """
    The skymodel class for radiointerferometic simulations
    """

    def __init__(self, path, source_name, cleaned_path=None):
        """
        Create a new Skymodel

        Parameters
        ----------

        path : str
            The path to the FITS file of the uncleaned (!) skymodel

        source_name : str
            Name of the source

        cleaned_path : str, optional
            The path to the FITS file of the cleaned (!) skymodel

        """

        self.name = source_name
        self.original_path = str(Path(path).resolve())
        self.cleaned_path = (
            str(Path(cleaned_path).resolve()) if cleaned_path is not None else None
        )

        if cleaned_path is not None:
            self.flux_per_beam = fits.open(cleaned_path)[0].header["BUNIT"] == "Jy/beam"

    def get_metadata(self):
        """
        Get the metadata of the skymodel
        """

        f = fits.open(self.cleaned_path)[0]
        header = f.header

        img_size = f.data[0, 0].shape[0]
        cell_size = np.abs(header["CDELT1"] * 3600)

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
            "beam": {
                "bmin": header["BMIN"] * 3600,
                "bmaj": header["BMAJ"] * 3600,
                "bpa": header["BPA"],
            },
        }

    def get_info(self):
        """
        Get an overview of the skymodel's properties
        """

        return {"source_name": self.name} | self.get_metadata()

    def get_cleaned_model(self):
        """
        Get the image data of the cleaned skymodel
        """

        return fits.open(self.cleaned_path)[0].data[0]

    def get_original_model(self):
        """
        Get the image data of the uncleaned skymodel
        """

        return fits.open(self.original_path)[0].data[0]

    def _get_filename(self):
        """
        Internal function to retrieve the filename of the original model
        """
        return self.original_path.split(".fits")[0]

    def clean(
        self,
        crop=([None, None], [None, None]),
        intensity_cut=0,
        rms_cut_args={"sigma": 2.9},
        dbscan_args={"min_brightness": 1e-4},
        output_path=None,
        flux_per_beam=False,
        overwrite=False,
    ):
        """
        Clean the original skymodel from noise and negative values

        Parameters
        ----------
        crop : tuple of arrays, optional
            The cutout of the image to use for the cleaned model
            (e.g. ([-10, 10], [-15, 15])

        intensity_cut : float, optional
            The intensity below which every value is assumed as zero

        rms_cut_args : dict, optional
            The arguments to pass to the radio_stats.rms_cut method

        dbscan_args : dict, optional
            The arguments to pass to the radio_stats.dbscan_clean method

        output_path : str, optional
            The path to save the cleaned model to (if None same directory
            is used and name is appended a '_cleaned' suffix)

        flux_per_beam : bool, optional
            Whether to save the cleaned model in the unit Jy/beam. If False
            Jy/px is used.

        overwrite : bool, optional
            Whether to overwrite an existing cleaned model
        """

        if output_path is None:
            cleaned_path = str(Path(f"{self._get_filename()}_cleaned.fits").resolve())
        else:
            cleaned_path = str(Path(output_path).resolve())

        self.cleaned_path = cleaned_path
        cleaned_file = Path(cleaned_path)

        if cleaned_file.is_file() and not overwrite:
            self.flux_per_beam = fits.open(cleaned_path)[0].header["BUNIT"] == "Jy/beam"
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

        self.flux_per_beam = flux_per_beam

        header = f[0].header
        beam_info = {
            "bmin": header["BMIN"] * 3600,
            "bmaj": header["BMAJ"] * 3600,
            "bpa": header["BPA"],
        }

        beam_correction = calculate_beam_correction(
            bmin=beam_info["bmin"],
            bmaj=beam_info["bmaj"],
            model_incell=np.abs(header["CDELT1"] * 3600),
        )

        f[0].data = (
            skymodel_cleaned[None, None]
            if flux_per_beam
            else skymodel_cleaned[None, None] / beam_correction
        )

        if not flux_per_beam:
            header["BUNIT"] = "Jy/pix "

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
        show_beam=True,
        annotation=None,
        fig=None,
        ax=None,
    ):
        """
        Plots the cleaned model

        Parameters
        ----------
        exp : float, optional
            The exponent for the matplotlib.colors.PowerNorm norm

        crop : tuple of arrays, optional
            The cutout of the image to use for the plot
            (e.g. ([-10, 10], [-15, 15])

        plot_args : dict, optional
            The arguments for the pyplot imshow plot of the cleaned model

        colorbar_shrink : float, optional
            The shrink parameter for the colorbar

        save_to : str, optional
            Path to save the figure to

        save_args : str, optional
            The arguments for the savefig function

        show_beam : bool, optional
            Whether to show the size and position angle of the
            beam as a white ellipse in the corner of the plot

        annotation : str, optional
            The label the plot is supposed to have, None for no label

        fig : matplotlib.figure.Figure, optional
            A figure to put the plot into

        ax : matplotlib.axes._axes.Axes, optional
            A axis to put the plot into

        """

        skymodel = fits.open(self.cleaned_path)[0].data[0, 0]

        if None in (fig, ax) and not all(x is None for x in (fig, ax)):
            raise KeyError(
                "The parameters ax and fig have to be both None or not None!"
            )

        if ax is None:
            fig, ax = plt.subplots()

        crop = ([0, skymodel.shape[0]], [0, skymodel.shape[0]])

        im = ax.imshow(skymodel, norm=PowerNorm(gamma=exp), **plot_args, origin="lower")

        if annotation is not None:
            _plot_label(annotation, ax)

        if show_beam:
            beam_info = self.get_metadata()["beam"]
            img_size = self.get_metadata()["img_size"]
            cell_size = self.get_metadata()["cell_size"]

            ax.add_patch(
                Ellipse(
                    (int(img_size / 10), int(img_size / 10)),
                    width=beam_info["bmin"] / cell_size,
                    height=beam_info["bmaj"] / cell_size,
                    angle=beam_info["bpa"],
                    facecolor="white",
                )
            )

        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")

        ax.set_xlim(crop[0][0], crop[0][1])
        ax.set_ylim(crop[1][0], crop[1][1])

        fig.colorbar(
            im,
            ax=ax,
            shrink=colorbar_shrink,
            label=f"Flussdichte in Jy/{'beam' if self.flux_per_beam else 'px'}",
        )

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
        show_beam=True,
        flux_per_beam=False,
        annotation=None,
        fig=None,
        ax=None,
    ):
        """
        Plots the uncleaned model

        Parameters
        ----------
        exp : float, optional
            The exponent for the matplotlib.colors.PowerNorm norm

        crop : tuple of arrays, optional
            The cutout of the image to use for the plot
            (e.g. ([-10, 10], [-15, 15])

        plot_args : dict, optional
            The arguments for the pyplot imshow plot of the uncleaned model

        colorbar_shrink : float, optional
            The shrink parameter for the colorbar

        save_to : str, optional
            Path to save the figure to

        save_args : str, optional
            The arguments for the savefig function

        show_beam : bool, optional
            Whether to show the size and position angle of the
            beam as a white ellipse in the corner of the plot

        annotation : str, optional
            The label the plot is supposed to have, None for no label

        fig : matplotlib.figure.Figure, optional
            A figure to put the plot into

        ax : matplotlib.axes._axes.Axes, optional
            A axis to put the plot into

        """

        f = fits.open(self.original_path)[0]
        skymodel = f.data[0, 0]

        if None in (fig, ax) and not all(x is None for x in (fig, ax)):
            raise KeyError(
                "The parameters ax and fig have to be both None or not None!"
            )

        if ax is None:
            fig, ax = plt.subplots()

        beam_info = self.get_metadata()["beam"]
        cell_size = np.abs(f.header["CDELT1"] * 3600)

        if not flux_per_beam:
            skymodel /= calculate_beam_correction(
                bmaj=beam_info["bmaj"],
                bmin=beam_info["bmin"],
                model_incell=cell_size,
            )

        crop = ([0, skymodel.shape[0]], [0, skymodel.shape[0]])

        im = ax.imshow(skymodel, norm=PowerNorm(gamma=exp), **plot_args, origin="lower")

        if annotation is not None:
            _plot_label(annotation, ax)

        if show_beam:
            ax.add_patch(
                Ellipse(
                    (
                        crop[0][0] + int((crop[0][1] - crop[0][0]) / 10),
                        crop[1][0] + int((crop[1][1] - crop[1][0]) / 10),
                    ),
                    width=beam_info["bmin"] / cell_size,
                    height=beam_info["bmaj"] / cell_size,
                    angle=beam_info["bpa"],
                    facecolor="white",
                )
            )

        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")

        ax.set_xlim(crop[0][0], crop[0][1])
        ax.set_ylim(crop[1][0], crop[1][1])

        fig.colorbar(
            im,
            ax=ax,
            shrink=colorbar_shrink,
            label=f"Flussdichte in Jy/{'beam' if flux_per_beam else 'px'}",
        )

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
        """
        Plots the uncleaned model

        Parameters
        ----------
        original_args : dict, optional
            The arguments to pass to the plot_clean function

        clean_args : dict, optional
            The arguments to pass to the plot_original function

        save_to : str, optional
            Path to save the figure to

        save_args : str, optional
            The arguments for the savefig function

        figsize : array_like, optional
            The size of the figure

        """

        fig, ax = plt.subplots(1, 2, layout="constrained", figsize=figsize)

        ax[0].set_title("Originales Modell")
        ax[1].set_title("Bereinigtes Modell")

        self.plot_original(fig=fig, ax=ax[0], colorbar_shrink=0.3, **original_args)
        self.plot_clean(fig=fig, ax=ax[1], colorbar_shrink=0.3, **clean_args)

        if save_to is not None:
            fig.savefig(save_to, **save_args)

        return fig, ax
