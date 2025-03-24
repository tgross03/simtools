import toml

from simtools.data import Dataset
from simtools.simulations import DatasetSimulation

import click

import warnings

warnings.filterwarnings("ignore")


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
def main(configuration_path):
    config = toml.load(configuration_path)
    patterns = config["dataset"]["required_patterns"]
    start_indices = config["simulation_options"]["start_indices"]

    for pattern in patterns[start_indices[0] :]:
        print(
            f"------------ Simulating for dataset with pattern {pattern} ------------"
        )
        dataset = Dataset(
            data_path=config["dataset"]["path"],
            required_pattern=pattern,
            cache_loaded=config["dataset"]["cache_loaded"],
            max_cache_size=config["dataset"]["max_cache_size"],
            cache_cleaning_policy=config["dataset"]["cache_cleaning_policy"],
        )
        simulation = DatasetSimulation(dataset=dataset, config=configuration_path)
        simulation.simulate_dataset(
            out=config["simulation_options"]["out"],
            out_prefix=config["simulation_options"]["out_prefix"],
            batch_size=config["simulation_options"]["batch_size"],
            fov_multiplier=config["simulation_options"]["fov_multiplier"],
            show_individual_progress=config["simulation_options"][
                "show_individual_progress"
            ],
            start_index=start_indices[1],
            generate_config=config["simulation_options"]["generate_config"],
            overwrite=config["simulation_options"]["overwrite"],
            verbose=config["simulation_options"]["verbose"],
        )
