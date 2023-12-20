#!/usr/bin/env python
"""\
Main executable module to run cosmological N-body simulations

Usage: python main.py -c param.ini
"""
__author__ = "Michel-Andrès Breton"
__copyright__ = "Copyright 2022-2023, Michel-Andrès Breton"
__version__ = "0.2.5"
__email__ = "michel-andres.breton@obspm.fr"
__status__ = "Development"

import ast
import numba
import numpy as np
import os
import cosmotable
import initial_conditions
import integration
import solver
import utils
from typing import Dict
import pandas as pd
import logging
from rich.logging import RichHandler


def run(param) -> None:
    """This is the main function to run N-body simulations

    Parameters
    ----------
    param : dict or pd.Series
        Parameter container
    """
    # Set logging/verbose level
    # Ideally it would have been error/info/debug, but the latter triggers extensive numba verbose
    if param["verbose"] == 0:
        logging_level = logging.ERROR
    elif param["verbose"] == 1:
        logging_level = logging.WARNING
    elif param["verbose"] == 2:
        logging_level = logging.INFO
    else:
        raise ValueError(f"{param['verbose']=}, should be 0, 1 or 2")
    logging.basicConfig(
        level=logging_level,
        format="%(message)s",
        datefmt="%d/%m/%Y %I:%M:%S %p",
        handlers=[
            RichHandler(
                show_level=False,
                show_path=False,
                enable_link_path=False,
                markup=True,
            )
        ],
        force=True,
    )
    # Check param type
    if isinstance(param, Dict):
        param = pd.Series(param)
    elif isinstance(param, pd.Series):
        pass
    else:
        raise ValueError(f"{type(param)=}, should be a dictionnary or a Pandas Series")
    # Threading
    if param["nthreads"] > 0:
        numba.set_num_threads(param["nthreads"])
    else:
        param["nthreads"] = numba.get_num_threads()
    logging.warning(f"{param['nthreads']=}")
    # Extra string
    extra = param["theory"].casefold()
    if extra.casefold() == "fr".casefold():
        extra += f"{param['fR_logfR0']}_n{param['fR_n']}"
    extra += f"_{param['linear_newton_solver']}_ncoarse{param['ncoarse']}"
    param["extra"] = extra
    z_out = ast.literal_eval(param["z_out"])
    # Create directories
    # Power dir
    power_directory = f"{param['base']}/power"
    os.makedirs(power_directory, exist_ok=True)
    for i in range(len(z_out) + 1):
        output_directory = f"{param['base']}/output_{i:05d}"
        os.makedirs(output_directory, exist_ok=True)
    ###################################################
    # Get cosmological table
    tables = cosmotable.generate(param)
    # aexp and t are overwritten if we read a snapshot
    param["aexp"] = 1.0 / (1 + param["z_start"])
    utils.set_units(param)
    # Initial conditions
    logging.warning(f"\n[bold blue]----- Initial conditions -----[/bold blue]\n")
    position, velocity = initial_conditions.generate(param, tables)
    param["t"] = tables[1](param["aexp"])
    logging.warning(f"{param['aexp']=} {param['t']=}")
    # Run code
    logging.warning(f"\n[bold blue]----- Run N-body -----[/bold blue]\n")
    param["nsteps"] = 0
    acceleration, potential, additional_field = solver.pm(position, param)
    aexp_out = 1.0 / (np.array(z_out) + 1)
    aexp_out.sort()
    t_out = tables[1](aexp_out)
    logging.info(f"{aexp_out=}")
    logging.info(f"{t_out}")
    i_snap = 1
    # Get output redshifts
    while param["aexp"] < 1.0:
        param["nsteps"] += 1
        param["i_snap"] = i_snap
        (
            position,
            velocity,
            acceleration,
            potential,
            additional_field,
        ) = integration.integrate(
            position,
            velocity,
            acceleration,
            potential,
            additional_field,
            tables,
            param,
            t_out[i_snap - 1],
        )  # Put None instead of potential if you do not want to use previous step

        if param["nsteps"] % param["n_reorder"] == 0:
            logging.warning("Reordering particles")
            utils.reorder_particles(position, velocity, acceleration)
        if param["write_snapshot"]:
            utils.write_snapshot_particles(position, velocity, param)
            i_snap += 1
        logging.warning(
            f"{param['nsteps']=} {param['aexp']=} z = {1.0 / param['aexp'] - 1}"
        )


def main():
    import argparse

    logging.warning("Read configuration file")
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="Configuration file", required=True)
    args = parser.parse_args()
    param = utils.read_param_file(args.config_file)
    logging.warning(param)
    run(param)


if __name__ == "__main__":
    from rich import print

    print(
        r"""
        _______           _______  _______  _______ 
        (  ____ )|\     /|(  ____ \(  ____ \(  ___  )
        | (    )|( \   / )| (    \/| (    \/| (   ) |
        | (____)| \ (_) / | (_____ | |      | |   | |
        |  _____)  \   /  (_____  )| |      | |   | |
        | (         ) (         ) || |      | |   | |
        | )         | |   /\____) || (____/\| (___) |
        |/          \_/   \_______)(_______/(_______)
                                                    
        """
    )
    print(f"VERSION: {__version__}")
    print(f"{__copyright__}")
    print(f"{'':{'-'}<{71}}\n")
    main()
    print("Run Completed!")
