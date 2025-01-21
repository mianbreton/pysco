#!/usr/bin/env python
"""\
Main executable module to run cosmological N-body simulations

Usage: python main.py -c param.ini
"""
__author__ = "Michel-Andrès Breton"
__version__ = "1.0.1"
__email__ = "michel-andres.breton@obspm.fr"
__status__ = "Production"

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
import iostream
from time import perf_counter
import sys


def run(param) -> None:
    """This is the main function to run N-body simulations

    Parameters
    ----------
    param : dict or pd.Series
        Parameter container
    """
    # Ideally it would have been error/info/debug, but the latter triggers extensive Numba verbose
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
                show_time=False,
                show_level=False,
                show_path=False,
                enable_link_path=False,
                markup=True,
            )
        ],
        force=True,
    )

    if isinstance(param, Dict):
        param = pd.Series(param)
    elif isinstance(param, pd.Series):
        pass
    else:
        raise ValueError(f"{type(param)=}, should be a dictionnary or a Pandas Series")
    param["write_snapshot"] = False

    if param["nthreads"] > 0:
        numba.set_num_threads(param["nthreads"])
    else:
        param["nthreads"] = numba.get_num_threads()
    logging.warning(f"{param['nthreads']=}")

    if "pyfftw" in sys.modules:
        logging.warning("\n[bold green]FFT module: PyFFTW[/bold green]")
    else:
        logging.warning("\n[bold green]FFT module: NumPy[/bold green]")

    extra = param["theory"].casefold()
    if extra.casefold() == "fr".casefold():
        extra += f"{param['fR_logfR0']}_n{param['fR_n']}"
    elif extra.casefold() == "mond".casefold():
        mond_function = param["mond_function"].casefold()
        extra += f"_g0_{param['mond_g0']}_exponent_{param['mond_scale_factor_exponent']}_{mond_function}"
        if "simple".casefold() != mond_function:
            extra += f"_{param['mond_alpha']}"
    elif extra.casefold() == "parametrized".casefold():
        extra += f"_mu0_{param['parametrized_mu0']}"
    extra += f"_{param['linear_newton_solver']}_ncoarse{param['ncoarse']}"
    param["extra"] = extra
    z_out = ast.literal_eval(param["z_out"])

    power_directory = f"{param['base']}/power"
    os.makedirs(power_directory, exist_ok=True)
    for i in range(len(z_out) + 1):
        output_directory = f"{param['base']}/output_{i:05d}"
        os.makedirs(output_directory, exist_ok=True)

    logging.warning(
        f"\n[bold blue]----- Compute background cosmology -----[/bold blue]\n"
    )
    tables = cosmotable.generate(param)
    # aexp and t are overwritten if we read a snapshot
    param["aexp"] = 1.0 / (1 + param["z_start"])
    utils.set_units(param)
    if not "nsteps" in param.index:
        param["nsteps"] = 0
    logging.warning(f"\n[bold blue]----- Initial conditions -----[/bold blue]\n")
    position, velocity = initial_conditions.generate(param, tables)
    param["t"] = tables[1](np.log(param["aexp"]))
    logging.warning(f"{param['aexp']=} {param['t']=}")
    logging.warning(f"\n[bold blue]----- Run N-body -----[/bold blue]\n")
    acceleration, potential, additional_field = solver.pm(position, param)
    aexp_out = 1.0 / (np.array(z_out) + 1)
    aexp_out.sort()
    t_out = tables[1](np.log(aexp_out))
    logging.info(f"{aexp_out=}")
    logging.info(f"{t_out}")
    if not "i_snap" in param.index:
        param["i_snap"] = 1
    else:
        param["i_snap"] += 1

    while param["aexp"] < aexp_out[-1]:
        param["nsteps"] += 1
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
            t_out[param["i_snap"] - 1],
        )  # Put None instead of potential if you do not want to use previous step

        if (param["nsteps"] % param["n_reorder"]) == 0:
            logging.info("Reordering particles")
            position, velocity, acceleration = utils.reorder_particles(
                position, velocity, acceleration
            )
        if param["write_snapshot"]:
            iostream.write_snapshot_particles(position, velocity, param)
            param["i_snap"] += 1
        logging.warning(
            f"{param['nsteps']=} {param['aexp']=} z = {1.0 / param['aexp'] - 1}"
        )


def main():
    import argparse

    print("Read configuration file")
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="Configuration file", required=True)
    args = parser.parse_args()
    param = iostream.read_param_file(args.config_file)
    print(param)
    t_start = perf_counter()
    run(param)
    t_end = perf_counter()
    print(f"Simulation run time: {t_end - t_start} seconds.")


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
    print(f"{__author__}")
    print(f"{'':{'-'}<{71}}\n")
    main()
    print("Run Completed!")
