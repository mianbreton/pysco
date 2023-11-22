#!/usr/bin/env python
"""\
Main executable module to run cosmological N-body simulations

Usage: python main.py -c param.ini
"""
__author__ = "Michel-Andrès Breton"
__copyright__ = "Copyright 2022-2023, Michel-Andrès Breton"
__version__ = "0.1.13"
__email__ = "michel-andres.breton@obspm.fr"
__status__ = "Development"

import ast
import logging

import numba
import numpy as np
import os
import cosmotable
import initial_conditions
import integration
import solver
import utils

from rich import print


def run(param):
    # Threading
    if param["nthreads"] > 0:
        numba.set_num_threads(param["nthreads"])
    print(f"{numba.get_num_threads()=}")
    # Debug verbose
    if param["DEBUG"].casefold() == "True".casefold():
        logging.basicConfig(level=logging.DEBUG)
    extra = param["theory"].casefold()
    if extra.casefold() == "fr".casefold():
        extra += f"{param['fR_logfR0']}_n{param['fR_n']}"
    extra += f"_ncoarse{param['ncoarse']}"
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
    logging.debug("Get table...")
    tables = cosmotable.generate(param)
    # aexp and t are overwritten if we read a snapshot
    param["aexp"] = 1.0 / (1 + param["z_start"])
    utils.set_units(param)
    # Initial conditions
    print(f"\n[bold blue]----- Initial conditions -----[/bold blue]\n")
    position, velocity = initial_conditions.generate(param, tables)
    param["t"] = tables[1](param["aexp"])
    print(f"{param['aexp']=} {param['t']=}")
    # Run code
    # Compute acceleration
    logging.debug("Compute initial acceleration")
    print(f"\n[bold blue]----- Run N-body -----[/bold blue]\n")
    param["nsteps"] = 0
    acceleration, potential, additional_field = solver.pm(position, param)
    aexp_out = 1.0 / (np.array(z_out) + 1)
    aexp_out.sort()
    t_out = tables[1](aexp_out)
    print(f"{aexp_out=}")
    print(f"{t_out}")
    i_snap = 1
    # Get output redshifts
    while param["aexp"] < 1.0:
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
            t_out[i_snap - 1],
        )  # Put None instead of potential if you don't want to use previous step
        # plt.imshow(potential[0])
        # plt.show()
        if param["nsteps"] % param["n_reorder"] == 0:
            print("Reordering particles")
            utils.reorder_particles(position, velocity, acceleration)
        if param["write_snapshot"]:
            snap_name = f"{param['base']}/output_{i_snap:05d}/particles_{extra}.parquet"
            print(f"Write snapshot...{snap_name=} {param['aexp']=}")
            utils.write_snapshot_particles_parquet(f"{snap_name}", position, velocity)
            param.to_csv(
                f"{param['base']}/output_{i_snap:05d}/param_{extra}_{i_snap:05d}.txt",
                sep="=",
                header=False,
            )
            i_snap += 1
        print(f"{param['nsteps']=} {param['aexp']=} z = {1.0 / param['aexp'] - 1}")


def main():
    import argparse

    print("Read configuration file")
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="Configuration file", required=True)
    args = parser.parse_args()
    param = utils.read_param_file(args.config_file)
    print(param)
    run(param)


if __name__ == "__main__":
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
