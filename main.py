#!/usr/bin/env python
"""\
Main executable module to run cosmological N-body simulations

Usage: python main.py -c param.ini
"""
__author__ = "Michel-Andrès Breton"
__copyright__ = "Copyright 2022-2023, Michel-Andrès Breton"
__version__ = "0.1.7"
__email__ = "michel-andres.breton@obspm.fr"
__status__ = "Production"

import ast
import logging

import numba
import numpy as np

import cosmotable
import initial_conditions
import integration
import solver
import utils


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
    ###################################################
    # Get cosmological table
    logging.debug("Get table...")
    tables = cosmotable.generate(param)
    # aexp and t are overwritten if we read a snapshot
    param["aexp"] = 1.0 / (1 + param["z_start"])
    utils.set_units(param)
    # Initial conditions
    logging.debug(f"Initial conditions")
    position, velocity = initial_conditions.generate(param)
    param["t"] = tables[1](param["aexp"])
    print(f"{param['aexp']=} {param['t']=}")
    # Run code
    # Compute acceleration
    logging.debug("Compute initial acceleration")
    param["nsteps"] = 0
    acceleration, potential, additional_field = solver.pm(position, param)
    z_out = ast.literal_eval(param["z_out"])
    aexp_out = 1.0 / (np.array(z_out) + 1)
    aexp_out.sort()
    print(f"{aexp_out=}")
    i_snap = 1
    # Get output redshifts
    while param["aexp"] < 1.0:
        (
            position,
            velocity,
            acceleration,
            potential,
            additional_field,
        ) = integration.integrate(
            position, velocity, acceleration, potential, additional_field, tables, param
        )  # Put None instead of potential if you don't want to use previous step
        param["nsteps"] += 1
        # plt.imshow(potential[0])
        # plt.show()
        if param["nsteps"] % param["n_reorder"] == 0:
            print("Reordering particles")
            utils.reorder_particles(position, velocity, acceleration)
        if param["aexp"] >= aexp_out[i_snap - 1]:
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
