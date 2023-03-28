import ast
import logging
import sys

import matplotlib.pyplot as plt
import numba
import numpy as np

import cosmotable
import initial_conditions
import integration
import solver
import units
import utils


def main():
    ### Inputs ###
    # Read parameter file # wrong
    print("Read parameter file")
    if len(sys.argv) < 2:
        raise ValueError("usage: " + sys.argv[0] + " <param_file>")
    param = utils.read_param_file(sys.argv[1])
    print(param)
    # Threading
    if param["nthreads"] > 0:
        numba.set_num_threads(param["nthreads"])
    print(f"{numba.get_num_threads()=}")
    # Debug verbose
    if param["DEBUG"].casefold() == "True".casefold():
        logging.basicConfig(level=logging.DEBUG)
    ###################################################
    # Get cosmological table
    logging.debug("Get table...")
    func_a_t, func_t_a = cosmotable.generate(param)
    # func_a_t, func_t_a = cosmotable.read_ascii_ramses('/home/mabreton/boxlen500_n64_lcdmw7v2/ramses_input_lcdmw7v2.dat')
    param["aexp"] = 1.0 / (1 + param["z_start"])
    units.set_units(param)
    t_start = func_t_a(param["aexp"])
    param["t"] = t_start
    # Initial conditions
    logging.debug(f"Initial conditions")
    # position, velocity = initial_conditions.read_hdf5('/home/mabreton/boxlen500_n64_lcdmw7v2/output_00001/pfof_cube_snap_part_data_boxlen500_n64_lcdmw7v2_00000_00000.h5', param)
    # param.t = func_t_a(param.a)
    position, velocity = initial_conditions.generate(param)
    # Run code
    # Compute acceleration
    logging.debug("Compute initial acceleration")
    # PROFILING
    acceleration, potential = solver.pm(position, param)
    # TESTS
    step = 0
    z_out = ast.literal_eval(param["z_out"])
    aexp_out = 1.0 / (np.array(z_out) + 1)
    aexp_out.sort()
    print(f"{aexp_out=}")
    i_snap = 1
    # Get output redshifts
    while param["aexp"] < 1.0:
        position, velocity, acceleration, potential = integration.integrate(
            position, velocity, acceleration, potential, func_a_t, func_t_a, param
        )  # Put None instead of potential if you don't want to use previous step
        step += 1
        # plt.imshow(potential[0])
        # plt.show()
        if step % param["n_reorder"] == 0:
            print("Reordering particles")
            utils.reorder_particles3(position, velocity, acceleration)
        if param["aexp"] > aexp_out[i_snap - 1]:
            snap_name = f"{param['base']}/output_{i_snap:05d}/particles.parquet"
            print(f"Write snapshot...{snap_name=} {param['aexp']=}")
            utils.write_snapshot_particles_parquet(f"{snap_name}", position, velocity)
            param.to_csv(
                f"{param['base']}/output_{i_snap:05d}/param.txt", sep="=", header=False
            )

            i_snap += 1
        print(f"{step=} {param['aexp']=} z = {1.0 / param['aexp'] - 1}")


if __name__ == "__main__":
    main()
    print("Run Completed!")
