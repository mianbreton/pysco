import logging

import numpy as np
import numpy.typing as npt
import pandas as pd

import units
import utils


def generate(
    param: pd.Series,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Generate initial conditions

    Args:
        param (pd.Series): Parameters

    Returns:
        tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: 3D position, velocity
    """
    flag = "random"
    if flag == "random":
        position, velocity = random(param)
    elif flag == "sphere":
        position, velocity = sphere(param)
    else:
        # TODO: Read aexp and compute param['t']
        position, velocity = read_hdf5(
            "/home/mabreton/boxlen500_n64_lcdmw7v2/output_00001/pfof_cube_snap_part_data_boxlen500_n64_lcdmw7v2_00000_00000.h5",
            param,
        )
    # Wrap particles
    utils.reorder_particles(position, velocity)
    # Write initial distribution
    snap_name = f"{param['base']}/output_00000/particles.parquet"
    print(f"Write initial snapshot...{snap_name=} {param['aexp']=}")
    utils.write_snapshot_particles_parquet(f"{snap_name}", position, velocity)
    param.to_csv(f"{param['base']}/output_00000/param.txt", sep="=", header=False)
    return position, velocity


def random(param: pd.Series) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Generate random initial conditions

    Args:
        param (pd.Series): Parameters containing "ncoarse" key

    Returns:
        tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: 3D position, velocity
    """
    np.random.seed(42)  # set the random number generator seed
    npart = 8 ** param["ncoarse"]
    # Generate positions (Uniform random between [0,1])
    position = np.random.rand(3, npart).astype(np.float32)
    utils.periodic_wrap(position)
    # Generate velocities
    velocity = 0.007 * np.random.randn(3, npart).astype(np.float32)
    return position, velocity


def sphere(param: pd.Series) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Generate spherical initial conditions

    Args:
        param (pd.Series): Parameters containing "ncoarse" key

    Returns:
        tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: 3D position, velocity
    """
    logging.debug("Spherical initial conditions")
    np.random.seed(42)  # set the random number generator seed
    npart = 8 ** param["ncoarse"]
    center = [0.01, 0.5, 0.5]
    radius = 0.15
    phi = 2 * np.pi * np.random.rand(npart)
    cth = np.random.rand(npart) * 2 - 1
    sth = np.sin(np.arccos(cth))
    x = center[0] + radius * np.cos(phi) * sth
    y = center[1] + radius * np.sin(phi) * sth
    z = center[2] + radius * cth

    utils.periodic_wrap(x)
    utils.periodic_wrap(y)
    utils.periodic_wrap(z)
    return np.vstack([x, y, z]), np.zeros((3, npart))


def read_hdf5(
    filename: str, param: pd.Series
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Read initial conditions from HDF5 Ramses snapshot

    Args:
        filename (str): HDF5 snapshot filename
        param (pd.Series): Parameters, will be modified for "aexp" key

    Returns:
        tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: 3D position, velocity
    """
    import h5py

    logging.debug("HDF5 initial conditions")
    # Open file
    f = h5py.File(filename, "r")
    # Get scale factor
    param["aexp"] = f["metadata/ramses_info"].attrs["aexp"][0]
    units.set_units(param)
    # Get positions
    npart = int(f["metadata/npart_file"][:])
    position = np.empty((3, npart))
    velocity = np.empty_like(position)
    npart_grp_array = f["metadata/npart_grp_array"][:]

    print(npart, npart_grp_array)
    data = f["data"]
    istart = 0
    for i in range(npart_grp_array.shape[0]):
        name = "group" + str(i + 1).zfill(8)
        # print (name + '/position_part', npart_grp_array[i])
        # print (data[name + '/position_part'][:])
        position[:, istart : istart + npart_grp_array[i]] = data[
            name + "/position_part"
        ][:].T
        velocity[:, istart : istart + npart_grp_array[i]] = data[
            name + "/velocity_part"
        ][:].T
        istart += npart_grp_array[i]
    # Write in parquet file
    utils.write_snapshot_particles_parquet(
        "output/snap_00000.parquet", position, velocity
    )

    return position, velocity
