"""
Cosmological Initial Conditions Generator

This script generates cosmological initial conditions for N-body simulations. 
It provides functions to generate density and force fields based on the linear power spectrum 
and various Lagrangian Perturbation Theory (LPT) approximations.
"""

import math
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple, List
from numba import config, njit, prange
import numpy.typing as npt
from scipy.interpolate import interp1d
import solver
import utils
from astropy.constants import pc
import logging
import iostream
import fourier


def generate(
    param: pd.Series, tables: List[interp1d]
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Generate initial conditions

    Parameters
    ----------
    param : pd.Series
        Parameter container

    tables : List[interp1d]
        Interpolated functions [lna(t), t(lna), H(lna), Dplus1(lna), f1(lna), Dplus2(lna), f2(lna), Dplus3a(lna), f3a(lna), Dplus3b(lna), f3b(lna), Dplus3c(lna), f3c(lna)]

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [Npart, 3]

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.utils import set_units
    >>> from pysco.initial_conditions import generate
    >>> from pysco import cosmotable
    >>> param = pd.Series({
    ...     'initial_conditions': '2LPT',
    ...     'z_start': 49.0,
    ...     "theory": "newton",
    ...     "H0": 70.0,
    ...     "Om_m": 0.3,
    ...     "T_cmb": 2.726,
    ...     "N_eff": 3.044,
    ...     "w0": -1.0,
    ...     "wa": 0.0,
    ...     "base": "./",
    ...     "extra": "test",
    ...     'power_spectrum_file': f"./examples/pk_lcdmw7v2.dat",
    ...     'boxlen': 500,
    ...     'npart': 32**3,
    ...     'seed': 42,
    ...     'fixed_ICS': False,
    ...     'paired_ICS': False,
    ...     'dealiased_ICS': False,
    ...     'position_ICS': "center",
    ...     'output_snapshot_format': "HDF5",
    ...     'nthreads': 2,
    ...  })
    >>> param["aexp"] = 1./(1 + param["z_start"])
    >>> set_units(param)
    >>> tables = cosmotable.generate(param)
    >>> position, velocity = generate(param, tables)
    """
    INITIAL_CONDITIONS = param["initial_conditions"]
    if isinstance(INITIAL_CONDITIONS, int):
        i_restart = int(INITIAL_CONDITIONS)
        param["initial_conditions"] = i_restart
        OUTPUT_SNAPSHOT_FORMAT = param["output_snapshot_format"].casefold()
        match OUTPUT_SNAPSHOT_FORMAT:
            case "parquet":
                filename = f"{param['base']}/output_{i_restart:05d}/particles_{param['extra']}.parquet"
                position, velocity = iostream.read_snapshot_particles_parquet(filename)
                param_filename = f"{param['base']}/output_{i_restart:05d}/param_{param['extra']}_{i_restart:05d}.txt"
                param_restart = iostream.read_param_file(param_filename)
                logging.warning(f"Parameter file read at ...{param_filename=}")
                for key in param_restart.index:
                    if key.casefold() is not "nthreads".casefold():
                        param[key] = param_restart[key]
            case "hdf5":
                import h5py

                filename = f"{param['base']}/output_{i_restart:05d}/particles_{param['extra']}.h5"
                position, velocity = iostream.read_snapshot_particles_hdf5(filename)
                with h5py.File(filename, "r") as h5r:
                    attrs = h5r.attrs
                    for key in attrs.keys():
                        if key.casefold() is not "nthreads".casefold():
                            param[key] = attrs[key]
            case _:
                raise ValueError(
                    f"{param['snapshot_format']=}, should be 'parquet' or 'hdf5'"
                )
        return position, velocity
    elif "LPT".casefold() in INITIAL_CONDITIONS.casefold():
        a_start = 1.0 / (1 + param["z_start"])
        lna_start = np.log(a_start)
        logging.warning(f"{param['z_start']=}")
        Hz = tables[2](lna_start)
        mpc_to_km = 1e3 * pc.value
        Hz *= param["unit_t"] / mpc_to_km  # km/s/Mpc to BU
        # psi_1lpt = generate_force(param)

        density_fourier = generate_density_fourier(param)

        fourier.inverse_laplacian(density_fourier)
        potential_1_fourier = density_fourier
        del density_fourier
        psi_1lpt_fourier = fourier.gradient(potential_1_fourier)
        psi_1lpt = fourier.ifft_3D_real_grad(psi_1lpt_fourier, param["nthreads"])
        psi_1lpt_fourier = 0
        # 1LPT
        logging.warning("Compute 1LPT contribution")
        dplus_1_z0 = tables[3](0)
        dplus_1 = np.float32(tables[3](lna_start) / dplus_1_z0)
        f1 = tables[4](lna_start)
        fH_1 = np.float32(f1 * Hz)
        position, velocity = initialise_1LPT(psi_1lpt, dplus_1, fH_1, param)
        psi_1lpt = 0
        if INITIAL_CONDITIONS.casefold() == "1LPT".casefold():
            position = position.reshape(param["npart"], 3)
            velocity = velocity.reshape(param["npart"], 3)
            finalise_initial_conditions(position, velocity, param, do_reorder=False)
            return position, velocity
        # 2LPT
        logging.warning("Compute 2LPT contribution")
        density_2 = compute_2ndorder_rhs(potential_1_fourier, param)
        density_2_fourier = fourier.fft_3D_real(density_2, param["nthreads"])
        density_2 = 0
        fourier.inverse_laplacian(density_2_fourier)
        potential_2_fourier = density_2_fourier
        del density_2_fourier
        psi_2lpt_fourier = fourier.gradient(potential_2_fourier)
        psi_2lpt = fourier.ifft_3D_real_grad(psi_2lpt_fourier, param["nthreads"])
        psi_2lpt_fourier = 0
        dplus_2 = np.float32(tables[5](lna_start) / dplus_1_z0**2)
        f2 = tables[6](lna_start)
        fH_2 = np.float32(f2 * Hz)
        add_nLPT(position, velocity, psi_2lpt, dplus_2, fH_2)
        # psi_2lpt = 0
        if INITIAL_CONDITIONS.casefold() == "2LPT".casefold():
            position = position.reshape(param["npart"], 3)
            velocity = velocity.reshape(param["npart"], 3)
            finalise_initial_conditions(position, velocity, param, do_reorder=False)
            return position, velocity
        # 3LPT
        dplus_3a = -np.float32(tables[7](lna_start) / dplus_1_z0**3)
        f3a = tables[8](lna_start)
        fH_3a = np.float32(f3a * Hz)
        dplus_3b = -np.float32(tables[9](lna_start) / dplus_1_z0**3)
        f3b = tables[10](lna_start)
        fH_3b = np.float32(f3b * Hz)
        dplus_3c = -np.float32(tables[11](lna_start) / dplus_1_z0**3)
        f3c = tables[12](lna_start)
        fH_3c = np.float32(f3c * Hz)
        logging.warning("Compute 3LPT a) contribution")
        psi_3lpt_a = compute_3a_displacement(potential_1_fourier, param)
        add_nLPT(position, velocity, psi_3lpt_a, dplus_3a, fH_3a)
        psi_3lpt_a = 0
        logging.warning("Compute 3LPT b) contribution")
        psi_3lpt_b = compute_3b_displacement(
            potential_1_fourier, potential_2_fourier, param
        )
        add_nLPT(position, velocity, psi_3lpt_b, dplus_3b, fH_3b)
        psi_3lpt_b = 0
        logging.warning("Compute 3LPT c) Ax contribution")
        psi_3lpt_c_Ax = compute_3c_Ax_displacement(
            potential_1_fourier, potential_2_fourier, param
        )
        add_nLPT(position, velocity, psi_3lpt_c_Ax, dplus_3c, fH_3c)
        psi_3lpt_c_Ax = 0
        logging.warning("Compute 3LPT c) Ay contribution")
        psi_3lpt_c_Ay = compute_3c_Ay_displacement(
            potential_1_fourier, potential_2_fourier, param
        )
        add_nLPT(position, velocity, psi_3lpt_c_Ay, dplus_3c, fH_3c)
        psi_3lpt_c_Ay = 0
        logging.warning("Compute 3LPT c) Az contribution")
        psi_3lpt_c_Az = compute_3c_Az_displacement(
            potential_1_fourier, potential_2_fourier, param
        )
        potential_1_fourier = 0
        potential_2_fourier = 0
        add_nLPT(position, velocity, psi_3lpt_c_Az, dplus_3c, fH_3c)
        psi_3lpt_c_Az = 0
        if INITIAL_CONDITIONS.casefold() == "3LPT".casefold():
            position = position.reshape(param["npart"], 3)
            velocity = velocity.reshape(param["npart"], 3)
            finalise_initial_conditions(position, velocity, param, do_reorder=False)
            return position, velocity
        else:
            raise ValueError(f"{INITIAL_CONDITIONS=}, should be 1LPT, 2LPT or 3LPT")
    elif INITIAL_CONDITIONS[-3:].casefold() == ".h5".casefold():
        position, velocity = read_hdf5(param)
        finalise_initial_conditions(position, velocity, param, do_reorder=True)
        return position, velocity
    else:  # Gadget format
        position, velocity = read_gadget(param)
        finalise_initial_conditions(position, velocity, param, do_reorder=True)
        return position, velocity


def finalise_initial_conditions(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    param: pd.Series,
    do_reorder: bool,
) -> None:
    """Wrap, reorder, and write initial conditions to output files.

    This function wraps and reorders particle positions, writes the initial
    distribution to an HDF5 or Parquet file.

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Particle positions [Npart, 3]
    velocity : npt.NDArray[np.float32]
        Particle velocities [Npart, 3]
    param : pd.Series
        Parameter container
    do_reorder: bool
        Reorder the particle

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.initial_conditions import finalise_initial_conditions
    >>> position = np.random.rand(64, 3).astype(np.float32)
    >>> velocity = np.random.rand(64, 3).astype(np.float32)
    >>> param = pd.Series({
    ...     'aexp': 0.8,
    ...     'base': '.',
    ...     'output_snapshot_format': 'HDF5',
    ...     'extra': 'test',
    ...  })
    >>> finalise_initial_conditions(position, velocity, param, True)
    """

    if "base" not in param:
        raise ValueError(f"{param.index=}, should contain 'base'")

    utils.periodic_wrap(position)
    if do_reorder:
        position, velocity = utils.reorder_particles(position, velocity)

    OUTPUT_SNAPSHOT_FORMAT = param["output_snapshot_format"].casefold()
    match OUTPUT_SNAPSHOT_FORMAT:
        case "parquet":
            snap_name = (
                f"{param['base']}/output_00000/particles_{param['extra']}.parquet"
            )
            iostream.write_snapshot_particles_parquet(snap_name, position, velocity)
            param.to_csv(
                f"{param['base']}/output_00000/param_{param['extra']}.txt",
                sep="=",
                header=False,
            )
        case "hdf5":
            snap_name = f"{param['base']}/output_00000/particles_{param['extra']}.h5"
            iostream.write_snapshot_particles_hdf5(snap_name, position, velocity, param)
        case _:
            raise ValueError(
                f"{param['snapshot_format']=}, should be 'parquet' or 'hdf5'"
            )
    logging.warning(f"Write initial snapshot...{snap_name=}")


def read_hdf5(
    param: pd.Series,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Read initial conditions from HDF5 Ramses snapshot

    Parameters
    ----------
    param : pd.Series
        Parameter container

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [3, Npart]

    Example
    -------
    >>> from pysco.initial_conditions import read_hdf5
    >>> param = pd.Series({
    ...     'initial_conditions': "file.h5",
    ...     'npart': 128**3,
    ...  })
    >>> position, velocity = read_hdf5(param)
    """
    import h5py

    logging.warning(f"Read {param['initial_conditions']}")
    f = h5py.File(param["initial_conditions"], "r")
    param["aexp"] = f["metadata/ramses_info"].attrs["aexp"][0]
    logging.warning(f"Initial redshift snapshot at z = {1./param['aexp'] - 1}")
    utils.set_units(param)

    npart = int(f["metadata/npart_file"][:])
    if npart != param["npart"]:
        raise ValueError(f"{npart=} and {param['npart']} should be equal.")
    position = np.empty((npart, 3), dtype=np.float32)
    velocity = np.empty_like(position, dtype=np.float32)
    npart_grp_array = f["metadata/npart_grp_array"][:]

    logging.info(f"{npart=}")
    data = f["data"]
    istart = 0
    for i in range(npart_grp_array.shape[0]):
        name = f"group{(i + 1):08d}"
        position[istart : istart + npart_grp_array[i]] = data[name + "/position_part"][
            :
        ]
        velocity[istart : istart + npart_grp_array[i]] = data[name + "/velocity_part"][
            :
        ]
        istart += npart_grp_array[i]

    return position, velocity


def read_gadget(
    param: pd.Series,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Read initial conditions from Gadget snapshot

    The snapshot can be divided in multiple files, such as \\
    snapshot_X.Y, where Y is the file number. \\
    In this case only keep snapshot_X

    Parameters
    ----------
    param : pd.Series
        Parameter container

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [3, Npart]

    Example
    -------
    >>> from pysco.initial_conditions import read_hdf5
    >>> param = pd.Series({
    ...     'initial_conditions': "file.h5",
    ...     'npart': 128**3,
    ...  })
    >>> position, velocity = read_gadget(param)
    """
    import readgadget  # From Pylians

    logging.warning(f"Read {param['initial_conditions']}")
    filename = param["initial_conditions"]
    ptype = 1  # DM particles
    header = readgadget.header(filename)
    BoxSize = header.boxsize  # in Gpc/h
    Nall = header.nall  # Total number of particles
    Omega_m = header.omega_m
    Omega_l = header.omega_l
    h = header.hubble
    redshift = header.redshift  # redshift of the snapshot
    aexp = 1.0 / (1 + redshift)
    param["aexp"] = aexp
    param["z_start"] = 1.0 / aexp - 1
    logging.warning(f"Initial redshift snapshot at z = {1./param['aexp'] - 1}")
    utils.set_units(param)

    npart = int(Nall[ptype])
    if npart != param["npart"]:
        raise ValueError(f"{npart=} and {param['npart']} should be equal.")
    if not np.allclose([Omega_m, Omega_l, 100 * h], [param["Om_m"], param["H0"]]):
        raise ValueError(
            f"Cosmology mismatch: {Omega_m=} {param['Om_m']=} {(100*h)=} {param['H0']=}"
        )

    position = readgadget.read_block(filename, "POS ", [ptype])
    velocity = readgadget.read_block(filename, "VEL ", [ptype])

    vel_factor = param["unit_t"] / param["unit_l"]
    utils.prod_vector_scalar_inplace(position, np.float32(1.0 / BoxSize))
    utils.prod_vector_scalar_inplace(velocity, np.float32(vel_factor))
    return position, velocity


@utils.time_me
def generate_density_fourier(param: pd.Series) -> npt.NDArray[np.complex64]:
    """Compute density initial conditions from power spectrum

    Parameters
    ----------
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.complex64]
        Fourier-space density field [N, N, N]

    Example
    -------
    >>> import pandas as pd
    >>> from pysco.initial_conditions import generate_density_fourier
    >>> param = pd.Series({
         'power_spectrum_file': './examples/pk_lcdmw7v2.dat',
         'npart': 64,
         'seed': 42,
         'boxlen': 100,
         'fixed_ICS': False,
         'paired_ICS': False,
     })
    >>> generate_density_fourier(param)
    """
    transfer_grid = get_transfer_grid(param)

    ncells_1d = int(math.cbrt(param["npart"]))
    seed = param["seed"]
    if seed < 0:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(param["seed"])

    if param["fixed_ICS"]:
        density_k = white_noise_fourier_fixed(ncells_1d, rng, param["paired_ICS"])
    else:
        density_k = white_noise_fourier(ncells_1d, rng)

    utils.prod_vector_vector_inplace(density_k, transfer_grid)
    transfer_grid = 0
    return density_k


@utils.time_me
def generate_density(param: pd.Series) -> npt.NDArray[np.float32]:
    """Compute density initial conditions from power spectrum

    Parameters
    ----------
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        Initial density field and velocity field (delta, vx, vy, vz)

    Example
    -------
    >>> import pandas as pd
    >>> from pysco.initial_conditions import generate_density
    >>> param = pd.Series({
         'power_spectrum_file': './examples/pk_lcdmw7v2.dat',
         'npart': 64,
         'seed': 42,
         'boxlen': 100,
         'fixed_ICS': False,
         'paired_ICS': False,
         'nthreads': 2,
     })
    >>> generate_density(param)
    """
    density_k = generate_density_fourier(param)
    density = fourier.ifft_3D_real(density_k, param["nthreads"])
    density_k = 0
    return density


@utils.time_me
def generate_force(param: pd.Series) -> npt.NDArray[np.float32]:
    """Compute force initial conditions from power spectrum

    Parameters
    ----------
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        Initial density field and velocity field (delta, vx, vy, vz)

    Example
    -------
    >>> import pandas as pd
    >>> from pysco.initial_conditions import generate_force
    >>> param = pd.Series({
         'power_spectrum_file': './examples/pk_lcdmw7v2.dat',
         'npart': 16**3,
         'seed': 42,
         'boxlen': 100,
         'fixed_ICS': False,
         'paired_ICS': False,
         'nthreads': 2,
     })
    >>> generate_force(param)
    """
    transfer_grid = get_transfer_grid(param)
    ncells_1d = int(math.cbrt(param["npart"]))
    seed = param["seed"]
    if seed < 0:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(param["seed"])

    if param["fixed_ICS"]:
        force = white_noise_fourier_fixed_force(ncells_1d, rng, param["paired_ICS"])
    else:
        force = white_noise_fourier_force(ncells_1d, rng)
    utils.prod_gradient_vector_inplace(force, transfer_grid)
    transfer_grid = 0
    force = fourier.ifft_3D_real_grad(force, param["nthreads"])
    return force


@utils.time_me
def get_transfer_grid(param: pd.Series) -> npt.NDArray[np.float32]:
    """Compute transfer 3D grid

    Computes sqrt(P(k,z)) on a 3D grid

    Parameters
    ----------
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        Initial density field and velocity field (delta, vx, vy, vz)

    Example
    -------
    >>> import pandas as pd
    >>> from pysco.initial_conditions import get_transfer_grid
    >>> param = pd.Series({
         'power_spectrum_file': './examples/pk_lcdmw7v2.dat',
         'npart': 16**3,
         'boxlen': 100,
     })
    >>> get_transfer_grid(param)
    """
    k, Pk = np.loadtxt(param["power_spectrum_file"]).T

    ncells_1d = int(math.cbrt(param["npart"]))
    if param["npart"] != ncells_1d**3:
        raise ValueError(f"{math.cbrt(param['npart'])=}, should be integer")
    kf = 2 * np.pi / param["boxlen"]
    k_dimensionless = k / kf
    sqrtPk = (np.sqrt(Pk / param["boxlen"] ** 3) * ncells_1d**3).astype(np.float32)
    k_1d = np.fft.fftfreq(ncells_1d, 1 / ncells_1d)
    k_grid = np.sqrt(
        k_1d[np.newaxis, np.newaxis, :] ** 2
        + k_1d[:, np.newaxis, np.newaxis] ** 2
        + k_1d[np.newaxis, :, np.newaxis] ** 2
    )
    k_1d = 0
    transfer_grid = np.interp(k_grid, k_dimensionless, sqrtPk)
    """ transfer_grid = 10 ** np.interp(
        np.log10(k_grid), np.log10(k_dimensionless), np.log10(sqrtPk)
    ) """
    return transfer_grid


@utils.time_me
@njit(
    fastmath=True,
    cache=True,
    parallel=True,
)
def white_noise_fourier(
    ncells_1d: int, rng: np.random.Generator
) -> npt.NDArray[np.complex64]:
    """Generate Fourier-space white noise on a regular 3D grid

    Parameters
    ----------
    ncells_1d : int
        Number of cells along one direction
    rng : np.random.Generator
        Random generator (NumPy)

    Returns
    -------
    npt.NDArray[np.complex64]
        3D white-noise field for density [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import white_noise_fourier
    >>> white_noise = white_noise_fourier(16, np.random.default_rng())
    >>> print(white_noise)
    """
    twopi = np.float32(2 * math.pi)
    ii = np.complex64(1j)
    one = np.float32(1)
    middle = ncells_1d // 2
    density = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.complex64)
    # Must compute random before parallel loop to ensure reproductability
    rng_amplitudes = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    rng_phases = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    for i in prange(middle + 1):
        im = -np.int32(i)
        for j in prange(ncells_1d):
            jm = -j
            for k in prange(ncells_1d):
                km = -k
                phase = twopi * rng_phases[i, j, k]
                amplitude = math.sqrt(
                    -math.log(
                        one - rng_amplitudes[i, j, k]
                    )  # rng.random in range [0,1), must ensure no NaN
                )  # Rayleigh sampling
                real = amplitude * math.cos(phase)
                imaginary = ii * amplitude * math.sin(phase)
                result_upper = real + imaginary
                result_lower = real - imaginary
                density[i, j, k] = result_upper
                density[im, jm, km] = result_lower
    rng_phases = 0
    rng_amplitudes = 0
    # Fix corners
    density[0, 0, 0] = 0
    density[0, 0, middle] = math.sqrt(-math.log(one - rng.random(dtype=np.float32)))
    density[0, middle, 0] = math.sqrt(-math.log(one - rng.random(dtype=np.float32)))
    density[0, middle, middle] = math.sqrt(
        -math.log(one - rng.random(dtype=np.float32))
    )
    density[middle, 0, 0] = math.sqrt(-math.log(one - rng.random(dtype=np.float32)))
    density[middle, 0, middle] = math.sqrt(
        -math.log(one - rng.random(dtype=np.float32))
    )
    density[middle, middle, 0] = math.sqrt(
        -math.log(one - rng.random(dtype=np.float32))
    )
    density[middle, middle, middle] = math.sqrt(
        -math.log(one - rng.random(dtype=np.float32))
    )

    return density


@utils.time_me
@njit(
    fastmath=True,
    cache=True,
    parallel=True,
)
def white_noise_fourier_fixed(
    ncells_1d: int, rng: np.random.Generator, is_paired: bool
) -> npt.NDArray[np.complex64]:
    """Generate Fourier-space white noise with fixed amplitude on a regular 3D grid

    Parameters
    ----------
    ncells_1d : int
        Number of cells along one direction
    rng : np.random.Generator
        Random generator (NumPy)
    is_paired : bool
        If paired, add π to the random phases

    Returns
    -------
    Tuple[npt.NDArray[np.complex64], npt.NDArray[np.complex64]]
        3D white-noise field for density [N_cells_1d, N_cells_1d, N_cells_1d]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import white_noise_fourier_fixed
    >>> paired_white_noise = white_noise_fourier_fixed(16, np.random.default_rng(), 1)
    >>> print(paired_white_noise)
    """
    twopi = np.float32(2 * np.pi)
    one = np.float32(1)
    ii = np.complex64(1j)
    middle = ncells_1d // 2
    if is_paired:
        shift = np.float32(math.pi)
    else:
        shift = np.float32(0)
    density = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.complex64)
    rng_phases = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    for i in prange(middle + 1):
        im = -np.int32(i)
        for j in prange(ncells_1d):
            jm = -j
            for k in prange(ncells_1d):
                km = -k
                phase = twopi * rng_phases[i, j, k] + shift
                real = math.cos(phase)
                imaginary = ii * math.sin(phase)
                result_upper = real + imaginary
                result_lower = real - imaginary
                density[i, j, k] = result_upper
                density[im, jm, km] = result_lower
    rng_phases = 0
    density[0, 0, 0] = 0
    density[0, 0, middle] = one
    density[0, middle, 0] = one
    density[0, middle, middle] = one
    density[middle, 0, 0] = one
    density[middle, 0, middle] = one
    density[middle, middle, 0] = one
    density[middle, middle, middle] = one
    return density


@utils.time_me
@njit(
    fastmath=True,
    cache=True,
    parallel=True,
    error_model="numpy",
)
def white_noise_fourier_force(
    ncells_1d: int, rng: np.random.Generator
) -> npt.NDArray[np.complex64]:
    """Generate Fourier-space white FORCE noise on a regular 3D grid

    Parameters
    ----------
    ncells_1d : int
        Number of cells along one direction
    rng : np.random.Generator
        Random generator (NumPy)

    Returns
    -------
    Tuple[npt.NDArray[np.complex64], npt.NDArray[np.complex64]]
        3D white-noise field for force [N_cells_1d, N_cells_1d, N_cells_1d, 3]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import white_noise_fourier_force
    >>> ncells_1d = 16
    >>> rng = np.random.default_rng(42)
    >>> white_noise_fourier_force(ncells_1d, rng)
    """
    invtwopi = np.float32(0.5 / np.pi)
    one = np.float32(1)
    twopi = np.float32(2 * np.pi)
    ii = np.complex64(1j)
    middle = ncells_1d // 2
    force = np.empty((ncells_1d, ncells_1d, ncells_1d, 3), dtype=np.complex64)
    # Must compute random before parallel loop to ensure reproductability
    rng_amplitudes = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    rng_phases = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    for i in prange(middle + 1):
        im = -np.int32(i)
        if i >= middle:
            kx = np.float32(i - ncells_1d)
        else:
            kx = np.float32(i)
        kx2 = kx**2
        for j in prange(ncells_1d):
            jm = -j
            if j >= middle:
                ky = np.float32(j - ncells_1d)
            else:
                ky = np.float32(j)
            kx2_ky2 = kx2 + ky**2
            for k in prange(ncells_1d):
                km = -k
                kz = np.float32(k)
                invk2 = one / (kx2_ky2 + kz**2)
                phase = twopi * rng_phases[i, j, k]
                amplitude = math.sqrt(
                    -math.log(
                        one - rng_amplitudes[i, j, k]
                    )  # rng.random in range [0,1), must ensure no NaN
                )  # Rayleigh sampling
                real = amplitude * math.cos(phase)
                imaginary = ii * amplitude * math.sin(phase)
                result_upper = real + imaginary
                result_lower = real - imaginary
                i_phi_upper = ii * invtwopi * result_upper * invk2
                i_phi_lower = ii * invtwopi * result_lower * invk2
                force[i, j, k, 0] = -kx * i_phi_upper
                force[i, j, k, 1] = -ky * i_phi_upper
                force[i, j, k, 2] = -kz * i_phi_upper
                force[im, jm, km, 0] = kx * i_phi_lower
                force[im, jm, km, 1] = ky * i_phi_lower
                force[im, jm, km, 2] = kz * i_phi_lower
    rng_phases = 0
    rng_amplitudes = 0
    # Fix edges
    inv2 = np.float32(0.5)
    inv3 = np.float32(1.0 / 3)
    invkmiddle = -np.float32((twopi * middle) ** (-1))

    force_1_1_0 = (
        invkmiddle * inv2 * math.sqrt(-math.log(one - rng.random(dtype=np.float32)))
    )
    force_0_1_1 = (
        invkmiddle * inv2 * math.sqrt(-math.log(one - rng.random(dtype=np.float32)))
    )
    force_1_0_1 = (
        invkmiddle * inv2 * math.sqrt(-math.log(one - rng.random(dtype=np.float32)))
    )
    force_1_1_1 = (
        invkmiddle * inv3 * math.sqrt(-math.log(one - rng.random(dtype=np.float32)))
    )

    # 0
    force[0, 0, 0, 0] = 0
    force[0, 0, 0, 1] = 0
    force[0, 0, 0, 2] = 0
    # 1
    force[0, middle, 0, 0] = 0
    force[0, 0, middle, 0] = 0
    force[0, middle, middle, 0] = 0
    force[middle, 0, 0, 1] = 0
    force[0, 0, middle, 1] = 0
    force[middle, 0, middle, 1] = 0
    force[middle, 0, 0, 2] = 0
    force[0, middle, 0, 2] = 0
    force[middle, middle, 0, 2] = 0

    force[middle, 0, 0, 0] = invkmiddle * math.sqrt(
        -math.log(one - rng.random(dtype=np.float32))
    )
    force[0, middle, 0, 1] = invkmiddle * math.sqrt(
        -math.log(one - rng.random(dtype=np.float32))
    )
    force[0, 0, middle, 2] = invkmiddle * math.sqrt(
        -math.log(one - rng.random(dtype=np.float32))
    )
    # 2
    force[middle, middle, 0, 0] = force_1_1_0
    force[middle, 0, middle, 0] = force_1_0_1
    force[middle, middle, 0, 1] = force_1_1_0
    force[0, middle, middle, 1] = force_0_1_1
    force[0, middle, middle, 2] = force_0_1_1
    force[0, middle, middle, 2] = force_0_1_1
    # 3
    force[middle, middle, middle, 0] = force_1_1_1
    force[middle, middle, middle, 1] = force_1_1_1
    force[middle, middle, middle, 2] = force_1_1_1
    return force


@utils.time_me
@njit(
    fastmath=True,
    cache=True,
    parallel=True,
    error_model="numpy",
)
def white_noise_fourier_fixed_force(
    ncells_1d: int, rng: np.random.Generator, is_paired: bool
) -> npt.NDArray[np.complex64]:
    """Generate Fourier-space white FORCE noise with fixed amplitude on a regular 3D grid

    Parameters
    ----------
    ncells_1d : int
        Number of cells along one direction
    rng : np.random.Generator
        Random generator (NumPy)
    is_paired : bool
        If paired, add π to the random phases

    Returns
    -------
    Tuple[npt.NDArray[np.complex64], npt.NDArray[np.complex64]]
        3D white-noise field for force [N_cells_1d, N_cells_1d, N_cells_1d, 3]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import white_noise_fourier_fixed_force
    >>> ncells_1d = 16
    >>> rng = np.random.default_rng(42)
    >>> is_paired = True
    >>> white_noise_fourier_fixed_force(ncells_1d, rng, is_paired)
    """
    invtwopi = np.float32(0.5 / np.pi)
    one = np.float32(1)
    twopi = np.float32(2 * np.pi)
    ii = np.complex64(1j)
    middle = ncells_1d // 2
    if is_paired:
        shift = np.float32(math.pi)
    else:
        shift = np.float32(0)
    force = np.empty((ncells_1d, ncells_1d, ncells_1d, 3), dtype=np.complex64)
    rng_phases = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    for i in prange(middle + 1):
        im = -np.int32(i)
        if i >= middle:
            kx = np.float32(i - ncells_1d)
        else:
            kx = np.float32(i)
        kx2 = kx**2
        for j in prange(ncells_1d):
            jm = -j
            if j >= middle:
                ky = np.float32(j - ncells_1d)
            else:
                ky = np.float32(j)
            kx2_ky2 = kx2 + ky**2
            for k in prange(ncells_1d):
                km = -k
                kz = np.float32(k)
                invk2 = one / (kx2_ky2 + kz**2)
                phase = twopi * rng_phases[i, j, k] + shift
                real = math.cos(phase)
                imaginary = ii * math.sin(phase)
                result_upper = real + imaginary
                result_lower = real - imaginary
                i_phi_upper = ii * invtwopi * result_upper * invk2
                i_phi_lower = ii * invtwopi * result_lower * invk2
                force[i, j, k, 0] = -kx * i_phi_upper
                force[i, j, k, 1] = -ky * i_phi_upper
                force[i, j, k, 2] = -kz * i_phi_upper
                force[im, jm, km, 0] = kx * i_phi_lower
                force[im, jm, km, 1] = ky * i_phi_lower
                force[im, jm, km, 2] = kz * i_phi_lower
    rng_phases = 0
    # Fix edges
    inv2 = np.float32(0.5)
    inv3 = np.float32(1.0 / 3)
    invkmiddle = -np.float32((twopi * middle) ** (-1))
    force_2 = invkmiddle * inv2
    force_3 = invkmiddle * inv3
    # 0
    force[0, 0, 0, 0] = 0
    force[0, 0, 0, 1] = 0
    force[0, 0, 0, 2] = 0
    # 1
    force[0, middle, 0, 0] = 0
    force[0, 0, middle, 0] = 0
    force[0, middle, middle, 0] = 0
    force[middle, 0, 0, 1] = 0
    force[0, 0, middle, 1] = 0
    force[middle, 0, middle, 1] = 0
    force[middle, 0, 0, 2] = 0
    force[0, middle, 0, 2] = 0
    force[middle, middle, 0, 2] = 0

    force[middle, 0, 0, 0] = invkmiddle
    force[0, middle, 0, 1] = invkmiddle
    force[0, 0, middle, 2] = invkmiddle
    # 2
    force[middle, middle, 0, 0] = force_2
    force[middle, 0, middle, 0] = force_2
    force[middle, middle, 0, 1] = force_2
    force[0, middle, middle, 1] = force_2
    force[0, middle, middle, 2] = force_2
    force[0, middle, middle, 2] = force_2
    # 3
    force[middle, middle, middle, 0] = force_3
    force[middle, middle, middle, 1] = force_3
    force[middle, middle, middle, 2] = force_3
    return force


def compute_2ndorder_rhs(
    phi_1_fourier: npt.NDArray[np.complex64], param: pd.Series
) -> npt.NDArray[np.float32]:
    """Compute 2LPT displacement [Scoccimarro 1998 Appendix B.2]

    Parameters
    ----------
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        2LPT displacement field [N, N, N]

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pysco.initial_conditions import compute_2ndorder_rhs
    >>> param = pd.Series({
    ...     'nthreads': 2,
    ...     'dealiased_ICS': False,
    ...  })
    >>> phi = np.random.random((16,16,9)).astype(np.complex64)
    >>> rhs_2ndorder = compute_2ndorder_rhs(phi, param)
    """
    one = np.float32(1)
    nthreads = param["nthreads"]

    if param["dealiased_ICS"]:
        phi_1_fourier = pad(phi_1_fourier)

    tmp = fourier.hessian(phi_1_fourier, (0, 0))
    phi_2 = fourier.ifft_3D_real(tmp, nthreads)
    tmp = fourier.sum_of_hessian(phi_1_fourier, (1, 1), (2, 2))
    tmp = fourier.ifft_3D_real(tmp, nthreads)
    utils.prod_vector_vector_inplace(phi_2, tmp)
    tmp1 = fourier.hessian(phi_1_fourier, (1, 1))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.hessian(phi_1_fourier, (2, 2))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    utils.add_vector_vector_inplace(phi_2, one, tmp1, tmp2)
    tmp1 = 0
    tmp2 = 0
    tmp = fourier.hessian(phi_1_fourier, (0, 1))
    tmp = fourier.ifft_3D_real(tmp, nthreads)
    utils.add_vector_vector_inplace(phi_2, -one, tmp, tmp)
    tmp = fourier.hessian(phi_1_fourier, (0, 2))
    tmp = fourier.ifft_3D_real(tmp, nthreads)
    utils.add_vector_vector_inplace(phi_2, -one, tmp, tmp)
    tmp = fourier.hessian(phi_1_fourier, (1, 2))
    tmp = fourier.ifft_3D_real(tmp, nthreads)
    utils.add_vector_vector_inplace(phi_2, -one, tmp, tmp)
    tmp = 0

    if param["dealiased_ICS"]:
        phi_2 = fourier.fft_3D_real(phi_2, nthreads)
        phi_2 = trim(phi_2)
        phi_2 = fourier.ifft_3D_real(phi_2, nthreads)
        utils.prod_vector_scalar_inplace(phi_2, np.float32(1.5**3))
    return phi_2


def compute_3a_rhs(
    phi_1_fourier: npt.NDArray[np.complex64], param: pd.Series
) -> npt.NDArray[np.float32]:
    """Compute 3aLPT displacement [Scoccimarro 1998 Appendix B.2]

    Parameters
    ----------
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        2LPT displacement field [N, N, N]

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pysco.initial_conditions import compute_3a_rhs
    >>> param = pd.Series({
    ...     'nthreads': 2,
    ...     'dealiased_ICS': False,
    ...  })
    >>> phi = np.random.random((16,16,16)).astype(np.complex64)
    >>> rhs = compute_3a_rhs(phi, param)
    """
    one = np.float32(1)
    two = np.float32(2)
    nthreads = param["nthreads"]

    if param["dealiased_ICS"]:
        phi_1_fourier = pad(phi_1_fourier)

    tmp = fourier.hessian(phi_1_fourier, (0, 0))
    phi_3a = fourier.ifft_3D_real(tmp, nthreads)
    tmp = fourier.hessian(phi_1_fourier, (1, 1))
    tmp = fourier.ifft_3D_real(tmp, nthreads)
    utils.prod_vector_vector_inplace(phi_3a, tmp)
    tmp = fourier.hessian(phi_1_fourier, (2, 2))
    tmp = fourier.ifft_3D_real(tmp, nthreads)
    utils.prod_vector_vector_inplace(phi_3a, tmp)

    tmp1 = fourier.hessian(phi_1_fourier, (0, 1))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.hessian(phi_1_fourier, (0, 2))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    tmp3 = fourier.hessian(phi_1_fourier, (1, 2))
    tmp3 = fourier.ifft_3D_real(tmp3, nthreads)
    utils.add_vector_vector_vector_inplace(phi_3a, two, tmp1, tmp2, tmp3)
    tmp1 = 0
    tmp2 = 0
    tmp3 = 0

    tmp1 = fourier.hessian(phi_1_fourier, (1, 2))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.hessian(phi_1_fourier, (0, 0))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    utils.add_vector_vector_vector_inplace(phi_3a, -one, tmp1, tmp1, tmp2)

    tmp1 = fourier.hessian(phi_1_fourier, (0, 2))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.hessian(phi_1_fourier, (1, 1))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    utils.add_vector_vector_vector_inplace(phi_3a, -one, tmp1, tmp1, tmp2)

    tmp1 = fourier.hessian(phi_1_fourier, (0, 1))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.hessian(phi_1_fourier, (2, 2))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    utils.add_vector_vector_vector_inplace(phi_3a, -one, tmp1, tmp1, tmp2)

    if param["dealiased_ICS"]:
        phi_3a = fourier.fft_3D_real(phi_3a, nthreads)
        phi_3a = trim(phi_3a)
        phi_3a = fourier.ifft_3D_real(phi_3a, nthreads)
        utils.prod_vector_scalar_inplace(phi_3a, np.float32(1.5**6))
    return phi_3a


def compute_3a_displacement(
    phi_1_fourier: npt.NDArray[np.complex64], param: pd.Series
) -> npt.NDArray[np.float32]:
    """Compute 3aLPT displacement [Scoccimarro 1998 Appendix B.2]

    Parameters
    ----------
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        2LPT displacement field [N, N, N]

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pysco.initial_conditions import compute_3a_displacement
    >>> param = pd.Series({
    ...     'nthreads': 2,
    ...     'dealiased_ICS': False,
    ...  })
    >>> phi_1 = np.random.random((16,16,16)).astype(np.complex64)
    >>> phi_2 = np.random.random((16,16,16)).astype(np.complex64)
    >>> rhs = compute_3a_displacement(phi_1, param)
    """
    density_3a = compute_3a_rhs(phi_1_fourier, param)
    density_3a_fourier = fourier.fft_3D_real(density_3a, param["nthreads"])
    psi_3lpt_a_fourier = fourier.gradient_inverse_laplacian(density_3a_fourier)
    psi_3lpt_a = fourier.ifft_3D_real_grad(psi_3lpt_a_fourier, param["nthreads"])

    return psi_3lpt_a


def compute_3b_rhs(
    phi_1_fourier: npt.NDArray[np.complex64],
    phi_2_fourier: npt.NDArray[np.complex64],
    param: pd.Series,
) -> npt.NDArray[np.float32]:
    """Compute 3bLPT displacement

    Parameters
    ----------
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    phi_2_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        3bLPT displacement field [N, N, N]

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pysco.initial_conditions import compute_3b_rhs
    >>> param = pd.Series({
    ...     'nthreads': 2,
    ...     'dealiased_ICS': False,
    ...  })
    >>> phi_1 = np.random.random((16,16,16)).astype(np.complex64)
    >>> phi_2 = np.random.random((16,16,16)).astype(np.complex64)
    >>> rhs = compute_3b_rhs(phi_1, phi_2, param)
    """
    one = np.float32(1)
    half = np.float32(0.5)
    nthreads = param["nthreads"]

    if param["dealiased_ICS"]:
        phi_1_fourier = pad(phi_1_fourier)
        phi_2_fourier = pad(phi_2_fourier)

    tmp = fourier.hessian(phi_1_fourier, (0, 0))
    phi_3b = fourier.ifft_3D_real(tmp, nthreads)
    tmp = fourier.sum_of_hessian(phi_2_fourier, (1, 1), (2, 2))
    tmp = fourier.ifft_3D_real(tmp, nthreads)
    utils.prod_vector_vector_scalar_inplace(phi_3b, tmp, half)
    tmp = 0

    tmp1 = fourier.hessian(phi_1_fourier, (1, 1))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.sum_of_hessian(phi_2_fourier, (0, 0), (2, 2))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    utils.add_vector_vector_inplace(phi_3b, half, tmp1, tmp2)

    tmp1 = fourier.hessian(phi_1_fourier, (2, 2))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.sum_of_hessian(phi_2_fourier, (0, 0), (1, 1))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    utils.add_vector_vector_inplace(phi_3b, half, tmp1, tmp2)

    tmp1 = fourier.hessian(phi_1_fourier, (0, 1))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.hessian(phi_2_fourier, (0, 1))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    utils.add_vector_vector_inplace(phi_3b, -one, tmp1, tmp2)

    tmp1 = fourier.hessian(phi_1_fourier, (0, 2))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.hessian(phi_2_fourier, (0, 2))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    utils.add_vector_vector_inplace(phi_3b, -one, tmp1, tmp2)

    tmp1 = fourier.hessian(phi_1_fourier, (1, 2))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.hessian(phi_2_fourier, (1, 2))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    utils.add_vector_vector_inplace(phi_3b, -one, tmp1, tmp2)

    if param["dealiased_ICS"]:
        phi_3b = fourier.fft_3D_real(phi_3b, nthreads)
        phi_3b = trim(phi_3b)
        phi_3b = fourier.ifft_3D_real(phi_3b, nthreads)
        utils.prod_vector_scalar_inplace(phi_3b, np.float32(1.5**3))
    return phi_3b


def compute_3b_displacement(
    phi_1_fourier: npt.NDArray[np.complex64],
    phi_2_fourier: npt.NDArray[np.complex64],
    param: pd.Series,
) -> npt.NDArray[np.float32]:
    """Compute 3bLPT displacement

    Parameters
    ----------
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    phi_2_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        3bLPT displacement field [N, N, N]

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pysco.initial_conditions import compute_3b_displacement
    >>> param = pd.Series({
    ...     'nthreads': 2,
    ...     'dealiased_ICS': False,
    ...  })
    >>> phi_1 = np.random.random((16,16,16)).astype(np.complex64)
    >>> phi_2 = np.random.random((16,16,16)).astype(np.complex64)
    >>> rhs = compute_3b_displacement(phi_1, phi_2, param)
    """
    density_3b = compute_3b_rhs(phi_1_fourier, phi_2_fourier, param)
    density_3b_fourier = fourier.fft_3D_real(density_3b, param["nthreads"])
    psi_3lpt_b_fourier = fourier.gradient_inverse_laplacian(density_3b_fourier)
    psi_3lpt_b = fourier.ifft_3D_real_grad(psi_3lpt_b_fourier, param["nthreads"])
    return psi_3lpt_b


def compute_3c_Ax_rhs(
    phi_1_fourier: npt.NDArray[np.complex64],
    phi_2_fourier: npt.NDArray[np.complex64],
    param: pd.Series,
) -> npt.NDArray[np.float32]:
    """Compute 3cLPT displacement

    Parameters
    ----------
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    phi_2_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        3cLPT displacement field [N, N, N]

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pysco.initial_conditions import compute_3c_Ax_rhs
    >>> param = pd.Series({
    ...     'nthreads': 2,
    ...     'dealiased_ICS': False,
    ...  })
    >>> phi_1 = np.random.random((16,16,16)).astype(np.complex64)
    >>> phi_2 = np.random.random((16,16,16)).astype(np.complex64)
    >>> rhs = compute_3c_Ax_rhs(phi_1, phi_2, param)
    """
    one = np.float32(1)
    nthreads = param["nthreads"]

    if param["dealiased_ICS"]:
        phi_1_fourier = pad(phi_1_fourier)
        phi_2_fourier = pad(phi_2_fourier)

    tmp = fourier.hessian(phi_1_fourier, (0, 2))
    phi_3c = fourier.ifft_3D_real(tmp, nthreads)
    tmp = fourier.hessian(phi_2_fourier, (0, 1))
    tmp = fourier.ifft_3D_real(tmp, nthreads)
    utils.prod_vector_vector_scalar_inplace(phi_3c, tmp, one)
    tmp = 0

    tmp1 = fourier.hessian(phi_2_fourier, (0, 2))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.hessian(phi_1_fourier, (0, 1))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    utils.add_vector_vector_inplace(phi_3c, -one, tmp1, tmp2)

    tmp1 = fourier.hessian(phi_1_fourier, (1, 2))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.diff_of_hessian(phi_2_fourier, (1, 1), (2, 2))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    utils.add_vector_vector_inplace(phi_3c, one, tmp1, tmp2)

    tmp1 = fourier.hessian(phi_2_fourier, (1, 2))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.diff_of_hessian(phi_1_fourier, (1, 1), (2, 2))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    utils.add_vector_vector_inplace(phi_3c, -one, tmp1, tmp2)

    if param["dealiased_ICS"]:
        phi_3c = fourier.fft_3D_real(phi_3c, nthreads)
        phi_3c = trim(phi_3c)
        phi_3c = fourier.ifft_3D_real(phi_3c, nthreads)
        utils.prod_vector_scalar_inplace(phi_3c, np.float32(1.5**3))
    return phi_3c


def compute_3c_Ax_displacement(
    phi_1_fourier: npt.NDArray[np.complex64],
    phi_2_fourier: npt.NDArray[np.complex64],
    param: pd.Series,
) -> npt.NDArray[np.float32]:
    """Compute 3aLPT displacement

    Parameters
    ----------
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    phi_2_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        3bLPT displacement field [N, N, N]

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pysco.initial_conditions import compute_3c_Ax_displacement
    >>> param = pd.Series({
    ...     'nthreads': 2,
    ...     'dealiased_ICS': False,
    ...  })
    >>> phi_1 = np.random.random((16, 16, 16)).astype(np.complex64)
    >>> phi_2 = np.random.random((16, 16, 16)).astype(np.complex64)
    >>> rhs = compute_3c_Ax_displacement(phi_1, phi_2, param)
    """
    density_3c_Ax = compute_3c_Ax_rhs(phi_1_fourier, phi_2_fourier, param)
    density_3c_Ax_fourier = fourier.fft_3D_real(density_3c_Ax, param["nthreads"])
    psi_3lpt_c_Ax_fourier = fourier.gradient_inverse_laplacian(density_3c_Ax_fourier)
    psi_3lpt_c_Ax = fourier.ifft_3D_real_grad(psi_3lpt_c_Ax_fourier, param["nthreads"])
    return psi_3lpt_c_Ax


def compute_3c_Ay_rhs(
    phi_1_fourier: npt.NDArray[np.complex64],
    phi_2_fourier: npt.NDArray[np.complex64],
    param: pd.Series,
) -> npt.NDArray[np.float32]:
    """Compute 3cLPT displacement

    Parameters
    ----------
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    phi_2_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        3cLPT displacement field [N, N, N]

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pysco.initial_conditions import compute_3c_Ay_rhs
    >>> param = pd.Series({
    ...     'nthreads': 2,
    ...     'dealiased_ICS': False,
    ...  })
    >>> phi_1 = np.random.random((16,16,16)).astype(np.complex64)
    >>> phi_2 = np.random.random((16,16,16)).astype(np.complex64)
    >>> rhs = compute_3c_Ay_rhs(phi_1, phi_2, param)
    """
    one = np.float32(1)
    nthreads = param["nthreads"]

    if param["dealiased_ICS"]:
        phi_1_fourier = pad(phi_1_fourier)
        phi_2_fourier = pad(phi_2_fourier)

    tmp = fourier.hessian(phi_1_fourier, (0, 1))
    phi_3c = fourier.ifft_3D_real(tmp, nthreads)
    tmp = fourier.hessian(phi_2_fourier, (1, 2))
    tmp = fourier.ifft_3D_real(tmp, nthreads)
    utils.prod_vector_vector_scalar_inplace(phi_3c, tmp, one)
    tmp = 0

    tmp1 = fourier.hessian(phi_2_fourier, (0, 1))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.hessian(phi_1_fourier, (1, 2))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    utils.add_vector_vector_inplace(phi_3c, -one, tmp1, tmp2)

    tmp1 = fourier.hessian(phi_1_fourier, (0, 2))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.diff_of_hessian(phi_2_fourier, (2, 2), (0, 0))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    utils.add_vector_vector_inplace(phi_3c, one, tmp1, tmp2)

    tmp1 = fourier.hessian(phi_2_fourier, (0, 2))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.diff_of_hessian(phi_1_fourier, (2, 2), (0, 0))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    utils.add_vector_vector_inplace(phi_3c, -one, tmp1, tmp2)

    if param["dealiased_ICS"]:
        phi_3c = fourier.fft_3D_real(phi_3c, nthreads)
        phi_3c = trim(phi_3c)
        phi_3c = fourier.ifft_3D_real(phi_3c, nthreads)
        utils.prod_vector_scalar_inplace(phi_3c, np.float32(1.5**3))
    return phi_3c


def compute_3c_Ay_displacement(
    phi_1_fourier: npt.NDArray[np.complex64],
    phi_2_fourier: npt.NDArray[np.complex64],
    param: pd.Series,
) -> npt.NDArray[np.float32]:
    """Compute 3aLPT displacement

    Parameters
    ----------
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    phi_2_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        3bLPT displacement field [N, N, N]

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pysco.initial_conditions import compute_3c_Ay_displacement
    >>> param = pd.Series({
    ...     'nthreads': 2,
    ...     'dealiased_ICS': False,
    ...  })
    >>> phi_1 = np.random.random((16, 16, 16)).astype(np.complex64)
    >>> phi_2 = np.random.random((16, 16, 16)).astype(np.complex64)
    >>> rhs = compute_3c_Ay_displacement(phi_1, phi_2, param)
    """
    density_3c_Ay = compute_3c_Ay_rhs(phi_1_fourier, phi_2_fourier, param)
    density_3c_Ay_fourier = fourier.fft_3D_real(density_3c_Ay, param["nthreads"])
    psi_3lpt_c_Ay_fourier = fourier.gradient_inverse_laplacian(density_3c_Ay_fourier)
    psi_3lpt_c_Ay = fourier.ifft_3D_real_grad(psi_3lpt_c_Ay_fourier, param["nthreads"])
    return psi_3lpt_c_Ay


def compute_3c_Az_rhs(
    phi_1_fourier: npt.NDArray[np.complex64],
    phi_2_fourier: npt.NDArray[np.complex64],
    param: pd.Series,
) -> npt.NDArray[np.float32]:
    """Compute 3cLPT displacement

    Parameters
    ----------
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    phi_2_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        3cLPT displacement field [N, N, N]

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pysco.initial_conditions import compute_3c_Az_rhs
    >>> param = pd.Series({
    ...     'nthreads': 2,
    ...     'dealiased_ICS': False,
    ...  })
    >>> phi_1 = np.random.random((16,16,16)).astype(np.complex64)
    >>> phi_2 = np.random.random((16,16,16)).astype(np.complex64)
    >>> rhs = compute_3c_Az_rhs(phi_1, phi_2, param)
    """
    one = np.float32(1)
    nthreads = param["nthreads"]

    if param["dealiased_ICS"]:
        phi_1_fourier = pad(phi_1_fourier)
        phi_2_fourier = pad(phi_2_fourier)

    tmp = fourier.hessian(phi_1_fourier, (1, 2))
    phi_3c = fourier.ifft_3D_real(tmp, nthreads)
    tmp = fourier.hessian(phi_2_fourier, (0, 2))
    tmp = fourier.ifft_3D_real(tmp, nthreads)
    utils.prod_vector_vector_scalar_inplace(phi_3c, tmp, one)
    tmp = 0

    tmp1 = fourier.hessian(phi_2_fourier, (1, 2))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.hessian(phi_1_fourier, (0, 2))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    utils.add_vector_vector_inplace(phi_3c, -one, tmp1, tmp2)

    tmp1 = fourier.hessian(phi_1_fourier, (0, 1))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.diff_of_hessian(phi_2_fourier, (0, 0), (1, 1))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    utils.add_vector_vector_inplace(phi_3c, one, tmp1, tmp2)

    tmp1 = fourier.hessian(phi_2_fourier, (0, 1))
    tmp1 = fourier.ifft_3D_real(tmp1, nthreads)
    tmp2 = fourier.diff_of_hessian(phi_1_fourier, (0, 0), (1, 1))
    tmp2 = fourier.ifft_3D_real(tmp2, nthreads)
    utils.add_vector_vector_inplace(phi_3c, -one, tmp1, tmp2)

    if param["dealiased_ICS"]:
        phi_3c = fourier.fft_3D_real(phi_3c, nthreads)
        phi_3c = trim(phi_3c)
        phi_3c = fourier.ifft_3D_real(phi_3c, nthreads)
        utils.prod_vector_scalar_inplace(phi_3c, np.float32(1.5**3))
    return phi_3c


def compute_3c_Az_displacement(
    phi_1_fourier: npt.NDArray[np.complex64],
    phi_2_fourier: npt.NDArray[np.complex64],
    param: pd.Series,
) -> npt.NDArray[np.float32]:
    """Compute 3aLPT displacement

    Parameters
    ----------
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    phi_2_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

    Returns
    -------
    npt.NDArray[np.float32]
        3bLPT displacement field [N, N, N]

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pysco.initial_conditions import compute_3c_Az_displacement
    >>> param = pd.Series({
    ...     'nthreads': 2,
    ...     'dealiased_ICS': False,
    ...  })
    >>> phi_1 = np.random.random((16,16,16)).astype(np.complex64)
    >>> phi_2 = np.random.random((16,16,16)).astype(np.complex64)
    >>> rhs = compute_3c_Az_displacement(phi_1, phi_2, param)
    """
    density_3c_Az = compute_3c_Az_rhs(phi_1_fourier, phi_2_fourier, param)
    density_3c_Az_fourier = fourier.fft_3D_real(density_3c_Az, param["nthreads"])
    psi_3lpt_c_Az_fourier = fourier.gradient_inverse_laplacian(density_3c_Az_fourier)
    psi_3lpt_c_Az = fourier.ifft_3D_real_grad(psi_3lpt_c_Az_fourier, param["nthreads"])
    return psi_3lpt_c_Az


def initialise_1LPT(
    psi_1lpt: npt.NDArray[np.float32],
    dplus_1: np.float32,
    fH: np.float32,
    param: pd.Series,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Initialise particles according to 1LPT (Zel'Dovich) displacement field

    Parameters
    ----------
    psi_1lpt : npt.NDArray[np.float32]
        1LPT displacement field [N, N, N, 3]
    dplus1 : np.float32
        First-order growth factor
    fH : np.float32
        First-order growth rate times Hubble parameter
    param : pd.Series
        Parameter container

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [N, N, N, 3]

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pysco.initial_conditions import initialise_1LPT
    >>> param = pd.Series({
    ...     'position_ICS': "center",
    ...  })
    >>> psi_1lpt = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> dplus_1 = 0.1
    >>> fH = 0.1
    >>> initialise_1LPT(psi_1lpt, dplus_1, fH, param)
    """
    POSITION = param["position_ICS"].casefold()
    if POSITION == "center":
        return initialise_1LPT_center(psi_1lpt, dplus_1, fH)
    elif POSITION == "edge":
        return initialise_1LPT_edge(psi_1lpt, dplus_1, fH)
    else:
        raise ValueError(f"{POSITION=}, should be 'center' or 'edge'")


@utils.time_me
@njit(
    ["UniTuple(f4[:,:,:,::1], 2)(f4[:,:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def initialise_1LPT_edge(
    psi_1lpt: npt.NDArray[np.float32], dplus_1: np.float32, fH: np.float32
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Initialise particles according to 1LPT (Zel'Dovich) displacement field

    Initialise at cell edges

    Parameters
    ----------
    psi_1lpt : npt.NDArray[np.float32]
        1LPT displacement field [N, N, N, 3]
    dplus1 : np.float32
        First-order growth factor
    fH : np.float32
        First-order growth rate times Hubble parameter

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [N, N, N, 3]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import initialise_1LPT_edge
    >>> psi_1lpt = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> dplus_1 = 0.1
    >>> fH = 0.1
    >>> initialise_1LPT_edge(psi_1lpt, dplus_1, fH)
    """
    ncells_1d = psi_1lpt.shape[0]
    h = np.float32(1.0 / ncells_1d)
    dfH_1 = dplus_1 * fH
    position = np.empty_like(psi_1lpt)
    velocity = np.empty_like(psi_1lpt)
    for i in prange(ncells_1d):
        x = i * h
        for j in prange(ncells_1d):
            y = j * h
            for k in prange(ncells_1d):
                z = k * h
                psix = -psi_1lpt[i, j, k, 0]
                psiy = -psi_1lpt[i, j, k, 1]
                psiz = -psi_1lpt[i, j, k, 2]
                position[i, j, k, 0] = x + dplus_1 * psix
                position[i, j, k, 1] = y + dplus_1 * psiy
                position[i, j, k, 2] = z + dplus_1 * psiz
                velocity[i, j, k, 0] = dfH_1 * psix
                velocity[i, j, k, 1] = dfH_1 * psiy
                velocity[i, j, k, 2] = dfH_1 * psiz
    return position, velocity


@utils.time_me
@njit(
    ["UniTuple(f4[:,:,:,::1], 2)(f4[:,:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def initialise_1LPT_center(
    psi_1lpt: npt.NDArray[np.float32], dplus_1: np.float32, fH: np.float32
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Initialise particles according to 1LPT (Zel'Dovich) displacement field

    Initialise at cell centers

    Parameters
    ----------
    psi_1lpt : npt.NDArray[np.float32]
        1LPT displacement field [N, N, N, 3]
    dplus1 : np.float32
        First-order growth factor
    fH : np.float32
        First-order growth rate times Hubble parameter

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [N, N, N, 3]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import initialise_1LPT_center
    >>> psi_1lpt = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> dplus_1 = 0.1
    >>> fH = 0.1
    >>> initialise_1LPT_center(psi_1lpt, dplus_1, fH)
    """
    ncells_1d = psi_1lpt.shape[0]
    h = np.float32(1.0 / ncells_1d)
    half_h = np.float32(0.5 / ncells_1d)
    dfH_1 = dplus_1 * fH
    position = np.empty_like(psi_1lpt)
    velocity = np.empty_like(psi_1lpt)
    for i in prange(ncells_1d):
        x = half_h + i * h
        for j in prange(ncells_1d):
            y = half_h + j * h
            for k in prange(ncells_1d):
                z = half_h + k * h
                psix = -psi_1lpt[i, j, k, 0]
                psiy = -psi_1lpt[i, j, k, 1]
                psiz = -psi_1lpt[i, j, k, 2]
                position[i, j, k, 0] = x + dplus_1 * psix
                position[i, j, k, 1] = y + dplus_1 * psiy
                position[i, j, k, 2] = z + dplus_1 * psiz
                velocity[i, j, k, 0] = dfH_1 * psix
                velocity[i, j, k, 1] = dfH_1 * psiy
                velocity[i, j, k, 2] = dfH_1 * psiz
    return position, velocity


@utils.time_me
@njit(
    ["void(f4[:,:,:,::1], f4[:,:,:,::1], f4[:,:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def add_nLPT(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    psi_nlpt: npt.NDArray[np.float32],
    dplus_n: np.float32,
    fH_n: np.float32,
) -> None:
    """Initialise particles according to nLPT displacement field

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        1LPT position [N, N, N, 3]
    velocity : npt.NDArray[np.float32]
        1LPT velocity [N, N, N, 3]
    psi_nlpt : npt.NDArray[np.float32]
        nLPT displacement field [N, N, N, 3]
    dplus_n : np.float32
        nth-order growth factor
    fH_n : np.float32
        nth-order growth rate times Hubble parameter

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import add_nLPT
    >>> position_1storder = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> velocity_1storder = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> psi_2lpt = np.random.random((64, 64, 64, 3)).astype(np.float32)
    >>> dplus_2 = 0.2
    >>> fH_2 = 0.2
    >>> add_nLPT(position_1storder, velocity_1storder, psi_2lpt, dplus_2, fH_2)
    """
    ncells_1d = psi_nlpt.shape[0]
    dfH_n = dplus_n * fH_n
    for i in prange(ncells_1d):
        for j in prange(ncells_1d):
            for k in prange(ncells_1d):
                psix = psi_nlpt[i, j, k, 0]
                psiy = psi_nlpt[i, j, k, 1]
                psiz = psi_nlpt[i, j, k, 2]
                position[i, j, k, 0] += dplus_n * psix
                position[i, j, k, 1] += dplus_n * psiy
                position[i, j, k, 2] += dplus_n * psiz
                velocity[i, j, k, 0] += dfH_n * psix
                velocity[i, j, k, 1] += dfH_n * psiy
                velocity[i, j, k, 2] += dfH_n * psiz


@utils.time_me
def pad(input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Extend dimensions by with a factor 3/2 and padd with zeros

    Parameters
    ----------
    input : npt.NDArray[np.float32]
        Input field [N,N,N//2+1]

    Returns
    -------
    npt.NDArray[np.float32]
        Padded field [3N/2, 3N/2, 3N/2 //2+1]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import pad
    >>> phi = np.random.random((16,16,9)).astype(np.float32)
    >>> pad(phi)
    """
    ncells_1d = len(input)
    ncells_1d_extended = 3 * ncells_1d // 2
    middle = ncells_1d // 2
    output = np.zeros(
        (ncells_1d_extended, ncells_1d_extended, ncells_1d_extended // 2 + 1),
        dtype=input.dtype,
    )
    output[:middle, :middle, :middle] = input[:middle, :middle, :middle]
    output[-middle + 1 :, :middle, :middle] = input[-middle + 1 :, :middle, :middle]
    output[:middle, -middle + 1 :, :middle] = input[:middle, -middle + 1 :, :middle]
    output[-middle + 1 :, -middle + 1 :, :middle] = input[
        -middle + 1 :, -middle + 1 :, :middle
    ]
    return output


@utils.time_me
def trim(input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Trim dimensions by with a factor 2/3

    Parameters
    ----------
    input : npt.NDArray[np.float32]
        Input field [3N/2, 3N/2, (3N/2)//2 +1]

    Returns
    -------
    npt.NDArray[np.float32]
        Trimmed field [N, N, N//2+1]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import trim
    >>> phi = np.random.random((24, 24, 13)).astype(np.float32)
    >>> trim(phi)
    """
    ncells_1d_extended = len(input)
    ncells_1d = 2 * ncells_1d_extended // 3
    middle = ncells_1d // 2
    output = np.zeros((ncells_1d, ncells_1d, middle + 1), dtype=input.dtype)
    output[:middle, :middle, :middle] = input[:middle, :middle, :middle]
    output[-middle + 1 :, :middle, :middle] = input[-middle + 1 :, :middle, :middle]
    output[:middle, -middle + 1 :, :middle] = input[:middle, -middle + 1 :, :middle]
    output[-middle + 1 :, -middle + 1 :, :middle] = input[
        -middle + 1 :, -middle + 1 :, :middle
    ]
    return output
