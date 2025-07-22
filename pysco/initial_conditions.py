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
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    param: pd.Series, tables: List[interp1d]
) -> None:
    """Generate initial conditions

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Position [Npart, 3]
    velocity : npt.NDArray[np.float32]
        Velocity [Npart, 3]
    param : pd.Series
        Parameter container

    tables : List[interp1d]
        Interpolated functions [lna(t), t(lna), H(lna), Dplus1(lna), f1(lna), Dplus2(lna), f2(lna), Dplus3a(lna), f3a(lna), Dplus3b(lna), f3b(lna), Dplus3c(lna), f3c(lna)]

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
                iostream.read_snapshot_particles_parquet(position, velocity, filename)
                param_filename = f"{param['base']}/output_{i_restart:05d}/param_{param['extra']}_{i_restart:05d}.txt"
                param_restart = iostream.read_param_file(param_filename)
                logging.warning(f"Parameter file read at ...{param_filename=}")
                for key in param_restart.index:
                    if key.casefold() is not "nthreads".casefold():
                        param[key] = param_restart[key]
            case "hdf5":
                import h5py

                filename = f"{param['base']}/output_{i_restart:05d}/particles_{param['extra']}.h5"
                iostream.read_snapshot_particles_hdf5(position, velocity, filename)
                with h5py.File(filename, "r") as h5r:
                    attrs = h5r.attrs
                    for key in attrs.keys():
                        if key.casefold() is not "nthreads".casefold():
                            param[key] = attrs[key]
            case _:
                raise ValueError(
                    f"{param['snapshot_format']=}, should be 'parquet' or 'hdf5'"
                )
    elif "LPT".casefold() in INITIAL_CONDITIONS.casefold():
        ncells_1d = int(math.cbrt(param["npart"]))
        a_start = 1.0 / (1 + param["z_start"])
        lna_start = np.log(a_start)
        logging.warning(f"{param['z_start']=}")
        Hz = tables[2](lna_start)
        mpc_to_km = 1e3 * pc.value
        Hz *= param["unit_t"] / mpc_to_km  # km/s/Mpc to BU
        gradient_shape = (ncells_1d, ncells_1d, ncells_1d, 3)
        part_shape = position.shape
        # psi_1lpt = generate_force(param)
        density_2 = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
        psi = np.empty((ncells_1d, ncells_1d, ncells_1d, 3), dtype=np.float32)
        psi_fourier = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1, 3), dtype=np.complex64)
        potential_1_fourier = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.complex64)
        potential_2_fourier = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)

        generate_density_fourier(potential_1_fourier, param)

        fourier.inverse_laplacian(potential_1_fourier)
        fourier.gradient(psi_fourier, potential_1_fourier)
        fourier.ifft_3D_real_grad(psi, psi_fourier, param["nthreads"])
        # 1LPT
        logging.warning("Compute 1LPT contribution")
        dplus_1_z0 = tables[3](0)
        dplus_1 = np.float32(tables[3](lna_start) / dplus_1_z0)
        f1 = tables[4](lna_start)
        fH_1 = np.float32(f1 * Hz)
        initialise_1LPT(position, velocity, psi, dplus_1, fH_1, param)
        if INITIAL_CONDITIONS.casefold() == "1LPT".casefold():
            finalise_initial_conditions(position, velocity, param, reorder=False)
            return
        # 2LPT
        logging.warning("Compute 2LPT contribution")
        compute_2ndorder_rhs(density_2, potential_1_fourier, param)
        fourier.fft_3D_real(potential_2_fourier, density_2, param["nthreads"])
        density_2 = 0
        fourier.inverse_laplacian(potential_2_fourier)
        fourier.gradient(psi_fourier, potential_2_fourier)
        fourier.ifft_3D_real_grad(psi, psi_fourier, param["nthreads"])
        psi_fourier = 0
        dplus_2 = np.float32(tables[5](lna_start) / dplus_1_z0**2)
        f2 = tables[6](lna_start)
        fH_2 = np.float32(f2 * Hz)

        position.shape = gradient_shape
        velocity.shape = gradient_shape
        add_nLPT(position, velocity, psi, dplus_2, fH_2)
        position.shape = part_shape
        velocity.shape = part_shape
        if INITIAL_CONDITIONS.casefold() == "2LPT".casefold():
            finalise_initial_conditions(position, velocity, param, reorder=False)
            return
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
        compute_3a_displacement(psi, potential_1_fourier, param)

        position.shape = gradient_shape
        velocity.shape = gradient_shape
        add_nLPT(position, velocity, psi, dplus_3a, fH_3a)

        logging.warning("Compute 3LPT b) contribution")
        compute_3b_displacement(psi, potential_1_fourier, potential_2_fourier, param)
        add_nLPT(position, velocity, psi, dplus_3b, fH_3b)
        logging.warning("Compute 3LPT c) Ax contribution")
        compute_3c_Ax_displacement(psi, potential_1_fourier, potential_2_fourier, param)
        add_nLPT(position, velocity, psi, dplus_3c, fH_3c)
        logging.warning("Compute 3LPT c) Ay contribution")
        compute_3c_Ay_displacement(psi, potential_1_fourier, potential_2_fourier, param)
        add_nLPT(position, velocity, psi, dplus_3c, fH_3c)
        logging.warning("Compute 3LPT c) Az contribution")
        compute_3c_Az_displacement(psi, potential_1_fourier, potential_2_fourier, param)

        potential_1_fourier = 0
        potential_2_fourier = 0
        add_nLPT(position, velocity, psi, dplus_3c, fH_3c)
        position.shape = part_shape
        velocity.shape = part_shape

        if INITIAL_CONDITIONS.casefold() == "3LPT".casefold():
            finalise_initial_conditions(position, velocity, param, reorder=False)
            return
        else:
            raise ValueError(f"{INITIAL_CONDITIONS=}, should be 1LPT, 2LPT or 3LPT")
    elif INITIAL_CONDITIONS[-3:].casefold() == ".h5".casefold():
        read_hdf5(position, velocity, param)
        finalise_initial_conditions(position, velocity, param, reorder=True)
    else:  # Gadget format
        read_gadget(position, velocity, param)
        finalise_initial_conditions(position, velocity, param, reorder=True)


def finalise_initial_conditions(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    param: pd.Series,
    reorder: bool,
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
    reorder: bool
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
    if reorder:
        utils.reorder_particles(position, velocity)

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
            raise NotImplementedError(
                f"{param['snapshot_format']=}, should be 'parquet' or 'hdf5'"
            )
    logging.warning(f"Write initial snapshot...{snap_name=}")


def read_hdf5(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    param: pd.Series,
) -> None:
    """Read initial conditions from HDF5 Ramses snapshot

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Particle positions [Npart, 3]
    velocity : npt.NDArray[np.float32]
        Particle velocities [Npart, 3]
    param : pd.Series
        Parameter container

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


def read_gadget(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    param: pd.Series,
) -> None:
    """Read initial conditions from Gadget snapshot

    The snapshot can be divided in multiple files, such as \\
    snapshot_X.Y, where Y is the file number. \\
    In this case only keep snapshot_X

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Particle positions [Npart, 3]
    velocity : npt.NDArray[np.float32]
        Particle velocities [Npart, 3]
    param : pd.Series
        Parameter container

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

    position[:] = readgadget.read_block(filename, "POS ", [ptype])
    velocity[:] = readgadget.read_block(filename, "VEL ", [ptype])

    vel_factor = param["unit_t"] / param["unit_l"]
    utils.prod_vector_scalar_inplace(position, np.float32(1.0 / BoxSize))
    utils.prod_vector_scalar_inplace(velocity, np.float32(vel_factor))


@utils.time_me
def generate_density_fourier(out: npt.NDArray[np.complex64], param: pd.Series) -> None:
    """Compute density initial conditions from power spectrum

    Parameters
    ----------
    out : npt.NDArray[np.complex64]
        Fourier-space density field [N, N, N]
    param : pd.Series
        Parameter container

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
    ncells_1d = out.shape[0]
    transfer_grid = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    get_transfer_grid(transfer_grid, param)
    seed = param["seed"]
    if seed < 0:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(param["seed"])

    if param["fixed_ICS"]:
        white_noise_fourier_fixed(out, rng, param["paired_ICS"])
    else:
        white_noise_fourier(out, rng)

    utils.prod_vector_vector_inplace(out, transfer_grid)


@utils.time_me
def generate_density(out: npt.NDArray[np.float32], param: pd.Series) -> None:
    """Compute density initial conditions from power spectrum

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Initial density field [N, N, N]
    param : pd.Series
        Parameter container

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
    ncells_1d = out.shape[0]
    density_k = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.complex64)
    generate_density_fourier(density_k, param)
    fourier.ifft_3D_real(out, density_k, param["nthreads"])


@utils.time_me
def generate_force(out: npt.NDArray[np.float32], param: pd.Series) -> None:
    """Compute force initial conditions from power spectrum

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Initial force field [N, N, N, 3]
    param : pd.Series
        Parameter container

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
    ncells_1d = out.shape[0]
    force = np.empty((ncells_1d, ncells_1d, ncells_1d, 3), dtype=np.complex64)
    transfer_grid = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)

    get_transfer_grid(transfer_grid, param)
    seed = param["seed"]
    if seed < 0:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(param["seed"])

    if param["fixed_ICS"]:
        white_noise_fourier_fixed_force(force, rng, param["paired_ICS"])
    else:
        white_noise_fourier_force(force, rng)
    utils.prod_gradient_vector_inplace(force, transfer_grid)
    fourier.ifft_3D_real_grad(out, force, param["nthreads"])


@utils.time_me
def get_transfer_grid(out: npt.NDArray[np.float32], param: pd.Series) -> None:
    """Compute transfer 3D grid

    Computes sqrt(P(k,z)) on a 3D grid

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        Transfer grid [N, N, N]
    param : pd.Series
        Parameter container

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

    ncells_1d = out.shape[0]
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

    out[:] = np.interp(k_grid, k_dimensionless, sqrtPk)
    """ out[:] = 10 ** np.interp(
        np.log10(k_grid), np.log10(k_dimensionless), np.log10(sqrtPk)
    ) """


@utils.time_me
@njit(
    fastmath=True,
    cache=True,
    parallel=True,
)
def white_noise_fourier(
    out: npt.NDArray[np.complex64],
    rng: np.random.Generator
) -> None:
    """Generate Fourier-space white noise on a regular 3D grid

    Parameters
    ----------
    out : npt.NDArray[np.complex64]
        3D white-noise field for density [N_cells_1d, N_cells_1d, N_cells_1d]
    rng : np.random.Generator
        Random generator (NumPy)

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
    ncells_1d = out.shape[0]
    middle = ncells_1d // 2

    # Must compute random before parallel loop to ensure reproductability
    rng_amplitudes = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    rng_phases = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    for i in prange(middle + 1):
        im = -np.int32(i)
        for j in range(ncells_1d):
            jm = -j
            for k in range(ncells_1d):
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
                out[i, j, k] = result_upper
                out[im, jm, km] = result_lower
    rng_phases = 0
    rng_amplitudes = 0
    # Fix corners
    out[0, 0, 0] = 0
    out[0, 0, middle] = math.sqrt(-math.log(one - rng.random(dtype=np.float32)))
    out[0, middle, 0] = math.sqrt(-math.log(one - rng.random(dtype=np.float32)))
    out[0, middle, middle] = math.sqrt(
        -math.log(one - rng.random(dtype=np.float32))
    )
    out[middle, 0, 0] = math.sqrt(-math.log(one - rng.random(dtype=np.float32)))
    out[middle, 0, middle] = math.sqrt(
        -math.log(one - rng.random(dtype=np.float32))
    )
    out[middle, middle, 0] = math.sqrt(
        -math.log(one - rng.random(dtype=np.float32))
    )
    out[middle, middle, middle] = math.sqrt(
        -math.log(one - rng.random(dtype=np.float32))
    )


@utils.time_me
@njit(
    fastmath=True,
    cache=True,
    parallel=True,
)
def white_noise_fourier_fixed(
    out: npt.NDArray[np.complex64],
    rng: np.random.Generator, is_paired: bool
) -> None:
    """Generate Fourier-space white noise with fixed amplitude on a regular 3D grid

    Parameters
    ----------
    out : npt.NDArray[np.complex64]
        3D white-noise field for density [N_cells_1d, N_cells_1d, N_cells_1d]
    rng : np.random.Generator
        Random generator (NumPy)
    is_paired : bool
        If paired, add π to the random phases

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
    ncells_1d = out.shape[0]
    middle = ncells_1d // 2
    if is_paired:
        shift = np.float32(math.pi)
    else:
        shift = np.float32(0)

    rng_phases = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    for i in prange(middle + 1):
        im = -np.int32(i)
        for j in range(ncells_1d):
            jm = -j
            for k in range(ncells_1d):
                km = -k
                phase = twopi * rng_phases[i, j, k] + shift
                real = math.cos(phase)
                imaginary = ii * math.sin(phase)
                result_upper = real + imaginary
                result_lower = real - imaginary
                out[i, j, k] = result_upper
                out[im, jm, km] = result_lower
    rng_phases = 0
    out[0, 0, 0] = 0
    out[0, 0, middle] = one
    out[0, middle, 0] = one
    out[0, middle, middle] = one
    out[middle, 0, 0] = one
    out[middle, 0, middle] = one
    out[middle, middle, 0] = one
    out[middle, middle, middle] = one


@utils.time_me
@njit(
    fastmath=True,
    cache=True,
    parallel=True,
    error_model="numpy",
)
def white_noise_fourier_force(
    out: npt.NDArray[np.complex64],
    rng: np.random.Generator
) -> None:
    """Generate Fourier-space white noise force on a regular 3D grid

    Parameters
    ----------
    out : npt.NDArray[np.complex64]
        3D white-noise field for force [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    rng : np.random.Generator
        Random generator (NumPy)

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
    ncells_1d = out.shape[0]
    middle = ncells_1d // 2

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
        for j in range(ncells_1d):
            jm = -j
            if j >= middle:
                ky = np.float32(j - ncells_1d)
            else:
                ky = np.float32(j)
            kx2_ky2 = kx2 + ky**2
            for k in range(ncells_1d):
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
                out[i, j, k, 0] = -kx * i_phi_upper
                out[i, j, k, 1] = -ky * i_phi_upper
                out[i, j, k, 2] = -kz * i_phi_upper
                out[im, jm, km, 0] = kx * i_phi_lower
                out[im, jm, km, 1] = ky * i_phi_lower
                out[im, jm, km, 2] = kz * i_phi_lower
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
    out[0, 0, 0, 0] = 0
    out[0, 0, 0, 1] = 0
    out[0, 0, 0, 2] = 0
    # 1
    out[0, middle, 0, 0] = 0
    out[0, 0, middle, 0] = 0
    out[0, middle, middle, 0] = 0
    out[middle, 0, 0, 1] = 0
    out[0, 0, middle, 1] = 0
    out[middle, 0, middle, 1] = 0
    out[middle, 0, 0, 2] = 0
    out[0, middle, 0, 2] = 0
    out[middle, middle, 0, 2] = 0

    out[middle, 0, 0, 0] = invkmiddle * math.sqrt(
        -math.log(one - rng.random(dtype=np.float32))
    )
    out[0, middle, 0, 1] = invkmiddle * math.sqrt(
        -math.log(one - rng.random(dtype=np.float32))
    )
    out[0, 0, middle, 2] = invkmiddle * math.sqrt(
        -math.log(one - rng.random(dtype=np.float32))
    )
    # 2
    out[middle, middle, 0, 0] = force_1_1_0
    out[middle, 0, middle, 0] = force_1_0_1
    out[middle, middle, 0, 1] = force_1_1_0
    out[0, middle, middle, 1] = force_0_1_1
    out[0, middle, middle, 2] = force_0_1_1
    out[0, middle, middle, 2] = force_0_1_1
    # 3
    out[middle, middle, middle, 0] = force_1_1_1
    out[middle, middle, middle, 1] = force_1_1_1
    out[middle, middle, middle, 2] = force_1_1_1


@utils.time_me
@njit(
    fastmath=True,
    cache=True,
    parallel=True,
    error_model="numpy",
)
def white_noise_fourier_fixed_force(
    out: npt.NDArray[np.complex64],
    rng: np.random.Generator, is_paired: bool
) -> None:
    """Generate Fourier-space white noise force with fixed amplitude on a regular 3D grid

    Parameters
    ----------
    out : npt.NDArray[np.complex64]
        3D white-noise field for force [N_cells_1d, N_cells_1d, N_cells_1d, 3]
    ncells_1d : int
        Number of cells along one direction
    rng : np.random.Generator
        Random generator (NumPy)
    is_paired : bool
        If paired, add π to the random phases

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
    ncells_1d = out.shape[0]
    middle = ncells_1d // 2
    if is_paired:
        shift = np.float32(math.pi)
    else:
        shift = np.float32(0)

    rng_phases = rng.random((middle + 1, ncells_1d, ncells_1d), dtype=np.float32)
    for i in prange(middle + 1):
        im = -np.int32(i)
        if i >= middle:
            kx = np.float32(i - ncells_1d)
        else:
            kx = np.float32(i)
        kx2 = kx**2
        for j in range(ncells_1d):
            jm = -j
            if j >= middle:
                ky = np.float32(j - ncells_1d)
            else:
                ky = np.float32(j)
            kx2_ky2 = kx2 + ky**2
            for k in range(ncells_1d):
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
                out[i, j, k, 0] = -kx * i_phi_upper
                out[i, j, k, 1] = -ky * i_phi_upper
                out[i, j, k, 2] = -kz * i_phi_upper
                out[im, jm, km, 0] = kx * i_phi_lower
                out[im, jm, km, 1] = ky * i_phi_lower
                out[im, jm, km, 2] = kz * i_phi_lower
    rng_phases = 0
    # Fix edges
    inv2 = np.float32(0.5)
    inv3 = np.float32(1.0 / 3)
    invkmiddle = -np.float32((twopi * middle) ** (-1))
    force_2 = invkmiddle * inv2
    force_3 = invkmiddle * inv3
    # 0
    out[0, 0, 0, 0] = 0
    out[0, 0, 0, 1] = 0
    out[0, 0, 0, 2] = 0
    # 1
    out[0, middle, 0, 0] = 0
    out[0, 0, middle, 0] = 0
    out[0, middle, middle, 0] = 0
    out[middle, 0, 0, 1] = 0
    out[0, 0, middle, 1] = 0
    out[middle, 0, middle, 1] = 0
    out[middle, 0, 0, 2] = 0
    out[0, middle, 0, 2] = 0
    out[middle, middle, 0, 2] = 0

    out[middle, 0, 0, 0] = invkmiddle
    out[0, middle, 0, 1] = invkmiddle
    out[0, 0, middle, 2] = invkmiddle
    # 2
    out[middle, middle, 0, 0] = force_2
    out[middle, 0, middle, 0] = force_2
    out[middle, middle, 0, 1] = force_2
    out[0, middle, middle, 1] = force_2
    out[0, middle, middle, 2] = force_2
    out[0, middle, middle, 2] = force_2
    # 3
    out[middle, middle, middle, 0] = force_3
    out[middle, middle, middle, 1] = force_3
    out[middle, middle, middle, 2] = force_3


def compute_2ndorder_rhs(
    out: npt.NDArray[np.float32],
    phi_1_fourier: npt.NDArray[np.complex64], param: pd.Series
) -> None:
    """Compute 2LPT RHS [Scoccimarro 1998 Appendix D.2]

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        2LPT source field [N, N, N]
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

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

    ncells_1d = out.shape[0]
    ncells_1d_extended = ncells_1d * 3 // 2
    phi_fourier_orig = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)
    phi_fourier_extended = np.empty((ncells_1d_extended, ncells_1d_extended, ncells_1d_extended // 2 + 1), dtype=np.complex64)

    if param["dealiased_ICS"]:
        utils.zero_initialise_c64(phi_fourier_extended)
        pad(phi_fourier_extended, phi_1_fourier)
        phi_1_fourier = phi_fourier_extended

    ncells_1d = phi_1_fourier.shape[0]
    out_tmp = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    phi_a = np.empty_like(out_tmp)
    phi_b = np.empty_like(out_tmp)
    hessian = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)

    fourier.hessian(hessian, phi_1_fourier, (0, 0))
    fourier.ifft_3D_real(out_tmp, hessian, nthreads)
    fourier.sum_of_hessian(hessian, phi_1_fourier, (1, 1), (2, 2))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    utils.prod_vector_vector_inplace(out_tmp, phi_a)

    fourier.hessian(hessian, phi_1_fourier, (1, 1))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.hessian(hessian, phi_1_fourier, (2, 2))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    utils.add_vector_vector_inplace(out_tmp, one, phi_a, phi_b)

    fourier.hessian(hessian, phi_1_fourier, (0, 1))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    utils.add_vector_vector_inplace(out_tmp, -one, phi_a, phi_a)

    fourier.hessian(hessian, phi_1_fourier, (0, 2))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    utils.add_vector_vector_inplace(out_tmp, -one, phi_a, phi_a)

    fourier.hessian(hessian, phi_1_fourier, (1, 2))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    utils.add_vector_vector_inplace(out_tmp, -one, phi_a, phi_a)

    if param["dealiased_ICS"]:
        fourier.fft_3D_real(hessian, out_tmp, nthreads)
        utils.zero_initialise_c64(phi_fourier_orig)
        trim(phi_fourier_orig, hessian)
        fourier.ifft_3D_real(out, phi_fourier_orig, nthreads)
        utils.prod_vector_scalar_inplace(out, np.float32(1.5**3))
    else:
        utils.injection(out, out_tmp)


def compute_3a_rhs(
    out: npt.NDArray[np.float32],
    phi_1_fourier: npt.NDArray[np.complex64], param: pd.Series
) -> None:
    """Compute 3aLPT RHS

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        3LPT source field [N, N, N]
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container
    

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

    ncells_1d = out.shape[0]
    ncells_1d_extended = ncells_1d * 3 // 2
    phi_fourier_orig = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)
    phi_fourier_extended = np.empty((ncells_1d_extended, ncells_1d_extended, ncells_1d_extended // 2 + 1), dtype=np.complex64)

    if param["dealiased_ICS"]:
        utils.zero_initialise_c64(phi_fourier_extended)
        pad(phi_fourier_extended, phi_1_fourier)
        phi_1_fourier = phi_fourier_extended

    ncells_1d = phi_1_fourier.shape[0]
    out_tmp = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    phi_a = np.empty_like(out_tmp)
    phi_b = np.empty_like(out_tmp)
    phi_c = np.empty_like(out_tmp)
    hessian = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)

    fourier.hessian(hessian, phi_1_fourier, (0, 0))
    fourier.ifft_3D_real(out_tmp, hessian, nthreads)
    fourier.hessian(hessian, phi_1_fourier, (1, 1))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    utils.prod_vector_vector_inplace(out_tmp, phi_a)

    fourier.hessian(hessian, phi_1_fourier, (2, 2))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    utils.prod_vector_vector_inplace(out_tmp, phi_a)

    fourier.hessian(hessian, phi_1_fourier, (0, 1))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.hessian(hessian, phi_1_fourier, (0, 2))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    fourier.hessian(hessian, phi_1_fourier, (1, 2))
    fourier.ifft_3D_real(phi_c, hessian, nthreads)
    utils.add_vector_vector_vector_inplace(out_tmp, two, phi_a, phi_b, phi_c)

    fourier.hessian(hessian, phi_1_fourier, (1, 2))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.hessian(hessian, phi_1_fourier, (0, 0))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    utils.add_vector_vector_vector_inplace(out_tmp, -one, phi_a, phi_a, phi_b)

    fourier.hessian(hessian, phi_1_fourier, (0, 2))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.hessian(hessian, phi_1_fourier, (1, 1))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    utils.add_vector_vector_vector_inplace(out_tmp, -one, phi_a, phi_a, phi_b)

    fourier.hessian(hessian, phi_1_fourier, (0, 1))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.hessian(hessian, phi_1_fourier, (2, 2))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    utils.add_vector_vector_vector_inplace(out_tmp, -one, phi_a, phi_a, phi_b)

    if param["dealiased_ICS"]:
        fourier.fft_3D_real(hessian, out_tmp, nthreads)
        utils.zero_initialise_c64(phi_fourier_orig)
        trim(phi_fourier_orig, hessian)
        fourier.ifft_3D_real(out, phi_fourier_orig, nthreads)
        utils.prod_vector_scalar_inplace(out, np.float32(1.5**6))
    else:
        utils.injection(out, out_tmp)


def compute_3a_displacement(
    out: npt.NDArray[np.float32],
    phi_1_fourier: npt.NDArray[np.complex64], param: pd.Series
) -> None:
    """Compute 3aLPT displacement

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        3LPT displacement field [N, N, N, 3]
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container


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
    ncells_1d = out.shape[0]
    density_3a = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    density_3a_fourier = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)
    psi_3lpt_a_fourier = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1, 3), dtype=np.complex64)

    compute_3a_rhs(density_3a, phi_1_fourier, param)
    fourier.fft_3D_real(density_3a_fourier, density_3a, param["nthreads"])
    fourier.gradient_inverse_laplacian(psi_3lpt_a_fourier, density_3a_fourier)
    fourier.ifft_3D_real_grad(out, psi_3lpt_a_fourier, param["nthreads"])


def compute_3b_rhs(
    out: npt.NDArray[np.float32],
    phi_1_fourier: npt.NDArray[np.complex64],
    phi_2_fourier: npt.NDArray[np.complex64],
    param: pd.Series,
) -> None:
    """Compute 3bLPT RHS

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        3bLPT source field [N, N, N]
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    phi_2_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

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

    ncells_1d = out.shape[0]
    ncells_1d_extended = ncells_1d * 3 // 2
    phi_fourier_orig = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)
    phi_1_fourier_extended = np.empty((ncells_1d_extended, ncells_1d_extended, ncells_1d_extended // 2 + 1), dtype=np.complex64)
    phi_2_fourier_extended = np.empty_like(phi_1_fourier_extended)

    if param["dealiased_ICS"]:
        utils.zero_initialise_c64(phi_1_fourier_extended)
        utils.zero_initialise_c64(phi_2_fourier_extended)
        pad(phi_1_fourier_extended, phi_1_fourier)
        pad(phi_2_fourier_extended, phi_2_fourier)
        phi_1_fourier = phi_1_fourier_extended
        phi_2_fourier = phi_2_fourier_extended

    ncells_1d = phi_1_fourier.shape[0]
    out_tmp = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    phi_a = np.empty_like(out_tmp)
    phi_b = np.empty_like(out_tmp)
    hessian = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)

    fourier.hessian(hessian, phi_1_fourier, (0, 0))
    fourier.ifft_3D_real(out_tmp, hessian, nthreads)
    fourier.sum_of_hessian(hessian, phi_2_fourier, (1, 1), (2, 2))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    utils.prod_vector_vector_scalar_inplace(out_tmp, phi_a, half)

    fourier.hessian(hessian, phi_1_fourier, (1, 1))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.sum_of_hessian(hessian, phi_2_fourier, (0, 0), (2, 2))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    utils.add_vector_vector_inplace(out_tmp, half, phi_a, phi_b)

    fourier.hessian(hessian, phi_1_fourier, (2, 2))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.sum_of_hessian(hessian, phi_2_fourier, (0, 0), (1, 1))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    utils.add_vector_vector_inplace(out_tmp, half, phi_a, phi_b)

    fourier.hessian(hessian, phi_1_fourier, (0, 1))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.hessian(hessian, phi_2_fourier, (0, 1))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    utils.add_vector_vector_inplace(out_tmp, -one, phi_a, phi_b)

    fourier.hessian(hessian, phi_1_fourier, (0, 2))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.hessian(hessian, phi_2_fourier, (0, 2))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    utils.add_vector_vector_inplace(out_tmp, -one, phi_a, phi_b)

    fourier.hessian(hessian, phi_1_fourier, (1, 2))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.hessian(hessian, phi_2_fourier, (1, 2))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    utils.add_vector_vector_inplace(out_tmp, -one, phi_a, phi_b)

    if param["dealiased_ICS"]:
        fourier.fft_3D_real(hessian, out_tmp, nthreads)
        utils.zero_initialise_c64(phi_fourier_orig)
        trim(phi_fourier_orig, hessian)
        fourier.ifft_3D_real(out, phi_fourier_orig, nthreads)
        utils.prod_vector_scalar_inplace(out, np.float32(1.5**3))
    else:
        utils.injection(out, out_tmp)


def compute_3b_displacement(
    out: npt.NDArray[np.float32],
    phi_1_fourier: npt.NDArray[np.complex64],
    phi_2_fourier: npt.NDArray[np.complex64],
    param: pd.Series,
) -> None:
    """Compute 3bLPT displacement

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        3bLPT displacement field [N, N, N, 3]
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    phi_2_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

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
    ncells_1d = out.shape[0]
    density_3b = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    density_3b_fourier = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)
    psi_3lpt_b_fourier = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1, 3), dtype=np.complex64)

    compute_3b_rhs(density_3b, phi_1_fourier, phi_2_fourier, param)
    fourier.fft_3D_real(density_3b_fourier, density_3b, param["nthreads"])
    fourier.gradient_inverse_laplacian(psi_3lpt_b_fourier, density_3b_fourier)
    fourier.ifft_3D_real_grad(out, psi_3lpt_b_fourier, param["nthreads"])


def compute_3c_Ax_rhs(
    out: npt.NDArray[np.float32],
    phi_1_fourier: npt.NDArray[np.complex64],
    phi_2_fourier: npt.NDArray[np.complex64],
    param: pd.Series,
) -> None:
    """Compute 3cLPT RHS

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        3cLPT source field [N, N, N]
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    phi_2_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

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

    ncells_1d = out.shape[0]
    ncells_1d_extended = ncells_1d * 3 // 2
    phi_fourier_orig = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)
    phi_1_fourier_extended = np.empty((ncells_1d_extended, ncells_1d_extended, ncells_1d_extended // 2 + 1), dtype=np.complex64)
    phi_2_fourier_extended = np.empty_like(phi_1_fourier_extended)

    if param["dealiased_ICS"]:
        utils.zero_initialise_c64(phi_1_fourier_extended)
        utils.zero_initialise_c64(phi_2_fourier_extended)
        pad(phi_1_fourier_extended, phi_1_fourier)
        pad(phi_2_fourier_extended, phi_2_fourier)
        phi_1_fourier = phi_1_fourier_extended
        phi_2_fourier = phi_2_fourier_extended

    ncells_1d = phi_1_fourier.shape[0]
    out_tmp = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    phi_a = np.empty_like(out_tmp)
    phi_b = np.empty_like(out_tmp)
    hessian = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)

    fourier.hessian(hessian, phi_1_fourier, (0, 2))
    fourier.ifft_3D_real(out_tmp, hessian, nthreads)
    fourier.hessian(hessian, phi_2_fourier, (0, 1))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    utils.prod_vector_vector_scalar_inplace(out_tmp, phi_a, one) # TODO: Just write a prod_vector_vector_inplace?

    fourier.hessian(hessian, phi_2_fourier, (0, 2))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.hessian(hessian, phi_1_fourier, (0, 1))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    utils.add_vector_vector_inplace(out_tmp, -one, phi_a, phi_b)

    fourier.hessian(hessian, phi_1_fourier, (1, 2))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.diff_of_hessian(hessian, phi_2_fourier, (1, 1), (2, 2))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    utils.add_vector_vector_inplace(out_tmp, one, phi_a, phi_b)

    fourier.hessian(hessian, phi_2_fourier, (1, 2))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.diff_of_hessian(hessian, phi_1_fourier, (1, 1), (2, 2))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    utils.add_vector_vector_inplace(out_tmp, -one, phi_a, phi_b)

    if param["dealiased_ICS"]:
        fourier.fft_3D_real(hessian, out_tmp, nthreads)
        utils.zero_initialise_c64(phi_fourier_orig)
        trim(phi_fourier_orig, hessian)
        fourier.ifft_3D_real(out, phi_fourier_orig, nthreads)
        utils.prod_vector_scalar_inplace(out, np.float32(1.5**3))
    else:
        utils.injection(out, out_tmp)


def compute_3c_Ax_displacement(
    out: npt.NDArray[np.float32],
    phi_1_fourier: npt.NDArray[np.complex64],
    phi_2_fourier: npt.NDArray[np.complex64],
    param: pd.Series,
) -> None:
    """Compute 3aLPT displacement

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        3cLPT displacement field [N, N, N, 3]
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    phi_2_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

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
    ncells_1d = out.shape[0]
    density_3c_Ax = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    density_3c_Ax_fourier = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)
    psi_3lpt_c_Ax_fourier = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1, 3), dtype=np.complex64)

    compute_3c_Ax_rhs(density_3c_Ax, phi_1_fourier, phi_2_fourier, param)
    fourier.fft_3D_real(density_3c_Ax_fourier, density_3c_Ax, param["nthreads"])
    fourier.gradient_inverse_laplacian(psi_3lpt_c_Ax_fourier, density_3c_Ax_fourier)
    fourier.ifft_3D_real_grad(out, psi_3lpt_c_Ax_fourier, param["nthreads"])


def compute_3c_Ay_rhs(
    out: npt.NDArray[np.float32],
    phi_1_fourier: npt.NDArray[np.complex64],
    phi_2_fourier: npt.NDArray[np.complex64],
    param: pd.Series,
) -> None:
    """Compute 3cLPT RHS

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        3cLPT source field [N, N, N]
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    phi_2_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

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

    ncells_1d = out.shape[0]
    ncells_1d_extended = ncells_1d * 3 // 2
    phi_fourier_orig = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)
    phi_1_fourier_extended = np.empty((ncells_1d_extended, ncells_1d_extended, ncells_1d_extended // 2 + 1), dtype=np.complex64)
    phi_2_fourier_extended = np.empty_like(phi_1_fourier_extended)

    if param["dealiased_ICS"]:
        utils.zero_initialise_c64(phi_1_fourier_extended)
        utils.zero_initialise_c64(phi_2_fourier_extended)
        pad(phi_1_fourier_extended, phi_1_fourier)
        pad(phi_2_fourier_extended, phi_2_fourier)
        phi_1_fourier = phi_1_fourier_extended
        phi_2_fourier = phi_2_fourier_extended

    ncells_1d = phi_1_fourier.shape[0]
    out_tmp = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    phi_a = np.empty_like(out_tmp)
    phi_b = np.empty_like(out_tmp)
    hessian = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)

    fourier.hessian(hessian, phi_1_fourier, (0, 1))
    fourier.ifft_3D_real(out_tmp, hessian, nthreads)
    fourier.hessian(hessian, phi_2_fourier, (1, 2))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    utils.prod_vector_vector_scalar_inplace(out_tmp, phi_a, one)

    fourier.hessian(hessian, phi_2_fourier, (0, 1))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.hessian(hessian, phi_1_fourier, (1, 2))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    utils.add_vector_vector_inplace(out_tmp, -one, phi_a, phi_b)

    fourier.hessian(hessian, phi_1_fourier, (0, 2))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.diff_of_hessian(hessian, phi_2_fourier, (2, 2), (0, 0))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    utils.add_vector_vector_inplace(out_tmp, one, phi_a, phi_b)

    fourier.hessian(hessian, phi_2_fourier, (0, 2))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.diff_of_hessian(hessian, phi_1_fourier, (2, 2), (0, 0))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    utils.add_vector_vector_inplace(out_tmp, -one, phi_a, phi_b)

    if param["dealiased_ICS"]:
        fourier.fft_3D_real(hessian, out_tmp, nthreads)
        utils.zero_initialise_c64(phi_fourier_orig)
        trim(phi_fourier_orig, hessian)
        fourier.ifft_3D_real(out, phi_fourier_orig, nthreads)
        utils.prod_vector_scalar_inplace(out, np.float32(1.5**3))
    else:
        utils.injection(out, out_tmp)


def compute_3c_Ay_displacement(
    out: npt.NDArray[np.float32],
    phi_1_fourier: npt.NDArray[np.complex64],
    phi_2_fourier: npt.NDArray[np.complex64],
    param: pd.Series,
) -> None:
    """Compute 3cLPT displacement

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        3cLPT displacement field [N, N, N, 3]
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    phi_2_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

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
    ncells_1d = out.shape[0]
    density_3c_Ay = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    density_3c_Ay_fourier = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)
    psi_3lpt_c_Ay_fourier = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1, 3), dtype=np.complex64)

    compute_3c_Ay_rhs(density_3c_Ay, phi_1_fourier, phi_2_fourier, param)
    fourier.fft_3D_real(density_3c_Ay_fourier, density_3c_Ay, param["nthreads"])
    fourier.gradient_inverse_laplacian(psi_3lpt_c_Ay_fourier, density_3c_Ay_fourier)
    fourier.ifft_3D_real_grad(out, psi_3lpt_c_Ay_fourier, param["nthreads"])


def compute_3c_Az_rhs(
    out: npt.NDArray[np.float32],
    phi_1_fourier: npt.NDArray[np.complex64],
    phi_2_fourier: npt.NDArray[np.complex64],
    param: pd.Series,
) -> None:
    """Compute 3cLPT RHS

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        3cLPT source field [N, N, N]
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    phi_2_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

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

    ncells_1d = out.shape[0]
    ncells_1d_extended = ncells_1d * 3 // 2
    phi_fourier_orig = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)
    phi_1_fourier_extended = np.empty((ncells_1d_extended, ncells_1d_extended, ncells_1d_extended // 2 + 1), dtype=np.complex64)
    phi_2_fourier_extended = np.empty_like(phi_1_fourier_extended)

    if param["dealiased_ICS"]:
        utils.zero_initialise_c64(phi_1_fourier_extended)
        utils.zero_initialise_c64(phi_2_fourier_extended)
        pad(phi_1_fourier_extended, phi_1_fourier)
        pad(phi_2_fourier_extended, phi_2_fourier)
        phi_1_fourier = phi_1_fourier_extended
        phi_2_fourier = phi_2_fourier_extended
    
    ncells_1d = phi_1_fourier.shape[0]
    out_tmp = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    phi_a = np.empty_like(out_tmp)
    phi_b = np.empty_like(out_tmp)
    hessian = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)

    fourier.hessian(hessian, phi_1_fourier, (1, 2))
    fourier.ifft_3D_real(out_tmp, hessian, nthreads)
    fourier.hessian(hessian, phi_2_fourier, (0, 2))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    utils.prod_vector_vector_scalar_inplace(out_tmp, phi_a, one)

    fourier.hessian(hessian, phi_2_fourier, (1, 2))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.hessian(hessian, phi_1_fourier, (0, 2))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    utils.add_vector_vector_inplace(out_tmp, -one, phi_a, phi_b)

    fourier.hessian(hessian, phi_1_fourier, (0, 1))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.diff_of_hessian(hessian, phi_2_fourier, (0, 0), (1, 1))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    utils.add_vector_vector_inplace(out_tmp, one, phi_a, phi_b)

    fourier.hessian(hessian, phi_2_fourier, (0, 1))
    fourier.ifft_3D_real(phi_a, hessian, nthreads)
    fourier.diff_of_hessian(hessian, phi_1_fourier, (0, 0), (1, 1))
    fourier.ifft_3D_real(phi_b, hessian, nthreads)
    utils.add_vector_vector_inplace(out_tmp, -one, phi_a, phi_b)

    if param["dealiased_ICS"]:
        fourier.fft_3D_real(hessian, out_tmp, nthreads)
        utils.zero_initialise_c64(phi_fourier_orig)
        trim(phi_fourier_orig, hessian)
        fourier.ifft_3D_real(out, phi_fourier_orig, nthreads)
        utils.prod_vector_scalar_inplace(out, np.float32(1.5**3))
    else:
        utils.injection(out, out_tmp)


def compute_3c_Az_displacement(
    out: npt.NDArray[np.float32],
    phi_1_fourier: npt.NDArray[np.complex64],
    phi_2_fourier: npt.NDArray[np.complex64],
    param: pd.Series,
) -> None:
    """Compute 3aLPT displacement

    Parameters
    ----------
    out : npt.NDArray[np.float32]
        3cLPT displacement field [N, N, N, 3]
    phi_1_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    phi_2_fourier : npt.NDArray[np.complex64]
        First-order Potential [N, N, N]
    param : pd.Series
        Parameter container

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
    ncells_1d = out.shape[0]
    density_3c_Az = np.empty((ncells_1d, ncells_1d, ncells_1d), dtype=np.float32)
    density_3c_Az_fourier = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1), dtype=np.complex64)
    psi_3lpt_c_Az_fourier = np.empty((ncells_1d, ncells_1d, ncells_1d // 2 + 1, 3), dtype=np.complex64)

    compute_3c_Az_rhs(density_3c_Az, phi_1_fourier, phi_2_fourier, param)
    fourier.fft_3D_real(density_3c_Az_fourier, density_3c_Az, param["nthreads"])
    fourier.gradient_inverse_laplacian(psi_3lpt_c_Az_fourier, density_3c_Az_fourier)
    fourier.ifft_3D_real_grad(out, psi_3lpt_c_Az_fourier, param["nthreads"])


def initialise_1LPT(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    psi_1lpt: npt.NDArray[np.float32],
    dplus_1: np.float32,
    fH: np.float32,
    param: pd.Series,
) -> None:
    """Initialise particles according to 1LPT (Zel'dovich) displacement field

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Particle positions [Npart, 3]
    velocity : npt.NDArray[np.float32]
        Particle velocities [Npart, 3]
    psi_1lpt : npt.NDArray[np.float32]
        1LPT displacement field [N, N, N, 3]
    dplus1 : np.float32
        First-order growth factor
    fH : np.float32
        First-order growth rate times Hubble parameter
    param : pd.Series
        Parameter container

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
    ncells_1d = psi_1lpt.shape[0]
    init_shape = position.shape
    shape = (ncells_1d, ncells_1d, ncells_1d, 3)
    if POSITION == "center":
        position.shape = shape
        velocity.shape = shape
        initialise_1LPT_center(position, velocity, psi_1lpt, dplus_1, fH)
        position.shape = init_shape
        velocity.shape = init_shape
        return
    elif POSITION == "edge":
        position.shape = shape
        velocity.shape = shape
        initialise_1LPT_edge(position, velocity, psi_1lpt, dplus_1, fH)
        position.shape = init_shape
        velocity.shape = init_shape
        return
    raise NotImplementedError(f"{POSITION=}, should be 'center' or 'edge'")


@utils.time_me
@njit(
    ["void(f4[:,:,:,::1], f4[:,:,:,::1], f4[:,:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def initialise_1LPT_edge(
    position: npt.NDArray[np.float32],  
    velocity: npt.NDArray[np.float32],
    psi_1lpt: npt.NDArray[np.float32], dplus_1: np.float32, fH: np.float32
) -> None:
    """Initialise particles according to 1LPT (Zel'Dovich) displacement field

    Initialise at cell edges

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Particle positions [N, N, N, 3]
    velocity : npt.NDArray[np.float32]
        Particle velocities [N, N, N, 3]
    psi_1lpt : npt.NDArray[np.float32]
        1LPT displacement field [N, N, N, 3]
    dplus1 : np.float32
        First-order growth factor
    fH : np.float32
        First-order growth rate times Hubble parameter

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
    for i in prange(ncells_1d):
        x = i * h
        for j in range(ncells_1d):
            y = j * h
            for k in range(ncells_1d):
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


@utils.time_me
@njit(
    ["void(f4[:,:,:,::1], f4[:,:,:,::1], f4[:,:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def initialise_1LPT_center(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    psi_1lpt: npt.NDArray[np.float32], dplus_1: np.float32, fH: np.float32
) -> None:
    """Initialise particles according to 1LPT (Zel'Dovich) displacement field

    Initialise at cell centers

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Particle positions [N, N, N, 3]
    velocity : npt.NDArray[np.float32]
        Particle velocities [N, N, N, 3]
    psi_1lpt : npt.NDArray[np.float32]
        1LPT displacement field [N, N, N, 3]
    dplus1 : np.float32
        First-order growth factor
    fH : np.float32
        First-order growth rate times Hubble parameter

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
    for i in prange(ncells_1d):
        x = half_h + i * h
        for j in range(ncells_1d):
            y = half_h + j * h
            for k in range(ncells_1d):
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
        for j in range(ncells_1d):
            for k in range(ncells_1d):
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
def pad(output: npt.NDArray[np.complex64], input: npt.NDArray[np.complex64]) -> None:
    """Extend dimensions by with a factor 3/2 and padd with zeros

    Parameters
    ----------
    output : npt.NDArray[np.complex64]
        Padded field [3N/2, 3N/2, 3N/2 //2+1]
    input : npt.NDArray[np.complex64]
        Input field [N,N,N//2+1]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import pad
    >>> phi = np.random.random((16,16,9)).astype(np.complex64)
    >>> pad(phi)
    """
    middle = input.shape[0] // 2
 
    output[:middle, :middle, :middle] = input[:middle, :middle, :middle]
    output[-middle + 1 :, :middle, :middle] = input[-middle + 1 :, :middle, :middle]
    output[:middle, -middle + 1 :, :middle] = input[:middle, -middle + 1 :, :middle]
    output[-middle + 1 :, -middle + 1 :, :middle] = input[
        -middle + 1 :, -middle + 1 :, :middle
    ]


@utils.time_me
def trim(output: npt.NDArray[np.complex64], input: npt.NDArray[np.complex64]) -> None:
    """Trim dimensions by with a factor 2/3

    Parameters
    ----------
    output : npt.NDArray[np.complex64]
        Trimmed field [N, N, N//2+1]
    input : npt.NDArray[np.complex64]
        Input field [3N/2, 3N/2, (3N/2)//2 +1]
    Example
    -------
    >>> import numpy as np
    >>> from pysco.initial_conditions import trim
    >>> phi = np.random.random((24, 24, 13)).astype(np.complex64)
    >>> trim(phi)
    """
    middle = output.shape[0] // 2

    output[:middle, :middle, :middle] = input[:middle, :middle, :middle]
    output[-middle + 1 :, :middle, :middle] = input[-middle + 1 :, :middle, :middle]
    output[:middle, -middle + 1 :, :middle] = input[:middle, -middle + 1 :, :middle]
    output[-middle + 1 :, -middle + 1 :, :middle] = input[
        -middle + 1 :, -middle + 1 :, :middle
    ]