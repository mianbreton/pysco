"""
This module contains various input/output functions.
"""

import utils
from typing import Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd
import logging


def read_param_file(name: str) -> pd.Series:
    """Read parameter file into Pandas Series

    Parameters
    ----------
    name : str
        Parameter file name

    Returns
    -------
    pd.Series
        Parameters container

    Examples
    --------
    >>> from pysco.iostream import read_param_file
    >>> params = read_param_file(f"./examples/param.ini")
    """
    param = pd.read_csv(
        name,
        delimiter="=",
        comment="#",
        skipinitialspace=True,
        skip_blank_lines=True,
        header=None,
    ).T
    # First row as header
    param = param.rename(columns=param.iloc[0]).drop(param.index[0])
    # Remove whitespaces from column names and values
    param = param.apply(lambda x: x.str.strip() if x.dtype == "object" else x).rename(
        columns=lambda x: x.strip()
    )
    param = param.astype("string")
    is_null = param.isnull()
    for key in param.columns:

        if is_null[key].item():
            param[key] = "False"

        if "true".casefold() == param[key].item().casefold():
            param[key] = "True"
        if "false".casefold() == param[key].item().casefold():
            param[key] = "False"

        try:
            value = eval(param[key].item())
            if isinstance(value, list):
                isDigit = False
            else:
                isDigit = True
        except:
            isDigit = False
        if isDigit:
            value = eval(param[key].item())
            param[key] = value

    return param.T.iloc[:, 0]


@utils.time_me
def read_snapshot_particles_hdf5(
    filename: str,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Read particles in snapshot from HDF5 file

    Parameters
    ----------
    filename : str
        Filename

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [N_part, 3]

    Examples
    --------
    >>> from pysco.iostream import read_snapshot_particles_hdf5
    >>> position, velocity = read_snapshot_particles_hdf5(f"snapshot.h5")
    """
    import h5py

    logging.warning(f"Read HDF5 snapshot {filename}")
    with h5py.File(filename, "r") as h5r:
        position = h5r["position"][:]
        velocity = h5r["velocity"][:]
    return (position, velocity)


@utils.time_me
def read_snapshot_particles_parquet(
    filename: str,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Read particles in snapshot from parquet file

    Parameters
    ----------
    filename : str
        Filename

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [N_part, 3]

    Examples
    --------
    >>> from pysco.iostream import read_snapshot_particles_parquet
    >>> position, velocity = read_snapshot_particles_parquet(f"snapshot.parquet")
    """
    import pyarrow.parquet as pq

    logging.warning(f"Read parquet snapshot {filename}")
    position = np.ascontiguousarray(
        np.array(pq.read_table(filename, columns=["x", "y", "z"])).T
    )
    velocity = np.ascontiguousarray(
        np.array(pq.read_table(filename, columns=["vx", "vy", "vz"])).T
    )
    return (position, velocity)


@utils.time_me
def write_snapshot_particles(
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    param: pd.Series,
) -> None:
    """Write snapshot with particle information in HDF5 or Parquet format

    Parameters
    ----------
    position : npt.NDArray[np.float32]
        Position [N_part, 3]
    velocity : npt.NDArray[np.float32]
        Velocity [N_part, 3]
    param : pd.Series
        Parameter container

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.iostream import write_snapshot_particles
    >>> position = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    >>> velocity = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    >>> parameters = pd.Series({"output_snapshot_format": "parquet", "base": f"./examples/", "i_snap": 0, "extra": "extra_info", "aexp": 1.0})
    >>> write_snapshot_particles(position, velocity, parameters)
    >>> parameters = pd.Series({"output_snapshot_format": "hdf5", "base": f"./examples/", "i_snap": 0, "extra": "extra_info", "aexp": 1.0})
    >>> write_snapshot_particles(position, velocity, parameters)
    """
    OUTPUT_SNAPSHOT_FORMAT = param["output_snapshot_format"].casefold()
    match OUTPUT_SNAPSHOT_FORMAT:
        case "parquet":
            filename = f"{param['base']}/output_{param['i_snap']:05d}/particles_{param['extra']}.parquet"
            write_snapshot_particles_parquet(filename, position, velocity)
            param_filename = f"{param['base']}/output_{param['i_snap']:05d}/param_{param['extra']}_{param['i_snap']:05d}.txt"
            param.to_csv(
                param_filename,
                sep="=",
                header=False,
            )
            logging.warning(f"Parameter file written at ...{param_filename=}")
        case "hdf5":
            filename = f"{param['base']}/output_{param['i_snap']:05d}/particles_{param['extra']}.h5"
            write_snapshot_particles_hdf5(filename, position, velocity, param)
        case _:
            raise ValueError(
                f"{param['snapshot_format']=}, should be 'parquet' or 'hdf5'"
            )

    logging.warning(f"Snapshot written at ...{filename=} {param['aexp']=}")


@utils.time_me
def write_snapshot_particles_parquet(
    filename: str,
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
) -> None:
    """Write snapshot with particle information in parquet format

    Parameters
    ----------
    filename : str
        Filename
    position : npt.NDArray[np.float32]
        Position [N_part, 3]
    velocity : npt.NDArray[np.float32]
        Velocity [N_part, 3]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.iostream import write_snapshot_particles_parquet
    >>> position = np.random.rand(32**3, 3).astype(np.float32)
    >>> velocity = np.random.rand(32**3, 3).astype(np.float32)
    >>> write_snapshot_particles_parquet(f"./examples/snapshot.parquet", position, velocity)
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "x": position[:, 0],
            "y": position[:, 1],
            "z": position[:, 2],
            "vx": velocity[:, 0],
            "vy": velocity[:, 1],
            "vz": velocity[:, 2],
        }
    )

    pq.write_table(table, filename)


@utils.time_me
def write_snapshot_particles_hdf5(
    filename: str,
    position: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    param: pd.Series,
) -> None:
    """Write snapshot with particle information in HDF5 format

    Parameters
    ----------
    filename : str
        Filename
    position : npt.NDArray[np.float32]
        Position [N_part, 3]
    velocity : npt.NDArray[np.float32]
        Velocity [N_part, 3]
    param : pd.Series
        Parameter container

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.iostream import write_snapshot_particles_hdf5
    >>> position = np.random.rand(32**3, 3).astype(np.float32)
    >>> velocity = np.random.rand(32**3, 3).astype(np.float32)
    >>> param = pd.Series({"Attribute_0": 0.0, "Attribute_1": 300.0})
    >>> write_snapshot_particles_hdf5(f"./examples/snapshot.h5", position, velocity, param)
    """
    import h5py

    with h5py.File(filename, "w") as h5f:
        h5f.create_dataset("position", data=position)
        h5f.create_dataset("velocity", data=velocity)
        for key, item in param.items():
            h5f.attrs[key] = item


def write_power_spectrum_to_ascii_file(
    k: npt.NDArray[np.float32],
    Pk: npt.NDArray[np.float32],
    Nmodes: npt.NDArray[np.float32],
    param: pd.Series,
) -> None:
    """Write P(k) to ascii file

    Parameters
    ----------
    k : npt.NDArray[np.float32]
        Wavelenght
    Pk : npt.NDArray[np.float32]
        Power spectrum
    Nmodes : npt.NDArray[np.float32]
        Number of Fourier modes
    param : pd.Series
        Parameter container

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pysco.iostream import write_power_spectrum_to_ascii_file
    >>> k = np.random.rand(32**3, 3).astype(np.float32)
    >>> Pk = np.random.rand(32**3, 3).astype(np.float32)
    >>> Nmodes = np.random.rand(32**3, 3).astype(np.float32)
    >>> param = pd.Series({"aexp": 0.0, "boxlen": 300.0, 'npart': 128, 'extra':"", 'base':".", "nsteps": 20})
    >>> write_power_spectrum_to_ascii_file(k, Pk, Nmodes, param)
    """
    output_pk = f"{param['base']}/power/pk_{param['extra']}_{param['nsteps']:05d}.dat"
    logging.warning(f"Write P(k) in {output_pk}")
    np.savetxt(
        f"{output_pk}",
        np.c_[k, Pk, Nmodes],
        header=f"aexp = {param['aexp']}\nboxlen = {param['boxlen']} Mpc/h \nnpart = {param['npart']} \nk [h/Mpc] P(k) [Mpc/h]^3 Nmodes",
    )
