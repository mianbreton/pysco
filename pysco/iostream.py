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
    >>> import os
    >>> this_dir = os.path.dirname(os.path.abspath(__file__))
    >>> params = read_param_file(f"{this_dir}/../examples/param.ini")
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
    for key in param.columns:
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

    param["write_snapshot"] = False

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
    >>> import os
    >>> this_dir = os.path.dirname(os.path.abspath(__file__))
    >>> position, velocity = read_snapshot_particles_hdf5(f"{this_dir}/../examples/snapshot.h5")
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
    >>> import os
    >>> this_dir = os.path.dirname(os.path.abspath(__file__))
    >>> position, velocity = read_snapshot_particles_parquet(f"{this_dir}/../examples/snapshot.parquet")
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
    >>> import os
    >>> this_dir = os.path.dirname(os.path.abspath(__file__))
    >>> position = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    >>> velocity = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    >>> parameters = pd.Series({"snapshot_format": "parquet", "base": f"{this_dir}/../examples/", "i_snap": 0, "extra": "extra_info", "aexp": 1.0})
    >>> write_snapshot_particles(position, velocity, parameters)
    >>> parameters = pd.Series({"snapshot_format": "hdf5", "base": f"{this_dir}/../examples/", "i_snap": 0, "extra": "extra_info", "aexp": 1.0})
    >>> write_snapshot_particles(position, velocity, parameters)
    """
    if "parquet".casefold() == param["output_snapshot_format"].casefold():
        filename = f"{param['base']}/output_{param['i_snap']:05d}/particles_{param['extra']}.parquet"
        write_snapshot_particles_parquet(filename, position, velocity)
        param_filename = f"{param['base']}/output_{param['i_snap']:05d}/param_{param['extra']}_{param['i_snap']:05d}.txt"
        param.to_csv(
            param_filename,
            sep="=",
            header=False,
        )
        logging.warning(f"Parameter file written at ...{param_filename=}")
    elif "hdf5".casefold() == param["output_snapshot_format"].casefold():
        filename = f"{param['base']}/output_{param['i_snap']:05d}/particles_{param['extra']}.h5"
        write_snapshot_particles_hdf5(filename, position, velocity, param)
    else:
        raise ValueError(f"{param['snapshot_format']=}, should be 'parquet' or 'hdf5'")

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
    >>> import os
    >>> this_dir = os.path.dirname(os.path.abspath(__file__))
    >>> position = np.random.rand(32**3, 3).astype(np.float32)
    >>> velocity = np.random.rand(32**3, 3).astype(np.float32)
    >>> write_snapshot_particles_parquet(f"{this_dir}/../examples/snapshot.parquet", position, velocity)
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
    >>> import os
    >>> this_dir = os.path.dirname(os.path.abspath(__file__))
    >>> position = np.random.rand(32**3, 3).astype(np.float32)
    >>> velocity = np.random.rand(32**3, 3).astype(np.float32)
    >>> param = pd.Series({"Attribute_0": 0.0, "Attribute_1": 300.0})
    >>> write_snapshot_particles_hdf5(f"{this_dir}/../examples/snapshot.h5", position, velocity, param)
    """
    import h5py

    with h5py.File(filename, "w") as h5f:
        h5f.create_dataset("position", data=position)
        h5f.create_dataset("velocity", data=velocity)
        for key, item in param.items():
            h5f.attrs[key] = item