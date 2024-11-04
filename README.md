<a id="top"></a>

<h3 align="center">PySCo: Python Simulations for Cosmology</h3>

  <p align="center">
    A Python library to run N-body simulations with modified gravity
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li> <a href="#about-the-project">About The Project</a> </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#testing-the-installation">Testing the installation</a></li>
      </ul>
    </li>
    <li>
       <a href="#usage">Usage</a>
       <ul>
        <li><a href="#as-command-line">As command line</a></li>
        <li><a href="#as-package">As package</a></li>
       </ul>
    </li>
    <li>
      <a href="#outputs">Outputs</a>
      <ul>
        <li><a href="#power-spectra">Power spectra</a></li>
        <li><a href="#particle-snapshots">Particle snapshots</a></li>
       </ul>
    </li>
    <li><a href="#library-utilities">Library utilities</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

PySCo is a multi-threaded Particle-Mesh code (no MPI parallelization) for cosmological simulations with various gravity models such as (currently)

- Newtonian gravity
- $f(R)$ model from [Hu & Sawicki (2007)](https://ui.adsabs.harvard.edu/abs/2007PhRvD..76f4004H/abstract).
- MOND gravity (quasi-linear formulation) from [Milgrom (2010)](https://ui.adsabs.harvard.edu/abs/2010MNRAS.403..886M/abstract).
- Parametrized gravity (scale independent)

The goal is to develop a Python-based N-body code that is user-friendly and efficient. Python was chosen for its widespread use and rapid development capabilities, making it well-suited for collaborative open-source projects. To address performance issues in Python, we utilize [Numba](https://github.com/numba/numba), a high-performance library that compiles Python functions using LLVM. Additionally, Numba facilitates straightforward loop parallelization.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

### Installation

The first method is to pip install pysco using

```sh
python -m pip install pysco-nbody
```

Otherwise, it is possible to install directly from source

```sh
git clone https://github.com/mianbreton/pysco.git
cd pysco
python -m pip install -e .
```

It is then possible to access other branches. If one wants to use the `feature/AwesomeNewFeature` branch but without having to download the source directory, it is possible to pip install directly from github

```sh
python -m pip install git+https://github.com/mianbreton/pysco.git@feature/AwesomeNewFeature
```

> :warning: **If the first method does not work because of a dependency issue** (for example, PyFFTW and icc-rt might not work for on some mac versions): use the second method, and comment the lines referring to the problematic dependencies in pyproject.toml

### Prerequisites

All dependencies will be automatically installed when using pip install (see [Installation](#installation)) so you can skip the remainder of this section.

However, if you prefer to install each of them by hand, then you will need the following libraries (the _conda_ installation is shown, but the same result can be achieved with _pip_).

- Numba

  ```sh
  conda install numba
  ```

- Numba Intel SVML (Optional, to improve performances)

  ```sh
  conda install -c numba icc_rt
  ```

- NumPy

  ```sh
  conda install -c anaconda numpy
  ```

- Pandas

  ```sh
  conda install -c anaconda pandas
  ```

- Astropy

  ```sh
  conda install -c conda-forge astropy
  ```

- Scipy

  ```sh
  conda install -c anaconda scipy
  ```

- Rich

  ```sh
  conda install -c conda-forge rich
  ```

- Pyarrow (optional, to read and write Parquet files)

  ```sh
  conda install -c conda-forge pyarrow
  ```

- H5py (optional, to read and write HDF5 files)

  ```sh
  conda install -c anaconda h5py
  ```

- PyFFTW (pip install works better than conda)

  ```sh
  python -m pip install pyfftw
  ```

  <p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

There are two ways to use Pysco, either as command line with parameter file, or as an external package.

> Numba uses _Just-in-Time_ and/or _Ahead-of-Time_ compilation, meaning that for the former the function is compiled when it is reached for the first time, while the latter is compiled before running the code (happens when the input/output types are specified in the function decorator). This means that when the code is run (or imported) for the first time, it will spend some time compiling the functions. These are then cached and subsequent runs will not have to compile the functions again.

#### As command line

_To run PySCo with command line it is not necessary to pip install the package. However, one must still at least install the depdendecies (see [Prerequisites](#prerequisites))_

Move to the main Pysco directory

```sh
cd pysco/
```

A example parameter file is available in `examples/param.ini`. **All strings (except paths and filenames) are case insensitive**.

```sh
# examples/param.ini
# Run this test from the pysco directory, as
# python pysco/main.py -c examples/param.ini
# All strings (except paths and filenames) are case insensitive
##################################################
nthreads = 1  # Number of threads to use in the simulation. For nthreads <= 0 use all threads
# Theoretical model
theory= newton # Cosmological theory to use, either "Newton", "fR", "mond" or "parametrized"
## f(R)
fR_logfR0 = 5 # Background value of the scalaron field today -log(fR0)
fR_n = 1 # Exponent on the curvature in the Hu & Sawicki model. Currently n = 1 or 2
## QUMOND
mond_function = simple # "simple", "n", "beta", "gamma" or "delta"
mond_g0 = 1.2 # Acceleration constant (in 1e-10 m/s²)
mond_scale_factor_exponent = 0 # Exponent N so that g0 -> a^N g0
mond_alpha = 1 #  Interpolating function parameter
## Parametrized
parametrized_mu0 = -0.1 # If null, then is equivalent to GR. Model from Abbott et al. (2019)
# Cosmology -- Put more parameters later
H0 = 72  # Hubble constant at redshift z=0 (in km/s/Mpc).
Om_m = 0.25733   # Matter density parameter
T_cmb = 2.726 # CMB temperature parameter
N_eff = 3.044 # Effective number of neutrino species (by default 3.044)
w0 = -1.0 # Equation of state for dark energy
wa = 0.0 # Evolution parameter for dark energy equation of state
# Simulation dimension
boxlen = 100  # Simulation box length (in Mpc/h)
ncoarse = 7 # Coarse level. Total number of cells = 2**(3*ncoarse)
npart = 128**3 # Number of particles in the simulation
# Initial conditions
z_start = 49 # Starting redshift of the simulation
seed = 42 # Seed for random number generation (completely random if negative)
position_ICS = center # Initial particle position on uniform grid. Put "center" or "edge" to start from cell centers or edges.
fixed_ICS = False # Use fixed initial conditions (Gaussian Random Field). If True, fixes the amplitude to match exactly the input P(k)
paired_ICS = False # Use paired initial conditions. If True, add π to the random phases (works only with fixed_ICS = True)
dealiased_ICS = False # Dealiasing 2LPT and 3LPT components using Orszag 3/2 rule
power_spectrum_file = examples/pk_lcdmw7v2.dat # File path to the power spectrum data
initial_conditions = 2LPT # Type of initial conditions. 1LPT, 2LPT, 3LPT or or snapshot number (for restart), or .h5 RayGal file. Else, assumes Gadget format
# Outputs
base = examples/boxlen100_n128_lcdmw7v2_00000/ # Base directory for storing simulation data
output_snapshot_format = HDF5 # Particle snapshot format. "parquet" or "HDF5"
z_out = [10, 5, 2, 1, 0.5, 0]  # List of redshifts for output snapshots. The simulation stops at the last redshift.
save_power_spectrum = yes # Save power spectra. Either 'no', 'z_out' for specific redshifts given by z_out or 'yes' to compute at every time step. Uses same mass scheme and grid size (ncoarse) as for the PM solver
# Particles
integrator = leapfrog # Integration scheme for time-stepping "Leapfrog" or "Euler"
mass_scheme = TSC # CIC or TSC
n_reorder = 50  # Re-order particles every n_reorder steps
# Time stepping
Courant_factor = 1.0 # Cell fraction for time stepping based on velocity/acceleration (Courant_factor < 1 means more time steps)
max_aexp_stepping = 10 # Maximum percentage [%] of scale factor that cannot be exceeded by a time step
# Newtonian solver
linear_newton_solver = multigrid # Linear solver for Newton's method: "multigrid", "fft", "fft_7pt" or "full_fft"
gradient_stencil_order = 5 # n-point stencil with n = 2, 3, 5 or 7
# Multigrid
Npre = 2  # Number of pre-smoothing Gauss-Seidel iterations
Npost = 1  # Number of post-smoothing Gauss-Seidel iterations
epsrel = 1e-2  # Maximum relative error on the residual norm
# Verbose
verbose = 1 # Verbose level. 0 : silent, 1 : basic infos, 2 : full timings
```

Run the command line

```sh
python pysco/main.py -c examples/param.ini
```

<p align="right">(<a href="#top">back to top</a>)</p>

#### As package

To obtain the same as above, one first need to import the pysco module, then build a dictionnary (or Pandas Series) containing the user inputs.

```python
# examples/example.py
from pathlib import Path
import pysco

path = Path(__file__).parent.absolute()

param = {
    "nthreads": 1,
    "theory": "newton",
    # "fR_logfR0": 5,
    # "fR_n": 1,
    # "mond_function": "simple",
    # "mond_g0": 1.2,
    # "mond_scale_factor_exponent": 0,
    # "mond_alpha": 1,
    # "parametrized_mu0": 0.1,
    "H0": 72,
    "Om_m": 0.25733,
    "T_cmb": 2.726,
    "N_eff": 3.044,
    "w0": -1.0,
    "wa": 0.0,
    "boxlen": 100,
    "ncoarse": 7,
    "npart": 128**3,
    "z_start": 49,
    "seed": 42,
    "position_ICS": "center",
    "fixed_ICS": False,
    "paired_ICS": False,
    "dealiased_ICS": False,
    "power_spectrum_file": f"{path}/pk_lcdmw7v2.dat",
    "initial_conditions": "2LPT",
    "base": f"{path}/boxlen100_n128_lcdmw7v2_00000/",
    "z_out": "[10, 5, 2, 1, 0.5, 0]",
    "output_snapshot_format": "HDF5",
    "save_power_spectrum": "yes",
    "integrator": "leapfrog",
    "n_reorder": 50,
    "mass_scheme": "TSC",
    "Courant_factor": 1.0,
    "max_aexp_stepping": 10,
    "linear_newton_solver": "multigrid",
    "gradient_stencil_order": 5,
    "Npre": 2,
    "Npost": 1,
    "epsrel": 1e-2,
    "verbose": 1,
}

# Run simulation
pysco.run(param)

print("Run completed!")

```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- OUTPUTS -->

## Outputs

Pysco produces power spectra, snapshots, and extra information.

#### Power spectra

Power spectra are written as ascii files in the directory `base/power/`
where _base_ in given by the user.

The name formatting is `pk_theory_ncoarseN_XXXXX_.dat`
where _theory_ and _N_ are user inputs.

The ascii file contains three columns: `k [h/Mpc], P(k) [Mpc/h]^3, N_modes`

Additionally, the file header contains the scale factor, the box length and number of particles.

<p align="right">(<a href="#top">back to top</a>)</p>

#### Background evolution

Background evolution file written as ascii files in `base/evolution_table_pysco.txt`
where _base_ in given by the user.

The ascii file contains three columns: `aexp, H/H0, t_supercomoving, dplus1, f1, dplus2, f2, dplus3a, f3a, dplus3b, f3b, dplus3c, f3c`

where `aexp` is the scale factor, `H/H0` the dimensionless Hubble parameter, `t_supercomoving` is the time in dimensionless supercomoving units, while `dplusN` and `fN` are the _N_-th order growth factor and growth rate respectively.

<p align="right">(<a href="#top">back to top</a>)</p>

#### Particle snapshots

Particles are written as either HDF5 or Parquet format. For the former, additional informations are written as attributes at the file root. For the latter, an extra ascii file is written.

- Positions are given in box units (between 0 and 1).

- **Velocities are given in supercomoving units. To recover km/s one need to multiply by _unit_l/unit_t_**, where both quantities are written either in the attributes for HDF5 file or in the associated information file for parquet format.

> Note that output_00000/ is used to write the initial conditions. The additional number of directories depend on the length of the input list z_out.

##### HDF5 format

Particle snapshots are written as HDF5 files in the directory `base/output_XXXXX`

The name formatting is `particles_theory_ncoarseN_XXXXX.h5`

The HDF5 file contains two datasets: "position" and "velocity"
as well as attributes. These files can be read as

```python
import h5py
with h5py.File('snapshot.h5', 'r') as h5r:
  pos = h5r['position'][:]
  vel = h5r['velocity'][:]
  unit_t = h5r.attrs['unit_t']
```

##### Parquet format

Particle snapshots can also be written as _parquet_ files in the directory `base/output_XXXXX`

The name formatting is `particles_theory_ncoarseN_XXXXX.parquet`

The parquet file contains six columns: `x, y, z, vx, vy, vz`. These files can be read as

```python
import pandas as pd
df = pd.read_parquet('snapshot.parquet')
x = df['x']
vx = df['vx']
```

Alongside every parquet file there is an associated ascii information file.

The name formatting is `param_theory_ncoarseN_XXXXX.txt`

It contains parameter file informations as well as useful quantities such as the scale factor and unit conversions (unit_l, unit_t, unit_d for length, time and density respectively) from Pysco units to SI.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Library -->

## Library utilities

As a side-product , PySCo can also be used as a library which contains utilities for particle and mesh computations.

Since PySCo does not use classes (not supported by Numba), we made it straightforward to use its functions on NumPy arrays for various purposes. **We give a few examples below**.

- Computing a density grid based on particle positions with a Triangular-Shaped Cloud scheme

```python
import numpy as np
from pysco.mesh import TSC
positions = np.random.rand(32**3, 3).astype(np.float32)
density_grid = TSC(positions, ncells_1d=64)
```

- Interpolating to a finer grid

```python
import numpy as np
from pysco.mesh import prolongation
coarse_field = np.random.rand(32, 32, 32).astype(np.float32)
fine_field = prolongation(coarse_field)
```

- Re-ordering particles according to [Morton](https://en.wikipedia.org/wiki/Z-order_curve) indexing (to increase data locality)

```python
import numpy as np
from pysco.utils import reorder_particles
position = np.random.rand(64, 3).astype(np.float32)
velocity = np.random.rand(64, 3).astype(np.float32)
position, velocity = reorder_particles(position, velocity)
```

- FFT a real-valued grid and compute power spectrum

```python
import numpy as np
from pysco.fourier import fourier_grid_to_Pk, fft_3D_real
nthreads = 2
density = np.random.rand(64, 64, 64).astype(np.float32)
density_k = fft_3D_real(density, nthreads)
MAS = 0 # Mass assignment scheme. # None = 0, NGP = 1, CIC = 2, TSC = 3
k, pk, modes = fourier_grid_to_Pk(density_k, MAS)
# For cosmological densities, need additional conversion factors
boxlen = 100 # Box length, in Mpc/h
pk *= (boxlen / len(density) ** 2) ** 3
k *= 2 * np.pi / boxlen
```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CITATION -->

## Citation

If you use PySCo in your work, please cite [Breton (2024)](https://arxiv.org/abs/2410.20501)

```bibtex
#biblio.bib
@ARTICLE{breton2024pysco,
       author = {{Breton}, Michel-Andr{\`e}s},
        title = "{PySCo: A fast Particle-Mesh $N$-body code for modified gravity simulations in Python}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2024,
        month = oct,
          eid = {arXiv:2410.20501},
        pages = {arXiv:2410.20501},
          doi = {10.48550/arXiv.2410.20501},
archivePrefix = {arXiv},
       eprint = {2410.20501},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv241020501B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Michel-Andrès Breton - michel-andres.breton@obspm.fr

Project Link: [https://github.com/mianbreton/pysco](https://github.com/mianbreton/pysco)

<p align="right">(<a href="#top">back to top</a>)</p>
