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
        <li><a href="#information-files">Information files</a></li>
       </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

PySCo is a multi-threaded Particle-Mesh code (no MPI parallelization) for cosmological simulations which currently contains Newtonian and [Hu & Sawicki (2007)](https://ui.adsabs.harvard.edu/abs/2007PhRvD..76f4004H/abstract) $f(R)$ gravity theories.

The goal is to develop a Python-based N-body code that is user-friendly and efficient. Python was chosen for its widespread use and rapid development capabilities, making it well-suited for collaborative open-source projects. To address performance issues in Python, we utilize [Numba](https://github.com/numba/numba), a high-performance library that compiles Python functions using LLVM. Additionally, Numba facilitates straightforward loop parallelization.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

To run Pysco, you will need the following dependencies (the _conda_ installation is shown, but the same result can be achieved with _pip_).

- Numba

  ```sh
  conda install numba
  ```

- Numba Intel SVML (Optional, to improve performances)
  ```sh
  conda install -c numba icc_rt
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
- PyFFTW
  ```sh
  python -m pip install pyfftw
  ```
- Pyarrow
  ```sh
  conda install -c conda-forge pyarrow
  ```
- Rich
  ```sh
  conda install -c conda-forge rich
  ```
- H5py (optional, to read HDF5 files)
  ```sh
  conda install -c anaconda h5py
  ```

### Installation

The first method is to pip install pysco using (not yet available)

```sh
python -m pip install ??
```

Otherwise, it is possible to install directly from source

```sh
git clone https://github.com/mianbreton/pysco.git
cd pysco
python -m pip install -e .
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

There are two ways to use Pysco, either as command line with parameter file, or as an external package

#### As command line

_To run PySCo with command line it is not necessary to pip install the package._

Move to the Pysco sub-directory

```sh
cd pysco/
```

Write a parameter file. **All strings (except paths and filenames) are case insensitive**. Here is an example

```sh
# param.ini
nthreads = 1  # Number of threads to use in the simulation. For nthreads <= 0 use all threads
# Theoretical model
theory= newton # Cosmological theory to use, either "Newton" or  "fR"
fR_logfR0 = 5 # Background value of the scalaron field today -log(fR0)
fR_n = 2 # Exponent on the curvature in the Hu & Sawicki model. Currently n = 1 or 2
# Cosmology -- Put more parameters later
H0 = 68  # Hubble constant at redshift z=0 (in km/s/Mpc).
Om_m = 0.31   # Matter density parameter
Om_lambda = 0.69 # Dark energy density parameter
w0 = -1.0 # Equation of state for dark energy
wa = 0.0 # Evolution parameter for dark energy equation of state
evolution_table = no # Table specifying the evolution of cosmological parameters (default: "no")
mpgrafic_table = no  # Table for initial conditions (default: "no")
# Simulation dimension
boxlen = 500  # Simulation box length (in Mpc/h)
ncoarse = 8 # Coarse level. Total number of cells = 2**(3*ncoarse)
npart = 256**3 # Number of particles in the simulation
# Initial conditions
z_start = 49 # Starting redshift of the simulation
seed = 42 # Seed for random number generation (completely random if negative)
fixed_ICS = 0 # Use fixed initial conditions. Gaussian Random Field, 1: Fixes the amplitude to match exactly the input P(k)
paired_ICS = 0 # Use paired initial conditions. If enabled, add π to the random phases (works only with fixed_ICS = 1)
power_spectrum_file = /home/user/power_spectra.dat # File path to the power spectrum data
initial_conditions = 3LPT # Type of initial conditions. 1LPT, 2LPT, 3LPT or .h5 file. Else, assumes Gadget format
# Outputs
base=/home/user/boxlen500_n256_lcdm/ # Base directory for storing simulation data
z_out = [1.5, 1.0, 0.66666, 0.53846, 0.25, 0.0] # List of redshifts for output snapshots
save_power_spectrum = all # Save power spectra. Either 'no', 'z_out' for specific redshifts given by z_out or 'all' to compute at every time step
# Particles
integrator = leapfrog # Integration scheme for time-stepping "Leapfrog" or "Euler"
n_reorder = 25  # Re-order particles every n_reorder steps
Courant_factor = 0.8 # Cell fraction for time stepping (Courant_factor < 1 means more time steps)
# Newtonian solver
linear_newton_solver = multigrid # Linear solver for Newton's method: "multigrid", "fft" or "full_fft"
# Multigrid
Npre = 2  # Number of pre-smoothing Gauss-Seidel iterations
Npost = 1  # Number of post-smoothing Gauss-Seidel iterations
epsrel = 1e-2  # Maximum relative error on the residual norm
# Verbose
verbose = 2 # Verbose level. 0 : silent, 1 : basic infos, 2 : full timings
```

Run the command line

```sh
python main.py -c param.ini
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### As package

To obtain the same as above, one first need to import the pysco module, then build a dictionnary (or Pandas Series) containing the user inputs.

```python
# examples/example.py
import pysco

param = {
    "nthreads": 1,
    "theory": "newton",
    "H0": 68,
    "Om_m": 0.31,
    "Om_lambda": 0.69,
    "w0": -1.0,
    "wa": 0.0,
    "evolution_table": "no",
    "mpgrafic_table": "no",
    "boxlen": 500,
    "ncoarse": 8,
    "npart": 256**3,
    "z_start": 49,
    "seed": 42,
    "fixed_ICS": 0,
    "paired_ICS": 0,
    "power_spectrum_file": "/home/user/power_spectrum.dat",
    "initial_conditions": "3LPT",
    "base": "/home/user/boxlen500_n256_lcdm_00000/",
    "z_out": "[10, 5, 2, 1, 0.5, 0]",
    "save_power_spectrum": "all",
    "integrator": "leapfrog",
    "n_reorder": 25,
    "Courant_factor": 0.8,
    "linear_newton_solver": "multigrid",
    "Npre": 2,
    "Npost": 1,
    "epsrel": 1e-2,
    "verbose" : 2
}

# Run simulation
pysco.run(param)

print("Run completed!")

```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- OUTPUTS -->

## Outputs

Pysco produces power spectra, snapshots, and information files.

#### Power spectra

Power spectra are written as ascii files in the directory _base_/power/
where _base_ in given by the user.

The name formatting is pk\__theory_\_ncoarse*N*\__XXXXX_.dat
where _theory_ and _N_ are user inputs.

The ascii file contains three columns: $k$ [$h$/Mpc], $P(k)$ [Mpc/$h$]$^3$, N_modes

Additionally, the file header contains the scale factor, the box length and number of particles.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### Particle snapshots

Particle snapshots are written as _parquet_ files in the directory _base_/output\__XXXXX_

> Note that output_00000/ is used to write the initial conditions. The additional number of directories depend on the length of the input list z_out.

The name formatting is particles\__theory_\_ncoarse*N*\__XXXXX_.parquet

The parquet file contains six columns: x, y, z, vx, vy, vz

Positions are given in box units, that is between 0 and 1.

Velocities are given in supercomoving units. To recover km/s one need to multiply by _unit_l/unit_t_, where both quantities are written in the associated information file.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### Information files

Alongside every snapshot file there is an associated ascii information file.

The name formatting is param\__theory_\_ncoarse*N*\__XXXXX_.txt

It contains parameter file informations as well as useful quantities such as the scale factor and unit conversions (unit_l, unit_t, unit_d for length, time and density respectively) from Pysco units to SI.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Michel-Andrès Breton - michel-andres.breton@obspm.fr
Project Link: [https://github.com/mianbreton/pysco](https://github.com/mianbreton/pysco)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
