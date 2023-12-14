<h3 align="center">PYSCO: PYthon Simulations for COsmology</h3>

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
        <li><a href="#command-line">Command line</a></li>
        <li><a href="#external-package">External package</a></li>
       </ul>
    </li>
    <li><a href="#outputs">Outputs</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

Pysco is a multi-threaded Particle-Mesh code (no MPI parallelization) which currently contains Newtonian and [Hu & Sawicki (2007)](https://ui.adsabs.harvard.edu/abs/2007PhRvD..76f4004H/abstract) $f(R)$ gravity theories.

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
- H5py (optional, to read HDF5 files)
  ```sh
  conda install -c anaconda h5py
  ```
- Rich (optional, for nicer _print_ statement)
  ```sh
  conda install -c conda-forge rich
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/mianbreton/pysco.git
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

There are two ways to use Pysco, either as command line with parameter file, or as an external package

#### Command line

Move to the Pysco directory

```sh
cd pysco/
```

Example of parameter file. All strings (except paths and filenames) are case insensitive.

```sh
### param.ini
nthreads = 1  # For nthreads <= 0 use all available threads
# Theoretical model
theory = newton # Newton, fR
fR_logfR0 = 5 # OPTIONAL: Background value of the scalaron field today -log(fR0)
fR_n = 2 # OPTIONAL: Exponent on the curvature in the Hu & Sawicki model. Currently n = 1 or 2
# Cosmology
H0 = 68  # in km/s/Mpc
Om_m = 0.31
Om_lambda = 0.69
w0 = -1.0
wa = 0.0
evolution_table = no # Can also give a RAMSES evolution file
mpgrafic_table = no # Can also give an MPGRAFIC file
# Simulation dimension
boxlen = 500  # Mpc/h
ncoarse = 8 # Number of cells = 2**(3*ncoarse)
npart = 256**3 # Number of particles (ignore if we use an already existing snapshot)
# Initial conditions
z_start = 49 # Used with initial_conditions = 1LPT, 2LPT or 3LPT
seed = 42 # Random seed for initial conditions (completely random if negative)
fixed_ICS = 0 # 0: Gaussian Random Field, 1: Fixes the amplitude to match exactly the input P(k)
paired_ICS = 0 # If enabled, add π to the random phases (works only with fixed_ICS = 1)
power_spectrum_file = /home/user/power_spectrum.dat # Power spectrum file
initial_conditions = 3LPT # 1LPT, 2LPT, 3LPT, .h5 file, else assumes Gadget format
# Outputs
base=/home/user/boxlen500_n256_lcdm_00000/ # Output directory
z_out = [10, 5, 2, 1, 0.5, 0]  # Output snapshots at these redshifts
save_power_spectrum = all # 'no', 'z_out' for specific redshifts given by z_out or 'all' to compute at every time step
# Particles
integrator = leapfrog # "leapfrog" or "euler"
n_reorder = 25  # Re-order particles every n_reorder steps
Courant_factor = 0.5 # Cell fraction for time stepping (Courant_factor < 1 means more time steps)
# Newtonian solver
linear_newton_solver = multigrid # Linear Poisson equation solver: "multigrid", "fft" or "full_fft"
# Multigrid parameters
Npre = 2  # Number of pre-smoothing Gauss-Seidel  iterations
Npost = 1  # Number of post-smoothing Gauss-Seidel iterations
epsrel = 1e-2  # Relative error on the residual norm
```

Run the command line

```sh
python main.py -c param.ini
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### External package

In progress

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

Alongside every power spectra or snapshot file there is an associated information file written in ascii.

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
