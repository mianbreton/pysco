"""
Python interface for running cosmological simulations using the PySCo library.
It provides a convenient way to set simulation parameters and run simulations for studying the evolution
of large-scale structures in the universe.
"""

import pysco

param = {
    "nthreads": 1,
    "theory": "newton",
    # "fR_logfR0": 5,
    # "fR_n": 1,
    # "mond_function": "simple",
    # "mond_g0": 1.2,
    # "mond_alpha": 1,
    # "parametrized_mu0": 0.1,
    "H0": 72,
    "Om_m": 0.25733,
    "T_cmb": 2.726,
    "N_eff": 3.044,
    "w0": -1.0,
    "wa": 0.0,
    "boxlen": 500,
    "ncoarse": 7,
    "npart": 128**3,
    "z_start": 49,
    "seed": 42,
    "position_ICS": "edge",
    "fixed_ICS": False,
    "paired_ICS": False,
    "dealiased_ICS": False,
    "power_spectrum_file": "/home/user/pysco/examples/pk_lcdmw7v2.dat",
    "initial_conditions": "3LPT",
    "base": "/home/user/boxlen500_n128_lcdm_00000/",
    "z_out": "[10, 5, 2, 1, 0.5, 0]",
    "output_snapshot_format": "HDF5",
    "save_power_spectrum": "yes",
    "integrator": "leapfrog",
    "n_reorder": 25,
    "mass_scheme": "TSC",
    "Courant_factor": 1.0,
    "max_aexp_stepping": 10,
    "linear_newton_solver": "multigrid",
    "gradient_stencil_order": 5,
    "Npre": 2,
    "Npost": 1,
    "epsrel": 1e-2,
    "verbose": 2,
}

# Run simulation
pysco.run(param)

print("Run Completed!")
