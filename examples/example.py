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
    # "qumond_a0": 1e-15,
    # "qumond_alpha": 1,
    # "parametrized_mu0": 0.1,
    "H0": 72,
    "Om_m": 0.25733,
    "Om_lambda": 0.742589237,
    "w0": -1.0,
    "wa": 0.0,
    "evolution_table": "no",
    "mpgrafic_table": "no",
    "boxlen": 500,
    "ncoarse": 7,
    "npart": 128**3,
    "z_start": 49,
    "seed": 42,
    "fixed_ICS": 0,
    "paired_ICS": 0,
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
    "linear_newton_solver": "multigrid",
    "Npre": 2,
    "Npost": 1,
    "epsrel": 1e-2,
    "verbose": 2,
}

# Run simulation
pysco.run(param)

print("Run Completed!")
