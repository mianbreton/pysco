import pysco
import pandas as pd

# Initialize parameter container (easier to write a dictionnary first, but can also directly be done with pd.Series)

dict = {
    "nthreads": 2,
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
    "power_spectrum_file": "/home/mabreton/CLPT_model/src/power_spectra/pk_lcdmw7v2.dat",
    "initial_conditions": "3LPT",
    "base": "/home/mabreton/boxlen500_n256_lcdm_00000/",
    "z_out": "[10, 5, 2, 1, 0.5, 0]",
    "save_power_spectrum": "all",
    "integrator": "leapfrog",
    "n_reorder": 25,
    "Courant_factor": 0.5,
    "linear_newton_solver": "multigrid",
    "Npre": 2,
    "Npost": 1,
    "epsrel": 1e-2,
}
param = pd.Series(dict)

# Run simulation
pysco.run(param)
print("Finished!")