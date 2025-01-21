import numpy as np
import pandas as pd
from pysco.solver import fft
from pysco.multigrid import V_cycle, F_cycle, W_cycle
from pysco.utils import prod_vector_scalar
from pysco.mesh import (
    TSC,
    CIC,
    derivative5,
    derivative7,
    invTSC_vec,
    invCIC_vec,
    invTSC,
    invCIC,
)
from pysco.utils import linear_operator_inplace
import ewald
import matplotlib.pyplot as plt
from pysco.utils import set_units
from astropy.constants import G, pc, c
import hernquist
from scipy.special import erfc
import matplotlib

matplotlib.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "Serif"
plt.rcParams["xtick.top"] = "True"
matplotlib.rcParams.update({"font.size": 16})
plt.rcParams["xtick.top"] = "True"


def force_point_mass(r):
    return 1.0 / r**2


def potential_point_mass(r):
    return -1.0 / r


def potential_sphere(r, R):
    inside_sphere = r < R
    potential = np.zeros_like(r)
    potential[inside_sphere] = -(3 * R**2 - r[inside_sphere] ** 2) / (2 * R**3)
    potential[~inside_sphere] = -1.0 / r[~inside_sphere]
    return potential


def force_sphere(r, R):
    inside_sphere = r < R
    force = np.zeros_like(r)
    force[inside_sphere] = r[inside_sphere] / R**3
    force[~inside_sphere] = 1.0 / r[~inside_sphere] ** 2
    return force


def force_sphere_periodic(x, y, z, R):
    dmax = 8
    res = np.zeros_like(x)
    norm = np.sqrt(x * x + y * y + z * z)
    nx = x / norm
    ny = y / norm
    nz = z / norm
    norm = 0
    for ix in range(-dmax, dmax + 1):
        for iy in range(-dmax, dmax + 1):
            for iz in range(-dmax, dmax + 1):
                rx = x + ix
                ry = y + iy
                rz = z + iz
                r2 = rx**2 + ry**2 + rz**2
                mask = r2 < 2.6**2
                res[mask] += (
                    force_sphere(np.sqrt(r2[mask]), R)
                    * (rx[mask] * nx[mask] + ry[mask] * ny[mask] + rz[mask] * nz[mask])
                    / np.sqrt(r2[mask])
                )
    return res


def force_sphere_periodic_full(x, y, z, R):
    alpha = 2
    dmax = 4
    res = np.zeros_like(x)
    norm = np.sqrt(x * x + y * y + z * z)
    nx = x / norm
    ny = y / norm
    nz = z / norm
    norm = 0
    for ix in range(-dmax, dmax + 1):
        for iy in range(-dmax, dmax + 1):
            for iz in range(-dmax, dmax + 1):
                rx = x + ix
                ry = y + iy
                rz = z + iz
                r2 = rx**2 + ry**2 + rz**2
                mask = r2 < 2.6**2
                res[mask] += (
                    force_sphere(np.sqrt(r2[mask]), R)
                    * (rx[mask] * nx[mask] + ry[mask] * ny[mask] + rz[mask] * nz[mask])
                    / np.sqrt(r2[mask])
                    * (
                        erfc(alpha * np.sqrt(r2[mask]))
                        + 2 * alpha / np.sqrt(np.pi) * np.exp(-(alpha**2) * r2[mask])
                    )
                )

    for ix in range(-dmax, dmax + 1):
        for iy in range(-dmax, dmax + 1):
            for iz in range(-dmax, dmax + 1):
                rx = x + ix
                ry = y + iy
                rz = z + iz
                r2 = rx**2 + ry**2 + rz**2
                if ix == 0 and iy == 0 and iz == 0:
                    continue
                mask = (ix**2 + iy**2 + iz**2) < 8
                res[mask] += (
                    -2
                    * (ix * nx[mask] + iy * ny[mask] + iz * nz[mask])
                    / (ix**2 + iy**2 + iz**2)
                    * np.exp(-np.pi**2 * (ix**2 + iy**2 + iz**2) / alpha**2)
                    * np.sin(
                        2 * np.pi * (ix * rx[mask] + iy * ry[mask] + iz * rz[mask])
                    )
                )

    return res / np.pi


param = pd.Series(
    {
        "theory": "newton",
        "nthreads": 4,
        "boxlen": 100.0,
        "npart": 64**3,
        "compute_additional_field": False,
        "ncoarse": 6,
        "Npre": 2,
        "Npost": 2,
        "aexp": 1.0,
        "H0": 70,
        "Om_m": 0.3,
        "MAS_index": 0,
    }
)

ncells_1d = 2 ** int(param["ncoarse"])
h = np.float32(1.0 / ncells_1d)
#
set_units(param)
g = G.value * 1e-9  # m3/kg/s2 -> km3/kg/s2
GM = g * param["mpart"]


h = np.float32(1.0 / ncells_1d)
# pos_point_mass = np.random.rand(3).astype(np.float32)
pos_point_mass_edge = np.array([0.5, 0.5, 0.5]).astype(np.float32)
pos_point_mass_center = np.array([0.5 + 0.5 * h, 0.5 + 0.5 * h, 0.5 + 0.5 * h]).astype(
    np.float32
)
pos_test_particles = (
    0.3 * (np.random.rand(128**3, 3) - 0.5) + pos_point_mass_edge
).astype(np.float32)

derivative = derivative7


def get_accelerations_and_plot(
    i, j, mas, inv_mas, pos_point_mass, label_mas, label_position
):
    r_test_particles = np.sqrt(
        (pos_test_particles[:, 0] - pos_point_mass[0]) ** 2
        + (pos_test_particles[:, 1] - pos_point_mass[1]) ** 2
        + (pos_test_particles[:, 2] - pos_point_mass[2]) ** 2
    )
    rhs = mas(np.array([pos_point_mass]), ncells_1d)
    """ print(label_mas, label_position, mas, inv_mas)
    print(f"{rhs[ncells_1d//2 - 1, ncells_1d//2  - 1]=}")
    print(f"{rhs[ncells_1d//2 - 1, ncells_1d//2]=}")
    print(f"{rhs[ncells_1d//2 - 1, ncells_1d//2 + 1]=}")
    print(f"{rhs[ncells_1d//2, ncells_1d//2  - 1]=}")
    print(f"{rhs[ncells_1d//2, ncells_1d//2]=}")
    print(f"{rhs[ncells_1d//2, ncells_1d//2 + 1]=}")
    print(f"{rhs[ncells_1d//2 + 1, ncells_1d//2 - 1]=}")
    print(f"{rhs[ncells_1d//2 + 1, ncells_1d//2]=}")
    print(f"{rhs[ncells_1d//2 + 1, ncells_1d//2 + 1]=}") """

    rhs *= 1.5 * param["Om_m"]
    # rhs *= 1.5 * param["Om_m"]
    param["linear_newton_solver"] = "fft"
    potential_fft = fft(rhs, param)
    force_fft = derivative(potential_fft)
    potential_fft = 0
    param["linear_newton_solver"] = "fft_7pt"
    potential_fft_7pt = fft(rhs, param)
    force_fft_7pt = derivative(potential_fft_7pt)
    potential_fft_7pt = 0
    param["linear_newton_solver"] = "multigrid"
    minus_one_sixth_h2 = np.float32(-(h**2) / 6)
    potential_V = prod_vector_scalar(rhs, minus_one_sixth_h2)
    # Call the V_cycle_FAS function
    V_cycle(potential_V, rhs, param)
    V_cycle(potential_V, rhs, param)
    force_2V = derivative(potential_V)
    potential_V = 0

    # Acceleration to test part
    acc_fft = inv_mas(force_fft, pos_test_particles)
    force_fft = 0
    acc_radial_fft = (
        acc_fft[:, 0] * (pos_test_particles[:, 0] - pos_point_mass[0])
        + acc_fft[:, 1] * (pos_test_particles[:, 1] - pos_point_mass[1])
        + acc_fft[:, 2] * (pos_test_particles[:, 2] - pos_point_mass[2])
    ) / r_test_particles
    acc_tangential_fft = (
        -acc_fft[:, 0] * (pos_test_particles[:, 1] - pos_point_mass[1])
        + acc_fft[:, 1] * (pos_test_particles[:, 0] - pos_point_mass[0])
    ) / r_test_particles
    acc_fft_7pt = inv_mas(force_fft_7pt, pos_test_particles)
    force_fft_7pt = 0
    acc_radial_fft_7pt = (
        acc_fft_7pt[:, 0] * (pos_test_particles[:, 0] - pos_point_mass[0])
        + acc_fft_7pt[:, 1] * (pos_test_particles[:, 1] - pos_point_mass[1])
        + acc_fft_7pt[:, 2] * (pos_test_particles[:, 2] - pos_point_mass[2])
    ) / r_test_particles
    acc_tangential_fft_7pt = (
        -acc_fft_7pt[:, 0] * (pos_test_particles[:, 1] - pos_point_mass[1])
        + acc_fft_7pt[:, 1] * (pos_test_particles[:, 0] - pos_point_mass[0])
    ) / r_test_particles
    acc_2V = inv_mas(force_2V, pos_test_particles)
    force_2V = 0
    acc_radial_2V = (
        acc_2V[:, 0] * (pos_test_particles[:, 0] - pos_point_mass[0])
        + acc_2V[:, 1] * (pos_test_particles[:, 1] - pos_point_mass[1])
        + acc_2V[:, 2] * (pos_test_particles[:, 2] - pos_point_mass[2])
    ) / r_test_particles
    acc_tangential_2V = (
        -acc_2V[:, 0] * (pos_test_particles[:, 1] - pos_point_mass[1])
        + acc_2V[:, 1] * (pos_test_particles[:, 0] - pos_point_mass[0])
    ) / r_test_particles

    markersize = 1.5
    alpha = 0.6
    axs[i, j].plot(
        r_test_particles / h,
        r_test_particles**2 * acc_radial_fft * norm,
        marker="o",
        markersize=markersize,
        linestyle="",
        alpha=alpha,
        color=colors[7],
        label=r"$\mathrm{FFT}$",
    )
    axs[i, j].plot(
        r_test_particles / h,
        r_test_particles**2 * acc_tangential_fft * norm,
        marker="v",
        markersize=markersize,
        linestyle="",
        alpha=alpha,
        color=colors[7],
    )
    axs[i, j].plot(
        r_test_particles / h,
        r_test_particles**2 * acc_radial_fft_7pt * norm,
        marker="o",
        markersize=markersize,
        linestyle="",
        alpha=alpha,
        color=colors[8],
        label=r"$\mathrm{FFT\_7PT}$",
    )

    axs[i, j].plot(
        r_test_particles / h,
        r_test_particles**2 * acc_tangential_fft_7pt * norm,
        marker="v",
        markersize=markersize,
        linestyle="",
        alpha=alpha,
        color=colors[8],
    )

    axs[i, j].plot(
        r_test_particles / h,
        r_test_particles**2 * acc_radial_2V * norm,
        marker="o",
        markersize=markersize,
        linestyle="",
        alpha=alpha,
        color=colors[0],
        label=r"$\mathrm{Multigrid}$",
    )
    axs[i, j].plot(
        r_test_particles / h,
        r_test_particles**2 * acc_tangential_2V * norm,
        marker="v",
        markersize=markersize,
        linestyle="",
        alpha=alpha,
        color=colors[0],
    )

    axs[i, j].text(0.15, 1.1, label_mas, fontdict=None)
    axs[i, j].text(0.15, 0.7, label_position, fontdict=None)


nr = 100
r = np.logspace(-3, 0, nr)
phi = 2 * np.pi * np.random.rand(nr)
theta = np.arccos(2 * np.random.rand(nr) - 1)
x = np.sin(phi) * np.sin(theta) * r
y = np.cos(phi) * np.sin(theta) * r
z = np.cos(theta) * r
phi = 0
theta = 0

norm = param["unit_l"] / param["unit_t"] ** 2 / (GM / param["unit_l"] ** 2)

colors = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]

f, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 8))


get_accelerations_and_plot(
    0, 0, CIC, invCIC_vec, pos_point_mass_center, "CIC", "Centre"
)
get_accelerations_and_plot(1, 0, CIC, invCIC_vec, pos_point_mass_edge, "CIC", "Edge")
get_accelerations_and_plot(
    0, 1, TSC, invTSC_vec, pos_point_mass_center, "TSC", "Centre"
)
get_accelerations_and_plot(1, 1, TSC, invTSC_vec, pos_point_mass_edge, "TSC", "Edge")

for ax in axs.ravel():
    ax.loglog(
        r / h,
        r**2 * force_sphere(r, h),
        "k",
    )
    ax.loglog(
        r / h,
        r**2 * force_sphere(r, np.cbrt(6.0 / np.pi) * h),
        "k:",
    )
    ax.loglog(
        r / h,
        r**2 * force_sphere_periodic(x, y, z, h),
        "k--",
    )
    ax.set_xlim(0.1, 20)
    ax.set_ylim(0.01, 3)


axs[0, 0].legend(
    loc="upper center",
    bbox_to_anchor=(1, 1.25),
    ncol=3,
    # fancybox=True,
    # shadow=True,
    fontsize=16,
    markerscale=10,
)
import matplotlib.ticker as mticker

for i in range(2):
    axs[i, 0].set_ylabel(r"$F/F_{\rm true}$")
    axs[1, i].set_xlabel(r"$r~\mathrm{(units~of~coarse~cells)}$")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: "{:g}".format(y)))

plt.subplots_adjust(hspace=0, wspace=0.0)

plt.savefig(
    "/home/mabreton/data/PySCo/fig/force_sphere.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300,
)
plt.savefig(
    "/home/mabreton/data/PySCo/fig/force_sphere.png", bbox_inches="tight", dpi=300
)

plt.show()
plt.close()
