#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from pathlib import Path
import json
import matplotlib.pylab as pl 
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

matplotlib.rcParams.update({"font.size": 12})                                                                                                                                           
matplotlib.rcParams["mathtext.fontset"] = "cm"                                                                                                                                          
plt.rcParams["font.family"] = "Serif"  


# In[2]:


def load_dat_with_json_header(filename):
    with open(filename, "r") as f:
        header_line = f.readline()[2:]  # remove # and whitespace
        meta = json.loads(header_line)

    data = np.loadtxt(filename, skiprows=1)

    return meta, data

def clean_key(key):
    if key == "lambda_max":
        return r"\lambda_{\rm max}"
    elif key == "lambda_min":
        return r"\lambda_{\rm min}"
    elif key == "w1":
        return r"w_1"
    elif key == "w2":
        return r"w_2"
    elif key == "w":
        return r"w"
    elif key == "f":
        return r"f"
    else:
        print(f"{key=}")
        raise ValueError(f"Missing clean key")

def clean_title(file):
    if "jacobi-jacobi" in file:
        return "Jacobi - Jacobi"
    elif "gs-gs" in file:
        return "Gauss-Seidel - Gauss-Seidel"
    elif "jacobi-gs" in file:
        return "Jacobi - Gauss-Seidel"
    elif "jacobi_weight" in file:
        return "Jacobi"
    elif "gauss_seidel" in file:
        return "Gauss-Seidel"
    elif "chebyshev1-jacobi" in file:
        return "Chebyshev 1st - Jacobi"
    elif "chebyshev1-gs" in file:
        return "Chebyshev 1st - Gauss-Seidel"
    elif "chebyshev4-jacobi" in file:
        return "Chebyshev 4th - Jacobi"
    elif "chebyshev4-gs" in file:
        return "Chebyshev 4th - Gauss-Seidel"
    elif "chebyshev4_opt-jacobi" in file:
        return "Chebyshev 4th (Opt) - Jacobi"
    elif "chebyshev4_opt-gs" in file:
        return "Chebyshev 4th (Opt) - Gauss-Seidel"
    elif "chebyshev1" in file:
        return "Chebyshev 1st"
    elif "chebyshev4" in file:
        return "Chebyshev 4th"
    elif "chebyshev4_opt" in file:
        return "Chebyshev 4th (Opt)"
    else:
        raise ValueError(f"Missing clean title, currently {file=}")


# In[3]:


path = "/local/home/mb280636/data/Dyablo/smoothers/pysco_tests"


# ## Plot smoothers used as Solvers

# In[4]:

print("Solvers")

""" files = Path(path).rglob("*/*solver*.dat")
for f in files:
    meta, data = load_dat_with_json_header(f)
    filename = str(f)
    key = next(iter(meta))
    output = f.with_suffix('.png')
    values_legend = meta[key]
    size = len(values_legend)
    colors = pl.cm.coolwarm(np.linspace(0, 1, size))
    colors[size//2] = np.array([0, 0, 0, 1])
    fig, ax = plt.subplots()
    for i in range(size):
        ax.semilogy(data[:,i], "o", color=colors[i], linestyle="-", label=fr"$w = {values_legend[i]:.3f}$")
    resmin = np.min(data)
    ax.set_ylabel("Residual")
    ax.set_xlabel("Iteration")
    ax.set_ylim(0.9*resmin, 1.1*data[0,0])
    ax.set_title(clean_title(filename))
    ax.legend(ncol=3, bbox_to_anchor=(1.0, 0.),loc="lower left",)
    fig.savefig(output, bbox_inches="tight")#,dpi=300
    #plt.show()
    plt.close(fig)
    del fig,ax """


# ## Plot smoothers used in Multigrid V cycles

# In[5]:


meta, data = load_dat_with_json_header(f"{path}/gauss_seidel/residual_source_hernquist_Vcycle_Npre2_Npost1.dat")
w = np.array(meta["w"])
idx = np.where(np.isclose(w, 1.25))[0][0]
print(f"{idx=} {meta['w'][idx]=}")
dat_ref = data[:,idx]


# In[ ]:

dirs = [
#"chebyshev1", 
#"chebyshev4-gs",
#"chebyshev4_opt",
#"chebyshev4_opt-jacobi",
#"gauss_seidel",
#"jacobi-gs",
#"jacobi_weight",
"chebyshev1-gs",
"chebyshev1-jacobi",
#"chebyshev4",
#"chebyshev4-jacobi",
#"chebyshev4_opt-gs",
#"gs-gs",
#"jacobi-jacobi",
]

for dir in dirs:
    print (dir)
    files = Path(path).rglob(f"{dir}/*Vcycle*.dat")
    for f in tqdm(files):
        meta, data = load_dat_with_json_header(f)
        filename = str(f)
        niter = len(data)
        listing = list(meta)    
        Npre = meta["Npre"]
        Npost = meta["Npost"]
        output = f.with_suffix('.png')
        values_legend = meta[listing[0]]
        size = len(values_legend)
        colors = pl.cm.coolwarm(np.linspace(0, 1, size))
        colors[size//2] = np.array([0, 0, 0, 1])
        fig, ax = plt.subplots()
        Npre_str = r"N_{\rm pre}"
        Npost_str = r"N_{\rm post}"
        xlabel = fr"V-cycle iteration $[{Npre_str}={Npre}$, ${Npost_str}={Npost}]$"
        for i in range(3, len(listing)):
            xlabel += fr", ${clean_key(listing[i])}={meta[listing[i]]:.3f}$"
        for i in range(size):
            label=fr"${clean_key(listing[0])} = {values_legend[i]:.3f}$"
            ax.semilogy(data[:,i], "o", color=colors[i], linestyle="-", label=label)
        ax.semilogy(dat_ref, "v", color="k", linestyle=":", label="GS 2-1")
        resmin = np.min(data[np.isfinite(data)])
        ax.set_ylabel("Residual")
        ax.set_xlabel(xlabel)
        ax.set_ylim(0.9*resmin, 1.1*data[0,0])
        ax.set_title(clean_title(filename))
        ax.semilogy(range(niter), data[0,0]*0.1**np.arange(niter), 'k--', label=r"$\mathrm{Decay}: 0.1$")
        ax.legend(ncol=3, bbox_to_anchor=(1.0, 0.),loc="lower left",)
        fig.savefig(output, bbox_inches="tight")#, dpi=300)
        #plt.show()
        plt.close(fig)
        del fig,ax


# In[ ]:




