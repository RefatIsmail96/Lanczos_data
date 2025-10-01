import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import pandas as pd
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from process_results import fetch_blist, eigensystem_from_blist, fetch_data
from scipy.spatial.distance import pdist, squareform

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["cmr10"]
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams['text.usetex'] = True
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 24
gold_color = "#e0be04"

system_type = 'nonintg'  # 'intg' or 'nonintg'
system_labels = {'intg': 'integrable',
    'nonintg': 'non-integrable'}
operator = 'Z1'  # 'X1', 'X2', or 'X3'
latex_operator = operator[0] + '_' + operator[1:]
blist, evals, evecs = fetch_data(5, system_type, operator)

init_op_squared = evecs[0]**2


fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(evals, init_op_squared, marker='o', linestyle="-", linewidth=0.1, color='darkred', markersize=1, markeredgewidth=0.2)
ax.set_yscale('log')
fig


c_k = 0
for idx, vec in enumerate(evecs):
    c_k += idx*np.dot(init_op_squared, vec**2)
    
x_list = np.arange(1, len(evecs)+1)/(len(evecs)+1)
omega_list = evals
evecs_squared = evecs**2

sigma = np.sqrt(sum(omega_list**2)/(len(omega_list)))
def boundary_function(omega):
    #normalized such that boundary(omega=0) = 1
    return np.exp(-omega**2/ (2 * sigma**2)) 
#matrix_to_plot = matrix_to_plot[::-1]

residual_matrix = np.zeros((len(omega_list), len(omega_list)))
nkry = len(evecs_squared)
for lanczos_step in range(nkry):
    if lanczos_step == 0:
        residual_matrix[nkry - lanczos_step-1] = evecs_squared[nkry- lanczos_step-1]
    else:
        # Add the current step to the previous residual
        residual_matrix[nkry - lanczos_step-1] = evecs_squared[nkry- lanczos_step-1] + residual_matrix[nkry - lanczos_step]

matrix_to_plot = residual_matrix
#matrix_to_plot = evecs_squared


def plot_residual_norms(ax, matrix_to_plot, omega_list, x_list):
    
    x,y = np.meshgrid( omega_list, x_list)

    #im_inst = ax.imshow(matrix_to_plot, cmap="OrRd", aspect = "auto", extent=(min(omega_list), max(omega_list), min(x_list), max(x_list)), interpolation = "nearest", norm = matplotlib.colors.LogNorm(vmin=1e-20, vmax = np.max(matrix_to_plot)))

    #im_inst = ax.pcolormesh(x,y, matrix_to_plot, cmap="OrRd", norm = matplotlib.colors.LogNorm(vmin=1e-20, vmax = np.max(matrix_to_plot)), shading='auto')
    im_inst = ax.pcolormesh(x,y, matrix_to_plot, cmap="OrRd", norm = matplotlib.colors.LogNorm(vmin=1e-15, vmax = 1e0), shading='auto', rasterized=True)
    #im_inst = ax.pcolormesh(x,y, matrix_to_plot, cmap="OrRd", shading='auto')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    ax.figure.colorbar(im_inst, cax=cax, label = "residual norm ($R_{\omega, n/N}$)" )
    ax.set_xlabel("energy gap ($\omega$)");
    ax.set_ylabel(r"Lanczos step fraction ($n/N$)");
    ax.set_ylim(0, 1.0)
    #ax.set_xlim(-1,1)
    ax.set_xlim(min(omega_list), max(omega_list))

    boundary_list = boundary_function(omega_list)
    ax.plot(omega_list, boundary_list, '-', color='black', markersize=1, markeredgewidth=0.5, label='threshold $x_c(\omega)$')
    ax.legend(loc='lower right', frameon=True, handlelength=0.5)
    return ax
fig, ax = plt.subplots(figsize=(6, 5))
ax = plot_residual_norms(ax, matrix_to_plot, omega_list, x_list)
fig



def density_function(omega):
    return np.exp(-omega**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))



def plot_density(ax, omega_list, evals):
    ax.hist(evals, bins=65, density=True, color='darkred', alpha=0.8)
    ax.set_xlabel("energy gap ($\omega$)");
    ax.set_ylabel("probability density")
    density_list = density_function(omega_list)
    ax.plot(omega_list, density_list, '--', color='black',  markersize=1, markeredgewidth=0.5,  label= r'analytical fit')
    ax.legend(loc='lower right', frameon=True)
    return ax
fig, ax = plt.subplots(figsize=(6, 5))
ax = plot_density(ax, omega_list, evals)
fig

def lanczos_function(x):
    return np.sqrt(- (np.log(x) * sigma**2) / 2)


def plot_bn(ax, xlist, blist):
    xlist = xlist[1:]
    ax.plot(xlist, blist, color = 'darkred', markersize=1, markeredgewidth=0.5)
    ax.set_xlabel("Lanczos step fraction ($x = n/N$)");
    ax.set_ylabel("Lanczos coefficients ($b_n$)")
    bn_fit_list = lanczos_function(xlist)
    ax.plot(xlist, bn_fit_list, '--', color='black',  markersize=1, markeredgewidth=0.5,  label= r'analytical fit')
    ax.legend(loc='upper right', frameon=True)
    return ax
fig, ax = plt.subplots(figsize=(6, 5))
ax = plot_bn(ax, x_list, blist)
fig

def plot_bn(ax, xlist, blist):
    xlist = xlist[1:]
    xlist = xlist[650:]
    blist = blist[650:]
    ax.plot(xlist[1::2], blist[1::2], 'ro', label='odd', alpha=0.5, markersize=3, markeredgewidth=0.2)
    ax.plot(xlist[0::2], blist[0::2], 'ko', label='even', alpha=0.5, markersize=3, markeredgewidth=0.2)

    #ax.plot(xlist, blist, "o", linestyle="", color = 'darkred', markersize=5, markeredgewidth=0.5)
    ax.set_xlabel("Lanczos step fraction ($x = n/N$)");
    ax.set_ylabel("Lanczos coefficients ($b_n$)")
    bn_fit_list = lanczos_function(xlist)
    ax.plot(xlist, bn_fit_list, '--', color='black',  markersize=0.5, markeredgewidth=0.2,  label= r'analytical fit')
    #ax.plot(xlist, bn_fit_list, '--', color='black',  markersize=1, markeredgewidth=0.5)
    ax.legend(loc='best', frameon=True, handlelength=0.5)
    return ax
fig, ax = plt.subplots(figsize=(6, 5))
ax = plot_bn(ax, x_list, blist)
fig