import numpy as np
import math

import numpy.typing as npt
from collections.abc import Callable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from pathlib import Path

from .interpolation.nodes import(
    generate_equidistant_nodes,
    generate_chebishev_nodes,
)

from .interpolation.methods import(
    lagrange,
    piecewise_interpolation,
    rbf,
    rbf_matrix,
    interpolation_bound,
)

from .interpolation.norms import(
    l2_norm,
    max_norm,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

def plot_chebishev_equidistant_lagrange(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], 
    interval: tuple[float, float], 
    n: int, 
    points: int = 1000,
    savefig: bool = False,
    ax: Axes|None = None,
    ) -> tuple[float, float, Figure, Axes]:
    """Plots lagrange interpolation for equidistant and chebishev nodes on function

    Args:
        fun (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]): Function to interpolate on
        interval (tuple[float, float]): Function interval
        n (int): n + 1 interpolation nodes
        points (int, optional): Granularity for plot, number of points to calculate on. Defaults to 1000.
        savefig (bool, optional): If figure is to be saved - saves to output folder. Defaults to False.
        ax (Axes | None, optional): Either appends ax to existing figure or creates new one if ax is none. Defaults to None.

    Returns:
        tuple[float, float, Figure, Axes]: Figure, axes, equidistant error, lagrange error
    """
    standalone = ax is None
    
    if ax is None:
        fig, ax = plt.subplots(figsize = (8,4))
    else:
        fig = ax.get_figure(root = True) #Include in AI declaration
        assert fig is not None
    
    x_arr = np.linspace(interval[0], interval[1], points)
    equidistant_nodes = generate_equidistant_nodes(interval = interval, n = n)
    chebishev_nodes = generate_chebishev_nodes(interval = interval, n = n)
    
    lagrange_equidistant = lagrange(equidistant_nodes, fun(equidistant_nodes), x_arr)
    lagrange_chebishev = lagrange(chebishev_nodes, fun(chebishev_nodes), x_arr)
    
    equidistant_max_error = max_norm(fun(x_arr), lagrange_equidistant)
    chebishev_max_error = max_norm(fun(x_arr), lagrange_chebishev)
    
    ax.plot(x_arr, lagrange_equidistant, label = "Equidistant nodes", linestyle = '--')
    ax.plot(x_arr, lagrange_chebishev, label = "Chebishev nodes", linestyle = '--')
    ax.plot(x_arr, fun(x_arr), label = "True function")
    
    error_text = (
        r'$L^\infty$ error'
        f'\nEquidistant: {equidistant_max_error:.2g}\n'
        f'Chebishev: {chebishev_max_error:.2g}\n'
    )
    
    #AI
    ax.text(
        0.02, 0.98,
        error_text,
        transform = ax.transAxes,
        verticalalignment = 'top',
        fontsize = 8,
        bbox = dict(boxstyle = "round", alpha = 0.8)
    )
    #AI
    
    if standalone:
        ax.set_title(f"Lagrange interpolation on {interval} with {n + 1} nodes")
        ax.legend() 
    else:
        ax.set_title(f"n = {n}")
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    ax.grid()
    
    if savefig: 
        save_dir = OUTPUT_DIR / 'lagrange_interpolation'
        OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
        
        filename = f'{interval[0]}_{interval[1]}_{n}.png'
        fig.savefig(save_dir / filename, bbox_inches = 'tight')
    
    return equidistant_max_error, chebishev_max_error, fig, ax

def plot_multiple_n_lagrange(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], 
    interval: tuple[float, float],
    n_arr: list[int],
    points: int = 1000,
    savefig: bool = False,
) -> tuple[Figure, Axes, list[float], list[float]]:
    """Plots multiple lagrange interpolation for equidistant and chebishev for multiple n-s

    Args:
        fun (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]): function to interpolate
        interval (tuple[float, float]): interval to interpolate on
        n_arr (list[int]): Array of n-s to plot for (nodes = n + 1)
        points (int, optional): Granularity of plot, number of points we calculate the interpolation value on. Defaults to 1000.
        savefig (bool, optional): If figure is to be saved. Defaults to False.

    Returns:
        tuple[Figure, Axes, list[float], list[float]]: Figure, axes, array of max norm error for equidistant, array of max norm error for chebishev
    """
    m = len(n_arr)
    
    ncols = math.ceil(math.sqrt(m))
    nrows = math.ceil(m / ncols)
    
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize = (5 * ncols, 4 * nrows),
    )
    
    equidistant_error_arr = []
    chebishev_error_arr = []
    
    axs_flat = np.atleast_1d(axs).ravel()
    
    for ax, n in zip(axs_flat, n_arr):
        
        equidistant_error, chebishev_error, _, _ = (
            plot_chebishev_equidistant_lagrange(
                fun = fun,
                interval = interval,
                n = n,
                points = points, 
                ax = ax,
            )
        )
        
        equidistant_error_arr.append(equidistant_error)
        chebishev_error_arr.append(chebishev_error)


    fig.suptitle(f"Lagrange interpolation of Runge function on [{interval[0]}, {interval[1]}]", fontsize = 16, y = 0.98)
    handles, labels = axs_flat[0].get_legend_handles_labels()
    
    fig.legend(handles, labels, loc = 'upper center', bbox_to_anchor = (0.5, 0.94), ncol = 3)
    fig.tight_layout(rect = (0, 0, 1, 0.95))
    
    for ax in axs_flat[len(n_arr):]:
        ax.set_visible(False)
    
    if savefig:
        save_dir = OUTPUT_DIR / 'lagrange_interpolation'
        save_dir.mkdir(parents = True, exist_ok = True)
        
        filename = f'{interval[0]}_{interval[1]}_multiple_n.png'
        fig.savefig(save_dir / filename, bbox_inches = 'tight')
        
    return fig, axs, equidistant_error_arr, chebishev_error_arr

def compare_l2_max_norm(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    interval: tuple[float, float],
    n_arr: npt.NDArray[np.int64],
    bound: bool,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], Figure, Axes]:
    """Compares L2 and  max norm as functions of n for fun. Also plots bound if appropriate function

    Args:
        fun (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]): Function to interpolate on
        interval (tuple[float, float]): Interval to interpolate on
        n_arr (npt.NDArray[np.int64]): array of ns to plot on
        bound (bool): If max error norm bound is to be included in plot

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], Figure, Axes]: equidistant_max_norm, chebishev_max_norm, 
        equidistiant_l2_norm, chebishev_l2_norm, figure, axes
    """
    equidistant_l2_norm_arr = np.zeros_like(n_arr, dtype = np.float64)
    chebishev_l2_norm_arr = np.zeros_like(n_arr, dtype = np.float64)
    
    equidistant_max_norm_arr = np.zeros_like(n_arr, dtype = np.float64)
    chebishev_max_norm_arr = np.zeros_like(n_arr, dtype = np.float64)
    
    bound_equidistant_arr = np.zeros_like(n_arr, dtype = np.float64)
    bound_chebishev_arr = np.zeros_like(n_arr, dtype = np.float64)
    
    N = 100 * max(n_arr)
    grid = np.linspace(interval[0], interval[1], N + 1)
    
    for i, n in enumerate(n_arr):
        equidistant_nodes = generate_equidistant_nodes(interval = interval, n = n)
        chebishev_nodes = generate_chebishev_nodes(interval = interval, n = n)
            
        lagrange_equidistant = lagrange(x_nodes = equidistant_nodes, y_nodes = fun(equidistant_nodes), x = grid)
        lagrange_chebishev = lagrange(x_nodes = chebishev_nodes, y_nodes = fun(chebishev_nodes), x = grid)
            
        equidistant_l2_norm_arr[i] = l2_norm(fun(grid), lagrange_equidistant, interval, N)
        chebishev_l2_norm_arr[i] = l2_norm(fun(grid), lagrange_chebishev, interval, N)
        
        equidistant_max_norm_arr[i] = max_norm(fun(grid), lagrange_equidistant)
        chebishev_max_norm_arr[i] = max_norm(fun(grid), lagrange_chebishev)
        
        if bound: 
            bound_equidistant_arr[i] = interpolation_bound(equidistant_nodes, grid)
            bound_chebishev_arr[i] = interpolation_bound(chebishev_nodes,grid)
    
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 4))
    
    axs[0].plot(n_arr, equidistant_max_norm_arr, label = 'equidistant', color = 'blue')
    axs[0].plot(n_arr, chebishev_max_norm_arr, label = 'chebishev', color = 'orange')
    
    if bound:
        axs[0].plot(n_arr, bound_equidistant_arr, label = 'equidistant bound', color = 'blue', linestyle = '--')
        axs[0].plot(n_arr, bound_chebishev_arr, label = 'chebishev bound', color = 'orange', linestyle = '--')
    
    axs[0].set_yscale('log')
    axs[0].grid()
    axs[0].set_ylabel('max')
    axs[0].set_xlabel('n')
    
    axs[1].plot(n_arr, equidistant_l2_norm_arr, label = 'equidistant', color = 'blue')
    axs[1].plot(n_arr, chebishev_l2_norm_arr, label = 'chebishev', color = 'orange')
    
    axs[1].set_yscale('log')
    axs[1].grid()
    axs[1].set_ylabel('$L^2$')
    axs[1].set_xlabel('n')
    
    fig.suptitle('$L^2$ and max norm as function of n')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'upper center', bbox_to_anchor = (0.5, 0.94), ncol = 3)
    
    fig.tight_layout(rect = (0, 0, 1, 0.90))
    
    return equidistant_max_norm_arr, chebishev_max_norm_arr, equidistant_l2_norm_arr, chebishev_l2_norm_arr, fig, axs

def piecewise_lagrangian_k(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x: npt.NDArray[np.float64], 
    k_arr: npt.NDArray[np.int64],
    n: int,
    interval: tuple[float, float],
) -> tuple[npt.NDArray[np.float64], Figure, Axes]:
    """Plot piecewise lagrangian maximum norm error as function of K"""
    max_error_arr = np.zeros_like(k_arr, dtype = float)
    
    fun_vals = fun(x)
    
    for i, k in enumerate(k_arr):
        piecewise_interpolant = piecewise_interpolation(fun, x, n, k, interval)
        
        max_error_arr[i] = np.max(np.abs(fun_vals - piecewise_interpolant))
        
    fig, ax = plt.subplots(figsize = (8, 4))
    
    ax.plot(k_arr, max_error_arr)
    ax.set_xlabel('K')
    ax.set_ylabel(r'$L^\infty$')
    ax.set_title('Max error of piecewise interpolation as function of K')
    ax.grid()
    
    return max_error_arr, fig, ax 

def compare_piecewise_global(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x: npt.NDArray[np.float64],
    k_arr: npt.NDArray[np.int64],
    n: int,
    interval: tuple[float, float],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], Figure, Axes]:
    """Compares max norm error as function of distinct nodes for piecewise interpolation and lagrange"""
    piecewise_err_arr = np.zeros_like(k_arr, dtype = float)
    equidistant_err_arr = np.zeros_like(k_arr, dtype = float)
    chebishev_err_arr = np.zeros_like(k_arr, dtype = float)
    
    num_nodes_arr = n*k_arr + 1
    
    fun_vals = fun(x)
    
    for i, k in enumerate(k_arr):
        piecewise_interpolant = piecewise_interpolation(fun, x, n, k, interval)
        
        piecewise_err_arr[i] = max_norm(fun_vals, piecewise_interpolant)
        
        global_degree = num_nodes_arr[i] - 1
        
        equidistant_nodes = generate_equidistant_nodes(interval, global_degree)
        chebishev_nodes = generate_chebishev_nodes(interval, global_degree)
        
        lagrange_equidistant = lagrange(equidistant_nodes, fun(equidistant_nodes), x)
        lagrange_chebishev = lagrange(chebishev_nodes, fun(chebishev_nodes), x)
        
        equidistant_err_arr[i] = max_norm(fun_vals, lagrange_equidistant)
        chebishev_err_arr[i] = max_norm(fun_vals, lagrange_chebishev)
        
    
    fig, ax = plt.subplots(figsize = (8, 4))
    
    ax.plot(num_nodes_arr, piecewise_err_arr, label = 'piecewise')
    ax.plot(num_nodes_arr, equidistant_err_arr, label = 'equidistant')
    ax.plot(num_nodes_arr, chebishev_err_arr, label = 'chebishev')
    
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Number of discretization nodes')
    ax.set_ylabel(r'$L^\infty$')
    ax.set_title('Interpolation error as function of nodes')
   
    ax.grid()
    ax.legend()
    
    return num_nodes_arr, piecewise_err_arr, equidistant_err_arr, chebishev_err_arr, fig, ax

def plot_cond_M(
    x_nodes: npt.NDArray[np.float64],
    epsilon_arr: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], Figure, Axes]:
    """Plots condition number of RBF matrix as function of epsilon"""
    cond_arr = np.zeros_like(epsilon_arr)
    
    for i, epsilon in enumerate(epsilon_arr):
        M = rbf_matrix(x_nodes, epsilon) #Or second norm??
        cond_arr[i] = np.linalg.cond(M, p = np.inf)
    
    fig, ax = plt.subplots(figsize = (8, 4))
    
    ax.set_yscale('log')
    
    ax.plot(epsilon_arr, cond_arr)
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$\kappa (M)$')
    ax.grid()
    
    return cond_arr, fig, ax
    
def plot_rbf_error_epsilon(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x_nodes: npt.NDArray[np.float64],
    y_nodes: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    epsilon_arr: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], Figure, Axes]:
    """Plots rbf error as function of epsilon"""
    rbf_max_err_arr = np.zeros_like(epsilon_arr, dtype = float)
    
    fun_vals = fun(x)
    
    for i, epsilon in enumerate(epsilon_arr):
        rbf_interpolant = rbf(x_nodes, y_nodes, x, epsilon)
        
        rbf_max_err_arr[i] = max_norm(fun_vals, rbf_interpolant)
    
    fig, ax = plt.subplot(figsize = (8, 4))
    
    ax.plot(epsilon_arr, rbf_max_err_arr)
    ax.grid()
    
    ax.set_title('RBF error as function of epsilon')
    
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$L^\infty$')
    
    return rbf_max_err_arr, fig, ax