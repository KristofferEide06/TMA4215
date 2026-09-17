import numpy as np

import numpy.typing as npt
from collections.abc import Callable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from pathlib import Path

from ...interpolation.nodes import(
    generate_equidistant_nodes,
)

from ...interpolation.methods import(
    piecewise_interpolation,
)

from ...interpolation.norms import (
    l2_norm_err,
    max_norm_err,
)

from ..computations.lagrange_computations import(
    lagrange_error_norms_multiple_n,
)

from ..computations.piecewise_computations import(
    piecewise_lagrange_k,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

def plot_piecewise_interpolation(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x: npt.NDArray[np.float64],
    k: int,
    n: int,
    interval: tuple[float, float],
    savefig: bool = False,
) -> tuple[Figure, Axes]:
    """Plot piecewise interpolant"""
    global_degree = n * k
        
    piecewise_values = piecewise_interpolation(
        fun = fun,
        x = x,
        n = n,
        k = k,
        interval = interval,
    )
        
    fig, ax = plt.subplots(figsize = (8, 4))
    
    fun_values = fun(x)
    
    max_err = max_norm_err(
        fun_val = fun_values,
        approximation_val = piecewise_values,
    )
    l2_err = l2_norm_err(
        fun_val = fun_values,
        approximation_val = piecewise_values,
        interval = interval,
        N = len(x) - 1,
    )
    
    ax.plot(x, fun_values, color = 'green', label = 'Exact function')
    ax.plot(x, piecewise_values, linestyle = '--', label = f'Piecewise, n = {n} and K = {k}')
        
    piecewise_nodes = generate_equidistant_nodes(interval = interval, n = global_degree)
    #ax.scatter(piecewise_nodes, fun(piecewise_nodes), zorder = 3, label = 'Piecewise nodes')
    
    error_text = (
        r'$\|f-p_{n,K}\|_\infty$'
        f' = {max_err:.2e}\n'
        r'$\|f-p_{n,K}\|_2$'
        f' = {l2_err:.2e}'
    )
    
    ax.text(
        0.02,
        0.98,
        error_text,
        transform = ax.transAxes,
        verticalalignment = 'top',
        fontsize = 9,
        bbox = dict(boxstyle = 'round', alpha = 0.8),
    )
    
    ax.set_xlabel('x')
    ax.set_ylabel('Value')
    ax.legend(ncol = 2, loc = 'upper right')
    
    if savefig:
        save_dir = OUTPUT_DIR / 'piecewise_interpolation'
        save_dir.mkdir(parents = True, exist_ok = True)
        
        filename = f'piecewise_interpolation_{interval[0]}_{interval[1]}_n{n}_K{k}.png'
        fig.savefig(save_dir / filename, bbox_inches = 'tight')
    
    return fig, ax

def plot_piecewise_max_error_k(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x: npt.NDArray[np.float64], 
    k_arr: npt.NDArray[np.int64],
    n: int,
    interval: tuple[float, float],
    savefig: bool = False,
) -> tuple[Figure, Axes]:
    """Plot piecewise lagrangian maximum norm error as function of K"""    
    
    max_err_arr, _ = piecewise_lagrange_k(
        fun = fun,
        x = x,
        n = n,
        interval = interval,
        k_arr = k_arr
    )
        
    fig, ax = plt.subplots(figsize = (8, 4))
    
    ax.plot(k_arr, max_err_arr)
    
    ax.set_yscale('log')
    
    ax.set_xlabel('K')
    ax.set_ylabel(r'$\|f-p_n\|_\infty$')
    ax.set_title('Max error of piecewise interpolation as function of K')
    ax.grid()
    
    if savefig:
        save_dir = OUTPUT_DIR / 'piecewise_interpolation'
        save_dir.mkdir(parents = True, exist_ok = True)
        
        filename = f'piecewise_max_error_{interval[0]}_{interval[1]}_degree_{n}.png'
        fig.savefig(save_dir / filename, bbox_inches = 'tight')


    return fig, ax 

def compare_piecewise_global_max_error(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x: npt.NDArray[np.float64],
    k_arr: npt.NDArray[np.int64],
    n: int,
    interval: tuple[float, float],
    savefig: bool = False, 
) -> tuple[Figure, Axes]:
    """Compares max norm error as function of distinct nodes for piecewise interpolation and lagrange"""   
    num_nodes_arr = n*k_arr + 1
    
    piecewise_max_err_arr, _ = piecewise_lagrange_k(
        fun = fun,
        x = x,
        n = n,
        interval = interval,
        k_arr = k_arr,
    )
    
    global_n_arr = n * k_arr
    
    (
        equidistant_max_err_arr,
        _,
        chebyshev_max_err_arr,
        _,
        _,
        _,
    ) = lagrange_error_norms_multiple_n(
        fun = fun,
        interval = interval,
        n_arr = global_n_arr,
        N = len(x) - 1,
        bound = False,
    )

    fig, ax = plt.subplots(figsize = (8, 4))
    
    ax.plot(num_nodes_arr, piecewise_max_err_arr, label = 'Piecewise')
    ax.plot(num_nodes_arr, equidistant_max_err_arr, label = 'Global equidistant')
    ax.plot(num_nodes_arr, chebyshev_max_err_arr, label = 'Global Chebyshev')
    
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Number of discretization nodes')
    ax.set_ylabel(r'$\|f-p_n\|_\infty$')
    ax.set_title('Interpolation error as function of nodes')
   
    ax.grid()
    ax.legend()
    
    if savefig:
        save_dir = OUTPUT_DIR / 'piecewise_interpolation'
        save_dir.mkdir(parents = True, exist_ok = True)
        
        filename = (
            f'piecewise_global_comparison_{interval[0]}_{interval[1]}_degree_{n}.png'
        )
        fig.savefig(save_dir / filename, bbox_inches = 'tight')

    return fig, ax