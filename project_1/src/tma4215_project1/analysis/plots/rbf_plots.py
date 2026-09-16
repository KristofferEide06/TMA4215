import numpy as np

import numpy.typing as npt
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from pathlib import Path

from ..computations.rbf_computations import (
    cond_M,
    rbf_error_epsilon,
    rbf_optimization,
    rbf_optimization_multiple_n,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

def plot_cond_M(
    x_nodes: npt.NDArray[np.float64],
    epsilon_arr: npt.NDArray[np.float64],
    savefig: bool = False, 
) -> tuple[Figure, Axes]:
    """Plots condition number of RBF matrix as function of epsilon"""
    cond_max_arr, _ = cond_M(
        x_nodes = x_nodes,
        epsilon_arr = epsilon_arr,
    )
    
    fig, ax = plt.subplots(figsize = (8, 4))
    
    ax.set_yscale('log')
    
    ax.plot(epsilon_arr, cond_max_arr)
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$\kappa (M)$')
    ax.grid()
    
    if savefig:
        save_dir = OUTPUT_DIR / 'rbf_interpolation'
        save_dir.mkdir(parents = True, exist_ok = True)
    
        filename = f'rbf_condition_number_{len(x_nodes)}_nodes.png'
        fig.savefig(save_dir / filename, bbox_inches = 'tight')
        
    return fig, ax
    
def plot_rbf_error_epsilon(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x_nodes: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    interval: tuple[float, float],
    epsilon_arr: npt.NDArray[np.float64],
    savefig: bool = False,
) -> tuple[Figure, Axes]:
    """Plots rbf error as function of epsilon"""
    rbf_max_err_arr, _ = rbf_error_epsilon(
        fun = fun,
        x_nodes = x_nodes,
        x = x,
        interval = interval,
        epsilon_arr = epsilon_arr,
    )
    
    fig, ax = plt.subplots(figsize = (8, 4))
    
    ax.plot(epsilon_arr, rbf_max_err_arr)
    ax.grid()
    
    ax.set_title('RBF error as function of epsilon')
    
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$L^\infty$')
    
    if savefig:
        save_dir = OUTPUT_DIR / 'rbf_interpolation'
        save_dir.mkdir(parents = True, exist_ok = True)
        
        filename = f'rbf_max_error_{interval[0]}_{interval[1]}_{len(x_nodes)}_nodes.png'

        fig.savefig(save_dir / filename, bbox_inches = 'tight')
        
    return fig, ax

def plot_cost_history(
    fun: Callable[[Any], Any],
    n: int,
    interval: tuple[float, float],
    init_epsilon: float,
    N: int,
    max_iter: int,
    max_backtracks: int,
    tol: float,
    L: float,
    rho: float,
    rho_bar: float,
    epsilon_min: float,
    savefig: bool = False,
) -> tuple[Figure, Axes]:
    """Plots the gd cost as function of accepted iterations"""
    result = rbf_optimization(
        fun = fun,
        n = n,
        interval = interval,
        init_epsilon = init_epsilon,
        N = N,
        max_iter = max_iter,
        max_backtracks = max_backtracks,
        tol = tol,
        L = L,
        rho = rho,
        rho_bar = rho_bar,
        epsilon_min = epsilon_min,
    )
    
    fig, ax = plt.subplots(figsize = (8, 4))
    
    cost_history = result['cost_history']
    iterations = np.arange(len(cost_history))
    
    ax.plot(iterations, cost_history)
    
    ax.set_xlabel('Accepted iteration')
    ax.set_ylabel(r'$C(\mathbf{x}, \epsilon)$')
    ax.set_title('Gradient-descent convergence')
    ax.set_yscale('log')
    
    ax.grid()
    
    if savefig:
        save_dir = OUTPUT_DIR / 'rbf_optimization'
        save_dir.mkdir(parents = True, exist_ok = True)
        
        filename = f'rbf_cost_history_{interval[0]}_{interval[1]}_n{n}_L_{L}.png'
        fig.savefig(save_dir / filename, bbox_inches = 'tight')
        
    return fig, ax

def plot_rbf_compare_n(
    fun: Callable[[Any], Any],
    n_arr: npt.NDArray[np.int64],
    interval: tuple[float, float],
    init_epsilon: float,
    N: int,
    max_iter: int,
    max_backtracks: int,
    tol: float,
    L: float,
    rho: float,
    rho_bar: float,
    epsilon_min: float,
    savefig: bool = False,
) -> tuple[Figure, Axes]:
    """Plots  RBF l2 errors for different node distributions over n"""
    result = rbf_optimization_multiple_n(
        fun = fun,
        n_arr = n_arr,
        interval = interval,
        init_epsilon = init_epsilon,
        N = N,
        max_iter = max_iter,
        max_backtracks = max_backtracks,
        tol = tol,
        L = L,
        rho = rho,
        rho_bar = rho_bar,
        epsilon_min = epsilon_min,
    )

    optimized_l2_err_arr = result['optimized_l2_error']
    equidistant_l2_err_arr = result['equidistant_l2_error']
    chebishev_l2_err_arr = result['chebishev_l2_error']
    
    fig, ax = plt.subplots(figsize = (8, 4))
    
    ax.plot(n_arr, optimized_l2_err_arr, label = 'Optimized')
    ax.plot(n_arr, equidistant_l2_err_arr, label = 'Equidistant')
    ax.plot(n_arr, chebishev_l2_err_arr, label = 'Chebishev')
    
    ax.set_xlabel('n')
    ax.set_ylabel('$L^2$')
    ax.set_title('$L^2$ error over n for different rbf node settings')
    
    ax.set_yscale('log')
    ax.grid()
    ax.legend()
    
    if savefig:
        save_dir = OUTPUT_DIR / 'rbf_optimization'
        save_dir.mkdir(parents = True, exist_ok = True)
            
        filename = f'rbf_l2_comparison_{interval[0]}_{interval[1]}_n_{n_arr[0]}_{n_arr[-1]}.png'
        fig.savefig(save_dir / filename, bbox_inches = 'tight')
    
    return fig, ax