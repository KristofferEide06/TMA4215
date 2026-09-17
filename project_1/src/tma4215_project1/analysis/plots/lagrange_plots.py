import numpy as np
import math

import numpy.typing as npt
from collections.abc import Callable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from pathlib import Path

from ...interpolation.nodes import(
    generate_equidistant_nodes,
    generate_chebyshev_nodes,
)

from ...interpolation.methods import (
    lagrange,
)

from ..computations.lagrange_computations import(
    lagrange_error_norms,
    lagrange_error_norms_multiple_n,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

def plot_chebyshev_equidistant_lagrange(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], 
    interval: tuple[float, float], 
    n: int, 
    points: int = 1000,
    savefig: bool = False,
    ax: Axes|None = None,
    ) -> tuple[Figure, Axes]:
    """Plots lagrange interpolation for equidistant and chebyshev nodes on function
    
    Args:
        fun (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]): Function to interpolate on
        interval (tuple[float, float]): Function interval
        n (int): n + 1 interpolation nodes
        points (int, optional): Granularity for plot, number of points to calculate on. Defaults to 1000.
        savefig (bool, optional): If figure is to be saved - saves to output folder. Defaults to False.
        ax (Axes | None, optional): Either appends ax to existing figure or creates new one if ax is none. Defaults to None.
    
    Returns: tuple[Figure, Axes]: Figure, axes
    """
        
    standalone = ax is None
            
    if ax is None:
        fig, ax = plt.subplots(figsize = (8,4))
    else:
        fig = ax.get_figure(root = True) #Include in AI declaration
        assert fig is not None
            
    grid = np.linspace(interval[0], interval[1], points)
    
    equidistant_nodes = generate_equidistant_nodes(interval = interval, n = n)
    
    chebyshev_nodes = generate_chebyshev_nodes(interval = interval, n = n)
            
    lagrange_equidistant = lagrange(
        x_nodes = equidistant_nodes, 
        y_nodes = fun(equidistant_nodes), 
        x = grid,
    )
    lagrange_chebyshev = lagrange(
        x_nodes = chebyshev_nodes, 
        y_nodes = fun(chebyshev_nodes), 
        x = grid,
    )
    
    (
        equidistant_max_err, 
        _, 
        chebyshev_max_err, 
        _,
        _,
        _,
    ) = lagrange_error_norms(
        fun = fun, 
        interval = interval,
        n = n,
        N = points - 1,
        bound = False
    )
    
    ax.plot(grid, lagrange_equidistant, label = "Equidistant nodes", linestyle = '--', color = 'blue')
    ax.plot(grid, lagrange_chebyshev, label = "Chebyshev nodes", linestyle = '--', color = 'orange')
    ax.plot(grid, fun(grid), label = "True function", color = 'green', linewidth = 2.5)
        
    ax.scatter(equidistant_nodes, fun(equidistant_nodes), color = 'blue', zorder = 3, s = 14)
    ax.scatter(chebyshev_nodes, fun(chebyshev_nodes), color = 'orange', zorder = 3, s = 14)
    
    error_text = (
        r'$\|f-p_n\|_\infty$'
        f'\nEquidistant: {equidistant_max_err:.2g}\n'
        f'Chebyshev: {chebyshev_max_err:.2g}\n'
    )
        
    #AI
    ax.text(
        0.02, 0.98,
        error_text,
        transform = ax.transAxes,
        verticalalignment = 'top',
        fontsize = 8,
        bbox = dict(boxstyle = "round", alpha = 0.8),
    )
    #AI
        
    if standalone:
        ax.set_title(
            f"Lagrange interpolation on [{interval[0]}, {interval[1]}] with {n + 1} nodes"
        )
        ax.legend() 
    else:
        ax.set_title(f"n = {n}")
        
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
        
    if savefig: 
        save_dir = OUTPUT_DIR / 'lagrange_interpolation'
        save_dir.mkdir(parents = True, exist_ok = True)
            
        filename = f'{interval[0]}_{interval[1]}_{n}.png'
        fig.savefig(save_dir / filename, bbox_inches = 'tight')
        
    return fig, ax

def plot_multiple_n_lagrange(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], 
    interval: tuple[float, float],
    n_arr: npt.NDArray[np.int64],
    points: int = 1000,
    savefig: bool = False,
) -> tuple[Figure, Axes]:
    """Plots lagrange interpolation for equidistant and chebyshev for different values of n"""
    m = len(n_arr)
    
    ncols = math.ceil(math.sqrt(m))
    nrows = math.ceil(m / ncols)
    
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize = (5 * ncols, 4 * nrows),
    )
    
    axs_flat = np.atleast_1d(axs).ravel()
    
    for ax, n in zip(axs_flat, n_arr):
        fig, ax = (
            plot_chebyshev_equidistant_lagrange(
                fun = fun,
                interval = interval,
                n = int(n),
                points = points, 
                ax = ax,
            )
        )

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
        
    return fig, axs

def compare_l2_max_norm(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    interval: tuple[float, float],
    n_arr: npt.NDArray[np.int64],
    bound: bool,
    savefig: bool = False,
    ) -> tuple[Figure, Axes]:
    """Compares L2 and  max norm as functions of n for fun. Also plots bound if appropriate function

    Args:
        fun (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]): Function to interpolate on
        interval (tuple[float, float]): Interval to interpolate on
        n_arr (npt.NDArray[np.int64]): array of ns to plot on
        bound (bool): If max error norm bound is to be included in plot

    Returns: tuple[Figure, Axes]: Figure, axes
    """
    N = 100 * max(n_arr)
    
    (
        equidistant_max_err_arr,
        equidistant_l2_err_arr,
        chebyshev_max_err_arr,
        chebyshev_l2_err_arr,
        equidistant_bound_arr,
        chebyshev_bound_arr,
    ) = lagrange_error_norms_multiple_n(
        fun = fun,
        interval = interval,
        n_arr = n_arr,
        N = N,
        bound = bound,
    )
    
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 4))
    
    axs[0].plot(n_arr, equidistant_max_err_arr, label = 'equidistant', color = 'blue')
    axs[0].plot(n_arr, chebyshev_max_err_arr, label = 'chebyshev', color = 'orange')
    
    if bound:
        axs[0].plot(n_arr, equidistant_bound_arr, label = 'equidistant bound', color = 'blue', linestyle = '--')
        axs[0].plot(n_arr, chebyshev_bound_arr, label = 'chebyshev bound', color = 'orange', linestyle = '--')
    
    axs[0].set_yscale('log')
    axs[0].grid()
    axs[0].set_ylabel(r'$\|f-p_n\|_\infty$')
    axs[0].set_xlabel('n')
    
    axs[1].plot(n_arr, equidistant_l2_err_arr, label = 'equidistant', color = 'blue')
    axs[1].plot(n_arr, chebyshev_l2_err_arr, label = 'chebyshev', color = 'orange')
    
    axs[1].set_yscale('log')
    axs[1].grid()
    axs[1].set_ylabel(r'$\|f-p_n\|_2$')
    axs[1].set_xlabel('n')
    
    fig.suptitle(r'$\|f-p_n\|_2$ and $\|f-p_n\|_\infty$ as functions of n')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'upper center', bbox_to_anchor = (0.5, 0.94), ncol = 3)
    
    fig.tight_layout(rect = (0, 0, 1, 0.90))
    
    if savefig:
        save_dir = OUTPUT_DIR / 'lagrange_interpolation'
        save_dir.mkdir(parents = True, exist_ok = True)
        
        bound_name = 'with_bound' if bound else 'without_bound'
        filename = f'lagrange_error_norms_{interval[0]}_{interval[1]}_{bound_name}.png'
        fig.savefig(save_dir / filename, bbox_inches = 'tight')
    
    return fig, axs