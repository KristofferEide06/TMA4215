import numpy as np

import numpy.typing as npt
from collections.abc import Callable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from pathlib import Path

from ..computations.rbf_computations import (
    cond_M,
    rbf_error_epsilon,
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
        save_dir = OUTPUT_DIR / 'rbf_inteprolation'
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
    figsave: bool = False,
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
    
    if figsave:
        save_dir = OUTPUT_DIR / 'rbf_interpolation'
        save_dir.mkdir(parents = True, exist_ok = True)
        
        filename = f'rbf_max_error_{interval[0]}_{interval[1]}_{len(x_nodes)}_nodes.png'

        fig.savefig(save_dir / filename, boox_inches = 'tight')
        
    return fig, ax