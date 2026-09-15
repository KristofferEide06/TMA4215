import numpy as np

import numpy.typing as npt
from collections.abc import Callable

from ...interpolation.methods import(
    rbf,
    rbf_matrix, 
)

from ...interpolation.norms import(
    l2_norm_err,
    max_norm_err,
)

def cond_M(
    x_nodes: npt.NDArray[np.float64],
    epsilon_arr: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculates the condition numer for rbf matrix; max, l2"""
    
    cond_arr_max = np.zeros_like(epsilon_arr, dtype = float)
    cond_arr_l2 = np.zeros_like(epsilon_arr, dtype = float)
        
    for i, epsilon in enumerate(epsilon_arr):
        M = rbf_matrix(x_nodes = x_nodes, epsilon = epsilon)
        cond_arr_max[i] = np.linalg.cond(M, p = np.inf)
        cond_arr_l2[i] = np.linalg.cond(M, p = 2)

    return cond_arr_max, cond_arr_l2

def rbf_error_epsilon(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x_nodes: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    interval: tuple[float, float],
    epsilon_arr: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculates norm errors for rbf interpolation method

    Args:
        fun (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]): Function to interpolate
        x_nodes (npt.NDArray[np.float64]): Nodes to interpolate on
        x (npt.NDArray[np.float64]): Points to evaluate interpolate on
        interval (tuple[float, float]): Interval to evaluate function on
        epsilon_arr (npt.NDArray[np.float64]): Shape parameter array
    
    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: max error of rbf, l2 error of rbf
    """
    rbf_max_err_arr = np.zeros_like(epsilon_arr, dtype = float)
    rbf_l2_err_arr = np.zeros_like(epsilon_arr, dtype = float)
    
    fun_vals = fun(x)
        
    for i, epsilon in enumerate(epsilon_arr):
        rbf_interpolant = rbf(
            x_nodes = x_nodes, 
            y_nodes = fun(x_nodes), 
            x = x, 
            epsilon = epsilon
        )
                
        rbf_max_err_arr[i] = max_norm_err(fun_val = fun_vals, approximation_val = rbf_interpolant)
        rbf_l2_err_arr[i] = l2_norm_err(
            fun_val = fun_vals,
            approximation_val = rbf_interpolant,
            interval = interval,
            N = len(x) - 1,
        )
        
    return rbf_max_err_arr, rbf_l2_err_arr