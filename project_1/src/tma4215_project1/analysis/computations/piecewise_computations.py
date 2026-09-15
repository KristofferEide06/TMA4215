import numpy as np

import numpy.typing as npt
from collections.abc import Callable

from ...interpolation.methods import(
    piecewise_interpolation,
)

from ...interpolation.norms import(
    l2_norm_err,
    max_norm_err,
)

def piecewise_lagrange_k(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x: npt.NDArray[np.float64], 
    n: int,
    interval: tuple[float, float],
     k_arr: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Errors for piecewise_lagrange across different values of k

    Args:
        fun (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]): Function to interpolate
        x (npt.NDArray[np.float64]): Points to evaluate interpolant on 
        n (int): Number of interpolation nodes - n + 1
        interval (tuple[float, float]): Interval to evaluate interpolant on
        k_arr (npt.NDArray[np.int64]): Array of k values - k is number of disjunt intervals

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: max error array, l2 error array for piecewise lagrangian interpolation
    """
    max_err_arr = np.zeros_like(k_arr, dtype = float)
    l2_err_arr = np.zeros_like(k_arr, dtype = float)
    
    fun_vals = fun(x)
    
    for i, k in enumerate(k_arr):
        piecewise_interpolant = piecewise_interpolation(
            fun = fun, 
            x = x, 
            n = n, 
            k = k,
            interval = interval
        )
        
        max_err_arr[i] = max_norm_err(fun_val = fun_vals, approximation_val = piecewise_interpolant)
        
        l2_err_arr[i] = l2_norm_err(
            fun_val = fun_vals, 
            approximation_val = piecewise_interpolant, 
            interval = interval,
            N = len(x) - 1,
            )
    
    return max_err_arr, l2_err_arr