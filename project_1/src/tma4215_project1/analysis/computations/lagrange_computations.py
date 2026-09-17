import numpy as np

import numpy.typing as npt
from collections.abc import Callable

from ...interpolation.nodes import(
    generate_equidistant_nodes,
    generate_chebyshev_nodes,
)

from ...interpolation.methods import(
    lagrange,
    interpolation_bound,
    interpolation_bound_chebyshev,
)

from ...interpolation.norms import(
    l2_norm_err,
    max_norm_err,
)

def lagrange_error_norms(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], 
    interval: tuple[float, float], 
    n: int, 
    N: int,
    bound: bool = False,
) -> tuple[float, float, float, float, float, float]: 
    """Calculates errors for Lagrange interpolation 

    Args:
        fun (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]): Function to interpolate
        interval (tuple[float, float]): Interval to evaluate interpolant on
        n (int): number of nodes to interpolate on - n + 1
        N (int): number of grid points   - N + 1
        bound (bool, optional): Theoretically estimated max error bound (only valid for f function). Defaults to False.

    Returns:
        tuple[float, float, float, float, float, float]: equidistant max error, equidistant l2 error, chebyshev max error,
        chebyshev l2 error, theoretical equidistant bound for max error, theoretical chebyshev bound for max error
    """
    grid = np.linspace(interval[0], interval[1], N + 1)
    
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
    
    fun_vals = fun(grid)
    
    equidistant_max_err = max_norm_err(
        fun_val = fun_vals, 
        approximation_val = lagrange_equidistant,
    )
    chebyshev_max_err = max_norm_err(
        fun_val = fun_vals, 
        approximation_val = lagrange_chebyshev,
    )
    
    equidistant_l2_err = l2_norm_err(
        fun_val = fun_vals, 
        approximation_val = lagrange_equidistant,
        interval = interval,
        N = N,
    )
    chebyshev_l2_err = l2_norm_err(
        fun_val = fun_vals, 
        approximation_val = lagrange_chebyshev,
        interval = interval,
        N = N,
    )
    
    equidistant_bound = np.nan
    chebyshev_bound = np.nan
    
    if bound:
        equidistant_bound = interpolation_bound(
            x_nodes = equidistant_nodes,
            grid = grid,
        )    
        
        chebyshev_bound = interpolation_bound_chebyshev(x_nodes = chebyshev_nodes)
        
    return (
        equidistant_max_err, 
        equidistant_l2_err,
        chebyshev_max_err, 
        chebyshev_l2_err,
        equidistant_bound,
        chebyshev_bound,
    )
    
def lagrange_error_norms_multiple_n(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    interval: tuple[float, float],
    n_arr: npt.NDArray[np.int64],
    N: int | None = None,
    bound: bool = False,
) -> tuple[
    npt.NDArray[np.float64], 
    npt.NDArray[np.float64], 
    npt.NDArray[np.float64], 
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    ]:
    """Calculates errors for Lagrange interpolation for multiple n

    Args:
        fun (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]): Function to interpolate
        interval (tuple[float, float]): Interval to evaluate interpolant on
        n_arr (npt.NDArray[np.int64]): Array of n, number of nodes - n + 1
        N (int | None = None): number of grid subintervals. If None, uses 100 times the larges tpolynmial degree. Defaults to None. 
        bound (bool, optional): True if function if f. Defaults to False.

    Returns:
        tuple[ npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], ]: arrays for
        equidistant max error, equidistant l2 error, chebyshev max error, chebyshev l2 error, equidistant theoretical bound, chebyshev theoretical bound
    """
    if N is None:
        N = 100 * int(np.max(n_arr))
    
    equidistant_max_err_arr = np.zeros_like(n_arr, dtype = float)
    equidistant_l2_err_arr = np.zeros_like(n_arr, dtype = float)
    equidistant_bound_arr = np.zeros_like(n_arr, dtype = float)
    
    chebyshev_max_err_arr = np.zeros_like(n_arr, dtype = float)
    chebyshev_l2_err_arr = np.zeros_like(n_arr, dtype = float)
    chebyshev_bound_arr = np.zeros_like(n_arr, dtype = float)

    
    for i, n in enumerate(n_arr):
        (
            equidistant_max_err_arr[i], 
            equidistant_l2_err_arr[i], 
            chebyshev_max_err_arr[i], 
            chebyshev_l2_err_arr[i],
            equidistant_bound_arr[i],
            chebyshev_bound_arr[i],
        ) = lagrange_error_norms(
            fun = fun,
            interval = interval,
            n = n,
            N = N,
            bound = bound, 
        )
    
    return (
        equidistant_max_err_arr, 
        equidistant_l2_err_arr, 
        chebyshev_max_err_arr, 
        chebyshev_l2_err_arr,
        equidistant_bound_arr,
        chebyshev_bound_arr,
    )