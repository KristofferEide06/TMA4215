import numpy as np

import numpy.typing as npt
from collections.abc import Callable

from ...interpolation.nodes import(
    generate_equidistant_nodes,
    generate_chebyshev_nodes,
)

from ...interpolation.methods import(
    L,
    lagrange,
    interpolation_bound_f,
    interpolation_bound_chebyshev_f,
    interpolation_bound_g,
    interpolation_bound_chebyshev_g,
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
    bound: bool,
    fun_type: str | None = None,
) -> tuple[float, float, float, float, float, float, float]: 
    """Calculates errors for Lagrange interpolation 

    Args:
        fun (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]): Function to interpolate
        interval (tuple[float, float]): Interval to evaluate interpolant on
        n (int): number of nodes to interpolate on - n + 1
        N (int): number of grid points  - N + 1
        bound (bool): if bound is to be calculated
        fun_type (str | None, optional): either f or g, if bound is true it returns the bound for either of these functions as described in the task and report. Defaults to None.

    Returns:
        tuple[float, float, float, float, float, float, float, float, float]: equidistant max error, equidistant l2 error, chebyshev max error, chebyshev l2 error, theoretical tight equidistant bound for max error, theoretical loose equidistant bound for max error, theoretical chebyshev bound for max error
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
    
    equidistant_bound_tight = np.nan
    equidistant_bound_loose = np.nan
    chebyshev_bound = np.nan
    
    if bound:
        if fun_type == 'f':
            equidistant_bound_tight = interpolation_bound_f(
                x_nodes = equidistant_nodes,
                grid = grid,
                use_numeric_calc = True,
            )    
            equidistant_bound_loose = interpolation_bound_f(
                x_nodes = equidistant_nodes,
                grid = grid,
                use_numeric_calc = False,
            )
            chebyshev_bound = interpolation_bound_chebyshev_f(x_nodes = chebyshev_nodes) 
        elif fun_type == 'g':
            equidistant_bound_tight = interpolation_bound_g(
                x_nodes = equidistant_nodes,
                grid = grid,
                use_numeric_calc = True,
            )    
            equidistant_bound_loose = interpolation_bound_g(
                x_nodes = equidistant_nodes,
                grid = grid,
                use_numeric_calc = False,
            ) 
            chebyshev_bound = interpolation_bound_chebyshev_g(x_nodes = chebyshev_nodes)
        
    return (
        equidistant_max_err, 
        equidistant_l2_err,
        chebyshev_max_err, 
        chebyshev_l2_err,
        equidistant_bound_tight,
        equidistant_bound_loose,
        chebyshev_bound,
    )
    
def lagrange_error_norms_multiple_n(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    interval: tuple[float, float],
    n_arr: npt.NDArray[np.int64],
    bound: bool, 
    fun_type: str | None = None,
    N: int | None = None,
) -> tuple[
    npt.NDArray[np.float64], 
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
        fun_type: if bound is true uses bounds for either f or g function as described in task and report. Defaults to None.
        N: resolution. Defaults t0 100*int(np.max(n_arr)) if None

    Returns:
        tuple[ npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], ]: arrays for equidistant max error, equidistant l2 error, chebyshev max error, chebyshev l2 error, equidistant tight theoretical bound, equidistant loose theoretical bound, chebyshev theoretical bound
    """
    if N is None:
        N = 100 * int(np.max(n_arr))
    
    equidistant_max_err_arr = np.zeros_like(n_arr, dtype = float)
    equidistant_l2_err_arr = np.zeros_like(n_arr, dtype = float)
    equidistant_bound_tight_arr = np.zeros_like(n_arr, dtype = float)
    equidistant_bound_loose_arr = np.zeros_like(n_arr, dtype = float)
    
    chebyshev_max_err_arr = np.zeros_like(n_arr, dtype = float)
    chebyshev_l2_err_arr = np.zeros_like(n_arr, dtype = float)
    chebyshev_bound_arr = np.zeros_like(n_arr, dtype = float)

    
    for i, n in enumerate(n_arr):
        (
            equidistant_max_err_arr[i], 
            equidistant_l2_err_arr[i], 
            chebyshev_max_err_arr[i], 
            chebyshev_l2_err_arr[i],
            equidistant_bound_tight_arr[i],
            equidistant_bound_loose_arr[i],
            chebyshev_bound_arr[i],
        ) = lagrange_error_norms(
            fun = fun,
            interval = interval,
            n = n,
            N = N,
            bound = bound, 
            fun_type = fun_type,
        )
    
    return (
        equidistant_max_err_arr, 
        equidistant_l2_err_arr, 
        chebyshev_max_err_arr, 
        chebyshev_l2_err_arr,
        equidistant_bound_tight_arr,
        equidistant_bound_loose_arr,
        chebyshev_bound_arr,
    )
    
def estimate_lagrange_max_sum(
    nodes: npt.NDArray[np.float64],
    grid: npt.NDArray[np.float64]
    ) -> float:
    """Calculuates max_x sum_j |L_j(x)|"""
    lagrange_max_sum = np.zeros_like(grid)
    
    for j in range(len(nodes)):
        lagrange_max_sum += np.abs(L(j, nodes, grid))
    
    return np.max(lagrange_max_sum)
