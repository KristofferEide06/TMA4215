import numpy as np

import numpy.typing as npt
from collections.abc import Callable
from typing import Any

from ...interpolation.methods import(
    rbf,
    rbf_matrix, 
)

from ...interpolation.norms import(
    l2_norm_err,
    max_norm_err,
)

from ...interpolation.nodes import(
    generate_equidistant_nodes,
    generate_chebishev_nodes,
    gradient_descent,
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

def rbf_optimization(
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
) -> dict[str, Any]:
    grid = np.linspace(interval[0], interval[1], N + 1)
    fun_vals = fun(grid)
    
    init_nodes = generate_equidistant_nodes(interval = interval, n = n)
    
    init_rbf = rbf(
        x_nodes = init_nodes,
        y_nodes = fun(init_nodes),
        x = grid,
        epsilon = init_epsilon,
    )
    
    (
        optimized_nodes,
        optimized_epsilon,
        z_history,
        cost_history,
        backtracks_history,
    ) = gradient_descent(
        fun = fun,
        n = n,
        interval = interval,
        epsilon = init_epsilon,
        N = N,
        max_iter = max_iter,
        max_backtracks = max_backtracks,
        tol = tol,
        L = L,
        rho = rho,
        rho_bar = rho_bar,
        epsilon_min = epsilon_min,
    )
    
    optimized_rbf = rbf(
        x_nodes = optimized_nodes,
        y_nodes = fun(optimized_nodes),
        x = grid,
        epsilon = optimized_epsilon,
    )
    
    init_l2_err = l2_norm_err(
        fun_val = fun_vals,
        approximation_val =  init_rbf,
        interval = interval,
        N = N,
    )
    
    optimized_l2_err = l2_norm_err(
        fun_val = fun_vals,
        approximation_val = optimized_rbf,
        interval = interval,
        N = N,
    )
    
    return {
        'grid': grid,
        'fun_values': fun_vals,
        'initial_nodes': init_nodes,
        'initial_epsilon': init_epsilon,
        'initial_rbf': init_rbf,
        'initial_l2_error': init_l2_err,
        'optimized_nodes': optimized_nodes,
        'optimized_epsilon': optimized_epsilon,
        'optimized_rbf': optimized_rbf,
        'optimized_l2_error': optimized_l2_err,
        'z_history': z_history,
        'cost_history': cost_history,
        'backtracks_history': backtracks_history,
    }
    
def rbf_optimization_multiple_n(
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
    epsilon_min: float   
) -> dict[str, Any]: 
    optimized_l2_err_arr = np.zeros_like(n_arr, dtype = np.float64)
    equidistant_l2_err_arr = np.zeros_like(n_arr, dtype = np.float64)
    chebishev_l2_err_arr = np.zeros_like(n_arr, dtype = np.float64)
    
    optimized_epsilon_arr = np.zeros_like(n_arr, dtype = np.float64)
    iteration_arr = np.zeros_like(n_arr, dtype = np.int64)
    optimized_nodes_lst: list[npt.NDArray[np.float64]] = []
    
    for i, n in enumerate(n_arr):
        n = int(n)
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
        grid = result['grid']
        fun_vals = result['fun_values']
        optimized_epsilon = result['optimized_epsilon']
        
        optimized_l2_err_arr[i] = result['optimized_l2_error']
        optimized_epsilon_arr[i] = optimized_epsilon
        iteration_arr[i] = len(result['cost_history']) - 1
        optimized_nodes_lst.append(result['optimized_nodes'])
        
        equidistant_nodes = generate_equidistant_nodes(interval = interval, n = n)
        chebishev_nodes = generate_chebishev_nodes(interval = interval, n = n)
        
        equidistant_rbf = rbf(
            x_nodes = equidistant_nodes, 
            y_nodes = fun(equidistant_nodes),
            x = grid,
            epsilon = optimized_epsilon,
        )
        
        chebishev_rbf = rbf(
            x_nodes = chebishev_nodes,
            y_nodes = fun(chebishev_nodes),
            x = grid,
            epsilon = optimized_epsilon,
        )
        
        equidistant_l2_err_arr[i] = l2_norm_err(
            fun_val = fun_vals,
            approximation_val = equidistant_rbf,
            interval = interval,
            N = N,
        )
        
        chebishev_l2_err_arr[i] = l2_norm_err(
            fun_val = fun_vals,
            approximation_val = chebishev_rbf,
            interval = interval,
            N = N,
        )
        
    return {
        'n_arr': n_arr,
        'optimized_l2_error': optimized_l2_err_arr,
        'equidistant_l2_error': equidistant_l2_err_arr,
        'chebishev_l2_error': chebishev_l2_err_arr,
        'optimized_epsilon': optimized_epsilon_arr,
        'optimized_nodes': optimized_nodes_lst,
        'iterations': iteration_arr,
    }