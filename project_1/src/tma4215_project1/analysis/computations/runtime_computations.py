import numpy as np

import numpy.typing as npt
from collections.abc import Callable
from typing import Any

from time import perf_counter

from ...interpolation.nodes import (
    generate_equidistant_nodes,
)

from ...interpolation.methods import (
    lagrange,
    piecewise_interpolation,
)

def interpolation_runtime_comparison(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x: npt.NDArray[np.float64],
    interval: tuple[float, float],
    n: int,
    k_arr: npt.NDArray[np.int64],
    repeats: int,
) -> dict[str, Any]:
    global_time_arr = np.zeros((len(k_arr), repeats), dtype = np.float64)
    piecewise_time_arr = np.zeros((len(k_arr), repeats), dtype = np.float64)
    num_nodes_arr = n * k_arr + 1

    for i, k in enumerate(k_arr):
        k = int(k)
        global_degree = n * k
        
        for j in range(repeats):            
            start = perf_counter()
            equidistant_nodes = generate_equidistant_nodes(interval = interval, n = global_degree) #Doesnt matter which nodes we use

            _ = lagrange(
                x_nodes = equidistant_nodes,
                y_nodes = fun(equidistant_nodes),
                x = x,
            )
            end = perf_counter()
            
            global_time_arr[i, j] = end - start
            
            start = perf_counter()
            _ = piecewise_interpolation(
                fun = fun,
                x = x,
                n = n,
                k = k,
                interval = interval,
            )
            end = perf_counter()
            
            piecewise_time_arr[i][j] = end - start
            
    global_time_median = np.median(global_time_arr, axis = 1)
    piecewise_time_median = np.median(piecewise_time_arr, axis = 1)
    
    global_std_arr = np.std(global_time_arr, axis = 1)
    piecewise_std_arr = np.std(piecewise_time_arr, axis = 1)
    
    return {
        'num_nodes': num_nodes_arr,
        'global_time': global_time_arr,
        'piecewise_time': piecewise_time_arr,
        'global_median_time': global_time_median,
        'piecewise_median_time': piecewise_time_median,
        'global_std_time': global_std_arr,
        'piecewise_std_time': piecewise_std_arr,
    }
