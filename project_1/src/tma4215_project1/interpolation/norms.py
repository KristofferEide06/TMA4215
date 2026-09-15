import numpy as np
import numpy.typing as npt

def max_norm_err(
    fun_val: npt.NDArray[np.float64], 
    approximation_val: npt.NDArray[np.float64],
    ) -> float:
    """Calculates max norm error of function and approximation array"""
    
    return np.max(np.abs(fun_val - approximation_val))

def l2_norm_err(
    fun_val: npt.NDArray[np.float64], 
    approximation_val: npt.NDArray[np.float64],
    interval: tuple[float, float],
    N: int,
    ) -> float:
    """Calculates l2 norm error of function and approximation array"""
    
    return np.sqrt(interval[1] - interval[0]) / np.sqrt(N) * np.sqrt(np.sum((fun_val - approximation_val)**2))