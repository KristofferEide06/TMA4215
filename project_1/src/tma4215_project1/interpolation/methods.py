import numpy as np
import autograd.numpy as anp

import math

from collections.abc import Callable
import numpy.typing as npt

# pyright: reportAttributeAccessIssue=false

def L(
    i: int, 
    x_nodes: npt.NDArray[np.float64], 
    x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
    """Calculates the i-th Lagrange basis polynomial"""
    L_i = np.ones_like(x, dtype = float)
    
    for j in range(len(x_nodes)):
        if j != i:
            L_i *= (x - x_nodes[j])/(x_nodes[i] - x_nodes[j])
    return L_i

def lagrange(
    x_nodes: npt.NDArray[np.float64], 
    y_nodes: npt.NDArray[np.float64], 
    x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
    """Lagrangian interpolation function values on x"""
    pol = np.zeros_like(x, dtype = float)
    for i in range(len(x_nodes)):
        pol += y_nodes[i]*L(i, x_nodes, x)
        
    return pol

def interpolation_bound(
    x_nodes: npt.NDArray[np.float64], 
    grid: npt.NDArray[np.float64]
    ) -> float:
    """Theoretical lagrange interpolation bound for cos(2pix) using 
    equidistional nodes """
    N = len(x_nodes)
    omega = np.ones_like(grid)
    
    for node in x_nodes:
        omega *= grid - node
    
    omega_max = np.max(np.abs(omega)) #Numerical estimate of omega_max
    
    return (2 * np.pi)**N/math.factorial(N) * omega_max

def interpolation_bound_chebishev(
    x_nodes: npt.NDArray[np.float64],
) -> float:
    """Theoretical lagrange interpolation bound for cos(2pix) using Chebyshev nodes
    on [0, 1]"""
    N = len(x_nodes)
    
    return (2 * np.pi)**N / (math.factorial(N) * 2**(2*N - 1))

def piecewise_interpolation( #Consider opening for chebyshev as well later
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x: npt.NDArray[np.float64], 
    n: int,
    k: int, 
    interval: tuple[float, float],
    ) -> npt.NDArray[np.float64]:
    """piecewise lagrangian interpolation 

    Args:
        fun (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]): _Function to interpolate
        x (npt.NDArray[np.float64]): Array to interpolate on
        n (int): Number of nodes - n + 1
        k (int): Number of subintervals
        interval (tuple[float, float]): interval to interpolate on

    Returns:
        npt.NDArray[np.float64]: Iinterpolation values on x
    """
    points = np.linspace(interval[0], interval[1], k + 1)
    intervals = np.column_stack((points[:-1], points[1:]))
        
    interpolation_arr = np.zeros_like(x, dtype = np.float64)
        
    for i, subinterval in enumerate(intervals):
            left, right = subinterval
            local_nodes = np.linspace(left, right, n + 1)
            if i < len(intervals) - 1:
                    mask = (x >= left) & (x < right)
            else:
                    mask = (x >= left) & (x <= right)
                        
            x_local = x[mask]
                
            interpolation_arr[mask] = lagrange(local_nodes, fun(local_nodes), x_local)
        
    return interpolation_arr

def phi(r, epsilon):
    """Calculates basis function for RBF"""
    return anp.exp(-(epsilon * r)**2)

def rbf_matrix(
    x_nodes,
    epsilon,
):
    """Calculates basis function matrix for RBF"""
    diff = x_nodes[:, None] - x_nodes[None, :]
    
    return phi(diff, epsilon)

def rbf(
    x_nodes,
    y_nodes,
    x,
    epsilon,
):
    """Radial basis function interpolation on function with values y_nodes on x_nodes
    
        Args:
            x_nodes (npt.NDArray[np.float64]): Points to interpolate on
            y_nodes (npt.NDArray[np.float64]): Function values on interpolation points
            x (npt.NDArray[np.float64]): Points to find interpolation value of
            epsilon (float): Shape parameter
    
        Returns:
            npt.NDArray[np.float64]: Interpolation values on x
        """
    M = rbf_matrix(x_nodes, epsilon)
    w = anp.linalg.solve(M, y_nodes)
    
    x_eval = anp.atleast_1d(x)
    diff = x_eval[:, None] - x_nodes[None, :] #No need for abs, removes NA issue
    Phi = phi(diff, epsilon)
    
    values = Phi @ w
    
    return values[0] if anp.ndim(x) == 0 else values