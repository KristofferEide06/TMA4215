import numpy as np
import autograd.numpy as anp
from autograd import grad

import numpy.typing as npt
from collections.abc import Callable
from typing import Any

from .methods import rbf

# pyright: reportAttributeAccessIssue=false

def generate_equidistant_nodes(
    interval: tuple[float, float], 
    n: int
    ) -> npt.NDArray[np.float64]:
    """Generates n + 1 equidistant nodes on interval"""
    return np.linspace(interval[0], interval[1], n + 1)

def generate_chebishev_nodes(
    interval: tuple[float, float],
    n: int, 
    ) -> npt.NDArray[np.float64]:
    """Generates n + 1 Chebishev nodes on interval"""
    j = np.arange(n + 1)
    nodes = np.cos((2 * j + 1) * np.pi / (2 * (n + 1)))
    
    a, b = interval
    return (b - a)*nodes/2 + (a + b)/2

def make_cost(
    fun: Callable[[Any], Any],
    interval: tuple[float, float],
    N: int,
) -> Callable[[Any], Any]:
    """Generates the cost function for gd"""
    a, b = interval
    
    eta = anp.linspace(a, b, N + 1)
    fun_vals = fun(eta)
    
    def cost(z: Any) -> Any:
        x_nodes = z[:-1]
        epsilon = z[-1]
        
        rbf_interpolant = rbf(x_nodes, fun(x_nodes), eta, epsilon)
        
        return (b - a) / N * anp.sum((fun_vals - rbf_interpolant)**2)
    
    return cost

def project_parameters(
    z: Any,
    interval: tuple[float, float],
    epsilon_min: float,
) -> Any:
    """Projects nodes and epsilon into valid region"""
    a, b = interval
    
    x_nodes = anp.clip(z[:-1], a, b)
    epsilon = anp.maximum(z[-1], epsilon_min)
    
    return anp.concatenate((x_nodes, anp.array([epsilon]))) 

def gradient_descent(
    fun: Callable[[Any], Any],
    n: int,
    interval: tuple[float, float],
    epsilon: float,
    N: int,
    max_iter: int,
    max_backtracks: int,
    tol: float,
    L: float,
    rho: float,
    rho_bar: float,
    epsilon_min: float, 
) -> tuple[
    npt.NDArray[np.float64],
    float,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64]
]:
    """Attempts to find ideal node placement and epsilon parameter using gradient descent
    with respect to the rbf interpolation of specific function

    Args:
        fun: Function to interpolate on
        n: Num nodes - n + 1
        interval: Interval to evaluate function and interpolate on
        epsilon: Initial shape parameter of rbf
        N: Grid size to evaluate function on
        L: GD adjustment parameter, tau = 1/L
        max_iter: Max iterations of gd algorithm
        max_backtracks: Max backtracts per iteration
        tol: Gradient tolerance
        rho: Acceptance scaling factor. Decrease L if step accepted. 0 < rho < 1
        rho_bar: Rejection scaling factor. Increase L if step rejected. rho_bar > 1
        epsilon_min: Minimum allowed shape parameter

    Returns: Updated x_nodes, epsilon, z history, cost history, backtrack history
    """
    iterations = 0
    
    x_nodes = generate_equidistant_nodes(interval, n)
    z = anp.concatenate((x_nodes, anp.array([epsilon])))
    z = project_parameters(z, interval, epsilon_min)
    
    cost = make_cost(fun, interval, N)
    grad_c = grad(cost) # pyright: ignore[reportCallIssue]
    
    z_history = [z]
    cost_history = [float(cost(z))] 
    backtracks_history = []
    
    while iterations < max_iter:
        g = grad_c(z)
        
        projected_gradient = L * (
            z - project_parameters(
                z = z - g / L,
                interval = interval,
                epsilon_min = epsilon_min,
            )
        )
        
        if anp.linalg.norm(projected_gradient) < tol:
            break
        
        phi = cost(z)
        
        backtracks = 0
        rejections = 0
        step_accepted = False
        
        while backtracks < max_backtracks:
            backtracks += 1
            z_new = project_parameters(
                z - 1 / L * g,
                interval = interval,
                epsilon_min = epsilon_min,
            )

            phi_new = cost(z_new)
            
            if phi_new <= phi + anp.dot(g, z_new - z) + L / 2 * anp.dot(z_new - z, z_new - z):
                z = z_new
                L = rho * L
                step_accepted = True
                
                break
            
            L = rho_bar * L
            rejections += 1
            
        if not step_accepted:
            raise RuntimeError('Backtracking couldnt find acceptable step')    
        
        iterations += 1
        
        z_history.append(z)
        backtracks_history.append(rejections)
        cost_history.append(float(phi_new))
            
    return (
        np.asarray(z[:-1], dtype = np.float64), 
        float(z[-1]),
        np.asarray(z_history, dtype = np.float64),
        np.asarray(cost_history, dtype = np.float64),
        np.asarray(backtracks_history, dtype = np.int64),
    )