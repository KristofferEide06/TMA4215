import numpy as np
import autograd.numpy as anp
from autograd import grad

import numpy.typing as npt

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
    fun,
    interval,
    N,
):
    a, b = interval
    
    eta = anp.linspace(a, b, N + 1)
    fun_vals = fun(eta)
    
    def cost(z):
        x_nodes = z[:-1]
        epsilon = z[-1]
        
        rbf_interpolant = rbf(x_nodes, fun(x_nodes), eta, epsilon)
        
        return (b - a) / N * anp.sum((fun_vals - rbf_interpolant)**2)
    
    return cost

def gradient_descent(
    fun,
    n,
    interval,
    epsilon,
    N,
    max_iter,
    tol,
    L,
    rho,
    rho_bar,
):
    """Attempts to find ideal node placement and epsilon parameter using gradient descent
    with respect to the rbf interpolation of specific function

    Args:
        fun: Function to interpolate on
        n: Num nodes - n + 1
        interval: Interval to evaluate function and interpolate on
        epsilon: Shape parameter of rbf
        N: Grid size to evaluate function on
        L: GD adjustment parameter, tau = 1/L
        max_iter: Max iterations of gd algorithm
        tol: Gradient tolerance
        rho: Acceptance scaling factor. Increase L if step accepted
        rho_bar: Rejection scaling factor. Decrease L if step rejected
        

    Returns:
        _type_: _description_
    """
    x_nodes = generate_equidistant_nodes(interval, n)
    
    z = anp.concatenate((x_nodes, anp.array([epsilon])))

    iterations = 0
    
    cost = make_cost(fun, interval, N)
    grad_c = grad(cost) # pyright: ignore[reportCallIssue]
    
    while iterations <= max_iter:
        g = grad_c(z)
        
        if anp.linalg.norm(g) < tol:
            break
        
        phi = cost(z)
        
        while True:
            z_new = z - 1/L * g
            phi_new = cost(z_new)
            
            if phi_new <= phi - 1 / (2 * L) * anp.dot(g, g):
                z = z_new
                L = rho * L
                
                break
            else:
                L = rho_bar * L
            
            iterations += 1
    
    return z[:-1], z[-1]