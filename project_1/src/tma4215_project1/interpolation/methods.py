import numpy as np
import numpy.typing as npt

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
    """_summary_

    Args:
        x_nodes (npt.NDArray[np.float64]): Nodes to interpolate on
        y_nodes (npt.NDArray[np.float64]): Function values on nodes
        x (npt.NDArray[np.float64]): Points to find interpolation value on

    Returns:
        npt.NDArray[np.float64]: Lagrangian interpolation function values on x
    """
    pol = np.zeros_like(x, dtype = float)
    for i in range(len(x_nodes)):
        pol += y_nodes[i]*L(i, x_nodes, x)
        
    return pol

def phi(
    r: npt.NDArray[np.float64], 
    epsilon: float
    ) -> npt.NDArray[np.float64]:
    """Calculates basis function for RBF"""
    return np.exp(-(epsilon * r)**2)

def rbf_matrix(
    x_nodes: npt.NDArray[np.float64],
    epsilon: float,
) -> npt.NDArray[np.float64]:
    """Calculates basis function matrix for RBF"""
    diff = x_nodes[:, None] - x_nodes[None, :]
    
    return phi(diff, epsilon)

def RBF(
    x_nodes: npt.NDArray[np.float64],
    y_nodes: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    epsilon: float,
) -> npt.NDArray[np.float64]:
    """radial basis function interpolation on function with values y_nodes on x_nodes

    Args:
        x_nodes (npt.NDArray[np.float64]): Points to interpolate on
        y_nodes (npt.NDArray[np.float64]): Function values on interpolation points
        x (npt.NDArray[np.float64]): Points to find interpolation value of
        epsilon (float): shape parameter

    Returns:
        npt.NDArray[np.float64]: Interpolation values on x
    """
    M = rbf_matrix(x_nodes, epsilon)
    w = np.linalg.solve(M, y_nodes)

    f = np.zeros_like(x, dtype = float)
    
    for i in range(len(x_nodes)):
        f += w[i] * phi(np.abs(x - x_nodes[i]), epsilon)
        
    return f