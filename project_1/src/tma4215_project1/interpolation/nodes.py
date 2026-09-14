import numpy as np
import numpy.typing as npt

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