import numpy as np
import numpy.typing as npt

def runge(x): return 1/(x**2 + 1)
def f(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: return np.cos(2 * np.pi * x)
def g(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: return np.exp(3 * x)*np.sin(2 * x)

