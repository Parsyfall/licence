import numpy as np
from enum import Enum

# Store bounds for test functions
class Bounds(Enum):
    rastrigin = [-5.12, 5.12],
    ackley= [-5, 5],
    schaffer = [-100, 100]


# Define the Rastrigin function for 2D
def rastrigin(x: float, y: float) -> float:
    return (
        10 * 2
        + (x**2 - 10 * np.sin(2 * np.pi * x))
        + (y**2 - 10 * np.sin(2 * np.pi * y))
    )

# Define the Ackley function for 2D
def ackley(x: float, y: float) -> float:
    return (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        + 20 + np.e
    )

# Define the Ackley function for 2D
def schaffer(x: float, y: float) -> float:
    return (
        0.5 + (np.sin(x**2 - y**2) ** 2 - 0.5)
        / ((1 + 0.001 * (x**2 + y**2)) ** 2)
    )