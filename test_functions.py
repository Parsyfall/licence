"""
test_functions.py

This module contains definitions of several commonly used test functions for optimization problems in two dimensions.
It includes the Rastrigin, Ackley, and Schaffer functions. These functions are often used to evaluate the performance
of optimization algorithms due to their complex landscapes and multiple local minima.

Classes:
    Bounds: An enumeration that stores the typical input bounds for each test function.

Functions:
    rastrigin(x: float, y: float) -> float:
        Computes the value of the Rastrigin function at the given (x, y) coordinates.
        
    ackley(x: float, y: float) -> float:
        Computes the value of the Ackley function at the given (x, y) coordinates.
        
    schaffer(x: float, y: float) -> float:
        Computes the value of the Schaffer function at the given (x, y) coordinates.
"""
import numpy as np
from enum import Enum

# Store bounds for test functions
class Bounds(Enum):
    RASTRIGIN = (-5.12, 5.12)
    ACKLEY = (-5, 5)
    SCHAFFER = (-100, 100)


# Define the Rastrigin function for 2D
def rastrigin(x: float, y: float) -> float:
    return (
        20 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))
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