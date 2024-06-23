from __future__ import annotations
from math import sin, pi, isclose
from typing import Callable, Tuple
import numpy as np


class Chromosome:
    fitness_function: Callable[[float, float], float] = None # type: ignore

    def __init__(self, X: float, Y: float) -> None:
        self.coordinate: Point = Point(X, Y)
        self.fitness: float = self.eval_fitness()

    def eval_fitness(self) -> float:
        '''Manually specify implementation before creating any class instance'''
        if self.fitness_function is None:
            raise NotImplementedError("Fitness function not implemented")
        return Chromosome.fitness_function(self.coordinate.x, self.coordinate.y)
    
    @classmethod
    def from_tuple(cls, coordinates: Tuple[float, float]) -> Chromosome:
        return cls(coordinates[0], coordinates[1])
    
    @classmethod
    def set_fitness_function(cls, func: Callable[[float, float], float]) -> None:
        '''Set the fitness evaluation function for all instances of the Chromosome class.'''
        cls.fitness_function = func


    def __iter__(self):
        yield self.coordinate.x
        yield self.coordinate.y


class Point:
    def __init__(self, X: float, Y: float) -> None:
        self.x: float = X
        self.y: float = Y

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self):
        return " ".join([str(self.x), str(self.y)])

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False

        return isclose(self.x, other.x, rel_tol=1e-8) and isclose(
            self.y, other.y, rel_tol=1e-8
        )

    def __hash__(self):
        return hash((self.x, self.y))
