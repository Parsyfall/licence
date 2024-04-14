from __future__ import annotations
from math import sin, pi
from typing import Tuple


class Chromosome:
    def __init__(self, point: Point) -> None:
        self.coordinate: Point = point
        self.fitness: float = self.eval_fitness()

    @classmethod
    def from_tuple(cls, coordinates: Tuple[float, float]) -> Chromosome:
        return cls(Point(coordinates[0], coordinates[1]))

    @classmethod
    def from_pair(cls, X: float, Y: float) -> Chromosome:
        return cls(Point(X, Y))

    def eval_fitness(self) -> float:
        # Rastrigin function for 2D
        return (
            10 * 2
            + (self.coordinate.x**2 - 10 * sin(2 * pi * self.coordinate.x))
            + (self.coordinate.y**2 - 10 * sin(2 * pi * self.coordinate.y))
        )


class Point:
    def __init__(self, X: float, Y: float) -> None:
        self.x: float = X
        self.y: float = Y

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self):
        return " ".join([str(self.x), str(self.y)])
