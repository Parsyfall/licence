from typing import Dict
import numpy as np
from chromosome import Chromosome, Point
import random as rd
from math import dist
from test_functions import *

# TODO: Refactor all vanila lists to numpy arrays

MutationProbability = 0.05


def generate_population(size: int) -> list[Chromosome]:
    return [
        Chromosome(rd.uniform(-5.12, 5.12), rd.uniform(-5.12, 5.12))
        for _ in range(size)
    ]


def tournament_selection(
    population: list[Chromosome], tournament_size: int, return_number: int
) -> list[Chromosome]:
    selected = []
    while len(selected) < return_number:
        participants = rd.sample(population, tournament_size)
        winner = min(participants, key=lambda x: x.fitness)
        selected.append(winner)
    return selected


def crossover(population: list[Chromosome]) -> list[Chromosome]:
    #  TODO: Add a mating restriction

    population.sort(key=lambda genome: genome.fitness)

    new_pop: list[Chromosome] = [population[0]]  # Elitism
    selected = tournament_selection(population, 10, len(population) - 1)

    for _ in range(len(selected)):
        index = rd.randint(0, len(selected) - 1)
        parent1 = population[index]
        index = rd.randint(0, len(selected) - 1)
        parent2 = population[index]

        child = breed_individuals(parent1, parent2)
        new_pop.append(child)

    # print(f"Best fitness: {population[0].fitness:.8f} at {population[0].coordinate}")
    # print(
    #     f"Second best fitness: {population[1].fitness:.8f} at {population[1].coordinate}"
    # )
    return new_pop


def breed_individuals(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    alpha: float = rd.random()
    # alpha = 0.5
    child = Chromosome(
        alpha * parent1.coordinate.x + (1 - alpha) * parent2.coordinate.x,
        alpha * parent1.coordinate.y + (1 - alpha) * parent2.coordinate.y,
    )

    mutation(child, MutationProbability)
    return child


def mutation(specimen: Chromosome, probability: float) -> None:
    # Bounds for Rastrigin function
    upper_bound = 5.12
    lower_bound = -5.12
    magnitude = 2

    if rd.random() < probability:
        # Change the magnitude of mutation if needed
        mutation_amount_x = rd.uniform(-magnitude, magnitude)
        mutation_amount_y = rd.uniform(-magnitude, magnitude)

        # Apply mutation and ensure bounds
        specimen.coordinate.x = np.clip(
            specimen.coordinate.x + mutation_amount_x, lower_bound, upper_bound
        )
        specimen.coordinate.y = np.clip(
            specimen.coordinate.y + mutation_amount_y, lower_bound, upper_bound
        )

        # Reevaluate fitness of mutants
        specimen.fitness = specimen.eval_fitness()


'''

def sharing(distance: float, sigma: float, *, alpha: float = 1) -> float:
    """
    Sharing function of my GA that decides if 2 individuals are close enought to share fitness
    based on distance between them and the threshold of dissimilarity (sigma)
    """
    # If distance between 2 individuals is less than sharing distance (sigma) -> they share fitness
    return 1 - (distance / sigma) ** alpha if distance < sigma else 0


def fitness_sharing(population: list[Chromosome], radius: float) -> None:
    """
    Fitness Sharing  is based on the principle of distribution of limited resources.

    An over-crowded niche will be penalized, and its population
    forced to move and populate more vacant niches or even discover new ones.
    """
    # TODO: Try clustering algorithms to divite population in smaller groups
    # TODO: Try kd-tree/ball-tree for radius search (problem name: fixed-radius nearest neighbors search)
    # TODO: Try caching

    niche_count = 0  # measures the approximate number of individuals with whom the fitness is shared
    for target in population:
        for indiv in population:
            niche_count += sharing(dist(indiv.coordinate, target.coordinate), radius)

        if niche_count == 0:
            continue

        # Adjust target fitness
        target.fitness = target.fitness / niche_count

        # Reset niche_count
        niche_count = 0
        
'''


def crowding(population: list[Chromosome]) -> list[Chromosome]:
    # Find the closest chromosome for each speciment in the population
    # Mate them and obtain 2 children
    # Now include the best parent and the best child in the new population

    # FIXME: After some generation population keeps getting striped down by 2 units

    new_population = []
    mating_cache: Dict[Point, bool] = {}

    for parent1 in population:
        parent2 = find_nearest(population, parent1, mating_cache)

        if parent2 is None:
            # In case parent2 is None -> there are no more available mates, break the loop
            break

        if parent1.coordinate in mating_cache or parent2.coordinate in mating_cache:
            continue

        # Cache individuals
        mating_cache[parent1.coordinate] = True
        mating_cache[parent2.coordinate] = True

        # Breed
        child1 = breed_individuals(parent1, parent2)
        child2 = breed_individuals(parent1, parent2)

        # Add to the new population
        new_population.append(parent1 if parent1.fitness < parent2.fitness else parent2)
        new_population.append(child1 if child1.fitness < child2.fitness else child2)

    difference = len(population) - len(new_population)
    if difference > 0:
        new_population.extend(generate_population(difference))

    return new_population


def find_nearest(
    population: list[Chromosome], individual: Chromosome, mating_register
) -> None | Chromosome:
    r"""
    Find and return the chromosome in the given population that is closest to the specified individual and not in the registry.
    Return None in case there are no more available mates, or population size was an odd number ¯\\_(ツ)_/¯
    """

    nearest = None
    smallest_distance = np.inf

    # Iterate through each chromosome in the population
    for elem in population:
        # Skip the individual itself
        if elem is individual:
            continue

        # Pontential mate have already mated
        if elem.coordinate in mating_register:
            continue

        distance = dist(individual.coordinate, elem.coordinate)

        if distance == 0.0:
            # Add small noise to avoid zero distance
            noise = np.random.normal(0, 1e-5, 2)
            elem.coordinate.x += noise[0]
            elem.coordinate.y += noise[1]
            elem.eval_fitness()
            distance = dist(individual.coordinate, elem.coordinate)
            print("Distance 0")

        if distance < smallest_distance:
            smallest_distance = distance
            nearest = elem

    return nearest


def run_evolution(
    max_generations: int,
    generation_size: int = 100
) -> list[list[Chromosome]]:
    new_pop: list[Chromosome] = []
    pop: list[Chromosome] = []

    # TODO: Use crowding
    # TODO: Use fitness sharing

    # Generate initial poppulation
    pop = generate_population(generation_size)

    pop_history = []

    current_generation: int = 0
    while current_generation < max_generations:
        print(f"Generation: {current_generation}")

        # Breeding
        new_pop = []

        # fitness_sharing(pop, 0.2)
        # new_pop.extend(crossover(pop))
        new_pop.extend(crowding(pop))

        pop = new_pop

        # Save population for history
        pop_history.append(pop)
        current_generation += 1

    return pop_history


if __name__ == "__main__":
    Chromosome.set_fitness_function(rastrigin)
    a = run_evolution(100, 10)

    print()
