import numpy as np
from chromosome import Chromosome
import random as rd
from math import dist

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
    selected = tournament_selection(population, 10, 99)

    for _ in range(len(selected)):
        index = rd.randint(0, len(selected) - 1)
        parent1 = population[index]
        index = rd.randint(0, len(selected) - 1)
        parent2 = population[index]

        child = breed_individuals(parent1, parent2)
        new_pop.append(child)

    print(f"Best fitness: {population[0].fitness:.8f} at {population[0].coordinate}")
    print(
        f"Second best fitness: {population[1].fitness:.8f} at {population[1].coordinate}"
    )
    return new_pop


def breed_individuals(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    alpha: float = rd.random()
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

    if rd.random() < probability:
        # Mutate X
        if rd.random() < 0.5:
            specimen.coordinate.x -= rd.random() * (specimen.coordinate.x - lower_bound)
        else:
            specimen.coordinate.x += rd.random() * (upper_bound - specimen.coordinate.x)
        # Mutate Y
        if rd.random() < 0.5:
            specimen.coordinate.y -= rd.random() * (specimen.coordinate.y - lower_bound)
        else:
            specimen.coordinate.y += rd.random() * (upper_bound - specimen.coordinate.y)

        # Reevaluate fitness of mutants
        specimen.eval_fitness()


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


def crowding(population: list[Chromosome]) -> list[Chromosome]:
    # Find the closest chromosome for each speciment in the population
    # Mate them and obtain 2 children
    # Now include the best parent and the best child in the new population

    # FIXME: Critical productivity drop, optimize asap
    # Cause, doubling the population size with each call
    new_population = []
    mating_cache = {}
    for parent1 in population:
        parent2 = find_nearest(population, parent1)

        if (parent1.coordinate, parent2.coordinate) in mating_cache or (
            parent2.coordinate,
            parent1.coordinate,
        ) in mating_cache:
            # Individuals have already mated
            continue

        # Cache pairs
        mating_cache[(parent1.coordinate, parent2.coordinate)] = True

        # Breed
        child1 = breed_individuals(parent1, parent2)
        child2 = breed_individuals(parent1, parent2)

        # Add to the new population
        new_population.append(parent1 if parent1.fitness < parent2.fitness else parent2)
        new_population.append(child1 if child1.fitness < child2.fitness else child2)

    print(
        f"Afer crowding, new population size: {len(new_population[: len(population)])}"
    )

    # FIXME: There are a lot of duplicates in new_population
    new_population.sort(key=lambda genome: genome.fitness)
    return new_population[: len(population)]


def find_nearest(population: list[Chromosome], individual: Chromosome) -> Chromosome:
    """
    Find and return the chromosome in the given population that is closest to the specified individual.
    """

    nearest = None
    smallest_distance = np.inf

    # Iterate through each chromosome in the population
    for elem in population:
        # Skip the individual itself
        if elem is individual:
            continue

        distance = dist(individual.coordinate, elem.coordinate)
        if distance < smallest_distance:
            smallest_distance = distance
            nearest = elem

    return nearest  # type: ignore


def run_evolution(max_generations) -> list[list[Chromosome]]:
    new_pop: list[Chromosome] = []
    pop: list[Chromosome] = []

    # TODO: Use crowding
    # TODO: Use fitness sharing

    pop = generate_population(100)

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


def plot():
    pass


if __name__ == "__main__":
    a = run_evolution(100)
    print()
