from chromosome import Chromosome
import random as rd
from math import dist

# [ ] Implement Crowding
# [x] Implement ftiness sharing


def generate_population(size: int) -> list[Chromosome]:
    return [
        Chromosome.from_pair(rd.uniform(-5.12, 5.12), rd.uniform(-5.12, 5.12))
        for _ in range(size)
    ]


def tournament_selection(
    population: list[Chromosome], tournament_size: int
) -> list[Chromosome]:
    selected: list[Chromosome] = list()
    while len(selected) < len(population):
        participants = rd.sample(population, tournament_size)
        winner = max(participants, key=lambda x: x.fitness)
        selected.append(winner)
    return selected


def crossover(population: list[Chromosome]) -> list[Chromosome]:
    #  TODO: Add a mating restriction

    new_pop: list[Chromosome] = []
    selected = tournament_selection(population, 10)

    for _ in range(len(selected)):
        index = rd.randint(0, len(selected) - 1)
        parent1: Chromosome = population[index]
        index = rd.randint(0, len(selected) - 1)
        parent2: Chromosome = population[index]

        child = breed_individuals(parent1, parent2)
        new_pop.append(child)

    return new_pop


def breed_individuals(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    alpha: float = rd.random()
    child = (
        alpha * parent1.coordinate.x + (1 - alpha) * parent2.coordinate.x,
        alpha * parent1.coordinate.y + (1 - alpha) * parent2.coordinate.y,
    )

    return Chromosome.from_tuple(child)


def mutation(population: list[Chromosome], probability: float) -> None:
    # Bounds for Rastrigin function
    upper_bound = 5.12
    lower_bound = -5.12

    for gene in population:
        if rd.random() < probability:
            # Mutate X
            if rd.random() < 0.5:
                gene.coordinate.x -= rd.random() * (gene.coordinate.x - lower_bound)
            else:
                gene.coordinate.x += rd.random() * (upper_bound - gene.coordinate.x)
            # Mutate Y
            if rd.random() < 0.5:
                gene.coordinate.y -= rd.random() * (gene.coordinate.y - lower_bound)
            else:
                gene.coordinate.y += rd.random() * (upper_bound - gene.coordinate.y)

            # Reevaluate fitness of mutants
            gene.eval_fitness()


def sharing(distance: float, sigma: float, alpha: float = 1) -> float:
    """
    Sharing function of my GA that decides if 2 individuals are close enought to share fitness
    based on distance between them and the threshold of dissimilarity (sigma)
    """
    # If distance between 2 individuals is less than sharing distance (sigma) -> they share fitness
    return 1 - (distance / sigma) ** alpha if distance < sigma else 0


def fitness_sharing(
    population: list[Chromosome], target_individual: Chromosome, radius: float
) -> None:
    # TODO: Try and use clustering algorithms to divite population in smaller groups
    niche_count = 0
    for indiv in population:
        niche_count += sharing(
            dist(indiv.coordinate, target_individual.coordinate), len(population) / 3
        )

    target_individual.fitness = target_individual.fitness / niche_count


def crowding(population: list[Chromosome], radius: float):
    # Split population into 2 samples
    rd.shuffle(population)
    sample1 = population[: len(population) // 2]
    sample2 = population[len(population) // 2 :]

    # Pair and mate individuals
    for index in range(len(population)):
        child1 = breed_individuals(sample1[index], sample2[index])
        child2 = breed_individuals(sample1[index], sample2[index])

        # Compete and replace
        replace_most_similar(population, child1, radius)
        replace_most_similar(population, child2, radius)


def replace_most_similar(
    population: list[Chromosome], child: Chromosome, radius: float
):
    for index in range(len(population)):
        if dist(population[index].coordinate, child.coordinate) < radius:
            # Replace first most similar individual
            population[index] = child
            return


def run_evolution(loops, max_generations) -> list[Chromosome]:
    best: list[Chromosome] = []
    new_pop: list[Chromosome] = []
    pop: list[Chromosome] = []
    for _ in range(loops):
        print(f"loop {_}")
        pop = generate_population(100)

        current_generation: int = 0
        while current_generation < max_generations:
            pop.sort(key=lambda genome: genome.fitness)
            print(f"Generation: {current_generation}")

            new_pop = []
            new_pop.append(pop[0])  # Elitism
            new_pop.extend(crossover(pop))

            mutation(new_pop, 0.08)

            pop = new_pop

            best.append(new_pop[0])
            current_generation += 1

    return best


if __name__ == "__main__":
    a = run_evolution(10, 100)
    a = a[:10]
    print(*[x.coordinate for x in a], sep="\n")
