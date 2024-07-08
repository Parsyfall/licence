from typing import Callable, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from chromosome import Chromosome
from evolution import run_evolution
import numpy as np
from datetime import datetime
import test_functions

def update_plot(frame):
    text.set_text(f"Generation: {frame + 1}")
    x = np.array(x_coord[frame])
    y = np.array(y_coord[frame])
    data = np.stack([x, y]).T
    animated_plot.set_offsets(data)
    # if frame in [0, 99]:
    #     plt.savefig(f'frame {frame}')
    return animated_plot


def split_coordinates_matrix(
    matrix: List[List[Tuple[float, float]]],
) -> Tuple[List[List[float]], List[List[float]]]:
    """Splits a matrix of coordinates into two matrices: one for X coordinates and one for Y coordinates."""

    x_list: List[List[float]] = []
    y_list: List[List[float]] = []
    for generation in matrix:
        x_coords = [chromosome[0] for chromosome in generation]
        y_coords = [chromosome[1] for chromosome in generation]
        x_list.append(x_coords)
        y_list.append(y_coords)
    return x_list, y_list


def generation_to_coordinate_lists(
    generations: List[List[Chromosome]],
) -> List[List[Tuple[float, float]]]:
    """
    Converts a list of lists of Chromosome objects into a nested list structure
    where each Chromosome is represented as a list of its coordinates.
    """
    result = []
    for generation in generations:
        coordinates = []
        for chromosome in generation:
            coordinates.append((chromosome.coordinate.x, chromosome.coordinate.y))
        result.append(coordinates)
    return result

def get_X_Y_Z(
        *,
        test_function: Callable[[float, float], float], 
        bounds: Tuple[float, float],
        num_points):
    '''Generates random points within specified bounds and evaluates a test function at those points'''
    x = np.random.uniform(bounds[0], bounds[1], num_points)
    y = np.random.uniform(bounds[0], bounds[1], num_points)
    value = test_function(x,y)
    return x, y, value

def animate(max_generations:int,
        population_size:int,
        *,
        test_function: Callable[[float, float], float], 
        test_function_bounds: Tuple[float, float],
        refresh_interval = 200,
        ):
    global animated_plot, x_coord, y_coord, text

    Chromosome.set_fitness_function(test_function)

    start_time = datetime.now()
    print(f"Starting at: {start_time}")

    # Initialize plot
    fig, axis = plt.subplots(figsize=(10, 8))

    # Drawing countour of target function search space
    points = 10000
    x, y, z = get_X_Y_Z(
                        test_function=test_function,
                        bounds=test_function_bounds,
                        num_points=points
                        )
    contour = plt.tricontourf(x, y, z, levels=50, cmap="viridis")

    animated_plot = axis.scatter([], [], color="red", marker="+")

    # Add a colorbar
    fig.colorbar(contour, ax=axis, label="Function value")

    # Configure axis
    axis.set(xlim=[test_function_bounds[0], test_function_bounds[1]], ylim=[test_function_bounds[0], test_function_bounds[1]])
    text = axis.text(6, 6, "", fontsize=12, ha="center")

    # Run evolution
    generations = run_evolution(max_generations, population_size, test_function_bounds)

    # print(f"Evolution done in: {datetime.now() - start_time}")

    # Parse generations
    generations = generation_to_coordinate_lists(generations)

    # Split into separe coordinates
    x_coord, y_coord = split_coordinates_matrix(generations)

    # Create animation
    anim = animation.FuncAnimation(
        fig=fig,
        func=update_plot,
        frames=max_generations,
        interval=refresh_interval,
        repeat=False,
    )

    # Display plot
    plt.grid(True)
    plt.show()
    

    # Save animation as a GIF
    # anim.save(filename="./animation.gif", writer="pillow")

    # Save animation as mp4
    # writer = animation.FFMpegWriter(
    #     fps=5, metadata=dict(artist="Me"), bitrate=1800, codec="libxvid"
    # )
    # anim.save("./animation.mp4", writer)

    print(f"Execution ended at: {datetime.now()}")
    print(f"Overall it took: {datetime.now() - start_time} seconds")


if __name__ == "__main__":
    
    animate(
        100,
        100,
        test_function=test_functions.schaffer,
        test_function_bounds=test_functions.Bounds.RASTRIGIN.value, # type: ignore
        refresh_interval = 100
        )
