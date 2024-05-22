from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from chromosome import Chromosome
from evolution import run_evolution
import numpy as np
from datetime import datetime


def update_plot(frame):
    text.set_text(f"Generation: {frame}")
    x = np.array(x_coord[frame])
    y = np.array(y_coord[frame])
    data = np.stack([x, y]).T
    animated_plot.set_offsets(data)
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


def main(max_generations):
    global animated_plot, x_coord, y_coord, text

    start_time = datetime.now()
    print(f"Starting at: {start_time}")

    # Initialize plot
    fig, axis = plt.subplots(figsize=(10, 8))

    # Drawing countour of Rastrigin search space
    num_points = 10000
    x = np.random.uniform(-5.12, 5.12, num_points)
    y = np.random.uniform(-5.12, 5.12, num_points)
    z = 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
    contour = axis.tricontourf(x, y, z, levels=100, cmap="viridis")

    animated_plot = axis.scatter([], [], color="red", marker="+")

    # Add a colorbar
    fig.colorbar(contour, ax=axis, label="Function value")

    # Configure axis
    axis.set(xlim=[-5.2, 5.2], ylim=[-5.2, 5.2])
    text = axis.text(6, 6, "", fontsize=12, ha="center")

    # Run evolution
    generations = run_evolution(max_generations)

    print(f"Evolution done in: {datetime.now() - start_time}")

    # Parse generations
    generations = generation_to_coordinate_lists(generations)

    # Split into separe coordinates
    x_coord, y_coord = split_coordinates_matrix(generations)

    # Create animation
    anim = animation.FuncAnimation(
        fig=fig,
        func=update_plot,
        frames=max_generations,
        interval=400,
        repeat=False,
        blit=True,
    )

    # Display plot
    plt.grid(True)
    plt.show()

    # Save animation as a GIF
    # anim.save(filename="./animation.gif", writer="pillow")

    writer = animation.FFMpegWriter(
        fps=5, metadata=dict(artist="Me"), bitrate=1800, codec="libxvid"
    )
    # anim.save("./animation.mp4", writer)

    print(f"Execution ended at: {datetime.now()}")
    print(f"Overall it took: {datetime.now() - start_time} seconds")


if __name__ == "__main__":
    main(200)
