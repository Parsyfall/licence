import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


def split(matrix):
    xc = []
    yc = []
    for generation in matrix:
        # Extract x and y coordinates from each chromosome in the generation
        x_coords = [chromosome[0] for chromosome in generation]
        y_coords = [chromosome[1] for chromosome in generation]
        xc.append(x_coords)
        yc.append(y_coords)
    return xc, yc


def generation2list(generations):
    new_list = []
    for chromosome_list in generations:
        tmp = []
        for chromosome in chromosome_list:
            tmp.append(list(chromosome))
        new_list.append(tmp)
    return new_list


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

    animated_plot = axis.scatter([], [], color="red", marker='+')

    # Add a colorbar
    fig.colorbar(contour, ax=axis, label='Function value')

    # Configure axis
    axis.set(xlim=[-5.2, 5.2], ylim=[-5.2, 5.2])
    text = axis.text(6, 6, "", fontsize=12, ha="center")

    # Run evolution
    generations = run_evolution(max_generations)

    print(f"Evolution done in: {datetime.now() - start_time}")

    # Parse generations
    generations = generation2list(generations)

    # Split into separe coordinates
    x_coord, y_coord = split(generations)

    # Create animation
    anim = animation.FuncAnimation(
        fig=fig,
        func=update_plot,
        frames=max_generations,
        interval=200,
        repeat=False,
        blit=True,
    )

    # Display plot
    plt.grid(True)
    plt.show()

    # Save animation as a GIF
    anim.save(filename="./animation.gif", writer="pillow")

    print(f"Execution ended at: {datetime.now()}")
    print(f"Overall it took: {datetime.now() - start_time} seconds")


if __name__ == "__main__":
    main(200)
