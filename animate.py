import matplotlib.pyplot as plt
import matplotlib.animation as animation
from evolution import run_evolution
import numpy as np


def update_plot(frame):
    text.set_text(f"Frame: {frame}")
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
    for ch_l in generations:
        tmp = []
        for ch in ch_l:
            tmp.append(list(ch))
        new_list.append(tmp)
    return new_list


fig, axis = plt.subplots()

animated_plot = axis.scatter([], [])

axis.set(xlim=[-6, 6], ylim=[-6, 6])

generations = run_evolution(10, 100)
generations = generation2list(generations)

text = axis.text(5, 5, "", fontsize=12, ha="center")

x_coord, y_coord = split(generations)


anim = animation.FuncAnimation(
    fig=fig, func=update_plot, frames=100, interval=800, repeat=False, blit=True
)
plt.grid(True)
plt.show()
