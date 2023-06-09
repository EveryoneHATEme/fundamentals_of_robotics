import numpy as np
from itertools import starmap, accumulate

from utils import DH, Plotter, Angles
from hw2 import configuration, get_dh_parameters


def forward_kinematics(angles: Angles) -> list[np.ndarray]:
    transforms = starmap(DH.transform, get_dh_parameters(angles))

    base_frame = np.identity(4)
    frames_history = list(accumulate(transforms, func=np.dot, initial=base_frame))

    return frames_history


def main():
    angles_example = Angles(-np.pi / 8, -np.pi / 8, np.pi / 4, np.pi / 2, np.pi / 6, np.pi / 8)
    print(angles_example)

    frames = forward_kinematics(angles_example)
    print(*frames, sep='\n')

    limit = sum(configuration) / 2
    plotter = Plotter(figure_size=(15, 15),
                      x_limit=(-limit, limit + 2),
                      y_limit=(-limit, limit),
                      z_limit=(-limit, limit))
    plotter.plot_frames(frames)
    plotter.plot_points(frames)
    plotter.annotate_points(frames)
    plotter.show()
    plotter.save('hw2/images/forward_some_angles.png')
