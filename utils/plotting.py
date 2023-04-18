import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import Iterable, Generator
from itertools import cycle


class Plotter:
    pose_colors = cycle(('r', 'g', 'b'))

    def __init__(self,
                 figure_size: tuple[int, int] = None,
                 x_limit: tuple[float, float] = None,
                 y_limit: tuple[float, float] = None,
                 z_limit: tuple[float, float] = None):
        self.figure: Figure = plt.figure(figsize=figure_size)
        self.axes: Axes = self.figure.add_subplot(projection='3d')
        if x_limit is not None:
            self.axes.set_xlim3d(x_limit)
        if y_limit is not None:
            self.axes.set_ylim3d(y_limit)
        if z_limit is not None:
            self.axes.set_zlim3d(z_limit)
        self.axes.set_xlabel('x')
        self.axes.set_ylabel('y')
        self.axes.set_zlabel('z')

    def plot_points(self, frames: Iterable[np.ndarray]):
        self.axes.plot(*zip(*self.extract_points(frames)))

    def plot_frames(self, frames: Iterable[np.ndarray]):
        for position, orientation in self.extract_poses(frames):
            for vector in orientation:
                self.plot_arrow(position, vector, next(self.pose_colors))

    def plot_arrow(self, position: np.ndarray,
                   orientation: np.ndarray,
                   color: str):
        self.axes.quiver(*position, *orientation, length=0.5, normalize=True, color=color)

    def annotate_points(self, frames: Iterable[np.ndarray]):
        for i, position in enumerate(self.extract_points(frames)):
            self.axes.text(*position, s=f'{i}')

    def show(self):
        self.figure.show()

    def save(self, filename: str = 'plot.png'):
        self.figure.savefig(filename)

    @staticmethod
    def extract_poses(frames: Iterable[np.ndarray]) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        for frame in frames:
            position = frame[0:3, 3]
            orientation = frame[0:3, 0:3].T
            yield position, orientation

    @staticmethod
    def extract_points(frames: Iterable[np.ndarray]) -> Generator[np.ndarray, None, None]:
        for frame in frames:
            yield frame[0:3, 3]
