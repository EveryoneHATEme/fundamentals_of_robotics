import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from configuration import Point


class Plotter:
    pose_colors = ('r', 'g', 'b')

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

    def plot_points(self, frames: list[np.ndarray]):
        points = self.extract_points(frames)
        self.axes.plot(*zip(*points))

    def plot_point(self, point: Point):
        self.axes.scatter(point.x, point.y, point.z)

    def plot_poses(self, frames: list[np.ndarray]):
        poses = self.extract_poses(frames)

        for position, orientation in poses:
            for vector, color in zip(orientation, self.pose_colors):
                self.plot_arrow(position, vector, color)

    def plot_arrow(self, position: tuple[float, float, float],
                   orientation: tuple[Point, ...],
                   color: str):
        self.axes.quiver(*position, *orientation, length=1, normalize=True, color=color)

    def show(self):
        self.figure.show()

    def save(self, filename: str = 'plot.png'):
        self.figure.savefig(filename)

    @staticmethod
    def extract_points(frames: list[np.ndarray]) -> list[Point]:
        points = []
        for frame in frames:
            points.append(Point(*frame[0:3, 3]))
        return points

    @staticmethod
    def extract_poses(frames: list[np.ndarray]) -> list[tuple[Point, tuple[Point, ...]]]:
        poses = []
        for frame in frames:
            position = Point(*frame[0:3, 3])
            orientation = tuple(Point(*vector) for vector in frame[0:3, 0:3].T)
            poses.append((position, orientation))

        return poses

    def temp(self, vec, pose):
        self.axes.plot((0, vec[0]), (0, vec[1]), (0, vec[2]))
        self.axes.quiver(*vec, *pose)

    def temp2(self, pose):
        self.axes.quiver(*pose[:3, 3], *pose[:3, :3])
