import numpy as np


class Translation:
    @staticmethod
    def on_x(distance: float) -> np.ndarray:
        return np.array([[1., 0., 0., distance],
                         [0., 1., 0., 0.],
                         [0., 0., 1., 0.],
                         [0., 0., 0., 1.]])

    @staticmethod
    def on_y(distance: float) -> np.ndarray:
        return np.array([[1., 0., 0., 0.],
                         [0., 1., 0., distance],
                         [0., 0., 1., 0.],
                         [0., 0., 0., 1.]])

    @staticmethod
    def on_z(distance: float) -> np.ndarray:
        return np.array([[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 1., distance],
                         [0., 0., 0., 1.]])
