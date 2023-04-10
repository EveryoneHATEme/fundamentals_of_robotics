import numpy as np


class Rotation:
    @staticmethod
    def around_x(theta: float) -> np.ndarray:
        return np.array([[1.,   0.,             0.,             0.],
                         [0.,   np.cos(theta),  -np.sin(theta), 0.],
                         [0.,   np.sin(theta),  np.cos(theta),  0.],
                         [0.,   0.,             0.,             1.]])

    @staticmethod
    def around_y(theta: float) -> np.ndarray:
        return np.array([[np.cos(theta),    0., np.sin(theta),  0.],
                         [0.,               1., 0.,             0.],
                         [-np.sin(theta),   0., np.cos(theta),  0.],
                         [0.,               0., 0.,             1.]])

    @staticmethod
    def around_z(theta: float) -> np.ndarray:
        return np.array([[np.cos(theta),    -np.sin(theta), 0., 0.],
                         [np.sin(theta),    np.cos(theta),  0., 0.],
                         [0.,               0.,             1., 0.],
                         [0.,               0.,             0., 1.]])
