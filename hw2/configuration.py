import numpy as np

from collections import namedtuple

from utils import Angles

configuration = namedtuple('Configuration', ['d_1', 'd_2', 'd_3', 'd_4', 'd_5'])(1., 3., 2., 1., 1.)
limits = (-sum(configuration) / 2, sum(configuration) / 2)


def get_dh_parameters(angles: Angles) -> list[tuple[float, float, float, float]]:
    dh_parameters = [
        # a                 d                                       alpha       theta
        (0,                 configuration.d_1,                      np.pi / 2,  angles.theta_1),
        (configuration.d_2, 0,                                      0,          angles.theta_2),
        (0,                 0,                                      -np.pi / 2, angles.theta_3 - np.pi / 2),
        (0,                 configuration.d_3 + configuration.d_4,  np.pi / 2,  angles.theta_4 - np.pi / 2),
        (0,                 0,                                      -np.pi / 2, angles.theta_5),
        (0,                 configuration.d_5,                      0,          angles.theta_6)
    ]
    return dh_parameters
