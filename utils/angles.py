from random import uniform
from collections import namedtuple
import numpy as np

Angles = namedtuple('Angles', ['theta_1', 'theta_2', 'theta_3', 'theta_4', 'theta_5', 'theta_6'])


def get_random_angles() -> Angles:
    return Angles(*[uniform(-np.pi, np.pi) for _ in range(6)])
