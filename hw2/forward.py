import numpy as np
from itertools import starmap, accumulate

from utils import Rotation, Translation, DH, Plotter

from collections import namedtuple

Angles = namedtuple('Angles', ['theta_1', 'theta_2', 'theta_3', 'theta_4', 'theta_5', 'theta_6'])
constants = namedtuple('Constants', ['d_1', 'd_2', 'd_3', 'd_4', 'd_5'])(1., 3., 2., 1., 1.)


def get_dh_parameters(angles: Angles) -> list[tuple[float, float, float, float]]:
    dh_parameters = [
        # a             d                               alpha       theta
        (0,             constants.d_1,                  np.pi / 2,  angles.theta_1),
        (constants.d_2, 0,                              0,          angles.theta_2),
        (0,             0,                              -np.pi / 2, angles.theta_3),
        (0,             constants.d_3 + constants.d_4,  np.pi / 2,  angles.theta_4),
        (0,             0,                              -np.pi / 2, angles.theta_5),
        (0,             constants.d_5,                  0,          angles.theta_6)
    ]
    return dh_parameters


def forward_kinematics(angles: Angles) -> list[np.ndarray]:

    transforms = starmap(DH.transform, get_dh_parameters(angles))

    base_frame = np.identity(4)
    frames_history = list(accumulate(transforms, func=np.dot, initial=base_frame))

    return frames_history


if __name__ == '__main__':
    angles_example = Angles(0., 0., 0., 0., 0., 0.)

    frames = forward_kinematics(angles_example)

    limit = sum(constants) / 2
    plotter = Plotter(figure_size=(15, 15), x_limit=(-limit, limit), y_limit=(-limit, limit), z_limit=(-limit, limit))
    plotter.plot_frames(frames)
    plotter.plot_points(frames)
    plotter.annotate_points(frames)
    plotter.show()
    plotter.save('forward_zeros.png')
