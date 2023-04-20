from math import atan2

import numpy as np
from itertools import starmap, accumulate
from operator import itemgetter

from utils import Translation, Rotation, Angles, DH, Plotter
from hw2 import configuration, get_dh_parameters, forward_kinematics


def inverse_kinematics(end_effector_frame: np.ndarray) -> list[Angles]:
    # get end effector position and orientation as a rotation matrix and a vector
    end_effector_orientation: np.ndarray = end_effector_frame[:3, :3].copy()
    end_effector_position: np.ndarray = end_effector_frame[:3, 3].copy()

    # unpack configuration
    a1, a2, a3, a4, a5 = configuration

    # get wrist position relatively to the link 1
    wrist_position = end_effector_position - a5 * end_effector_orientation[:3, 2]
    wrist_position[2] -= a1

    # get two solutions for theta_1
    theta_1_1 = atan2(wrist_position[1], wrist_position[0])
    theta_1_2 = atan2(-wrist_position[1], -wrist_position[0])

    # calculate cosine of theta_3
    cos_3_numerator = np.sum(wrist_position ** 2) - a2 ** 2 - (a3 + a4) ** 2
    cos_3_denominator = 2 * a2 * (a3 + a4)
    cos_3 = cos_3_numerator / cos_3_denominator

    # calculate two solutions for sine of theta_3
    sin_3_1 = np.sqrt(1 - cos_3 ** 2)
    sin_3_2 = -np.sqrt(1 - cos_3 ** 2)

    # get two solutions for theta_3
    theta_3_1 = atan2(sin_3_1, cos_3)
    theta_3_2 = atan2(sin_3_2, cos_3)

    # calculate four solutions for theta_2
    xy_norm = np.linalg.norm(wrist_position[:2])
    theta_2_1 = atan2((a2 + (a3 + a4) * cos_3) * wrist_position[2] - a3 * sin_3_1 * xy_norm,
                      (a2 + (a3 + a4) * cos_3) * xy_norm + a3 * sin_3_1 * wrist_position[2])
    theta_2_2 = atan2((a2 + (a3 + a4) * cos_3) * wrist_position[2] + a3 * sin_3_1 * xy_norm,
                      -(a2 + (a3 + a4) * cos_3) * xy_norm + a3 * sin_3_1 * wrist_position[2])
    theta_2_3 = atan2((a2 + (a3 + a4) * cos_3) * wrist_position[2] - a3 * sin_3_2 * xy_norm,
                      (a2 + (a3 + a4) * cos_3) * xy_norm + a3 * sin_3_2 * wrist_position[2])
    theta_2_4 = atan2((a2 + (a3 + a4) * cos_3) * wrist_position[2] + a3 * sin_3_2 * xy_norm,
                      -(a2 + (a3 + a4) * cos_3) * xy_norm + a3 * sin_3_2 * wrist_position[2])

    # get rotation matrices from frame 0 to frame 3
    partial_angles = [Angles(theta_1_1, theta_2_1, theta_3_1, 0, 0, 0),
                      Angles(theta_1_1, theta_2_3, theta_3_2, 0, 0, 0),
                      Angles(theta_1_2, theta_2_2, theta_3_1, 0, 0, 0),
                      Angles(theta_1_2, theta_2_4, theta_3_2, 0, 0, 0)]

    partial_dh = [get_dh_parameters(_angles)[:3] for _angles in partial_angles]
    partial_transforms = [starmap(DH.transform, dh) for dh in partial_dh]
    partial_frames = [
        list(accumulate(transforms, func=np.dot, initial=np.identity(4)))[-1] for transforms in partial_transforms
    ]

    rotations_0_3 = list(map(itemgetter((slice(None, 3), slice(None, 3))), partial_frames))

    # get rotation matrices from frame 3 to frame 6
    rotations_3_6 = [_rotation.T.dot(end_effector_orientation) for _rotation in rotations_0_3]

    angles: list[Angles] = []

    # calculate angles for the spherical wrist
    for _rotation, _angles in zip(rotations_3_6, partial_angles):
        theta_4 = atan2(-_rotation[0, 2], _rotation[1, 2])
        theta_5 = atan2(np.sqrt(_rotation[0, 2] ** 2 + _rotation[1, 2] ** 2), _rotation[2, 2])
        theta_6 = atan2(-_rotation[2, 1], _rotation[2, 0])

        angles.append(Angles(*_angles[:3], theta_4, theta_5, theta_6))

        theta_4 = atan2(_rotation[0, 2], -_rotation[1, 2])
        theta_5 = atan2(-np.sqrt(_rotation[0, 2] ** 2 + _rotation[1, 2] ** 2), _rotation[2, 2])
        theta_6 = atan2(_rotation[2, 1], -_rotation[2, 0])

        angles.append(Angles(*_angles[:3], theta_4, theta_5, theta_6))

    return angles


if __name__ == '__main__':
    desired_frame = np.array([[0.82361604, 0.07306063, 0.56242222, 5.68374257],
                              [0.07306063, 0.9697373, -0.23296291, -2.35428326],
                              [-0.56242222, 0.23296291, 0.79335334, 1.79335334],
                              [0., 0., 0., 1.]])

    inverse_results = inverse_kinematics(desired_frame)
    limit = sum(configuration) / 2

    for i, coordinates in enumerate(inverse_results):
        forward_results = forward_kinematics(coordinates)
        print(f'obtained: {forward_results[-1]}\ndesired: {desired_frame}\n\n')
        plotter = Plotter(figure_size=(15, 15),
                          x_limit=(-limit, limit + 2),
                          y_limit=(-limit, limit),
                          z_limit=(-limit, limit))
        plotter.plot_frames(forward_results)
        plotter.plot_points(forward_results)
        plotter.show()
        plotter.save(f'images/inverse_{i}.png')
