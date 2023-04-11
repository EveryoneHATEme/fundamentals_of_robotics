from math import atan2
import numpy as np

from collections import namedtuple
from functools import reduce

from hw1 import constants, Angles, Rotation, Translation, forward_kinematics, Plotter, Point


class UnreachablePointException(Exception):
    pass


desired_pose = np.array([[1, 0, 0, 4],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

base_position = np.identity(4)
wrist_position = desired_pose.dot(np.linalg.inv(Translation.on_y(constants.d_5)))

theta_1 = atan2(wrist_position[1, 3], wrist_position[0, 3])

c_3 = (np.linalg.norm(wrist_position[:3, 3]) - constants.d_2 ** 2 - constants.d_3 ** 2) / (2 * constants.d_2 * constants.d_3)
theta_3 = atan2(np.sqrt(1 - c_3 ** 2), c_3)

c_2_numerator = (constants.d_2 + constants.d_3 * np.cos(theta_3)) * np.linalg.norm(wrist_position[:2, 3]) + constants.d_3 * np.sin(theta_3) * wrist_position[2, 3]
s_2_numerator = wrist_position[2, 3] * (constants.d_2 + constants.d_3 * np.cos(theta_3)) - constants.d_3 * np.sin(theta_3) * np.linalg.norm(wrist_position[:2, 3])
denominator = constants.d_2 ** 2 + constants.d_3 ** 2 + 2 * constants.d_2 * constants.d_3 * np.cos(theta_3)

c_2 = c_2_numerator / denominator
s_2 = s_2_numerator / denominator

theta_2 = atan2(s_2, c_2)

R_0_3 = reduce(np.dot, [
    Rotation.around_z(theta_1),
    Rotation.around_x(theta_2),
    Rotation.around_x(theta_3)
])[:3, :3]

R_3_6 = np.linalg.inv(R_0_3).dot(desired_pose[:3, :3])

theta_4 = atan2(R_3_6[2, 1], -R_3_6[0, 1])
theta_5 = atan2(np.sqrt(R_3_6[1, 0] ** 2 + R_3_6[1, 2] ** 2), R_3_6[1, 1])
theta_6 = atan2(R_3_6[1, 2], R_3_6[1, 0])

angles = Angles(theta_1, theta_2, theta_3, theta_4, theta_5, theta_6)

frames = forward_kinematics(angles)

limit = sum(constants) / 2
plotter = Plotter(figure_size=(15, 15), x_limit=(-limit, limit), y_limit=(-limit, limit), z_limit=(-limit, limit))
plotter.plot_poses(frames[1::2])
plotter.plot_points(frames)
plotter.plot_point(Point(*desired_pose[:3, 3]))
plotter.show()
plotter.save('inverse_example.png')
