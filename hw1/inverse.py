from math import atan2
import numpy as np

from collections import namedtuple
from functools import reduce

from hw1 import constants, Angles, Rotation, Translation, forward_kinematics, Plotter, Point


class UnreachablePointException(Exception):
    pass


Pose = namedtuple('Pose', ['position', 'orientation'])
desired_pose = Pose(np.array((0, -5, 0)), np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]))

base_position = np.identity(4)
spherical_wrist_position = desired_pose.position + desired_pose.orientation[1] * constants.d_5

c_3 = (np.linalg.norm(spherical_wrist_position) - constants.d_2 ** 2 - constants.d_3 ** 2) / (2 * constants.d_2 * constants.d_3)
s_3 = np.sqrt(1 - c_3 ** 2)

theta_3 = atan2(s_3, c_3)

c_2 = (spherical_wrist_position[0] ** 2 + spherical_wrist_position[1] ** 2 * ())

wrist_xy = spherical_wrist_position[:2]
wrist_unit_xy = wrist_xy / np.linalg.norm(wrist_xy)

c_1, s_1 = wrist_unit_xy

theta_1 = atan2(s_1, c_1)

second_frame_position = base_position.dot(Rotation.around_z(theta_1)).dot(Translation.on_z(constants.d_1))[:3, 3]

l_vector = spherical_wrist_position - second_frame_position

c_3 = (np.linalg.norm(l_vector) - constants.d_2 ** 2 - constants.d_3 ** 2) / (2 * constants.d_2 * constants.d_3)

if abs(c_3) > 1:
    raise UnreachablePointException()

s_3 = (1 - c_3 ** 2) ** 0.5

theta_3 = atan2(s_3, c_3)

s_2 = ((constants.d_2 + constants.d_3 * c_3) * l_vector[1] - constants.d_3 * s_3 * l_vector[0]) / np.linalg.norm(l_vector)
c_2 = ((constants.d_2 + constants.d_3 * c_3) * l_vector[0] + constants.d_3 * s_3 * l_vector[1]) / np.linalg.norm(l_vector)
theta_2 = atan2(s_2, c_2)

R_0_3 = reduce(np.dot, [
    Rotation.around_z(theta_1),
    Rotation.around_x(theta_2),
    Rotation.around_x(theta_3)
])[:3, :3]

R_3_6 = R_0_3.T.dot(desired_pose.orientation)

theta_4 = atan2(-R_3_6[2, 1], R_3_6[0, 1])
theta_5 = atan2(np.sqrt(R_3_6[1, 0] ** 2 + R_3_6[1, 2] ** 2), R_3_6[1, 1])
theta_6 = atan2(R_3_6[1, 2], R_3_6[1, 0])

angles = Angles(theta_1, theta_2, theta_3, theta_4, theta_5, theta_6)

frames = forward_kinematics(angles)

limit = sum(constants) / 2
plotter = Plotter(figure_size=(15, 15), x_limit=(-limit, limit), y_limit=(-limit, limit), z_limit=(-limit, limit))
plotter.plot_poses(frames[::2])
plotter.plot_points(frames)
plotter.plot_point(Point(*desired_pose.position))
plotter.plot_point(Point(0, 0, 0))
plotter.temp(l_vector, desired_pose.orientation)
plotter.temp2(desired_pose)
plotter.show()
plotter.save()
