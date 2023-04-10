import numpy as np

from hw1 import Rotation, Translation, Plotter, constants, Angles


def forward_kinematics(angles: Angles):
    current_frame = np.identity(4)

    transforms = [
        Rotation.around_z(angles.theta_1), Translation.on_z(constants.d_1),
        Rotation.around_x(angles.theta_2), Translation.on_y(constants.d_2),
        Rotation.around_x(angles.theta_3), Translation.on_y(constants.d_3),
        Rotation.around_y(angles.theta_4), Translation.on_y(constants.d_4),
        Rotation.around_z(angles.theta_5), Translation.on_y(constants.d_5),
        Rotation.around_y(angles.theta_6)
    ]

    history = [current_frame.copy()]

    for transform in transforms:
        current_frame = current_frame.dot(transform)
        history.append(current_frame.copy())

    return history


if __name__ == '__main__':
    angles_example = Angles(*np.random.uniform(0, np.pi * 2, 6))

    frames = forward_kinematics(angles_example)

    limit = sum(constants) / 2
    plotter = Plotter(figure_size=(15, 15), x_limit=(-limit, limit), y_limit=(-limit, limit), z_limit=(-limit, limit))
    plotter.plot_poses(frames[::2])
    plotter.plot_points(frames)
    plotter.show()
    plotter.save()
