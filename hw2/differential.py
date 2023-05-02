import numpy as np

from utils import get_random_angles
from hw2.forward import forward_kinematics


def compute_jacobian(frames: list[np.ndarray]) -> np.ndarray:
    jacobian = np.zeros((6, 6), dtype=np.float64)

    end_effector_position = frames[-1][:3, 3]

    for i, frame in enumerate(frames[:-1]):
        position = frame[:3, 3]
        z_axis = frame[:3, 2]

        linear = np.cross(z_axis, end_effector_position - position)
        angular = z_axis

        jacobian[:, i] = np.array([*linear, *angular])

    return jacobian


def check_singular(jacobian: np.ndarray):
    return np.linalg.matrix_rank(jacobian) != 6


def cartesian_velocities(jacobian: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
    return jacobian.dot(q_dot)


def main():
    angles = get_random_angles()
    forward_results = forward_kinematics(angles)
    matrix = compute_jacobian(forward_results)
    print(f'Jacobian obtained:\n{matrix}')
    print(f'Manipulator is {"" if check_singular(matrix) else "not "}in singularity')
    random_velocities = np.array(get_random_angles())
    random_cartesian_velocities = cartesian_velocities(matrix, random_velocities)
    print(f'Vector if velocities in cartesian space: {random_cartesian_velocities}')
