import numpy as np
from functools import reduce
from utils import Translation, Rotation


class DH:
    @staticmethod
    def transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
        rotation_theta = Rotation.around_z(theta)
        translation_d = Translation.on_z(d)
        translation_a = Translation.on_x(a)
        rotation_alpha = Rotation.around_x(alpha)

        return reduce(np.dot, [rotation_theta, translation_d, translation_a, rotation_alpha])
