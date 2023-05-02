import numpy as np
import os

from hw2.forward import forward_kinematics
from hw2.inverse import inverse_kinematics
from hw2.configuration import limits

from utils import get_random_angles, Plotter


def run_test(plot: bool = False, test_counter: int = None):
    angles = get_random_angles()

    fr_results = forward_kinematics(angles)
    desired_frame = fr_results[-1]

    ik_results = map(forward_kinematics, inverse_kinematics(desired_frame))

    if plot:
        plotter = Plotter(figure_size=(15, 15),
                          x_limit=limits,
                          y_limit=limits,
                          z_limit=limits)
        plotter.plot_frames(fr_results)
        plotter.plot_points(fr_results)
        plotter.annotate_points(fr_results)

        try:
            os.mkdir(f'hw2/images/tests/{test_counter}')
        except FileExistsError:
            pass
        plotter.save(f'hw2/images/tests/{test_counter}/forward.png')

        for i, result in enumerate(ik_results):
            plotter = Plotter(figure_size=(15, 15),
                              x_limit=limits,
                              y_limit=limits,
                              z_limit=limits)
            plotter.plot_frames(result)
            plotter.plot_points(result)
            plotter.annotate_points(result)
            plotter.save(f'hw2/images/tests/{test_counter}/inverse_{i}.png')

    if all([np.allclose(desired_frame, result) for result in ik_results]):
        print(f'Test {test_counter} passed')
        return True
    else:
        print(f'Test {test_counter} failed')
        return False


def main():
    try:
        os.mkdir('hw2/images/tests')
    except FileExistsError:
        pass

    for counter in range(25):
        run_test(plot=True, test_counter=counter)


if __name__ == '__main__':
    main()
