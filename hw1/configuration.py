from collections import namedtuple

constants = namedtuple('Constants', ['d_1', 'd_2', 'd_3', 'd_4', 'd_5'])(0., 3., 2., 1., 1.)

Point = namedtuple('Point', ['x', 'y', 'z'])

Angles = namedtuple('Angles',
                    ['theta_1', 'theta_2', 'theta_3', 'theta_4', 'theta_5', 'theta_6'])
