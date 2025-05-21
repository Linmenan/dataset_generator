from ...algebra_lib.polynomial.polynomial import *

import math
import numpy as np


def fit_quintic(start_x, start_y, start_dy, start_ddy, target_x, target_y, target_dy, target_ddy) -> Polynomial:
    start_vec = [1]
    target_vec = [1]
    for i in range(1, 6):
        start_vec.append(start_vec[-1] * start_x)
        target_vec.append(target_vec[-1] * target_x)
    A = np.array([start_vec,
                 [0.0, 1.0, 2.0 * start_vec[1], 3.0 * start_vec[2], 4.0 * start_vec[3], 5.0 * start_vec[4]],
                 [0.0, 0.0, 2.0, 6.0 * start_vec[1], 12.0 * start_vec[2], 20.0 * start_vec[3]],
                 target_vec,
                 [0.0, 1.0, 2.0 * target_vec[1], 3.0 * target_vec[2], 4.0 * target_vec[3], 5.0 * target_vec[4]],
                 [0.0, 0.0, 2.0, 6.0 * target_vec[1], 12.0 * target_vec[2], 20.0 * target_vec[3]]])
    b = np.array([start_y, start_dy, start_ddy, target_y, target_dy, target_ddy])
    x = np.linalg.solve(A, b)
    return Polynomial(x.tolist())


def path_fit_quintic(start_x, start_y, start_yaw, start_kappa, target_x, target_y, target_yaw, target_kappa) \
        -> Polynomial:
    start_dy = math.tan(start_yaw)
    target_dy = math.tan(target_yaw)
    return fit_quintic(start_x, start_y, start_dy, start_kappa * (1.0 + start_dy * start_dy) ** 1.5,
                       target_x, target_y, target_dy, target_kappa * (1.0 + target_dy * target_dy))
