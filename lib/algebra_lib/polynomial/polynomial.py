from ...geometry_lib.trajectory.trajectory import *


import math


class Polynomial:
    def __init__(self, params):
        self.params = params

    def differential(self, x, order):
        temp = 1.0
        i = 1
        result = 0.0
        if order >= len(self.params):
            return result
        while i <= order:
            temp *= i
            i += 1
        result += temp * self.params[i - 1]
        while i < len(self.params):
            temp *= x * i / (i - order)
            result += temp * self.params[i]
            i += 1
        return result

    def y(self, x):
        return self.differential(x, 0)

    def dy(self, x):
        return self.differential(x, 1)

    def ddy(self, x):
        return self.differential(x, 2)

    def trajectory_point(self, x) -> TrajectoryPoint2D:
        dy_temp = self.dy(x)
        return TrajectoryPoint2D(0.0, x, self.y(x), math.atan(dy_temp),
                                   self.ddy(x) / ((1 + dy_temp * dy_temp) ** 1.5),
                                 0.0, 0.0, 0.0, -1.0)

    def discrete_to_trajectory_dx(self, start_x, end_x, dx) -> Trajectory2D:
        result = Trajectory2D([])
        if dx <= 0.0:
            return Trajectory2D([])
        current_x = start_x
        while current_x <= end_x:
            result.data.append(self.trajectory_point(current_x))
            current_x += dx
        return result
