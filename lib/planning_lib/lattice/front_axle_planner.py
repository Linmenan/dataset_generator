import math

from ...geometry_lib.trajectory.trajectory import *
from ...geometry_lib.vector.vector import *


class FrontAxlePlanner:
    def __init__(self, plan_distance=0.0):
        self.plan_distance = plan_distance

    def change_point_rear2front(self, rear_point: TrajectoryPoint2D) -> TrajectoryPoint2D:
        theta = math.tan(self.plan_distance * rear_point.curvature)
        return TrajectoryPoint2D(
                rear_point.s,
                rear_point.x + self.plan_distance * rear_point.cos,
                rear_point.y + self.plan_distance * rear_point.sin,
                rear_point.yaw + theta,
                rear_point.curvature * math.cos(theta),
                rear_point.v,
                rear_point.a,
                rear_point.j,
                rear_point.t
                )

    @staticmethod
    def change_point_front2rear(reference_rear_point: TrajectoryPoint2D,
                                reference_front_point: TrajectoryPoint2D,
                                front_point: TrajectoryPoint2D) -> TrajectoryPoint2D:
        direction_factor = 1 if reference_rear_point.cos * (front_point.x - reference_front_point.x) + \
                                reference_rear_point.sin * (front_point.y - reference_front_point.y) > 0 else -1
        mid_point = Point2D(0.5 * (front_point.x + reference_front_point.x),
                            0.5 * (front_point.y + reference_front_point.y))
        normal_vector = Vector2D(reference_front_point.y - front_point.y,
                                 front_point.x - reference_front_point.x)
        projection = reference_rear_point.cos * normal_vector.x + reference_rear_point.sin * normal_vector.y
        if abs(projection) < 1e-6:
            delta_s = direction_factor * front_point.distance_to(reference_front_point)
            return TrajectoryPoint2D(reference_rear_point.s + delta_s,
                                     reference_rear_point.x + delta_s * reference_rear_point.cos,
                                     reference_rear_point.y + delta_s * reference_rear_point.sin,
                                     reference_rear_point.yaw,
                                     0.0,
                                     reference_rear_point.v,
                                     reference_rear_point.a,
                                     reference_rear_point.j,
                                     reference_rear_point.t)
        else:
            off_length = -(reference_rear_point.cos * (mid_point.x - reference_rear_point.x) + \
                           reference_rear_point.sin * (mid_point.y - reference_rear_point.y)) / projection
            center_of_gyration = Point2D(mid_point.x + off_length * normal_vector.x,
                                         mid_point.y + off_length * normal_vector.y)
            radius_of_gyration = -reference_rear_point.sin * (center_of_gyration.x - reference_rear_point.x) + \
                                  reference_rear_point.cos * (center_of_gyration.y - reference_rear_point.y)
            delta_theta = 2.0 * math.asin(front_point.distance_to(mid_point) / \
                                          front_point.distance_to(center_of_gyration))
            if radius_of_gyration < 0.0:
                delta_theta *= -1.0
            delta_theta *= direction_factor
            return TrajectoryPoint2D(reference_rear_point.s + radius_of_gyration * delta_theta,
                                     reference_rear_point.x + radius_of_gyration * \
                                     (math.sin(reference_rear_point.yaw + delta_theta) - reference_rear_point.sin),
                                     reference_rear_point.y + radius_of_gyration * \
                                     (reference_rear_point.cos - math.cos(reference_rear_point.yaw + delta_theta)),
                                     reference_rear_point.yaw + delta_theta,
                                     1.0 / radius_of_gyration,
                                     reference_rear_point.v,
                                     reference_rear_point.a,
                                     reference_rear_point.j,
                                     reference_rear_point.t)

    def change_front2rear(self, front_trajectory: Trajectory2D,
                          reference_rear_point: TrajectoryPoint2D) -> Trajectory2D:
        if len(front_trajectory.data) == 0:
            return Trajectory2D([])
        rear_trajectory = Trajectory2D([reference_rear_point])
        for i in range(1, len(front_trajectory.data)):
            reference_front_point = self.change_point_rear2front(rear_trajectory.data[- 1])
            rear_trajectory.data.append(self.change_point_front2rear(rear_trajectory.data[i - 1], reference_front_point,
                                        front_trajectory.data[i]))
        return rear_trajectory
