from ..pose.pose import *
from ..vector.vector import *
from ..common.common import *


class Segment2D:
    def __init__(self, start_point: Point2D, end_point: Point2D) -> None:
        self.start_point = start_point
        self.end_point = end_point
        self.length = self.start_point.distance_to(self.end_point)
        self.unit_direction: Vector2D = Vector2D.from_points(start_point, end_point)
        self.unit_direction.normalize_self()
        self.heading_angle = self.unit_direction.angle()
        self.x_min = min(start_point.x, end_point.x)
        self.x_max = max(start_point.x, end_point.x)
        self.y_min = min(start_point.y, end_point.y)
        self.y_max = max(start_point.y, end_point.y)

    def distance_to_point(self, point):
        if self.length <= GEOMETRY_EPSILON:
            return self.start_point.distance_to(point)
        vector_sq = Vector2D.from_points(self.start_point, point)
        projection = vector_sq.inner_product(self.unit_direction)
        if projection <= 0.0:
            return vector_sq.norm()
        elif projection >= self.length:
            return self.end_point.distance_to(point)
        return abs(vector_sq.cross_product_to(self.unit_direction))
