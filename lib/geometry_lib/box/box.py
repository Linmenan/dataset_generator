import sys

from ..segment.segment import *
from ..common.common import *


class Box2D(Pose2D):
    def __init__(self, center_pose: Pose2D, length=0.0, width=0.0) -> None:
        super().__init__(center_pose.x, center_pose.y, center_pose.yaw)
        self.length = length
        self.width = width
        self.half_length = 0.5 * self.length
        self.half_width = 0.5 * self.width
        self.dx_length = self.cos * self.half_length
        self.dx_width = -self.sin * self.half_width
        self.dy_length = self.sin * self.half_length
        self.dy_width = self.cos * self.half_width
        self.x_min = self.x - abs(self.dx_length) - abs(self.dx_width)
        self.x_max = self.x + abs(self.dx_length) + abs(self.dx_width)
        self.y_min = self.y - abs(self.dy_length) - abs(self.dy_width)
        self.y_max = self.y + abs(self.dy_length) + abs(self.dy_width)
        self.corners = [
                        Point2D(self.x + self.dx_length + self.dx_width, self.y + self.dy_length + self.dy_width),
                        Point2D(self.x - self.dx_length + self.dx_width, self.y - self.dy_length + self.dy_width),
                        Point2D(self.x - self.dx_length - self.dx_width, self.y - self.dy_length - self.dy_width),
                        Point2D(self.x + self.dx_length - self.dx_width, self.y + self.dy_length - self.dy_width)
                        ]

    @classmethod
    def from_vehicle_box(cls, rear_center_pose: Pose2D, distance_to_front=0.0, distance_to_rear=0.0,
                         distance_to_left=0.0, distance_to_right=0.0):
        local_center_pose = Pose2D(0.5 * (distance_to_front - distance_to_rear),
                                   0.5 * (distance_to_left - distance_to_right),
                                   0.0 if distance_to_front + distance_to_rear > 0.0 else math.pi)
        return cls(local_center_pose.transform_from(rear_center_pose), distance_to_front + distance_to_rear,
                   distance_to_left + distance_to_right)

    def __repr__(self):
        return f"x:= {self.x}, y:= {self.y}, yaw:= {self.yaw}, length = {self.length}, width= {self.width}"

    def reset(self) -> None:
        self.half_length = 0.5 * self.length
        self.half_width = 0.5 * self.width
        self.dx_length = self.cos * self.half_length
        self.dx_width = -self.sin * self.half_width
        self.dy_length = self.sin * self.half_length
        self.dy_width = self.cos * self.half_width
        self.x_min = self.x - abs(self.dx_length) - abs(self.dx_width)
        self.x_max = self.x + abs(self.dx_length) + abs(self.dx_width)
        self.y_min = self.y - abs(self.dy_length) - abs(self.dy_width)
        self.y_max = self.y + abs(self.dy_length) + abs(self.dy_width)
        self.corners = [
            Point2D(self.x + self.dx_length + self.dx_width, self.y + self.dy_length + self.dy_width),
            Point2D(self.x - self.dx_length + self.dx_width, self.y - self.dy_length + self.dy_width),
            Point2D(self.x - self.dx_length - self.dx_width, self.y - self.dy_length - self.dy_width),
            Point2D(self.x + self.dx_length - self.dx_width, self.y + self.dy_length - self.dy_width)
        ]

    def distance_to_point(self, point: Point2D):
        local_point = point.transform_to(super())
        dx = abs(local_point.x) - self.half_length
        dy = abs(local_point.y) - self.half_width
        if dx <= 0.0:
            return max(0.0, dy)
        if dy <= 0.0:
            return dx
        return math.sqrt(dx * dx + dy * dy)

    def distance_to_segment(self, segment) -> float:
        center_point = Point2D(self.x, self.y)
        vector_cs = Vector2D.from_points(center_point, segment.start_point)
        start_point = Point2D(vector_cs.x * self.cos + vector_cs.y * self.sin,
                              vector_cs.x * self.sin - vector_cs.y * self.cos)
        box_x = self.half_length
        box_y = self.half_width
        gx1 = 1 if start_point.x >= box_x else -1 if start_point.x <= -box_x else 0
        gy1 = 1 if start_point.y >= box_y else -1 if start_point.y <= -box_y else 0
        if gx1 == 0 and gy1 == 0:
            return 0.0
        vector_ce = Vector2D.from_points(center_point, segment.end_point)
        end_point = Point2D(vector_ce.x * self.cos + vector_ce.y * self.sin,
                             vector_ce.x * self.sin - vector_ce.y * self.cos)
        gx2 = 1 if end_point.x >= box_x else -1 if end_point.x <= -box_x else 0
        gy2 = 1 if end_point.y >= box_y else -1 if end_point.y <= -box_y else 0
        if gx2 == 0 and gy2 == 0:
            return 0.0
        if gx1 < 0 or (gx1 == 0 and gx2 < 0):
            start_point.x = -start_point.x
            gx1 = -gx1
            end_point.x = -end_point.x
            gx2 = -gx2
        if gy1 < 0 or (gy1 == 0 and gy2 < 0):
            start_point.y = -start_point.y
            gy1 = -gy1
            end_point.y = -end_point.y
            gy2 = -gy2
        if gx1 < gy1 or (gx1 == gy1 and gx2 < gy2):
            start_point.x, start_point.y = start_point.y, start_point.x
            gx1, gy1 = gy1, gx1
            end_point.x, end_point.y = end_point.y, end_point.x
            gx2, gy2 = gy2, gx2
            box_x, box_y = box_y, box_x
        current_segment = Segment2D(start_point, end_point)
        if gx1 == 1 and gy1 == 1:
            scene = gx2 * 3 + gy2
            if scene == 4:
                return current_segment.distance_to_point(Point2D(box_x, box_y))
            elif scene == 3:
                return end_point.x - box_x if start_point.x > end_point.x else \
                       current_segment.distance_to_point(Point2D(box_x, box_y))
            elif scene == 2:
                return current_segment.distance_to_point(Point2D(box_x, -box_y)) if start_point.x > end_point.x else \
                       current_segment.distance_to_point(Point2D(box_x, box_y))
            elif scene == -1:
                return 0.0 if Vector2D.from_points(start_point, end_point).\
                    cross_product_to(Vector2D.from_points(start_point, Point2D(box_x, -box_y))) >= 0.0 else \
                    current_segment.distance_to_point(Point2D(box_x, -box_y))
            elif scene == -4:
                return current_segment.distance_to_point(Point2D(box_x, box_y)) if \
                                Vector2D.from_points(start_point, end_point).\
                                cross_product_to(Vector2D.from_points(start_point, Point2D(box_x, -box_y))) <= 0.0 \
                                else 0.0 if Vector2D.from_points(start_point, end_point).\
                                cross_product_to(Vector2D.from_points(start_point, Point2D(-box_x, box_y))) <= 0.0 else \
                                current_segment.distance_to_point(Point2D(-box_x, box_y))
        else:
            scene = gx2 * 3 + gy2
            if scene == 4:
                return (start_point.x - box_x) if (start_point.x < end_point.x) else \
                       current_segment.distance_to_point(Point2D(box_x, box_y))
            elif scene == 3:
                return min(start_point.x, end_point.x) - box_x
            elif scene == 1 or scene == -2:
                return 0.0 if Vector2D.from_points(start_point, end_point).\
                       cross_product_to(Vector2D.from_points(start_point, Point2D(box_x, box_y))) <= 0.0 else \
                       current_segment.distance_to_point(Point2D(box_x, box_y))
            elif scene == -3:
                return 0.0
        return 0.0

    def distance_to_box(self, box: "Box2D"):
        if self.is_collision_to_point(Point2D(box.x, box.y)):
            return 0.0
        result = sys.float_info.max
        for i in range(len(self.corners)):
            result = min(result,box.distance_to_segment(Segment2D(self.corners[i],
                                                                  self.corners[(i + 1) % len(self.corners)])))
        return result

    def is_collision_to_point(self, point):
            if point.x < self.x_min or point.x > self.x_max or \
               point.y < self.y_min or point.y > self.y_max:
                return False
            local_point = point.transform_to(Pose2D(self.x, self.y, self.yaw))
            return abs(local_point.x) <= self.half_length + COLLISION_THRESHOLD and \
                   abs(local_point.y) <= self.half_width + COLLISION_THRESHOLD

    def is_collision_to_segment(self, segment: Segment2D) -> bool:
        if segment.x_max < self.x_min or segment.x_min > self.x_max or \
           segment.y_max < self.y_min or segment.y_min > self.y_max:
            return False
        segment_mid_x = 0.5 * (segment.start_point.x + segment.end_point.x)
        segment_mid_y = 0.5 * (segment.start_point.y + segment.end_point.y)
        shift_x = segment_mid_x - self.x
        shift_y = segment_mid_y - self.y
        segment_cos = math.cos(segment.heading_angle)
        segment_sin = math.sin(segment.heading_angle)
        dx_length = 0.5 * segment_cos * segment.length
        dy_length = 0.5 * segment_sin * segment.length
        return abs(shift_x * self.cos + shift_y * self.sin) <= \
               abs(dx_length * self.cos + dy_length * self.sin) + self.half_length and \
               abs(shift_x * self.sin - shift_y * self.cos) <= \
               abs(dx_length * self.sin - dy_length * self.cos) + self.half_width and \
               abs(shift_x * segment_cos + shift_y * segment_sin) <= \
               abs(self.dx_length * segment_cos + self.dy_length * segment_sin) + \
               abs(self.dx_width * segment_cos + self.dy_width * segment_sin) + 0.5 * segment.length and \
               abs(shift_x * segment_sin - shift_y * segment_cos) <= \
               abs(self.dx_length * segment_sin - self.dy_length * segment_cos) + \
               abs(self.dx_width * segment_sin - self.dy_width * segment_cos)

    def is_collision_to_box(self, box: "Box2D") -> bool:
        if box.x_max < self.x_min or box.x_min > self.x_max or box.y_max < self.y_min or box.y_min > self.y_max:
            return False
        shift_x = box.x - self.x
        shift_y = box.y - self.y
        return abs(shift_x * self.cos + shift_y * self.sin) <= \
               abs(box.dx_length * self.cos + box.dy_length * self.sin) + \
               abs(box.dx_width * self.cos + box.dy_width * self.sin) + self.half_length and \
               abs(shift_x * self.sin - shift_y * self.cos) <= \
               abs(box.dx_length * self.sin - box.dy_length * self.cos) + \
               abs(box.dx_width * self.sin - box.dy_width * self.cos) + self.half_width and \
               abs(shift_x * box.cos + shift_y * box.sin) <= \
               abs(self.dx_length * box.cos + self.dy_length * box.sin) + \
               abs(self.dx_width * box.cos + self.dy_width * box.sin) + box.half_length and \
               abs(shift_x * box.sin - shift_y * box.cos) <= \
               abs(self.dx_length * box.sin - self.dy_length * box.cos) + \
               abs(self.dx_width * box.sin - self.dy_width * box.cos) + box.half_width

    def transform_to(self, base_pose: "Pose2D") -> "Box2D":
        return Box2D(super().transform_to(base_pose), self.length, self.width)

    def transform_self_to(self, base_pose: "Pose2D") -> None:
        super().transform_self_to(base_pose)
        self.reset()

    def transform_from(self, base_pose: "Pose2D") -> "Box2D":
        return Box2D(super().transform_from(base_pose), self.length, self.width)

    def transform_self_from(self, base_pose: "Pose2D") -> None:
        super().transform_self_from(base_pose)
        self.reset()
