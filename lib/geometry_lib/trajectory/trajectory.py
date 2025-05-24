from ..trajectory_point.trajectory_point import *
from ..vector.vector import *
from ..box.box import Box2D
from ..line.line import Line2D

from typing import List
import copy


class Trajectory2D:
    def __init__(self, data: List[TrajectoryPoint2D], time_stamp=0.0,
                 is_forward: bool = True, has_end_point: bool = False) -> None:
        self.data = data
        self.time_stamp = time_stamp
        self.is_forward = is_forward
        self.has_end_point = has_end_point

    @classmethod
    def from_points(cls, points: List[Point2D], time_stamp = 0.0,
                    is_forward: bool = True, has_end_point: bool = False):
        data = []
        s = 0.0
        if (len(points) == 0):
            data = []
        else:
            data.append(TrajectoryPoint2D(s = s, 
                                          x = points[0].x, 
                                          y = points[0].y))
            if (len(points) > 1):
                data[0].set_yaw(math.atan2(points[1].y -  points[0].y, 
                                           points[1].x - points[0].x))
                # print("origin yaw:= ")
                # print(math.atan2(points[1].y -  points[0].y, 
                #                  points[1].x - points[0].x))
                
                for i in range(1, len(points)):
                    s+=points[i].distance_to(points[i - 1])
                    data.append(TrajectoryPoint2D(s = s, 
                                                x = points[i].x, 
                                                y = points[i].y,
                                                yaw = math.atan2(points[i].y -  points[i - 1].y, 
                                                                points[i].x - points[i - 1].x)))        
        return cls(data, time_stamp, is_forward, has_end_point)

    def __repr__(self):
        return (f"time_stamp:= {self.time_stamp}, is_forward:= {self.is_forward}, has_end_point:= {self.has_end_point}\n"
                f"data:= \n{self.data}")

    def interpolation_by_s(self, s)->TrajectoryPoint2D:
        if len(self.data) == 0:
            return TrajectoryPoint2D()
        elif len(self.data) == 1:
            return self.data[0]
        if s < self.data[0].s:
            return TrajectoryPoint2D(s, self.data[0].x + (s - self.data[0].s) * self.data[0].cos,
                                     self.data[0].y + (s - self.data[0].s) * self.data[0].sin, self.data[0].yaw,
                                     0.0, self.data[0].v, 0.0, 0.0, 0.0)
        elif s > self.data[-1].s:
            return TrajectoryPoint2D(s, self.data[-1].x + (s - self.data[-1].s) * self.data[-1].cos,
                                     self.data[-1].y + (s - self.data[-1].s) * self.data[-1].sin, self.data[-1].yaw,
                                     0.0, self.data[0].v, 0.0, 0.0, 0.0)
        for i in range(1, len(self.data)):
            if (s - self.data[i - 1].s) * (self.data[i].s - s) >= 0.0:
                factor = (s - self.data[i - 1].s) / (self.data[i].s - self.data[i - 1].s)
                return self.data[i - 1].interpolation(self.data[i], factor)
        return TrajectoryPoint2D()

    def interpolation_by_x(self, x):
        if len(self.data) == 0:
            return TrajectoryPoint2D()
        elif len(self.data) == 1:
            return self.data[0]
        if x < self.data[0].x:
            return TrajectoryPoint2D(self.data[0].s + (x - self.data[0].x) / self.data[0].cos, x,
                                     self.data[0].y + (x - self.data[0].x) * math.tan(self.data[0].yaw),
                                     self.data[0].yaw, 0.0, self.data[0].v, 0.0, 0.0, -1.0)
        elif x > self.data[-1].x:
            return TrajectoryPoint2D(self.data[-1].s + (x - self.data[-1].x) / self.data[-1].cos, x,
                                     self.data[-1].y + (x - self.data[-1].x) * math.tan(self.data[-1].yaw),
                                     self.data[-1].yaw, 0.0, self.data[-1].v, 0.0, 0.0, -1.0)
        for i in range(1, len(self.data)):
            if (x - self.data[i - 1].x) * (self.data[i].x - x) >= 0.0:
                factor = (x - self.data[i - 1].x) / (self.data[i].x - self.data[i - 1].x)
                return self.data[i - 1].interpolation(self.data[i], factor)
        return TrajectoryPoint2D()

    def projection(self, point: Point2D) -> TrajectoryPoint2D:
        if len(self.data) == 0:
            return TrajectoryPoint2D()
        direction_factor = 1 if self.is_forward else -1
        start_point = self.data[0]
        point_before_start_point = copy.deepcopy(start_point)
        point_before_start_point.s -= 1.0
        point_before_start_point.x -= direction_factor * start_point.cos
        point_before_start_point.y -= direction_factor * start_point.sin
        vector_bsp = Vector2D.from_points(point_before_start_point, point)
        vector_bss = Vector2D.from_points(point_before_start_point, start_point)
        last_factor = vector_bsp.projection(vector_bss)
        if last_factor < 1.0:
            result = point_before_start_point.interpolation(start_point, last_factor)
            result.curvature = 0.0
            result.a = 0.0
            result.j = 0.0
            return result
        
        for i in range(len(self.data)-1):
            vector_sq = Vector2D.from_points(self.data[i], point)
            vector_se = Vector2D.from_points(self.data[i], self.data[i + 1])
            now_factor = vector_sq.projection(vector_se) / vector_se.norm()
            if 0.0 <= now_factor < 1.0:
                return self.data[i].interpolation(self.data[i + 1], now_factor)
            elif now_factor < 0.0 and last_factor >= 1.0:
                return self.data[i]
            last_factor = copy.deepcopy(now_factor)
        end_point = self.data[-1]
        point_after_end_point = copy.deepcopy(self.data[-1])
        point_after_end_point.s += 1.0
        point_after_end_point.x += direction_factor * end_point.cos
        point_after_end_point.y += direction_factor * end_point.sin
        vector_ep = Vector2D.from_points(end_point, point)
        vector_eae = Vector2D.from_points(end_point, point_after_end_point)
        now_factor = vector_ep.projection(vector_eae) / vector_eae.norm();
        if now_factor >= 0.0:
            result = end_point.interpolation(point_after_end_point, now_factor)
            result.curvature = 0.0
            result.a = 0.0
            result.j = 0.0
            return result
        elif now_factor < 0.0 and last_factor >= 1.0:
            return end_point
        return TrajectoryPoint2D()

    def append(self, trajectory: "Trajectory2D"):
        if len(self.data) == 0:
            self.data = trajectory.data
            self.is_forward = trajectory.is_forward
            self.has_end_point = trajectory.has_end_point
        if len(trajectory.data) < 2:
            return
        point_source = self.data[-1]
        point_target = trajectory.data[0]
        for i in range(1, len(trajectory.data)):
            self.data.append(trajectory.data[i].transform_to(point_source))
            self.data[-1].s += point_target.s - point_source.s
            self.data[-1].t += point_target.t - point_source.t

    def transform_to(self, base_pose: Pose2D) -> "Trajectory2D":
        result = Trajectory2D(self.data, self.time_stamp, self.is_forward, self.has_end_point)
        for trajectory_point in result.data:
            trajectory_point.transform_self_to(base_pose)
        return result

    def transform_self_to(self, base_pose: Pose2D) -> None:
        for trajectory_point in self.data:
            trajectory_point.transform_self_to(base_pose)

    def transform_from(self, base_pose: Pose2D) -> "Trajectory2D":
        result = Trajectory2D(self.data, self.time_stamp, self.is_forward, self.has_end_point)
        for trajectory_point in result.data:
            trajectory_point.transform_self_from(base_pose)
        return result

    def transform_self_from(self, base_pose: Pose2D) -> None:
        for trajectory_point in self.data:
            trajectory_point.transform_self_from(base_pose)

    def envelope_collition_check(self, length_front:float, length_rear:float, width:float, around:List[Box2D]) -> bool:
        for trajectory_point in self.data:
            for ego_box in around:
                if Box2D.from_vehicle_box(Pose2D(trajectory_point.x, trajectory_point.y, trajectory_point.yaw), length_front, length_rear, width*0.5,width*0.5).is_collision_to_box(ego_box):
                    return True
        return False

    # === 点在多边形内判定（Ray-Casting） ===
    @staticmethod
    def _point_in_poly(pt: "Point2D", poly: List["Point2D"]) -> bool:
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i].x, poly[i].y
            xj, yj = poly[j].x, poly[j].y
            intersect = ((yi > pt.y) != (yj > pt.y)) and \
                        (pt.x < (xj - xi) * (pt.y - yi) /
                         (yj - yi + 1e-12) + xi)
            if intersect:
                inside = not inside
            j = i
        return inside

    def envelope_in_range(self, length_front:float, length_rear:float, width:float, left_bound:Line2D, right_bound:Line2D) -> bool:
        """
        判断轨迹上所有位置的车辆矩形是否始终位于左右边界之间
        :param length_front: 车前保险杠到质心距离 (m)
        :param length_rear:  车后保险杠到质心距离
        :param width:        车辆宽 (m)
        :param left_bound:   左边界折线
        :param right_bound:  右边界折线
        :return:             True → 全部在界内；False → 任一点越界
        """
        if not self.data or not left_bound.data or not right_bound.data:
            print(f"路径数据长度{len(self.data)}，左右边界数据长度{len(left_bound.data)}/{len(right_bound.data)}")
            return False

        # 1. 生成道路多边形（确保顺时针 / 逆时针均可）
        road_poly: List[Point2D] = []
        road_poly.extend(left_bound.data)
        road_poly.extend(reversed(right_bound.data))

        # 2. 遍历轨迹点，检查四个角是否落在多边形内
        half_w = width * 0.5
        for tp in self.data:
            box = Box2D.from_vehicle_box(Pose2D(tp.x, tp.y, tp.yaw), length_front, length_rear, width*0.5, width*0.5)
            for corner_point in box.corners:
                if not self._point_in_poly(corner_point, road_poly):
                    return False   # 任一角出界 ⇒ 整条轨迹不合法

        return True  # 全部角点都在界内 ⇒ 轨迹合法

