import sys
from typing import Tuple

from .curve_fitting import *
from .front_axle_planner import *
from ...geometry_lib.line.line import *
from ...geometry_lib.box.box import *
from ...geometry_lib.trajectory.trajectory import *
from ...geometry_lib.vector.vector import *


class LatticePlannerConfig:
    def __init__(self, delta_s=0.0, plan_distance=0.0,
                 longitudinal_sampling_point_number=0, longitudinal_sampling_interval=0.0,
                 transverse_sampling_point_number=0, transverse_sampling_interval=0.0,
                 max_curvature=0.0, precision_weight=0.0, smooth_weight=0.0,
                 obstacle_weight=0.0, end_point_weight=0.0, distance_to_front=0.0,
                 distance_to_rear=0.0, distance_to_left=0.0, distance_to_right=0.0):
        self.delta_s = delta_s
        self.plan_distance = plan_distance
        self.longitudinal_sampling_point_number = longitudinal_sampling_point_number
        self.longitudinal_sampling_interval = longitudinal_sampling_interval
        self.transverse_sampling_point_number = transverse_sampling_point_number
        self.transverse_sampling_interval = transverse_sampling_interval
        self.max_curvature = max_curvature
        self.precision_weight = precision_weight
        self.smooth_weight = smooth_weight
        self.obstacle_weight = obstacle_weight
        self.end_point_weight = end_point_weight
        self.distance_to_front = distance_to_front
        self.distance_to_rear = distance_to_rear
        self.distance_to_left = distance_to_left
        self.distance_to_right = distance_to_right


class LatticeState(Pose2D):
    def __init__(self, x=0.0, y=0.0, yaw=0.0, curvature=0.0, v=0.0):
        super().__init__(x, y, yaw)
        self.curvature = curvature
        self.v = v


class LatticePlanner:
    def __init__(self, config: LatticePlannerConfig) -> None:
        self.config = config
        self.front_axle_planner = FrontAxlePlanner(self.config.plan_distance)
        self.safe_box = Box2D.from_vehicle_box(Pose2D(), self.config.distance_to_front, self.config.distance_to_rear,
                                                         self.config.distance_to_left, self.config.distance_to_right)
        self._obstacles: List[Box2D] = []
        self._left_bound: Line2D = Line2D([])
        self._right_bound: Line2D = Line2D([])
        self.start_state = LatticeState()
        self.reference_line = Trajectory2D([])
        self.save_targets: List[TrajectoryPoint2D] = []
        self.offset_reference_line: List[Trajectory2D] = []
        self.sample_targets: List[TrajectoryPoint2D] = []
        self.sample_paths: List[Trajectory2D] = []
        self.valid_length: List[float] = []
        self.min_distance_to_obstacle: List[float] = []

    def init(self, config: LatticePlannerConfig) -> None:
        self.config = config
        self.front_axle_planner = FrontAxlePlanner(self.config.plan_distance)
        self.safe_box = Box2D.from_vehicle_box(Pose2D(), self.config.distance_to_front, self.config.distance_to_rear,
                                               self.config.distance_to_left, self.config.distance_to_right)

    @property
    def obstacles(self) -> List[Box2D]:
        return self._obstacles

    def set_obstacles(self, obstacles: List[Box2D]) -> None:
        self._obstacles = obstacles

    @property
    def left_bound(self) -> Line2D:
        return self._left_bound

    def set_left_bound(self, left_bound: Line2D) -> None:
        self._left_bound = left_bound

    @property
    def right_bound(self) -> Line2D:
        return self._right_bound

    def set_right_bound(self, right_bound: Line2D) -> None:
        self._right_bound = right_bound

    def calculate_offset_reference_line(self) -> None:
        self.offset_reference_line = [Trajectory2D([TrajectoryPoint2D() \
                                                    for _ in range(len(self.reference_line.data))]) \
                                                    for __ in range(self.config.transverse_sampling_point_number * 2 + 1)]
        for i in range(len(self.reference_line.data)):
            base_point = self.reference_line.data[i]
            for j in range(-self.config.transverse_sampling_point_number,
                           self.config.transverse_sampling_point_number + 1):
                offset_length = j * self.config.transverse_sampling_interval
                normal_vector = Vector2D.from_points(self.reference_line.data[i - 1], self.reference_line.data[i])\
                                .normal_vector() if i == len(self.reference_line.data) - 1 else \
                                Vector2D.from_points(self.reference_line.data[i], self.reference_line.data[i + 1])\
                                .normal_vector()
                index = self.config.transverse_sampling_point_number - j
                self.offset_reference_line[index].data[i].x = base_point.x + offset_length * normal_vector.x
                self.offset_reference_line[index].data[i].y = base_point.y + offset_length * normal_vector.y
                self.offset_reference_line[index].data[i].set_yaw(base_point.yaw)
                self.offset_reference_line[index].data[i].curvature = base_point.curvature if \
                    abs(base_point.curvature) < 1e-3 else 1.0 / ((1.0 / base_point.curvature) - offset_length)
                self.offset_reference_line[index].data[i].s = base_point.s if i == 0 else \
                    self.offset_reference_line[index].data[i - 1].s + \
                    self.offset_reference_line[index].data[i - 1].\
                        distance_to(self.offset_reference_line[index].data[i])
                self.offset_reference_line[index].data[i].v = base_point.v
                self.offset_reference_line[index].data[i].a = base_point.a
                self.offset_reference_line[index].data[i].j = base_point.j
                self.offset_reference_line[index].data[i].t = base_point.t

    def calculate_sample_target(self) -> None:
        minimum_s = (self.start_state.v - 1.0) * 5.0
        for i in range(self.config.longitudinal_sampling_point_number):
            base_s = self.config.longitudinal_sampling_interval * 2 ** i
            if base_s < minimum_s:
                base_s = minimum_s
            index = 0
            while index < len(self.reference_line.data):
                if self.reference_line.data[index].s >= base_s:
                    break
                index += 1
            for j in range(-self.config.transverse_sampling_point_number,
                           self.config.transverse_sampling_point_number + 1):
                self.sample_targets.append(self.offset_reference_line\
                                           [self.config.transverse_sampling_point_number - j].data[index])
            for target_point in self.save_targets:
                target_projection = self.reference_line.projection(target_point)
                if target_projection.s >= base_s:
                    self.sample_targets.append(target_point)

    def calculate_sample_path(self) -> bool:
        max_size = 0
        start_point = self.front_axle_planner.\
            change_point_rear2front(TrajectoryPoint2D(0.0, 0.0, 0.0,0.0, self.start_state.curvature,
                                                      0.0, 0.0, 0.0, 0.0))
        front_paths = []
        for sample_target in self.sample_targets:
            front_target = self.front_axle_planner.change_point_rear2front(sample_target)
            front_path = path_fit_quintic(start_point.x, start_point.y, start_point.yaw, start_point.curvature,
                                          front_target.x, front_target.y, front_target.yaw, front_target.curvature).\
                         discrete_to_trajectory_dx(start_point.x, front_target.x, self.config.delta_s)
            if len(front_path.data) == 0:
                # print("Failed！")
                return False
            front_paths.append(front_path)
            max_size = max(max_size, len(front_path.data))
        for i in range(self.config.longitudinal_sampling_point_number):
            for j in range(2 * self.config.transverse_sampling_point_number + 1):
                index = j + i * (2 * self.config.transverse_sampling_point_number + 1)
                while len(front_paths[index].data) < max_size:
                    extend_point = self.offset_reference_line[j].\
                                        interpolation_by_x(front_paths[index].data[-1].x + self.config.delta_s)
                    extend_point.s = front_paths[index].data[-1].s + \
                                     extend_point.distance_to(front_paths[index].data[-1])
                    front_paths[index].data.append(extend_point)
        for front_path in front_paths:
            self.sample_paths.append(self.front_axle_planner.\
                                     change_front2rear(front_path, TrajectoryPoint2D(0.0, 0.0, 0.0,0.0,
                                                                                     self.start_state.curvature,
                                                                                     0.0, 0.0, 0.0, -1.0)))
        return True
        # self.sample_paths = front_paths

    def calculate_valid_path_length(self):
        for path in self.sample_paths:
            if len(path.data) == 0:
                self.valid_length.append(0.0)
                self.min_distance_to_obstacle.append(0.0)
                continue
            valid_length = path.data[-1].s
            min_distance_to_obstacle = sys.float_info.max
            left_index = 0
            right_index = 0
            for i in range(1, len(path.data)):
                if path.data[i].curvature > self.config.max_curvature:
                    valid_length = min(valid_length, path.data[i - 1].s)
                    break
                vehicle_box = self.safe_box.transform_from(path.data[i])
                # vehicle_box = Box2D(path.data[i], 13, 2.55)
                for obstacle in self._obstacles:
                    if vehicle_box.is_collision_to_box(obstacle):
                        valid_length = min(valid_length, path.data[i - 1].s)
                        break
                    min_distance_to_obstacle = min(min_distance_to_obstacle, vehicle_box.distance_to_box(obstacle))
                for j in range(left_index, len(self._left_bound.data) - 1):
                    if self._left_bound.data[j + 1].x < vehicle_box.x_min:
                        left_index += 1
                        continue
                    if vehicle_box.is_collision_to_segment(Segment2D(self._left_bound.data[j],
                                                                     self._left_bound.data[j + 1])):
                        valid_length = min(valid_length, path.data[i - 1].s)
                        break
                for j in range(right_index, len(self._right_bound.data) - 1):
                    if self._right_bound.data[j + 1].x < vehicle_box.x_min:
                        right_index += 1
                        continue
                    if vehicle_box.is_collision_to_segment(Segment2D(self._right_bound.data[j],
                                                                     self._right_bound.data[j + 1])):
                        valid_length = min(valid_length, path.data[i - 1].s)
                        break
            self.valid_length.append(valid_length)
            self.min_distance_to_obstacle.append(min_distance_to_obstacle)

    def get_optimal_path(self) -> Trajectory2D:
        min_cost = sys.float_info.max
        best_index = -1
        for i in range(len(self.sample_paths)):
            cost = 0.0
            j = 0
            while j < len(self.sample_paths[i].data):
                if self.sample_paths[i].data[j].s > self.valid_length[i]:
                    break
                cost += self.config.smooth_weight * abs(self.sample_paths[i].data[j].curvature)
                projection_point = self.reference_line.projection(self.sample_paths[i].data[j])
                cost += self.config.precision_weight * projection_point.distance_to(self.sample_paths[i].data[j])
                j += 1
            projection_point = self.reference_line.projection(self.sample_paths[i].data[-1])
            cost += self.config.end_point_weight * self.config.precision_weight * \
                    abs(projection_point.distance_to(self.sample_paths[i].data[-1]))
            if self.min_distance_to_obstacle[i] != sys.float_info.max:
                cost += self.config.obstacle_weight / (self.min_distance_to_obstacle[i] + 0.01)
            if j != len(self.sample_paths[i].data):
                cost += 10000.0
            if j>0:
                cost /= j
            cost -= 0.02 * j
            if cost < min_cost:
                min_cost = cost
                best_index = i
        return self.sample_paths[best_index]

    def plan(self, start_state: LatticeState, reference_line: Trajectory2D)->Tuple[int,Trajectory2D]:
        if len(reference_line.data) == 0:
            return -1, Trajectory2D([])
        reference_line.transform_self_to(start_state)
        self.reference_line.data.clear()
        self.reference_line.data.append(reference_line.interpolation_by_x(0.0))
        self.reference_line.data[-1].s = 0
        while self.reference_line.data[-1].s < self.config.longitudinal_sampling_interval * \
                  2 ** (self.config.longitudinal_sampling_point_number - 1) - \
                  self.config.delta_s + self.config.plan_distance:
            extend_point = reference_line.interpolation_by_x(self.reference_line.data[-1].x + \
                                                             self.config.delta_s)
            extend_point.s = self.reference_line.data[-1].x + \
                             extend_point.distance_to(self.reference_line.data[-1])
            self.reference_line.data.append(extend_point)
        # print(self.reference_line.data)
        for trajectory_point in self.save_targets:
            trajectory_point.transform_self_to(start_state)
        for obstacle in self._obstacles:
            obstacle.transform_self_to(start_state)
        self._left_bound.transform_self_to(start_state)
        self._right_bound.transform_self_to(start_state)
        self.calculate_offset_reference_line()
        self.calculate_sample_target()
        if not self.calculate_sample_path():
            print("失败")
            return -1,reference_line.transform_from(start_state)
        self.calculate_valid_path_length()
        optimal_path = self.get_optimal_path()
        for obstacle in self._obstacles:
            obstacle.transform_self_from(start_state)
        self._left_bound.transform_self_from(start_state)
        self._right_bound.transform_self_from(start_state)
        optimal_path.transform_self_from(start_state)
        return 1, optimal_path
