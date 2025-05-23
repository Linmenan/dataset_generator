import math
import numpy as np
from typing import List, Tuple, Optional

from lib.geometry_lib.point.point import Point2D
from lib.geometry_lib.pose.pose import Pose2D
from lib.geometry_lib.box.box import Box2D
from lib.geometry_lib.line.line import Line2D
from lib.geometry_lib.trajectory.trajectory import Trajectory2D
from lib.geometry_lib.trajectory_point.trajectory_point import TrajectoryPoint2D
from src.utils.common import normalize_angle
from .quintic_fitting import quintic_coeffs, d_eval

class FrenetLattice:
    """
    - lateral_sample_num_per_side:   当前车道中心线左右各采样 N 条 (不含 0)
    - lateral_sample_step:           相邻采样间隔 (m)
        → 总偏移集合大小 = 2*N + 1  (对称 + 中心)
    """
    def __init__(self,
                 lateral_sample_num_per_side: int = 5,
                 lateral_sample_step: float = 0.10,
                 delta_s: float = 1.0,
                 horizon_s: float = 30.0,
                 max_curvature: float = 0.1):
        self.n_lat = max(1, lateral_sample_num_per_side)
        self.lat_step = max(1e-3, lateral_sample_step)
        self.ds = delta_s
        self.horizon = horizon_s
        self.kappa_max = max_curvature

    # ---------- 对外主入口 ----------
    def plan(self,
             ego_pose: Pose2D,
             ego_kappa: float,
             length_front: float,
             length_rear:  float,
             width:        float,
             left_bound:   Line2D,
             right_bound:  Line2D,
             ref_traj:     Trajectory2D,
             obstacles:    List[Box2D]) -> Tuple[int,Trajectory2D]:

        # 1. 计算当前 Frenet 状态
        proj = ref_traj.projection(ego_pose)
        s0 = proj.s
        dx, dy = ego_pose.x - proj.x, ego_pose.y - proj.y
        d0 = math.cos(proj.yaw) * dy - math.sin(proj.yaw) * dx
        dd0 = (ego_kappa - proj.curvature) * (1 + proj.curvature * d0)
        d_dot0 = 0.0

        # 2. 剪裁参考线：仅保留 [s0, s0 + horizon]
        ref_clipped = self._clip_ref_traj(ref_traj, s0, s0 + self.horizon)

        # 3. 生成横向偏移集合 [-N*step, …, 0, …, +N*step]
        lat_offsets = [
            i * self.lat_step
            for i in range(-self.n_lat, self.n_lat + 1)
        ]  # 递增排序，0 在中间

        best_traj, best_cost = ref_traj, float("inf")

        for d_f in lat_offsets:
            coeffs = quintic_coeffs(
                d0, d_dot0, dd0, d_f, 0.0, 0.0, self.horizon
            )

            points: List[TrajectoryPoint2D] = []
            feas = True
            abs_d_sum = 0.0

            for i in range(int(self.horizon / self.ds) + 1):
                si = s0 + i * self.ds
                di = d_eval(coeffs, si - s0, 0)
                d_prime = d_eval(coeffs, si - s0, 1)
                d_pp = d_eval(coeffs, si - s0, 2)

                xi, yi, yawi = self._frenet_to_cartesian(
                    ref_clipped, si, di, d_prime
                )
                tref = ref_clipped.interpolation_by_s(si)
                kappa = (
                    (tref.curvature * math.cos(math.atan(d_prime)) + d_pp)
                    / (1 - tref.curvature * di)
                )
                if abs(kappa) > self.kappa_max:
                    feas = False
                    break

                points.append(
                    TrajectoryPoint2D(
                        s=si, x=xi, y=yi, yaw=yawi, curvature=kappa
                    )
                )
                abs_d_sum += abs(di)

            if not feas:
                continue

            cand = Trajectory2D(points, is_forward=True)

            # 4. 约束：碰撞 & 邻域
            if (
                cand.envelope_collition_check(
                    length_front, length_rear, width, around=obstacles
                )
                or cand.envelope_in_range(
                    length_front,
                    length_rear,
                    width,
                    left_bound,
                    right_bound,
                )
            ):
                continue

            # 5. 代价：|d_f| + 曲率 + 横向误差均值
            mean_abs_d = abs_d_sum / len(points)
            cost = abs(d_f) * 1.0 + self._integral_curv(points) * 0.5 + mean_abs_d

            if cost < best_cost:
                best_cost, best_traj = cost, cand

        return (1,best_traj)  # 可能为 None

    # ---------- 内部工具 ----------
    @staticmethod
    def _integral_curv(pts: List[TrajectoryPoint2D]) -> float:
        return sum(abs(p.curvature) for p in pts)

    @staticmethod
    def _clip_ref_traj(
        ref: Trajectory2D, s_start: float, s_end: float
    ) -> Trajectory2D:
        """返回 s∈[s_start,s_end] 的子轨迹（含端点插值）"""
        clipped: List[TrajectoryPoint2D] = []  
        cur_s = s_start
        while cur_s < s_end:
            clipped.append(ref.interpolation_by_s(cur_s))
            cur_s += 0.5
        return Trajectory2D(clipped, is_forward=ref.is_forward)

    @staticmethod
    def _frenet_to_cartesian(
        ref: Trajectory2D, s: float, d: float, d_prime: float
    ) -> Tuple[float, float, float]:
        tp = ref.interpolation_by_s(s)
        nx, ny = -math.sin(tp.yaw), math.cos(tp.yaw)
        x, y = tp.x + d * nx, tp.y + d * ny
        denom = max(1e-6, 1.0 - tp.curvature * d)
        yaw = tp.yaw + math.atan2(d_prime, denom)
        return x, y, normalize_angle(yaw)