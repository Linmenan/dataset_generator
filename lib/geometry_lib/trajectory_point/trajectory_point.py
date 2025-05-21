from ..pose.pose import *
from ..common.common import *


class TrajectoryPoint2D(Pose2D):
    def __init__(self, s=0.0, x=0.0, y=0.0, yaw=0.0, curvature=0.0, v=0.0, a=0.0, j=0.0, t=-1.0):
        super().__init__(x, y, yaw)
        self.s = s
        self.curvature = curvature
        self.v = v
        self.a = a
        self.j = j
        self.t = t

    def __repr__(self):
        return (f"s:= {self.s}, x:= {self.x}, y:= {self.y}, yaw:= {self.yaw}, curvature:= {self.curvature}, "
                f"v:= {self.v}, a:= {self.a}, j:= {self.j}, t:= {self.t}\n")

    @classmethod
    def from_speed_point(cls, s=0.0, v=0.0, a=0.0, j=0.0, t=0.0):
        return cls(s=s, v=v, a=a, j=j, t=t)

    def interpolation(self, next_trajectory_point: "TrajectoryPoint2D", factor):
        return TrajectoryPoint2D(self.s + factor * (next_trajectory_point.s - self.s),
                                 self.x + factor * (next_trajectory_point.x - self.x),
                                 self.y + factor * (next_trajectory_point.y - self.y),
                                 self.yaw + factor * angle_normalize(next_trajectory_point.yaw - self.yaw),
                                 self.curvature + factor * (next_trajectory_point.curvature - self.curvature),
                                 self.v + factor * (next_trajectory_point.v - self.v),
                                 self.a + factor * (next_trajectory_point.a - self.a),
                                 self.j + factor * (next_trajectory_point.j - self.j),
                                 self.t + factor * (next_trajectory_point.t - self.t))

    def transform_to(self, base_pose: Pose2D) -> "TrajectoryPoint2D":
        local_pose = Pose2D(self.x, self.y, self.yaw).transform_to(base_pose)
        return TrajectoryPoint2D(self.s, local_pose.x, local_pose.y, local_pose.yaw,
                                 self.curvature, self.v, self.a, self.j, self.t)

    def transform_self_to(self, base_pose: Pose2D) -> "None":
        super().transform_self_to(base_pose)

    def transform_from(self, base_pose: Pose2D) -> "TrajectoryPoint2D":
        local_pose = Pose2D(self.x, self.y, self.yaw).transform_from(base_pose)
        return TrajectoryPoint2D(self.s, local_pose.x, local_pose.y, local_pose.yaw,
                                 self.curvature, self.v, self.a, self.j, self.t)

    def transform_self_from(self, base_pose: Pose2D) -> "None":
        super().transform_self_from(base_pose)
