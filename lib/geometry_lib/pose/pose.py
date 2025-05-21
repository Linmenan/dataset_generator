from ..common.common import *
from ..point.point import Point2D

import math


class Pose2D(Point2D):
    def __init__(self, x=0.0, y=0.0, yaw=0.0) -> None:
        super().__init__(x, y)
        self._yaw = angle_normalize(yaw)
        self.sin = math.sin(self.yaw)
        self.cos = math.cos(self.yaw)

    @property
    def yaw(self):
        return self._yaw
    
    def set_yaw(self, yaw) -> None:
        self._yaw = angle_normalize(yaw)
        self.sin = math.sin(self.yaw)
        self.cos = math.cos(self.yaw)

    def transform_to(self, base_pose: "Pose2D") -> "Pose2D":
        local_point = super().transform_to(base_pose)
        return Pose2D(local_point.x, local_point.y, angle_normalize(self._yaw - base_pose.yaw))

    def transform_self_to(self, base_pose: "Pose2D") -> None:
        super().transform_self_to(base_pose)
        self.set_yaw(self._yaw - base_pose.yaw)

    def transform_from(self, base_pose: "Pose2D") -> "Pose2D":
        local_point = super().transform_from(base_pose)
        return Pose2D(local_point.x, local_point.y, angle_normalize(self._yaw + base_pose.yaw))

    def transform_self_from(self, base_pose: "Pose2D") -> None:
        super().transform_self_from(base_pose)
        self.set_yaw(self._yaw + base_pose.yaw)

    
