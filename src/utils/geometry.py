import math
import numpy as np
from typing import Tuple, List, Union

from ..utils.common import normalize_angle

class Point2D:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

class Pose2D(Point2D):
    def __init__(self, x, y, yaw) -> None:
        self.x = x
        self.y = y
        self.yaw = normalize_angle(yaw)

class Box2D(Pose2D):
    def __init__(self, x, y, yaw, width, length_front, length_rear) -> None:
        self.x = x
        self.y = y
        self.yaw = normalize_angle(yaw)
        self.width = width
        self.length_front = length_front
        self.length_rear = length_rear
    def get_corners(self)->List['Point2D']:
        """
        返回 [(x0,y0)…x4,y4]。
        左后 → 左前 → 右前 → 右后
        """
        hw = self.width * 0.5
        local = [(-self.length_rear, -hw),
                ( self.length_front, -hw),
                ( self.length_front,  hw),
                (-self.length_rear,  hw)]
        s, c = math.sin(self.yaw), math.cos(self.yaw)
        return [Point2D(self.x + lx*c - ly*s,
                self.y + lx*s + ly*c) for lx, ly in local]