import math
import numpy as np


class Point2D:
    def __init__(self, x=0.0, y=0.0) -> None:
        self.x = x
        self.y = y

    def __repr__(self):
        return f"x:= {self.x}, y:= {self.y}"

    def distance_to(self, other_point: "Point2D"):
        return math.sqrt((self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2)

    def transform_to(self, base_pose) -> "Point2D":
        ans = np.dot(np.array([self.x - base_pose.x, self.y - base_pose.y]),
                     np.array([[base_pose.cos, -base_pose.sin],
                               [base_pose.sin, base_pose.cos]]))
        return Point2D(ans[0], ans[1])

    def transform_self_to(self, base_pose) -> None:
        ans = np.dot(np.array([self.x - base_pose.x, self.y - base_pose.y]),
                     np.array([[base_pose.cos, -base_pose.sin],
                               [base_pose.sin, base_pose.cos]]))
        self.x = ans[0]
        self.y = ans[1]

    def transform_from(self, base_pose) -> "Point2D":
        ans = np.dot(np.array([self.x, self.y]),
                     np.array([[base_pose.cos, base_pose.sin],
                               [-base_pose.sin, base_pose.cos]]))
        return Point2D(base_pose.x + ans[0], base_pose.y + ans[1])

    def transform_self_from(self, base_pose) -> None:
        ans = np.dot(np.array([self.x, self.y]),
                     np.array([[base_pose.cos, base_pose.sin],
                               [-base_pose.sin, base_pose.cos]]))
        self.x = base_pose.x + ans[0]
        self.y = base_pose.y + ans[1]


