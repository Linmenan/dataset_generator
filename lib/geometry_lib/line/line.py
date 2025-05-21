from ..pose.pose import *
from typing import List


class Line2D:
        def __init__(self, data: List[Point2D], time_stamp=0.0) -> None:
            self.data = data
            self.time_stamp = time_stamp

        def transform_to(self, base_pose: "Pose2D") -> "Line2D":
            result = Line2D(self.data, self.time_stamp)
            for point in result.data:
                point.transform_self_to(base_pose)
            return result

        def transform_self_to(self, base_pose: "Pose2D") -> None:
            for point in self.data:
                point.transform_self_to(base_pose)

        def transform_from(self, base_pose: "Pose2D") -> "Line2D":
            result = Line2D(self.data, self.time_stamp)
            for point in result.data:
                point.transform_self_from(base_pose)
            return result

        def transform_self_from(self, base_pose: "Pose2D") -> None:
            for point in self.data:
                point.transform_self_from(base_pose)

