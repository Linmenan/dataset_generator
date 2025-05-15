import math
from ..utils.geometry import Point2D, Pose2D
from ..utils.common import normalize_angle
from typing import Union, List



# ---------- 坐标系转换 ----------
def transfer_to(
    obj: Union[Point2D, Pose2D, List[Point2D]],
    ref: Pose2D,
) -> Union[Point2D, Pose2D, List[Union[Point2D, Pose2D]]]:
    """
    把 **世界坐标** 下的 obj 表示成以 ref 为原点/朝向的局部坐标
    """
    if isinstance(obj, list):
        return [transfer_to(o, ref) for o in obj]
    # 平移到 ref 原点
    dx, dy = obj.x - ref.x, obj.y - ref.y
    c, s = math.cos(ref.yaw), math.sin(ref.yaw)
    # 旋转到 ref 方向
    x_local = dx * c + dy * s
    y_local = dx * -s + dy * c

    if isinstance(obj, Pose2D):
        yaw_local = normalize_angle(obj.yaw - ref.yaw)
        return Pose2D(x_local, y_local, yaw_local)
    return Point2D(x_local, y_local)


def transfer_from(
    obj: Union[Point2D, Pose2D, List[Point2D]],
    ref: Pose2D,
) -> Union[Point2D, Pose2D, List[Union[Point2D, Pose2D]]]:
    """
    把 **ref 局部坐标系** 下的 obj 转换回 **世界坐标**
    """
    if isinstance(obj, list):
        return [transfer_from(o, ref) for o in obj]
    # 绕 ref.yaw 旋转
    c, s = math.cos(ref.yaw), math.sin(ref.yaw)
    x_world = obj.x * c + obj.y * -s + ref.x
    y_world = obj.x * s + obj.y * c + ref.y

    if isinstance(obj, Pose2D):
        yaw_world = normalize_angle(obj.yaw + ref.yaw)
        return Pose2D(x_world, y_world, yaw_world)
    return Point2D(x_world, y_world)

