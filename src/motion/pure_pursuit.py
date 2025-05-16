from typing import List
import math

from ..utils.common import normalize_angle
from ..utils.geometry import Point2D,Pose2D

def pure_pursuit( pose: Pose2D, path: List['Point2D'], lookahead: float) -> float:
    """
    纯追踪法计算曲率 κ：
    κ = 2 * sin(alpha) / L_d
    其中 alpha = 目标点方向与车辆航向的夹角，
        L_d    = lookahead 预瞄距离。

    参数:
    position  当前车辆质心位置
    path      参考线采样点列表，至少两个点
    lookahead 预瞄距离

    返回:
    曲率 κ；若输入不合法或找不到预瞄点，返回 None。
    """

    # 1) 找到最近的路径点索引
    dists = [(pt.x - pose.x)**2 + (pt.y - pose.y)**2 for pt in path]
    idx0 = min(range(len(path)), key=lambda i: dists[i])

    # 2) 在后续点中寻找首个满足距离 >= lookahead 的预瞄点
    target = None
    L2 = lookahead**2
    for pt in path[idx0+1:]:
        if (pt.x - pose.x)**2 + (pt.y - pose.y)**2 >= L2:
            target = pt
            break
    # 若后面都不够远，就用最后一个点
    if target is None:
        target = path[-1]


    # 4) 计算车头指向“预瞄点”的角度 alpha
    dx = target.x - pose.x
    dy = target.y - pose.y
    angle_to_target = math.atan2(dy, dx)

    # 归一化角度差到 [-pi, +pi]
    alpha = normalize_angle(angle_to_target - pose.yaw)

    # 5) 计算曲率 κ = 2*sin(alpha) / L_d
    # sin(alpha) 左转为正，右转为负
    curvature = 2 * math.sin(alpha) / lookahead

    return curvature