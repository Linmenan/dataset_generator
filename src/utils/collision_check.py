from ..models.agent import TrafficAgent
import numpy as np
import math
from typing import Tuple

def _box_corners(agent: TrafficAgent) -> np.ndarray:
    """
    返回 4×2 ndarray，依次为左后 -> 左前 -> 右前 -> 右后（逆时针）。
    """
    half_w = agent.width * 0.5
    # 车辆局部坐标 (x 前 +，y 左 +)
    local = np.array([
        [-agent.length_rear, -half_w],  # 左后
        [ agent.length_front, -half_w],  # 左前
        [ agent.length_front,  half_w],  # 右前
        [-agent.length_rear,  half_w],  # 右后
    ])

    sin_h, cos_h = np.sin(agent.pos.yaw), np.cos(agent.pos.yaw)
    rot = np.array([[cos_h, -sin_h],
                    [sin_h,  cos_h]])              # 2×2 旋转矩阵

    world = local @ rot.T + np.array([agent.pos.x, agent.pos.y])
    return world          # shape (4, 2)

def _project(poly: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """
    将 poly 的所有点投影到轴 axis（已归一化）上，返回 (min, max)。
    """
    proj = poly @ axis    # 点积
    return proj.min(), proj.max()

def _overlaps(interval1, interval2, margin=0.0) -> bool:
    """
    两投影区间是否重叠（含安全 margin）。
    """
    a_min, a_max = interval1
    b_min, b_max = interval2
    return not (a_max + margin < b_min or b_max + margin < a_min)

def is_collision(agent1: TrafficAgent,
                 agent2: TrafficAgent,
                 margin_l: float = 0.0,
                 margin_w: float = 0.0,
                 ) -> bool:
    """
    使用分离轴定理检测两个车辆安全盒是否相交。
    margin > 0 使盒子“变大”形成安全距离；=0 刚性碰撞。
    """
    poly1, poly2 = _box_corners(agent1), _box_corners(agent2)

    # 取两车长方形的 2 条边向量，各取法向共 4 个轴
    axes = []
    for poly in (poly1, poly2):
        for i in range(2):                         # 只需 2 条边
            edge = poly[(i + 1) % 4] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)

    # 在所有分离轴上投影并检测
    for idx, axis in enumerate(axes):
        if not _overlaps(_project(poly1, axis),
                         _project(poly2, axis),
                         margin=margin_l if idx%2!=0 else margin_w,
                         ):
            return False      # 找到分离轴 => 无碰撞
    return True

def will_collision(agent1: TrafficAgent,
                 agent2: TrafficAgent,
                 pred_horizon: float = 2.0,
                 margin_l: float = 0.0,
                 margin_w: float = 0.0,) -> bool:
    t = 0.0
    while t<=pred_horizon:
        if is_collision(agent1=agent1.pred(dt=t),agent2=agent2.pred(dt=t),margin_l=margin_l,margin_w=margin_w):
            return True
        t+=0.5
    return False

def distance_between(agent1: TrafficAgent,
                 agent2: TrafficAgent)->float:
    return math.sqrt((agent1.pos.x-agent2.pos.x)**2+(agent1.pos.y-agent2.pos.y)**2)