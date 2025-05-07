import math

def normalize_angle(angle: float) -> float:
    """
    将弧度制角度规范到 (-π, π]
    
    :param angle: 待规范化的角度（弧度）
    :return: 规范化后的角度，范围为 (-π, π]
    """
    # 先将 angle 平移 π 后对 2π 取模，再平移回去
    a = (angle + math.pi) % (2 * math.pi) - math.pi
    # 模运算的结果 a ∈ [−π, π)，如果恰好 a==−π，则映射到 π
    if a <= -math.pi:
        return math.pi
    return a