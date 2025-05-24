import math


COLLISION_THRESHOLD = 1.0e-4
GEOMETRY_EPSILON = 1.0e-8


def angle_normalize(angle):
    a = (angle + math.pi) % (2 * math.pi) - math.pi
    # 模运算的结果 a ∈ [−π, π)，如果恰好 a==−π，则映射到 π
    if a <= -math.pi:
        return math.pi
    return a
