import math


COLLISION_THRESHOLD = 1.0e-4
GEOMETRY_EPSILON = 1.0e-8


def angle_normalize(angle):
    return angle - math.floor((angle + math.pi) / (2 * math.pi)) * math.pi
