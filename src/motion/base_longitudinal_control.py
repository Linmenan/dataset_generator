
def acc_idm(delta_s: float,
            v1: float,
            v2: float,
            v0: float,
            T: float = 1.5) -> float:
    """
    基于 IDM 的自适应巡航加速度计算。
    delta_s: 与前车净距 (m)
    v1: 当前车速 (m/s)
    v2: 前车车速 (m/s)
    v0: 期望车速上限 v_max (m/s)
    返回: 加速度 a (m/s^2)
    """
    safe_des = v1**2/2+2
    a = 2*((delta_s-safe_des)/(T**2)+(0.0*v2-v1)/T)
    if v1>v0:
        a1 = compute_acceleration(v0,v1)
        a = min(a,a1)
    return a

def compute_acceleration(target_speed: float, current_speed: float) -> float:
    return (target_speed**2 - current_speed**2)/(2)

def compute_stop_acceleration(target_speed:float, cruising_speed:float, current_speed: float, remain_distance: float) -> float:
    remain_distance = max(remain_distance,1e-3)
    if current_speed**2<2*remain_distance and remain_distance>0.1:
        return compute_acceleration(cruising_speed,current_speed)
    return (target_speed**2 - current_speed**2)/(2*remain_distance)

