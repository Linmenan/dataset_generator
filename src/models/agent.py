import numpy as np
from typing import Tuple, List
from .map_elements import Pose2D

class TrafficAgent:
    """
    道路上任意交通参与者的基类。
    仅保存几何尺寸与纵向状态，不含控制逻辑。
    """
    def __init__(
        self,
        id: str = "",
        pos: Pose2D=Pose2D(0, 0, 0),         # 位置,航向角 (m, m, rad左正)
        speed: float = 0.0,                  # 纵向车速 (m/s)
        accel: float = 0.0,                  # 纵向加速度 (m/s²)
        curvature: float = 0.0,              # 行驶曲率 (1/m)
        width: float = 1.96,                 # 车宽 (m)
        length_front: float = 3.945,         # 前保险杠到质心距离 (m)
        length_rear: float = 1.08,           # 后保险杠到质心距离 (m)
        wheel_base: float = 2.97,            # 轴距 (m)
        speed_limit: float = 10.0,                 # 最高速度 (m/s)
        a_max: float = 2.0,                  # 最大加速度 (m/s²)
        a_min: float = -5.0,                 # 最大减速度 (m/s²)
        agent_type: str = "car",             # "car" | "truck" | ...
        current_road_index: str = "",
        current_lane_index: str = "", 
        current_lane_unicode: int = -1,
        nearerst_agent = None,
        min_distance = float('inf'),
        current_s = float('inf')
    ) -> None:
        self.id = str(id)
        self.pos = pos if pos is not None else Pose2D(0, 0, 0)
        self.speed = float(speed)
        self.accel = float(accel)
        self.curvature = float(curvature)
        self.width = float(width)
        self.length_front = float(length_front)
        self.length_rear = float(length_rear)
        self.wheel_base = float(wheel_base)
        self.speed_limit = float(speed_limit)
        self.a_max = float(a_max)
        self.a_min = float(a_min)
        self.agent_type = agent_type
        self.current_road_index = str(current_road_index)
        self.current_lane_index = str(current_lane_index)
        self.current_lane_unicode = int(current_lane_unicode)
        self.road_map = []                   # 记录一连串lane路线规划
        self.ref_line = []                   # 拼接路线规划车道中心线
        self.plan_haul = []                  # 记录路线规划路段距离
        self.nearerst_agent = nearerst_agent # 记录前方车辆
        self.min_distance = min_distance
        self.current_s = current_s
    def step(self, a_cmd: float, dt: float, cur_cmd: float = 0.0) -> None:
        """
        仿真步进。
        """
        self.accel = a_cmd
        self.curvature = cur_cmd
        delta_speed = a_cmd * dt
        delta_s = 0.0
        if delta_speed + self.speed < 0:
            stop_time = -self.speed / a_cmd
            self.speed = 0.0
            delta_s = 0.5*(self.speed*stop_time)
        else:
            delta_s = self.speed * dt + 0.5 * a_cmd * dt**2
            self.speed += delta_speed
            
        if abs(cur_cmd)>1.0e-3:
            self.pos = Pose2D(
                self.pos.x + (np.sin(self.pos.yaw + cur_cmd * delta_s) - np.sin(self.pos.yaw))/cur_cmd, 
                self.pos.y + (np.cos(self.pos.yaw) - np.cos(self.pos.yaw + cur_cmd * delta_s))/cur_cmd,
                self.pos.yaw+cur_cmd * delta_s
                )
        else:
            self.pos = Pose2D(
                self.pos.x + delta_s*np.cos(self.pos.yaw), 
                self.pos.y+delta_s*np.sin(self.pos.yaw),
                self.pos.yaw+cur_cmd * delta_s
                )
        
class EgoVehicle(TrafficAgent):
    """
    自车（Ego）——在 TrafficAgent 基础上扩充动力学状态、
    控制输入与硬件/软件约束。
    """

    # —— 默认约束常量（可按车型改） ——
    _DEFAULT_STEER_LIMIT: Tuple[float, float] = (-0.6109, 0.6109)  # ±35° (rad)
    _DEFAULT_STEER_RATE_LIMIT: float = 8.0      # rad/s
    _DEFAULT_TORQUE_LIMIT: float = 400.0        # N·m
    _DEFAULT_POWER_LIMIT: float = 150_000.0     # W

    def __init__(
        self,
        # EgoVehicle 独有字段
        torque: float = 0.0,                    # 车桥扭矩 (N·m)
        front_wheel_angle: float = 0.0,         # 前轮转角 δ_f (rad)
        steer_angle: float = 0.0,               # 方向盘转角 θ_s (rad)
        steer_rate: float = 0.0,                # 方向盘角速度 ω_s (rad/s)
        # 限制参数（可选覆盖）
        steer_angle_limit: Tuple[float, float] = None,
        steer_rate_limit: float = None,
        torque_limit: float = None,
        power_limit: float = None,
        **kwargs
    ) -> None:
        # --- 1. 基类初始化 ---
        super().__init__(**kwargs)

        # --- 2. 运动 / 控制状态 ---
        self.torque = float(torque)
        self.front_wheel_angle = float(front_wheel_angle)
        self.steer_angle = float(steer_angle)
        self.steer_rate = float(steer_rate)

        # --- 3. 约束 ---
        self.steer_angle_limit: Tuple[float, float] = (
            steer_angle_limit if steer_angle_limit is not None
            else self._DEFAULT_STEER_LIMIT
        )
        self.steer_rate_limit: float = (
            float(steer_rate_limit) if steer_rate_limit is not None
            else self._DEFAULT_STEER_RATE_LIMIT
        )
        self.torque_limit: float = (
            float(torque_limit) if torque_limit is not None
            else self._DEFAULT_TORQUE_LIMIT
        )
        self.power_limit: float = (
            float(power_limit) if power_limit is not None
            else self._DEFAULT_POWER_LIMIT
        )