import numpy as np
import math
from dataclasses import dataclass, field
from typing import Tuple, List, Dict
from ..utils.geometry import Pose2D,Box2D

class NearbyDistricts:
    def __init__(self) -> None:
        self.front_agents = []        # 记录自身前方{智能体,距离,横向距离,纵向距离}
        self.left_agents = []         # 记录自身左侧与左后方{智能体,距离,横向距离,纵向距离}
        self.right_agents = []        # 记录自身右侧与右后方{智能体,距离,横向距离,纵向距离}
        self.rear_agents = []         # 记录自身正后方{智能体,距离,横向距离,纵向距离}
        self.collition_agents = []    # 记录自身碰撞{智能体,距离,横向距离,纵向距离}
    def clear(self):
        self.front_agents = []     
        self.left_agents = []      
        self.right_agents = []    
        self.rear_agents = []      
        self.collition_agents = [] 

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
        lane_change:Tuple[int,int] = (-1,-1),
        current_s:float = float('inf'),
        remain_s:float = float('inf'),
        way_right_level:int = int(-1),
    ) -> None:
        self.id = str(id)
        self.pos = pos if pos is not None else Pose2D(0, 0, 0)
        self.box = Box2D(self.pos.x, self.pos.y, self.pos.yaw, width, length_front, length_rear)
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
        # 确认的
        self.road_map = []                   # 记录一连串lane路线规划
        self.ref_line = []                   # 拼接路线规划车道中心线
        self.plan_haul = []                  # 记录路线规划路段距离
        # 计划中
        self.plan_road_map = []              # 记录一连串lane路线规划
        self.plan_ref_line = []              # 拼接路线规划车道中心线
        self.plan_plan_haul = []             # 记录路线规划路段距离

        self.nearest_route_agent = [] # 记录前方车辆[{agent,distance}]
        self.around_agents = NearbyDistricts() # 记录周边
        self.lane_change = lane_change
        self.current_s = current_s
        self.remain_s = remain_s
        self.cte = float('inf')
        self.out_lane = False
        self.ephi = float('inf')
        self.way_right_level = way_right_level

    def pred(self,dt: float)->'TrafficAgent':
        speed = self.speed
        delta_s = speed*dt
        if abs(self.curvature)>1.0e-3:
            pos = Pose2D(
                self.pos.x + (np.sin(self.pos.yaw + self.curvature * delta_s) - np.sin(self.pos.yaw))/self.curvature, 
                self.pos.y + (np.cos(self.pos.yaw) - np.cos(self.pos.yaw + self.curvature * delta_s))/self.curvature,
                self.pos.yaw+self.curvature * delta_s
                )
        else:
            pos = Pose2D(
                self.pos.x + delta_s*np.cos(self.pos.yaw), 
                self.pos.y+delta_s*np.sin(self.pos.yaw),
                self.pos.yaw+self.curvature * delta_s
                )
        return TrafficAgent(pos=pos, speed=speed, curvature=self.curvature,width=self.width,length_front=self.length_front,length_rear=self.length_rear)
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
            self.box.x = self.pos.x
            self.box.y = self.pos.y
            self.box.yaw = self.pos.yaw
        else:
            self.pos = Pose2D(
                self.pos.x + delta_s*np.cos(self.pos.yaw), 
                self.pos.y+delta_s*np.sin(self.pos.yaw),
                self.pos.yaw+cur_cmd * delta_s
                )
            self.box.x = self.pos.x
            self.box.y = self.pos.y
            self.box.yaw = self.pos.yaw

    from ..models.map_elements import Lane
    def bev_perception(self, agents: List['TrafficAgent']) -> None:
        """
        传感器感知。
        """
        from ..utils.transfer_of_axes import transfer_to
        self.around_agents.clear()
        for agent in agents:
            if agent.id == self.id:
                continue
            from ..utils.collision_check import distance_between

            dis = distance_between(agent1=self, agent2=agent)
            if dis < 30:

                pos_ego_frame = transfer_to(ref=self.pos, obj=agent.pos)

                corners_ego_frame = transfer_to(ref=self.pos, obj=agent.box.get_corners())
                xs = [p.x for p in corners_ego_frame]
                ys = [p.y for p in corners_ego_frame]
                a_min_x, a_max_x = min(xs), max(xs)
                a_min_y, a_max_y = min(ys), max(ys)

                e_front = self.length_front
                e_rear  = -self.length_rear
                e_left  =  self.width / 2
                e_right = -self.width / 2

                
                # ------------------ 2. 判断是否碰撞 ------------------
                overlap_x = (a_min_x <= e_front) and (a_max_x >= e_rear)
                overlap_y = (a_min_y <= e_left ) and (a_max_y >= e_right)
                if overlap_x and overlap_y:
                    self.around_agents.collition_agents.append((agent,dis,0.0,0.0))
                # ------------------ 3. 分类 ------------------
                # 优先级：front > left > right > rear
                elif a_min_x >= 0.0:# and 2.0*pos_ego_frame.x>math.sqrt(max(abs(pos_ego_frame.y)-0.5*self.width,0)):
                    # 完全在前方
                    lon = a_min_x - e_front
                    lat = 0.0 if overlap_y else (a_min_y - e_left if a_min_y > e_left else e_right - a_max_y)
                    self.around_agents.front_agents.append((agent,dis,lat,lon))

                elif a_min_y >= e_left:
                    # 完全在左侧
                    lat = a_min_y - e_left
                    lon = 0.0 if overlap_x else (a_min_x - e_front if a_min_x > e_front else e_rear - a_max_x)
                    self.around_agents.left_agents.append((agent,dis,lat,lon))

                elif a_max_y <= e_right:
                    # 完全在右侧
                    lat = e_right - a_max_y
                    lon = 0.0 if overlap_x else (a_min_x - e_front if a_min_x > e_front else e_rear - a_max_x)
                    self.around_agents.right_agents.append((agent,dis,lat,lon))

                else:
                    # 完全在后方
                    lon = e_rear - a_max_x
                    lat = 0.0 if overlap_y else (a_min_y - e_left if a_min_y > e_left else e_right - a_max_y)
                    self.around_agents.rear_agents.append((agent,dis,lat,lon))

        # self.around_agents.sort(key=lambda x: x.pos.distance(self.pos))
    def lane_location(self,lanes:Dict[int,'Lane'])->None:
        from ..utils.common import normalize_angle

        current_lane = lanes[self.current_lane_unicode]     
        self.current_s,self.cte,self.out_lane,_,head_ref = current_lane.projection(self.pos)
        self.ephi = normalize_angle(head_ref-self.pos.yaw)
        self.remain_s = current_lane.length - self.current_s

    def get_route_agent(self, agents: List['TrafficAgent'])->None:
        self.nearest_route_agent = []
        #更新前方最近智能体
        for check_agent in agents:
            if check_agent.id == self.id:
                continue
            pred_length = 0.0
            # if self.lane_change == (-1,-1):
            for check_lane in self.road_map:
                if (check_lane.unicode==check_agent.current_lane_unicode):
                    agent_s,_,_,_,_ = check_lane.projection(check_agent.pos)
                    if self.current_lane_unicode==check_agent.current_lane_unicode:
                        if agent_s > self.current_s:
                            delta_s = agent_s - self.current_s
                            self.nearest_route_agent.append((check_agent,delta_s))

                    else:
                        delta_s = pred_length + agent_s - self.current_s
                        self.nearest_route_agent.append((check_agent,delta_s))
                pred_length+=check_lane.length
                if pred_length - self.current_s > max(min(20,self.speed*2),50):
                    break
            # else:
            #     pass
        self.nearest_route_agent.sort(key=lambda x: x[1])



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