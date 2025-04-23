from ..models.agent import *
from ..parsers.parsers import MapParser
from ..models.map_elements import *
from ..utils.collision_check import *

import plotly.graph_objects as go
import numpy as np
import random
import enum
import asyncio
import time
import math
from datetime import datetime
from typing import Callable, Awaitable, Optional, List

from pyqtgraph.Qt import QtWidgets
from ..visualization.visualization import SimView

class Mode(enum.Enum):   # enum 支持位运算 :contentReference[oaicite:0]{index=0}
    SYNC = "sync"
    ASYNC = "async"
class SceneSimulator:
    def __init__(
            self, 
            mode:Mode = Mode.SYNC, 
            step:float = 0.05, 
            plot_step:int = 5, 
            map_file_path='', 
            yaml_path='', 
            perception_range=0, 
            ):
        """
        :param mode: 运行模式 (同步/异步)
        :param step: 每步仿真时长
        :param on_step: 回调, 参数 = 仿真时钟
        """
        self.mode = mode
        self.step = step
        self.plot_step = plot_step

        self.sim_frame:int = 0
        self._sim_time: float = 0.0                        # 累积仿真时间
        self._start_wall: Optional[float] = None    # 真实起始秒 (perf_counter)
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None


        self.perception_range = perception_range
        self.ego_vehicle = None
        self.agents = []
        self.map_parser = MapParser(file_path=map_file_path,yaml_path=yaml_path)
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        # 创建并 show 视图
        self.view = SimView(self)
        self.view.show()
        print("simulator init")
    # ----------- 公共查询接口 -----------
    @property
    def sim_time(self) -> float:
        """仿真运行时间 (s)"""
        return self._sim_time

    @property
    def wall_time(self) -> datetime:
        """绝对时间戳 (本地时区)"""
        return datetime.now()                       # time 系统时钟 :contentReference[oaicite:1]{index=1}

    @property
    def wall_elapsed(self) -> float:
        """仿真器启动以来消耗的真实时间 (s)"""
        return time.perf_counter() - (self._start_wall or time.perf_counter())  # :contentReference[oaicite:2]{index=2}

    # ----------- 主循环 (同步) -----------
    def step_once(self):
        """外部驱动一次步进 (同步模式用)"""
        self._sim_time += self.step
        self.sim_frame += 1
        #  zzzzzz
        self.update_states()
        if self.sim_frame%self.plot_step == 0:
            self.view.update()
            
   
        
    # ----------- 主循环 (异步) -----------
    async def _run_async(self):
        self._start_wall = time.perf_counter()
        self._running = True
        try:
            while self._running:
                await asyncio.sleep(self.step)      # 让 event‑loop 调度其他任务 :contentReference[oaicite:3]{index=3}
                #  zzzzzz
                self.step_once()
                
        finally:
            self._running = False

    # ----------- 启动 / 停止 -----------
    def start(self):
        if self.mode is Mode.SYNC:
            # 同步模式只记录起始时间，交由外部循环调用 step_once()
            self._start_wall = time.perf_counter()
            self._running = True
        else:
            # 异步：独占事件循环
            asyncio.run(self._run_async())
            # self._loop = asyncio.new_event_loop()          # :contentReference[oaicite:4]{index=4}
            # asyncio.run_coroutine_threadsafe(
            #     self._run_async(), self._loop)


    def stop(self):
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def init_ego_vehicle(self):
        while self.ego_vehicle == None: 
            k = random.choice(list(self.map_parser.roads))
            road = self.map_parser.roads[k]
            if (road.junction == "-1"):
                lane = random.choice(road.lanes)
                index = random.randrange(len(lane.sampled_points))
                rear_point = lane.sampled_points[index][0]
                hdg = lane.headings[index]
                speed = 0.0
                vehicle = EgoVehicle(id=str(0), 
                                     pos=Point2D(rear_point[0], 
                                                 rear_point[1]), 
                                     hdg=hdg, speed=speed)
                collision = False
                for agent in self.agents:
                    if is_collision(vehicle, agent):
                        collision = True
                        break
                if not collision:
                    self.ego_vehicle = vehicle
                    self.ego_vehicle.current_road_index = road.road_id
                    self.ego_vehicle.current_lane_index = lane.lane_id

    def generate_traffic_agents(self, number = 0):
        while len(self.agents) < number:   
            k = random.choice(list(self.map_parser.roads))
            road = self.map_parser.roads[k]
            if (road.junction == "-1"):
                lane = random.choice(road.lanes)
                index = random.randrange(len(lane.sampled_points))
                rear_point = lane.sampled_points[index][0]
                hdg = lane.headings[index]
                speed = 0.0
                traffic_agent = TrafficAgent(id = str(len(self.agents)+1), 
                                             pos=Point2D(rear_point[0], 
                                                         rear_point[1]), 
                                             hdg=hdg, speed=speed)
                collision = False
                for agent in self.agents:
                    if is_collision(traffic_agent, agent):
                        collision = True
                        break
                if not collision:
                    traffic_agent.current_road_index = road.road_id
                    traffic_agent.current_lane_index = lane.lane_id
                    self.agents.append(traffic_agent) 
                

    def extract_lanes_in_range(self, roads, current_pos):
        cx, cy = current_pos
        for road in roads.values():
            road_in_range = False
            for lane in road.lanes:
                for (x, y) in lane.sampled_points:
                    if np.linalg.norm([x-cx, y-cy]) <= self.perception_range:
                        lane.in_range = True
                        road_in_range = True
                        break
            road.on_route = road_in_range

    def get_road(self, road_index:str)->Road:
        for road in self.map_parser.roads.values():
            if road.road_id == road_index:
                return road

    def get_lane(self, road_index:str, lane_index:str)->Lane:
        for road in self.map_parser.roads.values():
            if road.road_id == road_index:
                for lane in road.lanes:
                    if lane.lane_id == lane_index:
                        return lane

    def acc_idm(self, delta_s: float,
                v1: float,
                v2: float,
                v0: float,
                s0: float = 2.0,
                T: float = 1.5,
                a_max: float = 1.0,
                b: float = 1.5,
                delta: float = 4.0) -> float:
        """
        基于 IDM 的自适应巡航加速度计算。
        delta_s: 与前车净距 (m)
        v1: 当前车速 (m/s)
        v2: 前车车速 (m/s)
        v0: 期望车速上限 v_max (m/s)
        返回: 加速度 a (m/s^2)
        """
        # 1) 期望安全间距 s* (动态头距)
        s_star = s0 + v1 * T + (v1 * (v1 - v2)) / (2 * math.sqrt(a_max * b))
        
        # 2) 自由道路项与交互项
        free_term = 1.0 - (v1 / v0) ** delta
        interaction_term = (s_star / delta_s) ** 2 if delta_s > 0 else float('inf')
        
        # 3) IDM 加速度
        a = a_max * (free_term - interaction_term)
        
        # 4) 防止过度制动/加速，并确保不超过限速 v0
        #    若 v1 > v0，优先刹车
        if v1 >= v0:
            a = min(a, -b)   # 轻松减速
        #    限制最大加速度
        a = max(min(a, a_max), -b)
        return a
    def compute_acceleration(self, target_speed: float, current_speed: float) -> float:
        return (target_speed**2 - current_speed**2)/(2)
    def compute_stop_acceleration(self, current_speed: float, remain_distance: float) -> float:
        return (0.0 - current_speed**2)/(2*remain_distance+1e-3)
    
    def pure_pursuit_curvature(self,
        position: Point2D,
        path: List[Point2D],
        lookahead: float
    ) -> float:
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
        dists = [(pt.x - position.x)**2 + (pt.y - position.y)**2 for pt in path]
        idx0 = min(range(len(path)), key=lambda i: dists[i])

        # 2) 在后续点中寻找首个满足距离 >= lookahead 的预瞄点
        target: Optional[Point2D] = None
        L2 = lookahead**2
        for pt in path[idx0+1:]:
            if (pt.x - position.x)**2 + (pt.y - position.y)**2 >= L2:
                target = pt
                break
        # 若后面都不够远，就用最后一个点
        if target is None:
            target = path[-1]

        # 3) 估计车辆航向：取最近点与其下一个点的连线方向
        if idx0 < len(path) - 1:
            next_pt = path[idx0 + 1]
        else:
            # 如果最近点已经是最后一个，则退而取前一个
            next_pt = path[idx0 - 1]
        heading = math.atan2(next_pt.y - path[idx0].y,
                            next_pt.x - path[idx0].x)

        # 4) 计算车头指向“预瞄点”的角度 alpha
        dx = target.x - position.x
        dy = target.y - position.y
        angle_to_target = math.atan2(dy, dx)

        # 归一化角度差到 [-pi, +pi]
        alpha = angle_to_target - heading
        alpha = (alpha + math.pi) % (2*math.pi) - math.pi

        # 5) 计算曲率 κ = 2*sin(alpha) / L_d
        # sin(alpha) 左转为正，右转为负
        curvature = 2 * math.sin(alpha) / lookahead

        return curvature

    def get_lane_traffic_light(self, road_id: str, lane_id: str) -> str:
        road = self.get_road(road_id)
        lane = self.get_lane(road_id, lane_id)
        light = 'grey'
        if road.junction != "-1":
                if road.signals is not None:
                    for signal in road.signals:
                        if signal.type=='reference':
                            id = signal.signal_id
                            from_lane = int(signal.from_lane)
                            to_lane =  int(signal.to_lane)
                            if int(lane.lane_id)>=from_lane and int(lane.lane_id)<=to_lane:
                                for controll in self.map_parser.traffic_lights.values():
                                    for control in controll.controls:
                                        if control.signal_id == id:
                                            if controll.signal_controller is not None:
                                                t = self.sim_time
                                                status = controll.signal_controller.state(t=t)
                                                light = status[id].value
        return light

    def update_states(self)->None:
        lookahead = 5.0  # 或者从配置获取
        for current_agent in self.agents:
            road_id = current_agent.current_road_index
            lane_id = current_agent.current_lane_index
            current_lane = self.get_lane(road_id, lane_id)
            current_s = current_lane.calculate_cumulate_s(current_agent.pos)
            remain_s = current_lane.length - current_s
            current_v = current_agent.speed
            closest_agent = None
            min_distance = float('inf')

            #更新前方最近智能体
            for agent in [self.ego_vehicle]+self.agents:
                if agent.id == current_agent.id:
                    continue
                if road_id == current_agent.current_road_index and lane_id == current_agent.current_lane_index:
                    agent_s = current_lane.calculate_cumulate_s(agent.pos)
                    if agent_s > current_s:
                        delta_s = agent_s - current_s
                        if delta_s < min_distance:
                            min_distance = delta_s
                            closest_agent = agent
            if closest_agent is not None:
                agent_v = closest_agent.speed
                acc = self.acc_idm(delta_s, current_v, agent_v, v0=current_agent.speed_limit)
                current_agent.step(a_cmd = acc,dt = self.step)
            road = self.get_road(road_id)
            acc_cmd = 0
            curvature_cmd = 0
            
            #纵向控制，路口状况判断
            if road.junction != "-1":
                next_lane_index = 0
                acc_cmd = self.compute_acceleration(3.0, current_v)
            else:
                if road.successor[0] == "road":
                    next_lane_index = 0
                    acc_cmd = self.compute_acceleration(3.0, current_v)
                else:
                    next_lane_index = random.randrange(len(current_lane.successor))
                    color = self.get_lane_traffic_light(current_lane.successor[next_lane_index][0], current_lane.successor[next_lane_index][1])
                    if color == 'green' or color == 'grey':
                        acc_cmd = self.compute_acceleration(3.0, current_v)
                    else:
                        delta_s = current_lane.length - current_s
                        acc_cmd = self.compute_stop_acceleration(current_v, delta_s)
            
            #横向控制
            # 1) 得到当前车道的参考线
            ref_line = current_lane.get_ref_line()  # List[Point2D]

            # 2) 如果剩余 < lookahead，拼接下一车道的参考线
            if remain_s < lookahead and current_lane.successor:
                # 取第一个后继车道
                next_road_id, next_lane_id = current_lane.successor[next_lane_index]
                next_lane = self.get_lane(next_road_id, next_lane_id)
                next_ref = next_lane.get_ref_line()
                # 只拼接 next_ref 的前 lookahead 段
                ref_line = ref_line + next_ref

            # 3) 调用纯追踪算法计算曲率
            curvature_cmd = self.pure_pursuit_curvature(
                position=current_agent.pos,
                path=ref_line,
                lookahead=lookahead
            )
            current_agent.step(a_cmd = acc_cmd, cur_cmd = curvature_cmd, dt = self.step)

            #更新当前道路和车道
            if remain_s < 0.1:
                if current_lane.successor:
                    current_agent.current_road_index = current_lane.successor[next_lane_index][0]
                    current_agent.current_lane_index = current_lane.successor[next_lane_index][1]