'''
Author: linmenan 314378011@qq.com
Date: 2025-04-21 10:57:44
LastEditors: linmenan 314378011@qq.com
LastEditTime: 2025-04-22 20:10:19
FilePath: /dataset_generator/src/scene_simulation/scene_simulator.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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

class Mode(enum.Enum):   # enum 支持位运算 :contentReference[oaicite:0]{index=0}
    SYNC = "sync"
    ASYNC = "async"
class SceneSimulator:
    def __init__(
            self, 
            mode:Mode = Mode.SYNC, 
            step:float = 0.05, 
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

        self.sim_frame:int = 0
        self._sim_time: float = 0.0                        # 累积仿真时间
        self._start_wall: Optional[float] = None    # 真实起始秒 (perf_counter)
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None


        self.perception_range = perception_range
        self.ego_vehicle = None
        self.agents = []
        self.map_parser = MapParser(file_path=map_file_path,yaml_path=yaml_path)
        self.fig = go.FigureWidget()
        self.fig.update_layout(
            title="Traffic Map",
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1, visible=False),
            width=1200, height=800
        )
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
        from ..visualization.visualization import visualize_traffic_agents,visualize_lanes
        self._sim_time += self.step
        self.sim_frame += 1
        #  zzzzzz
        self.update_states()
        if self.mode is not Mode.SYNC:
            if self.sim_frame%20 == 0:
                # visualize_traffic_agents(self.fig, [self.ego_vehicle]+self.agents)
                visualize_lanes(self.fig, self)
                self.fig.update_layout(
                    title="Traffic Map",
                    xaxis=dict(scaleanchor="y", scaleratio=1),
                    yaxis=dict(scaleanchor="x", scaleratio=1, visible=False),
                    width=1200, height=800
                )
        
    # ----------- 主循环 (异步) -----------
    async def _run_async(self):
        self._start_wall = time.perf_counter()
        self._running = True
        try:
            while self._running:
                await asyncio.sleep(self.step)      # 让 event‑loop 调度其他任务 :contentReference[oaicite:3]{index=3}
                self._sim_time += self.step
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
                speed = 1.0
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
                speed = 1.0
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
        return (0.0 - current_speed**2)/(2*remain_distance)
    
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
        for current_agent in self.agents:
            road_id = current_agent.current_road_index
            lane_id = current_agent.current_lane_index
            current_lane = self.get_lane(road_id, lane_id)
            current_s = current_lane.calculate_cumulate_s(current_agent.pos)
            current_v = current_agent.speed
            closest_agent = None
            min_distance = float('inf')
            for agent in self.agents:
                if agent.id == current_agent.id:
                    continue
                if road_id == current_agent.current_road_index and lane_id == current_agent.current_lane_index:
                    agent_s = current_lane.calculate_cumulate_s(agent.pos)
                    if agent_s > current_s:
                        delta_s = agent_s - current_s
                        if delta_s < min_distance:
                            min_distance = delta_s
                            closest_agent = agent
            if road_id == self.ego_vehicle.current_road_index and lane_id == self.ego_vehicle.current_lane_index:
                ego_vehicle_s = current_lane.calculate_cumulate_s(self.ego_vehicle.pos)
                if ego_vehicle_s > current_s:
                    delta_s = ego_vehicle_s - current_s
                    if delta_s < min_distance:
                            min_distance = delta_s
                            closest_agent = agent
            if closest_agent is not None:
                agent_v = closest_agent.speed
                acc = self.acc_idm(delta_s, current_v, agent_v, v0=current_agent.speed_limit)
                current_agent.step(a_cmd = acc,dt = self.step)
            road = self.get_road(road_id)
            
            if road.junction != "-1":
                current_agent.step(a_cmd = self.compute_acceleration(3.0, current_v), dt = self.step)
            else:
                if road.successor[0] == "road":
                    current_agent.step(a_cmd = self.compute_acceleration(3.0, current_v), dt = self.step)
                else:
                    next_lane_index = random.randrange(len(current_lane.successor))
                    color = self.get_lane_traffic_light(current_lane.successor[next_lane_index][0], current_lane.successor[next_lane_index][1])
                    if color == 'green' or color == 'grey':
                        current_agent.step(a_cmd = self.compute_acceleration(3.0, current_v), dt = self.step)
                    else:
                        delta_s = current_lane.length - current_s
                        current_agent.step(a_cmd = self.compute_stop_acceleration(current_v, delta_s), dt = self.step)

                    
