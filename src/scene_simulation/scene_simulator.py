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
from ..utils.data_recorder import DataRecorder
from IPython.display import clear_output
import logging
from ..motion.agent_motion import LateralMPC_NL
from ..utils.color_print import RED,RESET,GREEN,YELLOW,BLUE,CYAN,MAGENTA

class Mode(enum.Enum):   # enum 支持位运算 :contentReference[oaicite:0]{index=0}
    SYNC = "sync"
    ASYNC = "async"
class SceneSimulator:
    def __init__(
            self, 
            window_size = (1000, 600),
            mode:Mode = Mode.SYNC, 
            step:float = 0.05, 
            plot_step:int = 5, 
            map_file_path='', 
            yaml_path='', 
            data_path='',
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
        self.mpc = LateralMPC_NL(pred_horizon=30.0, ctrl_horizon=10.0, step_len=1.0)

        self.cruising_speed = 10.0
        self.crossing_speed = 5.0
        self.perception_range = perception_range
        self.ego_vehicle = None
        self.agents = []
        self.map_parser = MapParser(file_path=map_file_path,yaml_path=yaml_path)
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.data_recorder = DataRecorder(data_path)
        # 创建并 show 视图
        self.view = SimView(self, size=window_size)
        self.view.show()
        logging.debug("simulator init")
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
                if lane.length>10.0 and len(lane.sampled_points)>10:
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
                        self.ego_vehicle.current_lane_unicode = lane.unicode

    def generate_traffic_agents(self, number = 0):
        while len(self.agents) < number:   
            k = random.choice(list(self.map_parser.roads))
            road = self.map_parser.roads[k]
            if (road.junction == "-1") :
                lane = random.choice(road.lanes)
                if lane.length>10.0 and len(lane.sampled_points)>10:
                    index = random.randrange(5,len(lane.sampled_points)-5)
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
                        traffic_agent.current_lane_unicode = lane.unicode
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
                T: float = 1.5) -> float:
        """
        基于 IDM 的自适应巡航加速度计算。
        delta_s: 与前车净距 (m)
        v1: 当前车速 (m/s)
        v2: 前车车速 (m/s)
        v0: 期望车速上限 v_max (m/s)
        返回: 加速度 a (m/s^2)
        """
        safe_des = v1**2/2+1
        a = 2*((delta_s-safe_des)/(T**2)+(0.0*v2-v1)/T)
        if v1>v0:
            a1 = self.compute_acceleration(v0,v1)
            a = min(a,a1)
        return a

    def compute_acceleration(self, target_speed: float, current_speed: float) -> float:
        return (target_speed**2 - current_speed**2)/(2)
    
    def compute_stop_acceleration(self, target_speed:float, current_speed: float, remain_distance: float) -> float:
        remain_distance = max(remain_distance,1e-3)
        if current_speed**2<2*remain_distance and remain_distance>0.1:
            return self.compute_acceleration(self.cruising_speed,current_speed)
        return (target_speed**2 - current_speed**2)/(2*remain_distance)
    
    def pure_pursuit_curvature(self, position: Point2D, heading: float, path: List[Point2D], lookahead: float) -> float:
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

    def get_lane_traffic_light(self, road_id: str, lane_id: str) -> Tuple[str,float]:
        road = self.get_road(road_id)
        lane = self.get_lane(road_id, lane_id)
        light = ('grey',0.0)
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
                                                status = controll.signal_controller.state_with_countdown(t=t)
                                                now_ = status[id]
                                                light = (now_[0].value,now_[1])
        return light

    def agent_route(self, current_agent:TrafficAgent)->None:
        lane_map = ""
        for lane in current_agent.road_map:
            lane_map+="R"+lane.belone_road.road_id+"L"+lane.lane_id+"->"
        self.data_recorder.add_data(current_agent.id,'LaneMap',lane_map)
        current_lane = self.map_parser.lanes[current_agent.current_lane_unicode]
        remain_s = current_lane.length - current_agent.current_s
        if current_agent.road_map:
            hold_lane = current_agent.road_map[-1]
            pd_l = -current_agent.current_s+sum(current_agent.plan_haul)
        else:
            hold_lane = current_lane
            current_agent.road_map.append(current_lane)
            current_agent.ref_line.append(current_lane.get_ref_line())
            pd_l = remain_s
        while pd_l<100:
            hold_lane_id = random.randrange(len(hold_lane.successor))
            hold_lane = self.get_lane(hold_lane.successor[hold_lane_id][0],hold_lane.successor[hold_lane_id][1])
            current_agent.road_map.append(hold_lane)
            current_agent.ref_line.append(hold_lane.get_ref_line())
            current_agent.plan_haul.append(hold_lane.length)
            pd_l+=hold_lane.length

    def get_interactive_agent(self, current_agent:TrafficAgent)->None:
        current_agent.nearerst_agent = None
        current_agent.min_distance = float('inf')
        current_agent.current_s
        #更新前方最近智能体
        for agent in [self.ego_vehicle]+self.agents:
            if agent.id == current_agent.id:
                continue
            pred_length = 0.0
            for check_lane in current_agent.road_map:
                if (check_lane.unicode==agent.current_lane_unicode):
                    agent_s,_,_,_,_ = check_lane.projection(agent.pos)
                    if current_agent.current_lane_unicode==agent.current_lane_unicode:
                        if agent_s > current_agent.current_s:
                            delta_s = agent_s - current_agent.current_s
                            if delta_s < current_agent.min_distance:
                                current_agent.min_distance = delta_s
                                current_agent.nearerst_agent = agent
                    else:
                        delta_s = pred_length + agent_s - current_agent.current_s
                        if delta_s < current_agent.min_distance:
                            current_agent.min_distance = delta_s
                            current_agent.nearerst_agent = agent
                pred_length+=check_lane.length
                # if pred_length-current_s>30:
                #     break

    def agent_longitudinal_control(self, current_agent:TrafficAgent)->float:
        #纵向控制
        current_lane = self.map_parser.lanes[current_agent.current_lane_unicode]
        road = current_lane.belone_road
        current_v = current_agent.speed
        
        acc_cmd = current_agent.a_max
        if current_agent.nearerst_agent is not None:
            if current_agent.min_distance>10:
                logging.debug(f"\t{YELLOW}最近智能体:{current_agent.nearerst_agent.id},在前方{current_agent.min_distance:.3f}m{RESET}")
            else:
                logging.debug(f"\t{RED}最近智能体:{current_agent.nearerst_agent.id},在前方{current_agent.min_distance:.3f}m{RESET}")
            agent_v = current_agent.nearerst_agent.speed
            acc_follow = self.acc_idm(delta_s=current_agent.min_distance-current_agent.length_front-current_agent.nearerst_agent.length_rear, v1=current_v, v2=agent_v, v0=self.crossing_speed if road.junction != "-1" else self.cruising_speed)
            logging.debug(f"\t跟车acc:{acc_follow:.3f}")
            acc_cmd = min(acc_cmd, acc_follow)
            self.data_recorder.add_data(current_agent.id,'ClosestAgentId',current_agent.nearerst_agent.id)
            self.data_recorder.add_data(current_agent.id,'ClosestAgentDis',current_agent.min_distance)
            self.data_recorder.add_data(current_agent.id,'ClosestAgentSpd',agent_v)
            self.data_recorder.add_data(current_agent.id,'FollowAcc',acc_follow)

        
        
        if road.junction != "-1":
            logging.debug(f"\t当前在路口,前方道路type:{road.successor[0]}")
            current_lane = self.map_parser.lanes[current_agent.current_lane_unicode]
            (color,countdown) = self.get_lane_traffic_light(current_lane.belone_road.road_id, current_lane.lane_id)
            if color=='grey':
                acc_in_junc = self.compute_acceleration(self.cruising_speed, current_v)
            else:
                acc_in_junc = self.compute_acceleration(self.crossing_speed, current_v)

            self.data_recorder.add_data(current_agent.id,'JunctionAcc',acc_in_junc)
            logging.debug(f"\t路口acc:{acc_in_junc:.3f}")
            acc_cmd = min(acc_cmd, acc_in_junc)
            current_agent.choise_lock = None
            #TODO 增加路口路权让行机制
        else:
            pred_length = 0.0
            for check_lane in current_agent.road_map:
                if check_lane.belone_road.junction!='-1':
                    (color,countdown) = self.get_lane_traffic_light(check_lane.belone_road.road_id, check_lane.lane_id)
                    lane_remain_s = pred_length-current_agent.current_s
                    signal_remain_s = lane_remain_s-current_agent.length_front-0.5
                    self.data_recorder.add_data(current_agent.id,'Signal',color)
                    self.data_recorder.add_data(current_agent.id,'SignalRemainS',signal_remain_s)

                    if (color == 'green'):
                        logging.debug(f"\t前方{lane_remain_s:.3f}m路口{GREEN}{color}{RESET}灯")
                    elif (color=='red'):
                        logging.debug(f"\t前方{lane_remain_s:.3f}m路口{RED}{color}{RESET}灯")
                    stop_s = abs(current_agent.speed**2/(2*current_agent.a_min))
                    if signal_remain_s>max(stop_s, 30.0):
                        acc_safe = self.compute_acceleration(self.cruising_speed, current_v)
                        self.data_recorder.add_data(current_agent.id,'SignalSafeacc',acc_safe)
                        logging.debug(f"\t灯前安全acc{acc_safe:.3f}")
                        acc_cmd = min(acc_cmd, acc_safe)
                    else: 
                        if color == 'green':
                            if countdown>3.0:
                                acc_green = self.compute_acceleration(self.crossing_speed, current_v)
                                self.data_recorder.add_data(current_agent.id,'JunctionAcc',acc_green)
                                logging.debug(f"\t灯前通行acc{acc_green:.3f}")
                                acc_cmd = min(acc_cmd, acc_green)
                            else:
                                acc_wait = self.compute_stop_acceleration(
                                    target_speed=0.0, current_speed=current_v, remain_distance=signal_remain_s
                                    )
                                self.data_recorder.add_data(current_agent.id,'WaitAcc',acc_wait)
                                logging.debug(f"\t灯前等待acc{acc_wait:.3f}")
                                acc_cmd = min(acc_cmd, acc_wait)
                        elif color == 'grey':
                            acc_in_road = self.compute_acceleration(self.cruising_speed, current_v)
                            self.data_recorder.add_data(current_agent.id,'NoControlJunctionAcc',acc_in_road)
                            logging.debug(f"\t道路acc:{acc_in_road:.3f}")
                            acc_cmd = min(acc_cmd, acc_in_road)
                        else:
                            acc_red =  self.compute_stop_acceleration(
                                target_speed=0.0, current_speed=current_v, remain_distance=signal_remain_s
                                )
                            self.data_recorder.add_data(current_agent.id,'RedStopAcc',acc_red)
                            logging.debug(f"\t红灯acc:{acc_red:.3f}")
                            acc_cmd = min(acc_cmd,acc_red)
                pred_length+=check_lane.length
                if pred_length-current_agent.current_s>30:
                    acc_in_road1 = self.compute_acceleration(self.cruising_speed, current_v)
                    self.data_recorder.add_data(current_agent.id,'RoadAcc',acc_in_road1)
                    logging.debug(f"\t道路acc:{acc_in_road1:.3f}")
                    acc_cmd = min(acc_cmd, acc_in_road1)
                    break
            else:
                logging.debug("\t前方无路")
                acc_cmd = min(acc_cmd, current_agent.a_min)
        
        acc_cmd = min(max(acc_cmd,current_agent.a_min),current_agent.a_max)
        return acc_cmd
    
    def agent_leternal_control(self,current_agent)->float:
        #横向控制

        ref_line = [element for sub_list in current_agent.ref_line for element in sub_list]
        # self.view.add_temp_path(ref_line)

        if current_agent.id == 0:
            curvature_cmd = self.mpc.solve(current_agent,current_agent.ref_line)
        else:
            curvature_cmd = self.pure_pursuit_curvature(
                position=current_agent.pos,
                heading=current_agent.hdg,
                path=ref_line,
                lookahead=5.0
            )
        return curvature_cmd

    def update_states(self)->None:
        clear_output(wait=True)
        self.view.clear_temp_paths()

        for current_agent in [self.ego_vehicle]+self.agents:
            self.data_recorder.add_data(current_agent.id,'sim_time',self.sim_time)

            road_id = current_agent.current_road_index
            lane_id = current_agent.current_lane_index
            current_lane = self.map_parser.lanes[current_agent.current_lane_unicode]
            logging.debug(f"Agent {current_agent.id},R{road_id},L{lane_id}:")
            self.data_recorder.add_data(current_agent.id,'RoadId_LaneId_JuncId',road_id+"_"+lane_id+"_"+current_lane.belone_road.junction)
            current_agent.current_s,current_b,current_out,_,_ = current_lane.projection(current_agent.pos)
            remain_s = current_lane.length - current_agent.current_s
            if current_out:
                logging.debug(f"\t{RED}偏离 R:{road_id},L:{lane_id}{RESET}")
            else:
                logging.debug(f"\ts:{current_agent.current_s:.3f}, 剩余{remain_s:.3f}m, 横向误差:{current_b:.3f}")
            self.data_recorder.add_data(current_agent.id,'RemainS',remain_s)
            self.data_recorder.add_data(current_agent.id,'Cte',current_b)
            
            current_v = current_agent.speed
            self.data_recorder.add_data(current_agent.id,'Speed',current_v)
            
            self.agent_route(current_agent)
            self.get_interactive_agent(current_agent)
 
            acc_cmd = self.agent_longitudinal_control(current_agent=current_agent)
            curvature_cmd = self.agent_leternal_control(current_agent=current_agent)

            logging.debug(f"\t v:{current_v:.3f}, acc_cmd:{acc_cmd:.3f}, cur_cmd:{curvature_cmd:.4f}")
            self.data_recorder.add_data(current_agent.id,'AccCmd',acc_cmd)
            self.data_recorder.add_data(current_agent.id,'CurCmd',curvature_cmd)

            current_agent.step(a_cmd = acc_cmd, cur_cmd = curvature_cmd, dt = self.step)

            #更新当前道路和车道
            if remain_s < 0.1:
                if current_lane.successor:
                    current_agent.current_road_index = current_agent.road_map[1].belone_road.road_id
                    current_agent.current_lane_index = current_agent.road_map[1].lane_id
                    current_agent.current_lane_unicode = current_agent.road_map[1].unicode

                    current_agent.road_map = current_agent.road_map[1:]
                    current_agent.ref_line = current_agent.ref_line[1:]
                    current_agent.plan_haul = current_agent.plan_haul[1:]