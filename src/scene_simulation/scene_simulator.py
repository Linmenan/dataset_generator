from ..models.agent import *
from ..parsers.parsers import MapParser
from ..models.map_elements import *
from ..utils.collision_check import is_collision, envelope_collision_check

import plotly.graph_objects as go
import numpy as np
import random
import enum
import asyncio
import time
import math
from datetime import datetime
from typing import Callable, Awaitable, Optional, List
import logging
from IPython.display import clear_output

from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from ..visualization.visualization import SimView
from ..utils.data_recorder import DataRecorder
from ..utils.color_print import RED,RESET,GREEN,YELLOW,BLUE,CYAN,MAGENTA

from ..motion.lateral_mpc_nl import LateralMPC_NL
from ..motion.base_longitudinal_control import *
from ..motion.pure_pursuit import pure_pursuit


from collections import OrderedDict   # 放在其他 import 之后

class Mode(enum.Enum):   # enum 支持位运算 :contentReference[oaicite:0]{index=0}
    SYNC = "sync"
    ASYNC = "async"
    REPLAY  = "replay"

class SceneSimulator:
    def __init__(
            self, 
            window_size = (1000, 600),
            mode:Mode = Mode.SYNC, 
            step:float = 0.05, 
            plot_on:bool = False, 
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
        self.plot_on = plot_on
        self.plot_step = plot_step

        self.sim_frame:int = 0
        self._sim_time: float = 0.0                        # 累积仿真时间
        self._start_wall: Optional[float] = None    # 真实起始秒 (perf_counter)
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.mpc = LateralMPC_NL(
            pred_horizon=5.0, ctrl_horizon=2.0, step_len=1.0, weights=
            {
                'cte':   1.0,
                'ephi':   0.5,
                'curv':  0.005,
                'dcurv': 0.01
            }
            )

        self.cruising_speed = 10.0
        self.crossing_speed = 5.0
        self.lane_changing_probability = 0.01
        self.perception_range = perception_range
        self.ego_vehicle = None
        self.agents = []
        self.map_parser = MapParser(file_path=map_file_path,yaml_path=yaml_path)

        # ---------- UI ----------
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        # 创建并 show 视图
        if self.plot_on:
            self.view = SimView(self, size=window_size)
        
        # ---------- 数据记录器 ----------
        self.data_recorder = DataRecorder(data_path)
        
        # ---------- 日志 ----------
        logging.debug("simulator init")

        # ---------- 回放相关 ----------
        self._replay_ready   = False     # 是否加载了回放文件
        self._replay_data    = {}        # {agent_id: {col: list}}
        self._replay_frames  = 1
        self._replay_index   = 0
        self._data_dt        = 0.1       # 默认 0.1 s ，可由文件推断
        self._replay_ids     = []        # agent 顺序
        self._replay_agents  = OrderedDict()  # 便于稳定迭代顺序
        self.replay_speed = 1.0        # 当前倍速
        self._vis_dt      = 1 / 10.0     # 可视化 10 fps   (自己喜欢可调)
        self._replay_speed_accum  = 0.0        # ← 新增：小数倍速累加器
        self._vis_next_ts = 0.0          # 下次应渲染到的 sim_time

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
        """外部驱动一次步进；根据运行模式调用不同逻辑"""
        if self.mode is Mode.REPLAY:
            self._replay_step_once()
            return

        # ---------- 原同步逻辑保持不变 ----------
        self._sim_time += self.step
        self.sim_frame += 1
        self.update_states()
        if self.plot_on and self.sim_frame % self.plot_step == 0:
            self.view.update()

    def _replay_step_once(self):
        """回放模式推进一帧。到尾帧后自动停止计时器 / 循环。"""
        if not self._replay_ready:
            return
        # A) 计算本 tick 应推进的帧数
        frames_per_tick = (self.replay_speed *
                       self._timer.interval() / 1000.0 /
                       self._data_dt)                          # 结合渲染刷新率,数据帧率,倍速计算推进帧数

        self._replay_speed_accum += frames_per_tick            # 累加跳帧数
        step = int(self._replay_speed_accum)                   # 取整数前进帧数
        if step == 0:                                          # <0.5× 等情况
            return
        self._replay_speed_accum -= step                       # 剩余小数留待下次

        # B) 若剩余帧不足，直接到末帧并暂停
        if self._replay_index + step >= self._replay_frames:
            self._replay_index = self._replay_frames - 1
            self._apply_replay_frame(self._replay_index)
            self.view.update()
            self._timer.stop()
            self.view.replay_finished()
            return

        # C) 正常推进 step 帧（不刷新 UI）
        for _ in range(step):
            self._replay_index += 1
            self._apply_replay_frame(self._replay_index)

        # D) 每 plot_step 帧刷新一次 UI
        if self._replay_index % self.plot_step == 0:
            self.view.update()

    def _apply_replay_frame(self, i: int):
        """把所有智能体更新到第 i 帧，但不触发 view.update()"""
        for aid, ag in self._replay_agents.items():
            row = self._replay_data[aid]
            ag.pos.x, ag.pos.y, ag.pos.yaw = row["PosX"][i], row["PosY"][i], row["Yaw"][i]
            ag.speed = row.get("Speed", [0]*self._replay_frames)[i]
        self._sim_time = self._replay_data[self.ego_vehicle.id]["SimTime"][i]


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
        if self.mode is Mode.REPLAY:
            self.load_replay()
            self._timer = QtCore.QTimer()
            self._timer.setInterval(int(self._vis_dt * 1000))  # 固定 30 fps
            self._timer.timeout.connect(self.step_once)
            # self._timer.start() #不自动播放
            self.app.exec_()
        elif self.mode is Mode.SYNC:
            self._running = True
            if self.plot_on:
                self.view.show()
        else:            # ASYNC
            asyncio.run(self._run_async())


    def stop(self):
        self._running = False
        if hasattr(self, "_timer"):
            self._timer.stop()
            del self._timer
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop())

    def init_ego_vehicle(self):
        while self.ego_vehicle == None: 
            k = random.choice(list(self.map_parser.roads))
            road = self.map_parser.roads[k]
            if (road.junction == "-1"):
                lane = random.choice(road.lanes)
                if lane.length>12.0 and len(lane.sampled_points)>100:
                    index = random.randrange(50,len(lane.sampled_points)-50)
                    rear_point = lane.sampled_points[index][0]
                    hdg = lane.headings[index]
                    speed = 0.0
                    vehicle = EgoVehicle(id=str(0), 
                                        pos=Pose2D(
                                            rear_point[0], 
                                            rear_point[1],
                                            hdg
                                        ),
                                        speed=speed)
                    collision = False
                    for agent in self.agents:
                        if is_collision(vehicle, agent, margin_l=1.0, margin_w=0.0):
                            collision = True
                            break
                    if not collision:
                        self.ego_vehicle = vehicle
                        self.ego_vehicle.current_road_index = road.road_id
                        self.ego_vehicle.current_lane_index = lane.lane_id
                        self.ego_vehicle.current_lane_unicode = lane.unicode

    def generate_traffic_agents(self, number = 0):
        count = 0
        while len(self.agents) < number:   
            k = random.choice(list(self.map_parser.roads))
            road = self.map_parser.roads[k]
            if (road.junction == "-1") :
                lane = random.choice(road.lanes)
                if lane.length>12.0 and len(lane.sampled_points)>100:
                    index = random.randrange(50,len(lane.sampled_points)-50)
                    rear_point = lane.sampled_points[index][0]
                    hdg = lane.headings[index]
                    speed = 0.0
                    traffic_agent = TrafficAgent(id = str(len(self.agents)+1), 
                                                pos=Pose2D(
                                                    rear_point[0], 
                                                    rear_point[1],
                                                    hdg
                                                ), 
                                                speed=speed)
                    collision = False
                    for agent in self.agents:
                        if is_collision(traffic_agent, agent, margin_l=1.0, margin_w=0.0):
                            collision = True
                            break
                    if not collision:
                        traffic_agent.current_road_index = road.road_id
                        traffic_agent.current_lane_index = lane.lane_id
                        traffic_agent.current_lane_unicode = lane.unicode
                        self.agents.append(traffic_agent) 
            count+=1
            if count>10000:
                logging.warning(f"已尝试{count}次，生成智能体{len(self.agents)}/{number},未找到合适位置放置其余{number-len(self.agents)}个车辆，请检查地图文件")
                break

    def update_states(self)->None:
            clear_output(wait=True)
            if self.plot_on:
                self.view.clear_temp_paths()
            else:
                logging.info(f"仿真时间:{self.sim_time:.2f}s")
            for current_agent in [self.ego_vehicle]+self.agents:
                current_agent.bev_perception([self.ego_vehicle]+self.agents) # 感知周边智能体
                current_agent.lane_location(self.map_parser.lanes) # 计算当前车道以及纵向、横向、航向信息
                current_agent.get_route_agent([self.ego_vehicle]+self.agents) # 计算沿途智能体（属于后台作弊）
                current_agent.get_right_of_way(
                    current_lane=self.map_parser.lanes[current_agent.current_lane_unicode], 
                    signal_map=self.map_parser.traffic_lights.values(), 
                    sim_time=self.sim_time
                    ) # 获取路权信息
            for current_agent in [self.ego_vehicle]+self.agents:
                self.data_recorder.add_data(current_agent.id,'SimTime',self.sim_time)
                self.data_recorder.add_data(current_agent.id,'PosX',current_agent.pos.x)
                self.data_recorder.add_data(current_agent.id,'PosY',current_agent.pos.y)
                self.data_recorder.add_data(current_agent.id,'Yaw',current_agent.pos.yaw)
                self.data_recorder.add_data(current_agent.id,'LengthFront',current_agent.length_front)
                self.data_recorder.add_data(current_agent.id,'LengthRear',current_agent.length_rear)
                self.data_recorder.add_data(current_agent.id,'Width',current_agent.width)
                self.data_recorder.add_data(
                    current_agent.id,
                    'RoadId_LaneId_JuncId',
                    current_agent.current_road_index+"_"+current_agent.current_lane_index+"_"+self.map_parser.lanes[current_agent.current_lane_unicode].belone_road.junction
                    )
               
                self.data_recorder.add_data(current_agent.id,'RemainS',current_agent.remain_s)
                self.data_recorder.add_data(current_agent.id,'Cte',current_agent.cte)
                self.data_recorder.add_data(current_agent.id,'Speed',current_agent.speed)
                
                logging.debug(f"Agent {current_agent.id},R{current_agent.current_road_index},L{current_agent.current_lane_index}:")
                if current_agent.out_lane:
                    logging.debug(f"\t{RED}偏离 R:{current_agent.current_road_index},L:{current_agent.current_road_index}{RESET}")
                else:
                    logging.debug(f"\ts:{current_agent.current_s:.3f}, 剩余{current_agent.remain_s:.3f}m, 横向误差:{current_agent.cte:.3f}")
                
                # 规划
                from ..motion.agrnt_route_planner import agent_random_route
                agent_random_route(self.lane_changing_probability, current_agent, self.map_parser.lanes, self.map_parser.lanes_serch) # 智能体道路规划（变道决策）
                
                self.data_recorder.add_data(
                    current_agent.id,
                    "LaneMap",
                    "->".join(f"R{ln.belone_road.road_id}L{ln.lane_id}" for ln in current_agent.road_map)
                )
                # self.data_recorder.add_data(
                #     current_agent.id,
                #     "PlanLaneMap",
                #     "->".join(f"R{ln.belone_road.road_id}L{ln.lane_id}" for ln in current_agent.plan_road_map)
                # )

                # 控制
                acc_cmd = self.agent_longitudinal_control(current_agent=current_agent)
                curvature_cmd = self.agent_leternal_control(current_agent=current_agent)
                
                logging.debug(f"\t v:{current_agent.speed:.3f}, acc_cmd:{acc_cmd:.3f}, cur_cmd:{curvature_cmd:.4f}")
                self.data_recorder.add_data(current_agent.id,'AccCmd',acc_cmd)
                self.data_recorder.add_data(current_agent.id,'CurCmd',curvature_cmd)
                self.data_recorder.add_data(current_agent.id,'Shifting',current_agent.shifting)

                # 更新智能体状态
                current_agent.step(a_cmd = acc_cmd, cur_cmd = curvature_cmd, dt = self.step)
                
                # 数据曲线数据刷新
                if self.plot_on and current_agent.id=="0":
                    self.view.add_data("ego_velocity (m/s)", self.sim_time, current_agent.speed)
                    self.view.add_data("ego_cte (m)", self.sim_time, current_agent.cte)
                    self.view.add_data("ego_ephi (deg)", self.sim_time, current_agent.ephi/math.pi*180)

                #更新当前道路和车道
                if current_agent.remain_s < 0.1:
                    if self.map_parser.lanes[current_agent.current_lane_unicode].successor:
                        
                        current_agent.current_road_index = current_agent.road_map[1].belone_road.road_id
                        current_agent.current_lane_index = current_agent.road_map[1].lane_id
                        current_agent.current_lane_unicode = current_agent.road_map[1].unicode

                        current_agent.road_map = current_agent.road_map[1:]
                        current_agent.ref_line = current_agent.ref_line[1:]
                        current_agent.plan_haul = current_agent.plan_haul[1:]

                        if current_agent.lane_change != (-1,-1):
                            current_agent.plan_road_map.clear()
                            current_agent.plan_ref_line.clear()
                            current_agent.plan_plan_haul.clear()
                            current_agent.lane_change = (-1,-1)
                            current_agent.shifting = False
                            logging.warning(f"Agent {current_agent.id} 变道失败，放弃变道！")

    

    def agent_longitudinal_control(self, current_agent:TrafficAgent)->float:
        #纵向控制
        current_lane = self.map_parser.lanes[current_agent.current_lane_unicode]
        road = current_lane.belone_road
        current_v = current_agent.speed
        
        acc_cmd = current_agent.a_max

        # 智能体
        if current_agent.nearest_route_agent is not None and current_agent.nearest_route_agent:
            nearerst_agent,dis = current_agent.nearest_route_agent[0]
            if current_agent.nearest_route_agent[0][1]>10:
                logging.debug(f"\t{YELLOW}最近智能体:{nearerst_agent.id},在前方{dis:.3f}m{RESET}")
            else:
                logging.debug(f"\t{RED}最近智能体:{nearerst_agent.id},在前方{dis:.3f}m{RESET}")
            agent_v = nearerst_agent.speed
            delta_s = dis-current_agent.length_front-nearerst_agent.length_rear
            acc_follow = acc_idm(delta_s=delta_s, v1=current_v, v2=agent_v, v0=self.crossing_speed if road.junction != "-1" else self.cruising_speed)
            logging.debug(f"\t路线跟车acc:{acc_follow:.3f}")
            acc_cmd = min(acc_cmd, acc_follow)
            self.data_recorder.add_data(current_agent.id,'OnRouteClosestAgentId',nearerst_agent.id)
            self.data_recorder.add_data(current_agent.id,'OnRouteClosestAgentDis',delta_s)
            self.data_recorder.add_data(current_agent.id,'OnRouteClosestAgentSpd',agent_v)
            self.data_recorder.add_data(current_agent.id,'OnRouteFollowAcc',acc_follow)
        # 已碰撞,原地不动
        if current_agent.around_agents.collition_agents:
            acc_cmd = min(acc_cmd,current_agent.a_min)
            logging.warning(
                f"已发生碰撞,当前车id:{current_agent.id},碰撞车id:"+
                ",".join(ag[0].id for ag in current_agent.around_agents.collition_agents)
                )
        # 开阔空间智能体避碰,保底
        if current_agent.around_agents.front_agents:
            current_agent.around_agents.front_agents.sort(key=lambda x: x[3])
            open_agent = None
            min_ttc = float('inf')
            for agent,dis,_,lon in current_agent.around_agents.front_agents:
                col,ttc = envelope_collision_check(current_agent,agent,pred_horizon=3,margin_l=1.0,margin_w=0.0)
                if col:
                    open_agent = agent
                    min_ttc = min(min_ttc,ttc)
                    break
            if open_agent is not None:
                agent_v = open_agent.speed
                acc_follow = acc_idm(delta_s=0.5*min_ttc*current_v, v1=current_v, v2=agent_v, v0=self.crossing_speed if road.junction != "-1" else self.cruising_speed)
                logging.debug(f"\t开阔空间跟车acc:{acc_follow:.3f}")
                acc_cmd = min(acc_cmd, acc_follow)
                self.data_recorder.add_data(current_agent.id,'OpenSpaceClosestAgentId',open_agent.id)
                self.data_recorder.add_data(current_agent.id,'OpenSpaceClosestAgentDis',lon)
                self.data_recorder.add_data(current_agent.id,'OpenSpaceClosestAgentSpd',agent_v)
                self.data_recorder.add_data(current_agent.id,'FollowAcc',acc_follow)
        
        # 路口中路权礼让
        if road.junction != "-1":
            color,_,_ = get_lane_traffic_light(current_lane,self.map_parser.traffic_lights.values(),sim_time=self.sim_time)
            if color=='grey':
                acc_in_junc = compute_acceleration(self.cruising_speed, current_v)
            else:
                acc_in_junc = compute_acceleration(self.crossing_speed, current_v)
            # 增加有信号灯控制路口路权让行机制
            around_agents = current_agent.around_agents.front_agents
            if abs(current_agent.curvature)<0.01:
                around_agents+=current_agent.around_agents.left_agents+current_agent.around_agents.right_agents
            elif current_agent.curvature>=0.01:
                around_agents+=current_agent.around_agents.left_agents
            else:
                around_agents+=current_agent.around_agents.right_agents
            for check_agent,dis,lat,lon in around_agents:
                if check_agent.id == current_agent.id:
                    continue
                if envelope_collision_check(current_agent,check_agent,agent1_speed=max(0.3,current_agent.speed),agent2_speed=max(0.3,check_agent.speed),pred_horizon=3,margin_l=1.0,margin_w=0.0)[0]:
                    if dis>20 and check_agent.way_right_level>current_agent.way_right_level:
                        self.data_recorder.add_data(current_agent.id,'JunctionEnv',"faraway_right_lower_stop_"+check_agent.id)
                        acc_in_junc = min(acc_in_junc,current_agent.a_min)
                        self.data_recorder.add_data(current_agent.id,'JunctionAgent',check_agent.id)
                    else:
                        check_agent_give_way = envelope_collision_check(
                            current_agent,
                            check_agent,
                            agent1_speed=max(0.3,current_agent.speed),
                            agent2_speed=0.0,
                            pred_horizon=3,
                            margin_l=0.0,
                            margin_w=0.0
                            )[0]
                        current_agent_give_way = envelope_collision_check(
                            current_agent,
                            check_agent,
                            agent1_speed=0.0,
                            agent2_speed=max(0.3,check_agent.speed),
                            pred_horizon=3,
                            margin_l=0.0,
                            margin_w=0.0
                            )[0]
                        if not current_agent_give_way and not check_agent_give_way:
                            # 如果双方都可以让行,则看谁路权低,路权低让行
                            if current_agent.way_right_level<check_agent.way_right_level:
                                self.data_recorder.add_data(current_agent.id,'JunctionEnv',"way_right_lower_stop_"+check_agent.id)
                                acc_in_junc = min(acc_in_junc,current_agent.a_min)
                            elif current_agent.way_right_level==check_agent.way_right_level:
                                if current_agent.speed<check_agent.speed:
                                    self.data_recorder.add_data(current_agent.id,'JunctionEnv',"slower_than_stop_"+check_agent.id)
                                    acc_in_junc = min(acc_in_junc,current_agent.a_min)
                                elif current_agent.speed==check_agent.speed:
                                    # 路权速度都相同,则id大的让行(这回不可能一样了)
                                    if current_agent.id>check_agent.id:
                                        self.data_recorder.add_data(current_agent.id,'JunctionEnv',"id_large_than_stop_"+check_agent.id)
                                        acc_in_junc = min(acc_in_junc,current_agent.a_min)
                            else:
                                #路权高,不用管
                                self.data_recorder.add_data(current_agent.id,'JunctionEnv',"way_right_higher_than_"+check_agent.id)
                                pass
                        elif not current_agent_give_way and check_agent_give_way:
                            # 只能当前车让行
                            self.data_recorder.add_data(current_agent.id,'JunctionEnv',"ego_stop_"+check_agent.id)
                            acc_in_junc = min(acc_in_junc,current_agent.a_min)
                        elif current_agent_give_way and not check_agent_give_way:
                            # 只能对方让行,不用管
                            self.data_recorder.add_data(current_agent.id,'JunctionEnv',"other_stop_"+check_agent.id)
                            pass
                        else:
                            # 两者都过不去
                            acc_in_junc = min(acc_in_junc,current_agent.a_min)
                            self.data_recorder.add_data(current_agent.id,'JunctionEnv',"traffic_jam_"+check_agent.id)
                            logging.warning(f"路口内塞车,双方都无法让行,当前车id:{current_agent.id},对方车id:{check_agent.id}")
            
            self.data_recorder.add_data(current_agent.id,'JunctionAcc',acc_in_junc)
            self.data_recorder.add_data(current_agent.id,'WayRightLevel',current_agent.way_right_level)
            logging.debug(f"\t路口acc:{acc_in_junc:.3f}")
            acc_cmd = min(acc_cmd, acc_in_junc)
        
        # 找信号灯
        if (len(current_agent.road_map)>1):
            pred_length = current_agent.remain_s
            for check_lane in current_agent.road_map[1:]:
                if check_lane.belone_road.junction!='-1':
                    color, countdown, _ = get_lane_traffic_light(check_lane,self.map_parser.traffic_lights.values(),sim_time=self.sim_time)
                    signal_remain_s = pred_length-current_agent.length_front-0.5
                    if color == 'grey':
                        # 无控制路口,迭代距离,下一条
                        pred_length+=check_lane.length
                        continue
                    self.data_recorder.add_data(current_agent.id,'Signal',color)
                    self.data_recorder.add_data(current_agent.id,'SignalRemainS',signal_remain_s)

                    if (color == 'green'):
                        logging.debug(f"\t前方{signal_remain_s:.3f}m路口{GREEN}{color}{RESET}灯")
                    elif (color=='red'):
                        logging.debug(f"\t前方{signal_remain_s:.3f}m路口{RED}{color}{RESET}灯")
                    
                    stop_s = abs(current_agent.speed**2/(2*current_agent.a_min))
                    # 信号灯超出制动距离，不用管
                    if signal_remain_s>max(stop_s, 30.0):
                        acc_safe = compute_acceleration(self.cruising_speed, current_v)
                        self.data_recorder.add_data(current_agent.id,'SignalSafeAcc',acc_safe)
                        logging.debug(f"\t灯前安全acc{acc_safe:.3f}")
                        acc_cmd = min(acc_cmd, acc_safe)

                    else:
                        if color == 'green':
                            if countdown>3.0:
                                # 绿灯没闪，路口减速观察通过
                                acc_green = compute_acceleration(self.crossing_speed, current_v)
                                self.data_recorder.add_data(current_agent.id,'CrossingObservationAcc',acc_green)
                                logging.debug(f"\t灯前通行acc{acc_green:.3f}")
                                acc_cmd = min(acc_cmd, acc_green)
                            else:
                                # 绿灯闪烁，不抢灯，停车
                                acc_wait = compute_stop_acceleration(
                                    target_speed=0.0, cruising_speed=self.cruising_speed, current_speed=current_v, remain_distance=signal_remain_s
                                    )
                                self.data_recorder.add_data(current_agent.id,'WaitAcc',acc_wait)
                                logging.debug(f"\t灯前等待acc{acc_wait:.3f}")
                                acc_cmd = min(acc_cmd, acc_wait)
                        else:
                            # 红黄灯，停车等待
                            acc_red =  compute_stop_acceleration(
                                target_speed=0.0, cruising_speed=self.cruising_speed, current_speed=current_v, remain_distance=signal_remain_s
                                )
                            self.data_recorder.add_data(current_agent.id,'RedStopAcc',acc_red)
                            logging.debug(f"\t红灯acc:{acc_red:.3f}")
                            acc_cmd = min(acc_cmd,acc_red)
                pred_length+=check_lane.length

                if pred_length>30:
                    # 前方30m内无信号灯,则可以加速(逻辑存在风险，主要是为了程序加速)
                    acc_in_road1 = compute_acceleration(self.cruising_speed, current_v)
                    self.data_recorder.add_data(current_agent.id,'RoadAcc',acc_in_road1)
                    logging.debug(f"\t道路acc:{acc_in_road1:.3f}")
                    acc_cmd = min(acc_cmd, acc_in_road1)
                    break
            else:
                logging.warning(f"\tAgent {current_agent.id} 前方无路!")
                acc_cmd = min(acc_cmd, current_agent.a_min)
        
        acc_cmd = min(max(acc_cmd,current_agent.a_min),current_agent.a_max)
        return acc_cmd
    
    def agent_leternal_control(self,current_agent:TrafficAgent)->float:
        #横向控制
        ref_line = [element for sub_list in current_agent.ref_line for element in sub_list]
        curvature_cmd = pure_pursuit(
                        pose=current_agent.pos,
                        path=ref_line,
                        lookahead=5.0 
                    )
        if self.plot_on and current_agent.id == "0":
            self.view.add_temp_path(ref_line,color="c",line_width=8,alpha=0.2,z_value=0)

        # 存在变道计划
        if current_agent.lane_change != (-1,-1) and current_agent.plan_ref_line:
            shift_ref_line = [element for sub_list in current_agent.plan_ref_line for element in sub_list]
            shift_curvature_cmd = pure_pursuit(
                        pose=current_agent.pos,
                        path=shift_ref_line,
                        lookahead=10.0
                    )
            can_shift = True
            around_agents = current_agent.around_agents.front_agents
            if abs(shift_curvature_cmd)<0.01:
                around_agents+=current_agent.around_agents.left_agents+current_agent.around_agents.right_agents
            elif shift_curvature_cmd>=0.01:
                around_agents+=current_agent.around_agents.left_agents
            else:
                around_agents+=current_agent.around_agents.right_agents
            for check_agent,dis,lat,lon in around_agents:
                if check_agent.id == current_agent.id:
                    continue
                col,ttc = envelope_collision_check(current_agent,check_agent,agent1_cur=shift_curvature_cmd)
                if col:
                    can_shift = False
                    break

            target_lane = self.map_parser.lanes[current_agent.lane_change[1]]
            s,b,is_out,_,_ = target_lane.projection(current_agent.pos)
            remain_s = target_lane.length - s
            #到路口剩余距离
            for lane in current_agent.plan_road_map[1:]:
                if lane.belone_road.junction!="-1":# and get_lane_traffic_light(lane=lane,controlls=self.map_parser.traffic_lights,sim_time=self.sim_time)[0]!="grey":
                    break
                remain_s+=lane.length
            if remain_s < 20+current_agent.length_front:
                # 变道空间不足,取消变道计划
                current_agent.lane_change = (0,0)
                current_agent.plan_road_map.clear()
                current_agent.plan_ref_line.clear()
                current_agent.plan_plan_haul.clear()
                current_agent.shifting = False
                logging.info(f"Agent {current_agent.id} 变道空间不足,取消变道计划！")
            
            elif not is_out:
                # 完成变道
                logging.info(f"Agent {current_agent.id} 完成变道！")
                current_agent.road_map = current_agent.plan_road_map.copy()  # 浅拷贝
                current_agent.ref_line = current_agent.plan_ref_line.copy()  # 浅拷贝
                current_agent.plan_haul = current_agent.plan_plan_haul.copy()  # 浅拷贝
                current_agent.plan_road_map.clear()
                current_agent.plan_ref_line.clear()
                current_agent.plan_plan_haul.clear()
                current_agent.current_lane_index = target_lane.lane_id
                current_agent.current_road_index = target_lane.belone_road.road_id
                current_agent.current_lane_unicode = target_lane.unicode
                current_agent.lane_change = (-1,-1)
                curvature_cmd = shift_curvature_cmd
                current_agent.shifting = False
            else:
                # 变道中
                if can_shift or current_agent.shifting:
                    logging.info(f"Agent {current_agent.id} 变道中")
                    curvature_cmd = shift_curvature_cmd
                    current_agent.shifting = True
                    self.data_recorder.add_data(current_agent.id,'ChangingLane',current_agent.current_road_index+"_"+current_agent.current_lane_index+"->"+target_lane.belone_road.road_id+"_"+target_lane.lane_id)

        return curvature_cmd

    def load_replay(self, fps: float = None) -> bool:
        """
        读取 Excel 录制文件并进入 REPLAY 模式。
        file_name  只需文件名，路径由构造函数里传入的 data_path 决定
        fps        指定帧率；None 时按 SimTime 差分估算
        """
        ok, agents_dict, file_name = self.data_recorder.load()
        if not ok:
            logging.error("Replay 文件读取失败")
            return False
        self.view.setWindowTitle(f"Replay ({file_name})")
        self.view.show()
        self._replay_data   = agents_dict
        self._replay_ids    = sorted(agents_dict.keys(), key=int)
        self._replay_frames = len(agents_dict[self._replay_ids[0]]["SimTime"])
        self._replay_index  = 0

        self._data_dt = agents_dict[next(iter(agents_dict))]["SimTime"][1] - agents_dict[next(iter(agents_dict))]["SimTime"][0]

        # 创建静态 TrafficAgent 对象池
        self._replay_agents.clear()
        for aid in self._replay_ids:
            d = agents_dict[aid]
            ag = TrafficAgent(
                id   = aid,
                pos  = Pose2D(d["PosX"][0], d["PosY"][0], d["Yaw"][0]),
                length_front = d.get("LengthFront", [1.5])[0],
                length_rear  = d.get("LengthRear",  [1.5])[0],
                width        = d.get("Width",       [0.6])[0],
                speed        = d.get("Speed",       [0.0])[0],
            )
            self._replay_agents[aid] = ag

        # 指定 ego & agents 引用，让 SimView 能拿到
        ego_id = "0" if "0" in self._replay_agents else self._replay_ids[0]
        self.ego_vehicle = self._replay_agents[ego_id]
        self.agents      = [a for a in self._replay_agents.values() if a.id != ego_id]

        self._replay_ready = True
        # ===== 新增：预渲染第 0 帧 =====
        self._replay_index = 0
        self._apply_replay_frame(0)   # 把状态切到首帧
        if hasattr(self, "view") and hasattr(self.view, "update"):
            self.view.update()     # 立即刷新画面
        # 通知可视化更新进度条
        if hasattr(self, "view") and hasattr(self.view, "update_replay_slider"):
            self.view.update_replay_slider()
        
        logging.info(f"Replay 模式载入成功：{len(self._replay_ids)} 个智能体，{self._replay_frames} 帧")

        return True

    def set_replay_speed(self, speed: float):
        """回放模式修改倍速"""
        self.replay_speed = max(0.01, speed)
