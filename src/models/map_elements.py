import numpy as np
from typing import List, Tuple
class Point2D:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

class Control:
    def __init__(self, id , type) -> None:
        self.signal_id = str(id)
        self.type = str(type)

class Controller:
    def __init__(self, id, sequence) -> None:
        self.signal_controller = None #信号灯语计算器
        self.controller_id = id
        self.sequence = sequence
        self.controls = [] # 由Control构成的list
class Signal:
    def __init__(self, id, s, t, type='', from_lane='',to_lane='',turn_relation='') -> None:
        self.signal_id = str(id)
        self.type = type
        self.s = s
        self.t = t
        self.from_lane = from_lane
        self.to_lane = to_lane
        self.turn_relation = turn_relation

# 定义 ReferenceLine 类，用于保存参考线信息
class ReferenceLine:
    def __init__(self):
        self.sampled_points = []  # 每个采样点(x, y)
        self.headings = []        # 对应采样点处的局部切向角（弧度）
        self.geometries = []      # 按序存储原始 geometry 参数字典，字典包含：s, x, y, hdg, length, type，以及 arc/spiral 时的特有参数
        self.s_values = []        # 每个采样点对应的 s 值
class Object:
    def __init__(self, id, type, s, t, z_offset, hdg, width, length) -> None:
        self.crosswalk_id = str(id)
        self.type = type
        self.s = s
        self.t = t
        self.z_offset = z_offset
        self.hdg = hdg
        self.width = width
        self.length = length
        self.corners = []  # 存放该 object 角点坐标


# 定义 Lane 类
class Lane:
    def __init__(self, lane_id, lane_type, sampled_points, headings, travel_dir, lane_change, in_range=False):
        self.lane_id = str(lane_id)      # 车道编号
        self.lane_type = lane_type       # "left" 或 "right"
        self.sampled_points = sampled_points  # [[(),(),()],[(),(),()]]采样点列表，用于绘制车道中心线、左边界、右边界线
        self.travel_dir = str(travel_dir) # 行驶方向包含 undirected 、 backward 、 forward 三种   
        self.lane_change = str(lane_change) # 变道选项包含 none 、 both 、 increase 、 decrease 四种   
        self.headings = headings  # 采样点列表，用于绘制车道中心线
        self.in_range = in_range         # 默认 False
        self.predecessor = []  # [(道路ID，轨道ID)]
        self.successor = []    # [(道路ID，轨道ID)]
        self.midpoint = (0, 0)  # 稍后计算
        self.length = self.calculate_lane_length()
    def compute_midpoint(self)-> None: # 用于计算拓扑图中车道节点位置
        pts = []
        if self.sampled_points:
            pts.append(self.sampled_points[len(self.sampled_points) // 2][0])
        if pts:
            self.midpoint = tuple(np.mean(np.array(pts), axis=0))
        else:
            self.midpoint = (0, 0)
    def calculate_lane_length(self)->float:
        """
        计算车道长度
        """
        if not self.sampled_points or len(self.sampled_points) < 2:
            return 0.0
        centers: List[Tuple[float, float]] = [c[0] for c in self.sampled_points]
        centers_np = np.asarray(centers)        # (N,2)
        seg_vecs = centers_np[1:] - centers_np[:-1]          # (N-1, 2)
        seg_lens = np.linalg.norm(seg_vecs, axis=1)          # 各段长度
        lane_length = np.sum(seg_lens)                        # 总长度
        return lane_length
    
    def calculate_cumulate_s(self, pos: Point2D)->float:
        """
        根据输入点 `pos` 计算其沿车道中心线的累计里程 s (单位: 米)。
        - 如果行驶方向为  forward / undirected  → 返回正值
        - 如果行驶方向为  backward            → 返回负值
        若无法投影（点在中心线延长线之外），返回 None。
        """
        if not self.sampled_points or len(self.sampled_points) < 2:
            return 0                         # 无法计算

        # ------- 1. 取中心线坐标 ------------
        centers: List[Tuple[float, float]] = [c[0] for c in self.sampled_points]
        centers_np = np.asarray(centers)        # (N,2)

        # ------- 2. 逐段寻找投影 ------------
        seg_vecs = centers_np[1:] - centers_np[:-1]          # (N-1, 2)
        seg_lens = np.linalg.norm(seg_vecs, axis=1)          # 各段长度
        lane_length = np.sum(seg_lens)                        # 总长度
        seg_dirs = seg_vecs / seg_lens[:, None]              # 单位向量

        p = np.array([pos.x, pos.y])
        cum_len = 0.0                                       # 已累计 s

        for i, (p0, dir_vec, L) in enumerate(zip(centers_np[:-1], seg_dirs, seg_lens)):
            # 向量投影公式: proj_len = (p - p0)·dir
            proj_len = np.dot(p - p0, dir_vec)
            if 0.0 <= proj_len <= L:                       # 投影落在当前段
                s = cum_len + proj_len
                break
            cum_len += L
        else:                                              # for‑loop 未 break
            s = lane_length                                    # 点不在中心线范围内

        # ------- 3. 根据行驶方向给正负 ------
        if self.travel_dir == "backward":
            s = lane_length-s
        # undirected / forward 保持正值

        return s
    def get_ref_line(self)-> List[Point2D]:
        """
        返回车道中心线的 Point2D 列表：
        - 对于 forward 或 undirected，按采样顺序返回；
        - 对于 backward，返回逆序的采样中心线。
        """
        # 如果没有采样点，返回空列表
        if not self.sampled_points:
            return []

        # 1) 从 sampled_points 中提取中心点 (第一个元素)
        ref_pts = [Point2D(x, y) 
                   for (x, y), (_, _), (_, _) in self.sampled_points]

        # 2) 根据行驶方向决定顺序
        if self.travel_dir == "backward":
            ref_pts.reverse()

        return ref_pts

        

# 定义 Road 类
class Road:
    def __init__(self, road_id, predecessor, successor, junction, type, length, speed_limit=0, on_route=False):
        self.road_id = str(road_id)
        self.predecessor = predecessor  # 例如 (elementType, elementId) 或 None
        self.successor = successor      # 例如 (elementType, elementId, contactPoint) 或 None
        self.junction = str(junction)   # "-1" 表示无 junction，否则为 junction id
        self.type = type        # 
        self.speed_limit = speed_limit  # 
        self.on_route = on_route        # 全局路径规划用，默认 False
        self.lanes = []                 # 存放 Lane 对象（仅解析 type="driving" 的车道）
        self.length = length            # 路段长度（float）
        self.midpoint = (0, 0)          # 稍后计算各车道中点的平均值
        self.reference_line = ReferenceLine()  # 保存该 road 的参考线信息
        self.objects = []  # 存放 road 内的 object 信息
        self.signals = []  # 存放 road 内的 signal 信息
    def compute_midpoint(self): # 用于计算拓扑图中道路节点位置
        pts = []
        for lane in self.lanes:
            if lane.sampled_points:
                pts.append(lane.sampled_points[len(lane.sampled_points) // 2][0])
        if pts:
            self.midpoint = tuple(np.mean(np.array(pts), axis=0))
        else:
            self.midpoint = (0, 0)
            