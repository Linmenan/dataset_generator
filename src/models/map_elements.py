import numpy as np
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
        self.controller_id = id
        self.sequence = sequence
        self.controls = {} # 由signal_id与Control构成的字典

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
class Signal:
    def __init__(self, id, s, t, z_offset) -> None:
        self.signal_id = str(id)
        self.s = s
        self.t = t
        self.z_offset = z_offset
        self.reference = []  # 存放该 signal 的参考(id, from_lane, to_lane, turn_relation)

# 定义 Lane 类
class Lane:
    def __init__(self, lane_id, lane_type, sampled_points, headings, travel_dir, lane_change, in_range=False):
        self.lane_id = str(lane_id)      # 车道编号
        self.lane_type = lane_type       # "left" 或 "right"
        self.sampled_points = sampled_points  # 采样点列表，用于绘制车道中心线
        self.travel_dir = str(travel_dir) # 行驶方向包含 undirected 、 backward 、 forward 三种   
        self.lane_change = str(lane_change) # 变道选项包含 none 、 both 、 increase 、 decrease 四种   
        self.headings = headings  # 采样点列表，用于绘制车道中心线
        self.in_range = in_range         # 默认 False
        self.predecessor = []  # [(道路ID，轨道ID)]
        self.successor = []    # [(道路ID，轨道ID)]
        self.midpoint = (0, 0)  # 稍后计算
    def compute_midpoint(self)-> None: # 用于计算拓扑图中车道节点位置
        pts = []
        if self.sampled_points:
            pts.append(self.sampled_points[len(self.sampled_points) // 2])
        if pts:
            self.midpoint = tuple(np.mean(np.array(pts), axis=0))
        else:
            self.midpoint = (0, 0)
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
                pts.append(lane.sampled_points[len(lane.sampled_points) // 2])
        if pts:
            self.midpoint = tuple(np.mean(np.array(pts), axis=0))
        else:
            self.midpoint = (0, 0)
            