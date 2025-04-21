import numpy as np

# 定义 ReferenceLine 类，用于保存参考线信息
class ReferenceLine:
    def __init__(self):
        self.sampled_points = []  # 每个采样点(x, y)
        self.headings = []        # 对应采样点处的局部切向角（弧度）
        self.geometries = []      # 按序存储原始 geometry 参数字典，字典包含：s, x, y, hdg, length, type，以及 arc/spiral 时的特有参数
        self.s_values = []        # 每个采样点对应的 s 值
        
# 定义 Lane 类
class Lane:
    def __init__(self, lane_id, lane_type, sampled_points, in_range=False):
        self.lane_id = str(lane_id)      # 车道编号
        self.lane_type = lane_type       # "left" 或 "right"
        self.sampled_points = sampled_points  # 采样点列表，用于绘制车道中心线
        self.in_range = in_range         # 默认 False
        self.predecessor = []  # [(道路ID，轨道ID)]
        self.successor = []    # [(道路ID，轨道ID)]
        
# 定义 Road 类
class Road:
    def __init__(self, road_id, predecessor, successor, junction, length, on_route=False):
        self.road_id = str(road_id)
        self.predecessor = predecessor  # 例如 (elementType, elementId) 或 None
        self.successor = successor      # 例如 (elementType, elementId, contactPoint) 或 None
        self.junction = str(junction)   # "-1" 表示无 junction，否则为 junction id
        self.on_route = on_route        # 全局路径规划用，默认 False
        self.lanes = []                 # 存放 Lane 对象（仅解析 type="driving" 的车道）
        self.length = length            # 路段长度（float）
        self.midpoint = (0, 0)          # 稍后计算各车道中点的平均值
        self.reference_line = ReferenceLine()  # 保存该 road 的参考线信息

    def compute_midpoint(self):
        pts = []
        for lane in self.lanes:
            if lane.sampled_points:
                pts.append(lane.sampled_points[len(lane.sampled_points) // 2])
        if pts:
            self.midpoint = tuple(np.mean(np.array(pts), axis=0))
        else:
            self.midpoint = (0, 0)
            