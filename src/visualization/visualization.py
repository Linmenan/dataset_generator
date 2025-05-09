import math
from typing import List
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from pyqtgraph import TextItem

from ..models.agent import TrafficAgent
from ..models.map_elements import Point2D

import networkx as nx
import plotly.graph_objects as go
import random

# 常见论文颜色（Tableau 10 + Set1）
PAPER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#a65628", "#f781bf",
]

# 线型可选项（Qt 支持的 pen style）
LINE_STYLES = [
    QtCore.Qt.SolidLine,
    QtCore.Qt.DashLine,
    QtCore.Qt.DotLine,
    QtCore.Qt.DashDotLine,
    QtCore.Qt.DashDotDotLine,
]

# 全局把所有新建窗口的背景设成白色
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')  # 如果想把文字/坐标轴改成黑色
pg.setConfigOptions(antialias=True)

def box_points(a: TrafficAgent):
    """返回 [(x0,y0)…x4,y4]，首尾重复以闭合。左后→左前→右前→右后"""
    hw = a.width * 0.5
    local = [(-a.length_rear, -hw),
             ( a.length_front, -hw),
             ( a.length_front,  hw),
             (-a.length_rear,  hw),
             (-a.length_rear, -hw)]
    s, c = math.sin(a.pos.yaw), math.cos(a.pos.yaw)
    return [(a.pos.x + lx*c - ly*s,
             a.pos.y + lx*s + ly*c) for lx, ly in local]

def generate_circle_points(center, radius, num_points=50):
    cx, cy = center
    theta = np.linspace(0, 2*np.pi, num_points)
    x_circle = (cx + radius * np.cos(theta)).tolist()
    y_circle = (cy + radius * np.sin(theta)).tolist()
    return x_circle, y_circle

class SimView(QtWidgets.QMainWindow):
    def __init__(self, sim, size=(900, 600), title="Real-Time Traffic"):
        super().__init__()
        self.sim = sim
        self.setWindowTitle(title)
        self.resize(*size)

        # ========================
        # 菜单栏：视图选项
        # ========================
        self.menu_bar = self.menuBar()
        view_menu = self.menu_bar.addMenu("视图选项")

        # ========================
        # 1.1 绘图区域（左侧）
        # ========================
        self.canvas = pg.GraphicsLayoutWidget()
        self.plot = self.canvas.addPlot()
        self.plot.setAspectLocked(True)

        self.dock_plot = QtWidgets.QDockWidget("地图场景", self)
        self.dock_plot.setObjectName("地图场景")
        self.dock_plot.setWidget(self.canvas)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock_plot)
        plot_action = self.dock_plot.toggleViewAction()  # 显式获取动作
        view_menu.addAction(plot_action)
        self.dock_plot.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.dock_plot.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable | 
            QtWidgets.QDockWidget.DockWidgetMovable | 
            QtWidgets.QDockWidget.DockWidgetFloatable
            )

        # ========================
        # 1.2 信息栏（右上）
        # ========================
        self.info_label = QtWidgets.QLabel('')
        self.info_label.setMinimumHeight(80)
        self.info_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.info_label.setStyleSheet(
            "background: rgba(255,255,255,180);"
            "font: 12pt 'Arial';"
            "padding: 4px;"
        )
        self.dock_info = QtWidgets.QDockWidget("信息栏", self)
        self.dock_info.setObjectName("信息栏")
        self.dock_info.setWidget(self.info_label)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock_info)

        info_action = self.dock_info.toggleViewAction()  # 显式获取动作
        view_menu.addAction(info_action)
        self.dock_info.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable | 
            QtWidgets.QDockWidget.DockWidgetMovable | 
            QtWidgets.QDockWidget.DockWidgetFloatable
            )

        # ========================
        # 1.3 数据曲线（右下）
        # ========================
        self.data_plot_widget = pg.PlotWidget(title="")
        self.data_plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.data_plot_widget.setBackground((240, 240, 240, 255))  # 浅灰背景
        self.legend = self.data_plot_widget.addLegend(offset=(500, 20))
        self.data_plot_widget.setLabel('left', 'Value')
        self.data_plot_widget.setLabel('bottom', 'Time', 's')

        self.data_lines = {}
        self.color_idx = 0  # 颜色循环索引
        self.style_idx = 0  # 线型循环索引

        self.dock_curve = QtWidgets.QDockWidget("数据曲线", self)
        self.dock_curve.setObjectName("数据曲线")
        self.dock_curve.setWidget(self.data_plot_widget)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock_curve)  # 暂时放右边
        
        self.splitDockWidget(self.dock_info, self.dock_curve, QtCore.Qt.Vertical)  # 将其垂直放在 info 下方
        curve_action = self.dock_curve.toggleViewAction()  # 显式获取动作
        view_menu.addAction(curve_action)
        self.dock_curve.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable | 
            QtWidgets.QDockWidget.DockWidgetMovable | 
            QtWidgets.QDockWidget.DockWidgetFloatable
            )
        self.resizeDocks([self.dock_plot, self.dock_info], [700, 300], QtCore.Qt.Horizontal)
        # ========================
        # 可视化内容
        # ========================

        self.lane_curves = {}
        self.agent_items = {}
        self.agent_arrows = {}
        self.agent_texts = {}
        self.lane_countdowns = {}
        self._temp_paths = {}
        
        self.text_style = {
            "color": "black",
            "font-size": "10pt",
            "anchor": (0.5, 0.5),
        }

        self._init_lanes()

        
    def add_temp_path(
        self,
        path_points: List[Point2D],
        name: str = None,
        color: str = "blue",
        line_width: float = 2.0,
        line_style: str = "solid",
        alpha: float = 1.0,
        z_value: float = 0  # 新增层级控制参数
    ) -> None:
        """
            path_points: 一串 Point2D 点，用于绘制折线
            name:       可选的标识符，用于后续移除（若为 None 则自动生成）
            color:      线条颜色（支持颜色名称/十六进制/RGB/RGBA，如 "red", "#FF0000", "rgb(255,0,0)", "rgba(255,0,0,0.5)"）
            line_width: 线条宽度（默认 2.0）
            line_style: 线型（"solid", "dash", "dot", "dashdot", "dashdotdot"）
            alpha:      透明度（0.0完全透明 ~ 1.0完全不透明，默认1.0）
        """
        # 线型映射字典
        style_map = {
            "solid": QtCore.Qt.SolidLine,
            "dash": QtCore.Qt.DashLine,
            "dot": QtCore.Qt.DotLine,
            "dashdot": QtCore.Qt.DashDotLine,
            "dashdotdot": QtCore.Qt.DashDotDotLine,
        }
        
        # 处理颜色和透明度
        if isinstance(color, str) and color.startswith('rgba'):
            # 直接使用RGBA字符串（如 "rgba(255,0,0,0.5)"）
            pen_color = color
        else:
            # 将颜色名称/HEX/RGB转换为QColor并添加透明度
            qcolor = pg.mkColor(color)
            qcolor.setAlphaF(alpha)  # 设置透明度
            pen_color = qcolor
        
        # 创建画笔（Pen）
        pen = pg.mkPen(
            color=pen_color,
            width=line_width,
            style=style_map.get(line_style.lower(), QtCore.Qt.SolidLine)
        )
        
        # 提取坐标
        xs = [pt.x for pt in path_points]
        ys = [pt.y for pt in path_points]
        
        # 绘制路径
        item = self.plot.plot(xs, ys, pen=pen)
        item.setZValue(z_value)  # 设置层级
        # 存储引用
        key = name or f"temp_path_{id(item)}"
        self._temp_paths[key] = item

    def clear_temp_paths(self) -> None:
        """
        移除所有通过 add_temp_path 绘制的临时路径。
        """
        for key, item in self._temp_paths.items():
            self.plot.removeItem(item)
        self._temp_paths.clear()
    

    def _init_lanes(self):
        for road in self.sim.map_parser.roads.values():
            for lane in road.lanes:
                if not lane.sampled_points:
                    continue
                centers = np.array([c[0] for c in lane.sampled_points])
                lefts   = np.array([c[1] for c in lane.sampled_points])
                rights  = np.array([c[2] for c in lane.sampled_points])

                normal_vectors = np.array([[-np.sin(h), np.cos(h)] for h in lane.headings])
                offset = 0.1
                offset_lefts = lefts - offset * normal_vectors
                offset_rights = rights + offset * normal_vectors

                lane_change = lane.lane_change
                style_map = {
                    'both': (QtCore.Qt.DashLine, QtCore.Qt.DashLine),
                    'increase': (QtCore.Qt.DashLine, QtCore.Qt.SolidLine),
                    'decrease': (QtCore.Qt.SolidLine, QtCore.Qt.DashLine),
                }
                l_style, r_style = style_map.get(lane_change, (QtCore.Qt.SolidLine, QtCore.Qt.SolidLine))

                curve_c = self.plot.plot(centers[:, 0], centers[:, 1],
                                         pen=pg.mkPen('grey', width=0.1, style=QtCore.Qt.DashLine))
                self.lane_curves[(road.road_id, lane.lane_id, 'c', lane_change)] = curve_c

                mid_x, mid_y = centers[0]
                txt = TextItem('', color='black', anchor=(0.0, 0.0))
                txt.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
                txt.setPos(mid_x, mid_y)
                self.plot.addItem(txt)
                self.lane_countdowns[(road.road_id, lane.lane_id)] = txt

                curve_l = self.plot.plot(offset_lefts[:, 0], offset_lefts[:, 1],
                                         pen=pg.mkPen('grey', width=1, style=l_style))
                self.lane_curves[(road.road_id, lane.lane_id, 'l', lane_change)] = curve_l

                curve_r = self.plot.plot(offset_rights[:, 0], offset_rights[:, 1],
                                         pen=pg.mkPen('grey', width=1, style=r_style))
                self.lane_curves[(road.road_id, lane.lane_id, 'r', lane_change)] = curve_r
    
    def add_data(self, dataname:str, x:float, y:float)->None:
        if dataname not in self.data_lines:
            self.data_lines[dataname] = {"x": [x], "y": [y]}
        self.data_lines[dataname]["x"].append(x)
        self.data_lines[dataname]["y"].append(y)

    def update(self):
        """每步仿真后调用：更新车道颜色 & 智能体安全盒。"""
        # 1) 车道颜色
        # ---------- 1)  车道信号 ----------
        for (rid, lid, kind, lane_change), curve in self.lane_curves.items():
            

            road = self.sim.get_road(rid)
            if road.junction == "-1":      # 没有信号控制
                # curve.setPen(pg.mkPen('grey', width=1, style=QtCore.Qt.DashLine))
                self.lane_countdowns[(rid, lid)].setText('')   # 清空数字
                continue

            # 有信号灯
            color, countdown, _ = self.sim.get_lane_traffic_light(rid, lid)
            if color!='grey':
                if kind == 'c':      # 中心线
                    # curve.setPen(pg.mkPen('grey', width=0.1)) 
                    continue
                curve.setPen(pg.mkPen(color))

                # ---- 显示倒计时（向上取整）----
                txt_item = self.lane_countdowns[(rid, lid)]
                countdown_show = ''
                countdown = math.ceil(countdown)
                #设置红绿灯倒计时超出显示模式
                if countdown>30:
                    countdown_show = 'H'
                else:
                    countdown_show = str(countdown)
                txt_item.setText('('+countdown_show+')')
                # 文字颜色同信号色，更直观
                txt_item.setColor(color)


        # ---- 2) 更新智能体多边形 & 箭头 ----
        for ag in [self.sim.ego_vehicle] + self.sim.agents:
            # 2.1 多边形安全盒
            pts = box_points(ag)
            poly = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in pts])

            item = self.agent_items.get(ag.id)
            if item is None:
                item = QtWidgets.QGraphicsPolygonItem(poly)
                pen   = pg.mkPen('#fc5f00' if ag.id=='0' else 'royalblue', width=2)
                brush = pg.mkBrush(255, 0, 0, 50) if ag.id=='0' else pg.mkBrush(65,105,225,50)
                item.setPen(pen)
                item.setBrush(brush)
                self.plot.addItem(item)
                self.agent_items[ag.id] = item
            else:
                item.setPolygon(poly)

            # 2.2 朝向箭头
            center_pos = (ag.pos.x+ag.length_front*np.cos(ag.pos.yaw), ag.pos.y+ag.length_front*np.sin(ag.pos.yaw))
            angle_deg = 180.0-math.degrees(ag.pos.yaw)  # PyQtGraph 默认以水平向右为 0°
            arrow = self.agent_arrows.get(ag.id)
            if arrow is None:
                # 新建箭头
                pen_arrow = pg.mkPen('#fc5f00' if ag.id=='0' else 'royalblue', width=2)
                brush_arrow = pg.mkBrush('#fc5f00' if ag.id=='0' else 'royalblue')
                arrow = pg.ArrowItem(
                    pos=center_pos,
                    angle=angle_deg,
                    pen=pen_arrow,
                    brush=brush_arrow,
                    headLen=10
                )
                self.plot.addItem(arrow)
                self.agent_arrows[ag.id] = arrow
            else:
                # 更新位置和角度
                arrow.setPos(center_pos[0], center_pos[1])
                arrow.setStyle(angle=angle_deg)

            text_item = self.agent_texts.get(ag.id)
            if text_item is None:
                text_item = TextItem(
                    text=ag.id,
                    color=self.text_style["color"],
                    anchor=self.text_style["anchor"],
                )
                text_item.setFont(QtGui.QFont("Arial", 10))
                self.plot.addItem(text_item)
                self.agent_texts[ag.id] = text_item
            text_item.setPos(ag.pos.x, ag.pos.y)

        t = self.sim.sim_time
        ego = self.sim.ego_vehicle

        # 更新信息栏
        self.info_label.setText(
            f"Time: {t:.2f}s\n"
            f"Ego: (x: {ego.pos.x:.2f} m, y: {ego.pos.y:.2f} m, yaw: {ego.pos.yaw / math.pi * 180:.1f} deg)\n"
            f"speed: {ego.speed:.1f} m/s"
        )

        # 更新所有已注册数据曲线
        for name in list(self.data_lines.keys()):  # 使用list避免字典修改异常
            data = self.data_lines[name]
            # 自动创建曲线（如果未初始化）
            if "curve" not in data:
                # 自动分配颜色和线型
                color = PAPER_COLORS[self.color_idx % len(PAPER_COLORS)]
                style = LINE_STYLES[self.style_idx % len(LINE_STYLES)]
                self.color_idx += 1
                self.style_idx += 1

                # 创建曲线对象
                pen = pg.mkPen(color=color, width=2, style=style)
                curve = self.data_plot_widget.plot([], [], name=name, pen=pen)
                data["curve"] = curve
                data["color"] = color
                data["style"] = style


            data["curve"].setData(data["x"], data["y"])


        # 设置时间轴范围（最近10秒）
        window_width = 10.0
        start_time = max(0.0, t - window_width) if t is not None else 0.0
        end_time = max(window_width, t) if t is not None else window_width
        self.data_plot_widget.setXRange(start_time, end_time, padding=0)


def build_topology_graph_lanes(roads):
    """
    构建 lane 级别的拓扑图：
      - 节点为每个包含采样点的 Lane，节点ID 格式为 "roadID_laneID"；
      - 节点位置为 Lane 采样点列表中位于中点处的点；
      - 边基于 Lane.predecessor 和 Lane.successor 建立：
          如果当前 Lane 的 predecessor 非 None，则认为其目标 Lane 位于所属 Road 的前驱 Road 中，
          即构造新键为 (road.predecessor[1], lane.predecessor)；
          同理，successor 使用 (road.successor[1], lane.successor)；
      - 附带属性：road_id, lane_id, lane_type, junction.
    """
    G = nx.DiGraph()
    # 添加 Lane 节点；节点ID 以 "roadID_lane_laneID" 命名
    for road in roads.values():
        for lane in road.lanes:
            if not lane.sampled_points:
                continue
   
            pos = lane.midpoint
            node_id = f"{road.road_id}_lane_{lane.lane_id}"
            # 保存所属 road 的 junction 信息（用于后续着色）
            G.add_node(node_id, pos=pos, road_id=road.road_id, lane_id=lane.lane_id,
                       lane_type=lane.lane_type, junction=road.junction)
    
    # 构建辅助字典，key 为 (road_id, lane_id)，value 为节点ID
    lane_node_dict = {}
    for road in roads.values():
        for lane in road.lanes:
            if not lane.sampled_points:
                continue
            node_id = f"{road.road_id}_lane_{lane.lane_id}"
            lane_node_dict[(road.road_id, lane.lane_id)] = node_id
    # print(f"lane_node_dict{lane_node_dict}")
    # 添加边：检查每个 Lane 的 predecessor 和 successor
    for road in roads.values():
        for lane in road.lanes:
            # print(f"road{road.road_id} lane{lane.lane_id}")
            if not lane.sampled_points:
                continue
            current_node = lane_node_dict.get((road.road_id, lane.lane_id))
            if not current_node:
                continue

            # 处理 predecessor
            if lane.predecessor is not None:
                for pred in lane.predecessor:
                    if road.predecessor is not None:
                        if pred in lane_node_dict:
                            pred_node = lane_node_dict[pred]
                            pos1 = np.array(G.nodes[pred_node]['pos'])
                            pos2 = np.array(G.nodes[current_node]['pos'])
                            weight = 0
                            # print(f"添加前续边 from {pred_node} to {current_node}")
                            G.add_edge(pred_node, current_node, weight=weight)

            # 处理 successor
            if lane.successor is not None:
                for succ in lane.successor:
                    if road.successor is not None:
                        if succ in lane_node_dict:
                            succ_node = lane_node_dict[succ]
                            pos1 = np.array(G.nodes[current_node]['pos'])
                            pos2 = np.array(G.nodes[succ_node]['pos'])
                            weight = 0
                            # print(f"添加后继边 from {current_node} to {succ_node}")
                            G.add_edge(current_node, succ_node, weight=weight)
    return G

def build_topology_graph_roads(roads):
    """
    构建 Road 级别的拓扑图，然后将所有 junction != "-1" 的 Road 节点合并为 aggregated 节点。
    
    具体步骤：
      1. 构建 Road 节点，其位置取 Road.midpoint，边根据 predecessor/successor 关系构建，
         权重为两节点的欧氏距离；
      2. 将所有 junction != "-1" 的 Road 节点分组，每组生成一个 aggregated 节点，
         该节点位置取组内所有节点的中点平均值，并重构与外部节点的边关系；
         
    返回：合并后的 networkx.DiGraph 图。
    """
    # 构建 Road 拓扑图
    G = nx.DiGraph()
    for road in roads.values():
        # 仅当 road 内存在 driving 车道时才加入图中
        if not road.lanes:
            continue
        is_junction_road = (road.junction != "-1")
        G.add_node(road.road_id,
                   pos=road.midpoint,
                   junction=road.junction,
                   road_id=road.road_id,
                   is_junction_road=is_junction_road)
    # 添加边：基于 predecessor/successor 关系
    for road in roads.values():
        if road.road_id not in G.nodes:
            continue
        pred = road.predecessor
        succ = road.successor
        if pred is not None:
            etype, eid = pred
            if etype == "road" and eid in G.nodes:
                pos1 = np.array(G.nodes[eid]['pos'])
                pos2 = np.array(G.nodes[road.road_id]['pos'])
                weight = 0
                G.add_edge(eid, road.road_id, weight=weight)
        if succ is not None:
            etype, eid, cp = succ
            if etype == "road" and eid in G.nodes:
                pos1 = np.array(G.nodes[road.road_id]['pos'])
                pos2 = np.array(G.nodes[eid]['pos'])
                weight = 0
                G.add_edge(road.road_id, eid, weight=weight)
    
    # 简化图：合并所有 junction != "-1" 的 Road 节点
    groups = {}
    for node, data in G.nodes(data=True):
        junction = data.get("junction", "-1")
        if junction != "-1":
            groups.setdefault(junction, []).append(node)
    
    simplified_G = nx.DiGraph()
    # 添加所有非 junction 节点（junction == "-1"）
    non_group_nodes = [node for node, data in G.nodes(data=True) if data.get("junction", "-1") == "-1"]
    for node in non_group_nodes:
        simplified_G.add_node(node, **G.nodes[node])
    # 添加非分组节点间的边
    for u, v, d in G.edges(data=True):
        if u in non_group_nodes and v in non_group_nodes:
            simplified_G.add_edge(u, v, **d)
    # 对每个 junction 分组生成 aggregated 节点
    for junc, nodes in groups.items():
        positions = [np.array(G.nodes[n]['pos']) for n in nodes if 'pos' in G.nodes[n]]
        center = tuple(np.mean(positions, axis=0)) if positions else (0, 0)
        agg_node = f"Junction_{junc}"
        # 注意：aggregated 节点也使用橙色显示，junction 信息保留
        simplified_G.add_node(agg_node, pos=center, junction=junc, is_junction_road=True)
        # 对该组内的所有节点，将它们与外部的边重写到 aggregated 节点
        for n in nodes:
            for u, v, d in G.out_edges(n, data=True):
                if v in nodes:
                    continue
                else:
                    simplified_G.add_edge(agg_node, v, **d)
            for u, v, d in G.in_edges(n, data=True):
                if u in nodes:
                    continue
                else:
                    simplified_G.add_edge(u, agg_node, **d)
    return simplified_G

def visualize_topology_combined(roads, detailed=True):
    """
    使用 Plotly 可视化拓扑图：
      - detailed=True：显示每个 lane 的节点，节点位置为 lane 采样点的中点，
                         边基于 lane 的 predecessor/successor 关系；
      - detailed=False：显示 Road 之间的拓扑关系，节点位置取 Road.midpoint，
                         并对 junction 路段进行聚合显示。
      对于属于 junction（junction != "-1"）的节点，均显示为橙黄色。
    """
    if detailed:
        G = build_topology_graph_lanes(roads)
    else:
        G = build_topology_graph_roads(roads)
        
    pos = nx.get_node_attributes(G, 'pos')
    
    # 构建边集
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x, node_y, node_text, node_color = [], [], [], []
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # 在 detailed 模式下显示 lane 信息，同时判断所属 road 的 junction 状态
        if detailed:
            # text = f"{data['road_id']}\nLane {data['lane_id']} ({data['lane_type']})"
            text = f"{data['road_id']}_{data['lane_id']}"
        else:
            text = str(node)
        node_text.append(text)
        # 若节点关联的 junction 不为 "-1"（包括aggregated节点），则使用橙色显示，否则使用天蓝色
        if data.get('junction', "-1") != "-1":
            node_color.append('orange')
        else:
            node_color.append('skyblue')
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_color,
            size=20,
            line_width=2
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(text='Topology Graph'if detailed else 'Topology Graph (Simplify)', font=dict(size=16)),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False),
                        width=700,
                        height=500,
                    ))
    fig.show()