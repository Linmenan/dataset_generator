import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from pyqtgraph import TextItem

import math
from typing import List
from ..models.agent import TrafficAgent
from ..models.map_elements import Point2D
import networkx as nx
import numpy as np
import plotly.graph_objects as go

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
    s, c = math.sin(a.hdg), math.cos(a.hdg)
    return [(a.pos.x + lx*c - ly*s,
             a.pos.y + lx*s + ly*c) for lx, ly in local]

def generate_circle_points(center, radius, num_points=50):
    cx, cy = center
    theta = np.linspace(0, 2*np.pi, num_points)
    x_circle = (cx + radius * np.cos(theta)).tolist()
    y_circle = (cy + radius * np.sin(theta)).tolist()
    return x_circle, y_circle

class SimView(QtWidgets.QWidget):
    """
    PyQtGraph 窗口，用于实时渲染车道线和智能体。
    创建时先绘制所有车道；每次 update() 只更新颜色和多边形顶点，
    完全在 GPU/Qt 侧渲染，性能大幅提升。
    """
    def __init__(self, sim, size=(900, 600), title="Real-Time Traffic"):
        super().__init__(parent=None)
        self.sim = sim
        
        self.setWindowTitle(title)     # 标题栏文字
        self.resize(*size)             # 默认宽高
        # ---------- 1) 搭积木 ----------
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)

        # 1.1 绘图区域
        self.canvas = pg.GraphicsLayoutWidget()
        vbox.addWidget(self.canvas, stretch=1)

        self.plot = self.canvas.addPlot(row=0, col=0)
        self.plot.setAspectLocked(True)

        # 1.2 信息栏
        self.info_label = QtWidgets.QLabel('')
        self.info_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.info_label.setMinimumHeight(32)           # 给点高度
        self.info_label.setStyleSheet(
            "background: rgba(255,255,255,180);"
            "font: 12pt 'Arial';"
            "padding: 4px;"
        )
        vbox.addWidget(self.info_label, stretch=0)

        self.lane_curves = {}    # {(road_id, lane_id): PlotDataItem}
        self.agent_items = {}    # {agent.id: QGraphicsPolygonItem}
        self.agent_arrows  = {}   # {agent.id: ArrowItem}
        self.agent_texts = {}  # {agent.id: TextItem}
        self.text_style = {
            "color": "black",
            "font-size": "10pt",
            "anchor": (0.5, 0.5),  # 文本锚点位于安全盒顶部中心
        }
        self.lane_curves      = {}    # {(road_id, lane_id, 'c/l/r'): PlotDataItem}
        self.lane_countdowns  = {}    # {(road_id, lane_id): TextItem}   # <— 新增

        self._init_lanes()

        # 新增：临时路径存储
        self._temp_paths = {}      # {name or id: PlotDataItem}
        


        
    def add_temp_path(
        self,
        path_points: List[Point2D],
        pen: pg.mkPen = None,
        name: str = None
    ) -> pg.PlotDataItem:
        """
        在视图中绘制一条临时路径（折线）。
        
        参数：
            path_points: 一串 (x, y) 点，用于绘制折线
            pen:        一个 pyqtgraph.mkPen 对象，用于设置颜色、宽度等
            name:       可选的标识符，用于后续移除，若为 None 则自动生成
        返回：
            对应的 PlotDataItem
        """
        # 生成默认 pen
        if pen is None:
            pen = pg.mkPen(color='blue', width=2)  # 红色，宽度 2

        # 从 Point2D 列表中提取 x, y
        xs = [pt.x for pt in path_points]
        ys = [pt.y for pt in path_points]


        # 绘制
        item = self.plot.plot(xs, ys, pen=pen)
        
        # 存储
        key = name or id(item)
        self._temp_paths[key] = item
        return item

    def clear_temp_paths(self) -> None:
        """
        移除所有通过 add_temp_path 绘制的临时路径。
        """
        for key, item in self._temp_paths.items():
            self.plot.removeItem(item)
        self._temp_paths.clear()
    
    def _init_lanes(self):
        """首次绘制所有车道中心线为灰色虚线。"""
        """首次绘制车道并在中线中点放一个空白 countdown 文本。"""
        for road in self.sim.map_parser.roads.values():
            for lane in road.lanes:
                if not lane.sampled_points:
                    continue

                # 拆解：center, left, right
                centers = np.array([c[0] for c in lane.sampled_points])
                lefts   = np.array([c[1] for c in lane.sampled_points])
                rights  = np.array([c[2] for c in lane.sampled_points])
                
                # 设置横向偏移量
                offset = 0.1

                # 计算法向量，通过切向角旋转 90 度得到
                normal_vectors = np.array([[-np.sin(heading), np.cos(heading)] for heading in lane.headings])
                # 对 lefts 和 rights 采样点进行偏移
                offset_lefts = lefts - offset * normal_vectors
                offset_rights = rights + offset * normal_vectors
                
                lane_change = lane.lane_change
                
                if lane_change == 'both':
                    l_bound_line_style = QtCore.Qt.DashLine
                    r_bound_line_style = QtCore.Qt.DashLine
                elif lane_change == 'increase':
                    l_bound_line_style = QtCore.Qt.DashLine
                    r_bound_line_style = QtCore.Qt.SolidLine
                elif lane_change == 'decrease':
                    l_bound_line_style = QtCore.Qt.SolidLine
                    r_bound_line_style = QtCore.Qt.DashLine
                else:
                    l_bound_line_style = QtCore.Qt.SolidLine
                    r_bound_line_style = QtCore.Qt.SolidLine
                # 中心线
                curve_c = self.plot.plot(
                    centers[:,0], centers[:,1],
                    pen=pg.mkPen('grey', width=0.1, style=QtCore.Qt.DashLine)
                )
                self.lane_curves[(road.road_id, lane.lane_id, 'c', lane_change)] = curve_c

                # ----- Countdown 文字 -----
                # mid_x, mid_y = centers[len(centers) // 2]
                mid_x, mid_y = centers[0]
                # txt = TextItem('', color='black', anchor=(0.5, -0.3))
                txt = TextItem('', color='black', anchor=(0.0, 0.0))
                txt.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
                txt.setPos(mid_x, mid_y)
                self.plot.addItem(txt)
                self.lane_countdowns[(road.road_id, lane.lane_id)] = txt

                # 左边界
                curve_l = self.plot.plot(
                    offset_lefts[:,0], offset_lefts[:,1],
                    pen=pg.mkPen('grey', width=1, style=l_bound_line_style)
                )
                self.lane_curves[(road.road_id, lane.lane_id, 'l', lane_change)] = curve_l

                # 右边界
                curve_r = self.plot.plot(
                    offset_rights[:,0], offset_rights[:,1],
                    pen=pg.mkPen('grey', width=1, style=r_bound_line_style)
                )
                self.lane_curves[(road.road_id, lane.lane_id, 'r', lane_change)] = curve_r
    

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
            color, countdown = self.sim.get_lane_traffic_light(rid, lid)
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
            center_pos = (ag.pos.x+ag.length_front*np.cos(ag.hdg), ag.pos.y+ag.length_front*np.sin(ag.hdg))
            angle_deg = 180.0-math.degrees(ag.hdg)  # PyQtGraph 默认以水平向右为 0°
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
            # ---- 新增：更新ID文本 ----
            # 计算文本位置：安全盒顶部中点（取左前和右前点的中点）
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

        # ---- 3) 更新外部信息标签 ----
        t = self.sim.sim_time
        ego = self.sim.ego_vehicle.pos
        if ego is not None and t is not None:
            self.info_label.setText(
                f"Time: {t:.2f}s    "
                f"Ego: ({ego.x:.2f}, {ego.y:.2f})"
            )




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