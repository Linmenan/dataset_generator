import networkx as nx
import numpy as np
import plotly.graph_objects as go
from typing import List
from ..models.agent import TrafficAgent
from ..scene_simulation.scene_simulator import SceneSimulator
import math

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

def visualize_traffic_agents(fig: go.FigureWidget,
                             traffic_agents) -> go.FigureWidget:
    """
    更新（或新增）TrafficAgent 安全盒及其箭头、标签。
    """
    # 1) 安全盒 traces
    for ag in traffic_agents:
        name = 'ego_vehicle' if ag.id == '0' else f'agent{ag.id}'
        xs, ys = zip(*box_points(ag))
        default_color = "#fc5f00" if ag.id == '0' else "royalblue"
        fill_color    = "rgba(255,0,0,0.22)" if ag.id == '0' else "rgba(65,105,225,0.22)"

        # 查找已有的“闭合”scatter trace（fill="toself"）:
        matching = [t for t in fig.data
                    if isinstance(t, go.Scatter) 
                       and t.name == name 
                       and t.fill == "toself"]
        if matching:
            # 就地更新
            trace = matching[0]
            trace.x = xs
            trace.y = ys
            trace.line.color = default_color
            trace.fillcolor = fill_color
        else:
            # 不存在则新增
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                name=name, fill="toself", fillcolor=fill_color,
                line=dict(color=default_color, width=2),
                showlegend=True
            ))

    # 2) 清除旧 annotations 再添加箭头与标签
    fig.layout.annotations = []

    for ag in traffic_agents:
        default_color = "#fc5f00" if ag.id == '0' else "royalblue"
        fx = ag.pos.x + 0.7 * ag.length_front * math.cos(ag.hdg)
        fy = ag.pos.y + 0.7 * ag.length_front * math.sin(ag.hdg)

        # 2.1 箭头
        fig.add_annotation(
            x=fx, y=fy, xref="x", yref="y",
            ax=ag.pos.x, ay=ag.pos.y, axref="x", ayref="y",
            arrowhead=3, arrowwidth=1.5, arrowcolor=default_color,
            showarrow=True
        )
        # 2.2 标签
        fig.add_annotation(
            x=ag.pos.x, y=ag.pos.y,
            text=ag.id, showarrow=False,
            font=dict(color=default_color)
        )

    return fig

def visualize_lanes(fig: go.FigureWidget,
                    sim: SceneSimulator) -> go.FigureWidget:
    """
    首次绘制或更新所有车道线。首次若不存在对应 trace 则新增，
    否则就地更新 x, y 和颜色。
    """
    roads = sim.map_parser.roads

    # 遍历所有 road / lane
    for road in roads.values():
        for lane in road.lanes:
            if not lane.sampled_points:
                continue

            # 解压坐标
            xs, ys, lxs, lys, rxs, rys = zip(*[
                (x, y, lx, ly, rx, ry)
                for (x, y), (lx, ly), (rx, ry) in lane.sampled_points
            ])

            # 统一 label：即使 junction=-1 也给个不重复的 name
            label_center = f"R{road.road_id}L{lane.lane_id}"
            label_left   = label_center + "_L"
            label_right  = label_center + "_R"

            # 颜色：路口内实时查询，否则灰色
            if road.junction != "-1":
                color = sim.get_lane_traffic_light(road.road_id, lane.lane_id)
            else:
                color = 'grey'

            # ------- 1) 中心线 -------

            # 在 fig.data 中查找既有中心线 trace（mode='lines' 且 name=label_center）
            match = next((t for t in fig.data
                          if isinstance(t, go.Scatter)
                             and t.mode == 'lines'
                             and t.name == label_center), None)
            if match:
                # 更新坐标与样式
                match.x = xs
                match.y = ys
                match.line.color = color
                match.line.width = 2
                match.line.dash  = 'dash'
            else:
                # 首次不存在则新增
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode='lines',
                    name=label_center, showlegend=True,
                    line=dict(color=color, width=2, dash="dash")
                ))

            # ------- 2) 左边界 -------

            match = next((t for t in fig.data
                          if isinstance(t, go.Scatter)
                             and t.mode == 'lines'
                             and t.name == label_left), None)
            if match:
                match.x = lxs
                match.y = lys
            else:
                fig.add_trace(go.Scatter(
                    x=lxs, y=lys, mode='lines',
                    name=label_left, showlegend=False,
                    line=dict(color='grey', width=2, dash="solid")
                ))

            # ------- 3) 右边界 -------

            match = next((t for t in fig.data
                          if isinstance(t, go.Scatter)
                             and t.mode == 'lines'
                             and t.name == label_right), None)
            if match:
                match.x = rxs
                match.y = rys
            else:
                fig.add_trace(go.Scatter(
                    x=rxs, y=rys, mode='lines',
                    name=label_right, showlegend=False,
                    line=dict(color='grey', width=2, dash="solid")
                ))

    return fig

# def visualize_traffic_agents(fig, traffic_agents)->go.Figure:
#     for idx, ag in enumerate(traffic_agents):
#         default_color: str = "royalblue"
#         fill_color = "rgba(65,105,225,0.22)"      # royalblue 带透明度
#         name = ''
#         if idx==0:
#             default_color = "#fc5f00"
#             fill_color = "rgba(255,0,0,0.22)"
#             name = 'ego_vehicle'
#         else:
#             name = 'agent'+ag.id
#         xs, ys = zip(*box_points(ag))
#         # 1) 安全盒
#         fig.add_trace(go.Scatter(
#             x=xs, y=ys, mode="lines",
#             line=dict(color=default_color, width=2),
#             fill="toself", fillcolor=fill_color,
#             name=name,
#             showlegend=True,
#             )
#         )

#         # 2) 朝向箭头
#         fx = ag.pos.x + 0.7 * ag.length_front * math.cos(ag.hdg)
#         fy = ag.pos.y + 0.7 * ag.length_front * math.sin(ag.hdg)

#         fig.add_annotation(
#             x=fx, y=fy, xref="x", yref="y",
#             ax=ag.pos.x, ay=ag.pos.y, axref="x", ayref="y",
#             arrowhead=3, arrowwidth=1.5,
#             arrowcolor=default_color,
#             showarrow=True
#         )

#         # 3) 标签
#         fig.add_annotation(
#             x=ag.pos.x, y=ag.pos.y,
#             text=ag.id, showarrow=False,
#             font=dict(color=default_color)
#         )

#     # fig.update_layout(
#         # title="TrafficAgent safety boxes (single color)",
#         # xaxis=dict(scaleanchor="y", scaleratio=1),
#         # yaxis=dict(scaleanchor="x", scaleratio=1, visible=False),
#         # legend_title="Agents",
#         # width=700, height=500
#     # )
#     return fig

# def visualize_lanes(fig, sim:SceneSimulator)->go.Figure:
#     roads = sim.map_parser.roads
#     for road in roads.values():
#         for lane in road.lanes:
#             if not lane.sampled_points:
#                 continue
#             xs, ys, lxs, lys, rxs, rys, = list(zip(*[(x, y, lx, ly, rx, ry)
#                   for (x, y), (lx, ly), (rx, ry) in lane.sampled_points]))
            
#             if road.junction != "-1":
#                 color = sim.get_lane_traffic_light(road.road_id, lane.lane_id)
#             else:
#                 color = 'grey'
#             # label = f'Road {road.road_id} Lane {lane.lane_id} ({lane.lane_type})'
                
#             if road.junction =='-1':
#                 label = f'R{road.road_id}L{lane.lane_id}'
#             else:
#                 label = f'R{road.road_id}L{lane.lane_id}J{road.junction}'
#             fig.add_trace(go.Scatter(
#                 x=xs, y=ys,
#                 mode='lines',
#                 line=dict(color=color, width=2, dash="dash"),
#                 name=label,
#                 showlegend=True,
#             ))
#             fig.add_trace(go.Scatter(
#                 x=lxs, y=lys,
#                 mode='lines',
#                 line=dict(color='grey', width=2, dash="solid"),
#                 name=label,
#                 showlegend=False,
#             ))
#             fig.add_trace(go.Scatter(
#                 x=rxs, y=rys,
#                 mode='lines',
#                 line=dict(color='grey', width=2, dash="solid"),
#                 name=label,
#                 showlegend=False,
#             ))
#     return fig
    
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