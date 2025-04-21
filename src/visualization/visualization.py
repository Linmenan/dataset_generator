import networkx as nx
import plotly.graph_objects as go

def generate_circle_points(center, radius, num_points=50):
    cx, cy = center
    theta = np.linspace(0, 2*np.pi, num_points)
    x_circle = (cx + radius * np.cos(theta)).tolist()
    y_circle = (cy + radius * np.sin(theta)).tolist()
    return x_circle, y_circle

def visualize_lanes(roads, current_pos, sensing_range):
    fig = go.Figure()
    for road in roads.values():
        for lane in road.lanes:
            if not lane.sampled_points:
                continue
            xs, ys = zip(*lane.sampled_points)
            if lane.in_range:
                color = 'blue'
            elif road.junction != "-1":
                color = '#f27a0d'
            else:
                color = 'grey'
            # label = f'Road {road.road_id} Lane {lane.lane_id} ({lane.lane_type})'
            if road.junction =='-1':
                label = f'R{road.road_id}L{lane.lane_id}'
            else:
                label = f'R{road.road_id}L{lane.lane_id}J{road.junction}'
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode='lines',
                line=dict(color=color, width=2),
                name=label
            ))
        
    cx, cy = current_pos
    fig.add_trace(go.Scatter(
        x=[cx], y=[cy],
        mode='markers',
        marker=dict(color='red', size=12, symbol='x'),
        name='Current Position'
    ))
    circle_x, circle_y = generate_circle_points(current_pos, sensing_range)
    fig.add_trace(go.Scatter(
        x=circle_x, y=circle_y,
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Sensing Range'
    ))
    fig.update_layout(
        title='Driving Lanes Visualization',
        xaxis_title='X',
        yaxis_title='Y',
        legend_title='Legend',
        xaxis=dict(scaleanchor="y", scaleratio=1),
        template="plotly_white",
        width=700,
        height=500,
    )
    fig.show()
    
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
            mid_index = len(lane.sampled_points) // 2
            pos = lane.sampled_points[mid_index]
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