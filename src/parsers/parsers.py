import xml.etree.ElementTree as ET
from ..models.map_elements import *
from ..models.signal_elements import *
from ..utils.sampling import *
import yaml


class MapParser:
    def __init__(self, file_path='', yaml_path='') -> None:
        self.file_path = file_path
        self.yaml_path = yaml_path
        self.tree = ET.parse(file_path)
        self.roads = {}
        self.lanes = {}
        self.lane_code = int(0)
        self.traffic_lights = {} # traffic light id 与 Controlle对象构成的字典
        self.parse_oxdr_all()
        self.parse_traffic_lights()

    def parse_oxdr_all(self) -> None:
        """
        解析 xodr 文件，提取参考线、车道及链路信息，并构造 Road 对象（包含 Lane 列表）。
        对于 lane 链接，直接采用 xodr 文件中 lane 链接信息进行赋值，不作默认假设。
        """
        self.roads = {}
        self.lanes = {}
        self.lane_code = int(0)
        root = self.tree.getroot()
        
        # 遍历所有 road 元素，构造 Road 对象
        for road_elem in root.findall('./road'):
            road_id = road_elem.get('id')
            length = float(road_elem.get('length', '0'))
            link = road_elem.find('link')
            type_elem = road_elem.find('type')
            type = ""
            speed_limit = 0
            
            if type_elem is not None:
                type = type_elem.get('type')
                speed_ = type_elem.find("speed")
                factor = 1.0
                if speed_ is not None:
                    speed_limit = float(speed_.get('max','0'))
                    unit = speed_.get('unit', None)
                    if unit is not None:
                        if unit == 'mph':
                            factor = 1609.344 / 3600 
                        elif unit == 'km/h' or unit == 'kmph'or unit == 'kph':
                            factor = 1000 / 3600 

            speed_limit = speed_limit*factor
            predecessor = None
            successor = None
            if link is not None:
                pred = link.find('predecessor')
                succ = link.find('successor')
                if pred is not None:
                    predecessor = (pred.get('elementType'), pred.get('elementId'))
                if succ is not None:
                    successor = (succ.get('elementType'), succ.get('elementId'), succ.get('contactPoint'))
            junction = road_elem.get('junction', "-1")
            road_obj = Road(road_id, predecessor, successor, junction, type, length, speed_limit, on_route=False)
            self.compute_reference_line(road_elem, road_obj.reference_line)
            self.parse_driving_lanes(road_elem, road_obj)
            self.parse_objects(road_elem, road_obj)
            road_obj.compute_midpoint()
            self.roads[road_id] = road_obj


        for current_road in self.roads.values():
            road_id = current_road.road_id
            for current_lane in current_road.lanes:
                lane_id = current_lane.lane_id

                # 1) 检查前驱的“后续”是否包含自己
                for pred in list(current_lane.predecessor):
                    pred_road_id, pred_lane_id = str(pred[0]), str(pred[1])
                    pred_road = self.roads.get(pred_road_id)
                    if not pred_road:
                        continue
                    # 在前驱 road 中找对应的 lane 对象
                    for pl in pred_road.lanes:
                        if pl.lane_id == pred_lane_id:
                            # 如果 pl.successor 中没有当前 lane，就添加
                            if (road_id, lane_id) not in pl.successor:
                                pl.successor.append((road_id, lane_id))
                            break

                # 2) 检查后续的“前驱”是否包含自己
                for succ in list(current_lane.successor):
                    succ_road_id, succ_lane_id = str(succ[0]), str(succ[1])
                    succ_road = self.roads.get(succ_road_id)
                    if not succ_road:
                        continue
                    for sl in succ_road.lanes:
                        if sl.lane_id == succ_lane_id:
                            if (road_id, lane_id) not in sl.predecessor:
                                sl.predecessor.append((road_id, lane_id))
                            break
        # for junc in root.findall('./junction'):

    def parse_objects(self, road_elem, road_obj):
        """
        解析 road 元素中的 objects 元素，提取信号灯、交通标志等对象信息，并构造 Object 对象。
        """
        objects_elem = road_elem.find('objects')
        if objects_elem is not None:
            for obj_elem in objects_elem.findall('object'):
                obj_id = obj_elem.get('id')
                obj_type = obj_elem.get('type')
                s = obj_elem.get('s')
                t = obj_elem.get('t')
                z_offset = obj_elem.get('zOffset')
                hdg = obj_elem.get('hdg')
                width = obj_elem.get('width')
                length = obj_elem.get('length')
                road_obj.objects.append(Object(obj_id, obj_type, s, t, z_offset, hdg, width, length))
        
        signals_elem = road_elem.find('signals')
        if signals_elem is not None:
            for signal_elem in signals_elem.findall('signal'):
                signal_id = signal_elem.get('id')
                s = signal_elem.get('s')
                t = signal_elem.get('t')
                road_obj.signals.append(Signal(id=signal_id, s=s, t=t, type='signal'))

            for signal_reference_elem in signals_elem.findall('signalReference'):
                id = signal_reference_elem.get('id')
                s = signal_reference_elem.get('s')
                t = signal_reference_elem.get('t')
                validity_elem = signal_reference_elem.find('validity')
                if validity_elem is not None:
                    from_lane = validity_elem.get('fromLane')
                    to_lane = validity_elem.get('toLane')
                user_data_elem = signal_reference_elem.find('userData')
                if user_data_elem is not None:
                    vector_signal_elem = user_data_elem.find('vectorSignal')
                    if vector_signal_elem is not None:
                        turn_relation = vector_signal_elem.get('turnRelation')
                        reference = Signal(id=id, s=s, t=t, type='reference', from_lane=from_lane, to_lane=to_lane, turn_relation=turn_relation)
                        road_obj.signals.append(reference)
            
    def parse_traffic_lights(self):
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        root = self.tree.getroot()
        for controller_elem in root.findall('./controller'): 
            id = controller_elem.get('id')
            sequence = controller_elem.get('sequence')
            controller = Controller(id, sequence)
            for control_elem in controller_elem.findall('control'):
                signal_id = control_elem.get('signalId')
                type = control_elem.get('type')
                controller.controls.append(Control(signal_id, type))
            ctrl_cfg = next(
                (item for item in cfg["controllers"] if str(item["id"]) == id),
                None
            )
            if ctrl_cfg is not None:
                controller.signal_controller = SignalController(ctrl_cfg)
            self.traffic_lights[id] = controller

    def compute_reference_line(self, road_elem, ref_line):
        """
        解析 road 的 planView 中所有 geometry 节点，按 s 排序，
        并使用每段的原始参数对全局 s 进行插值计算，
        将计算得到的参考线采样点、局部切向角以及原始 geometry 参数保存到 ref_line 成员中。
        
        ref_line.sampled_points: 全局采样点列表 [(x,y), ...]
        ref_line.headings: 每个采样点对应的局部切向角列表
        ref_line.geometries: 按顺序排列的 geometry 参数字典列表，每个字典包含：s, x, y, hdg, length, type,（及 arc/spiral 参数）
        """
        plan_view = road_elem.find('planView')
        ref_line.sampled_points = []
        ref_line.headings = []
        ref_line.geometries = []
        if plan_view is None:
            # 默认直线参考线
            length = float(road_elem.get('length', '0'))
            num = max(int(length/0.1), 2)
            s_vals = np.linspace(0, length, num=num)
            for s in s_vals:
                ref_line.sampled_points.append((s, 0))
                ref_line.headings.append(0.0)
            ref_line.geometries.append({'s': 0, 'x': 0, 'y': 0, 'hdg': 0.0, 'length': length, 'type': 'line'})
            return

        geoms = plan_view.findall('geometry')
        geoms.sort(key=lambda g: float(g.get('s', '0')))
        for g in geoms:
            seg = {
                's': float(g.get('s', '0')),
                'x': float(g.get('x', '0')),
                'y': float(g.get('y', '0')),
                'hdg': float(g.get('hdg', '0')),
                'length': float(g.get('length', '0'))
            }
            arc_tag = g.find('arc')
            spiral_tag = g.find('spiral')
            if arc_tag is not None:
                seg['type'] = 'arc'
                seg['k'] = float(arc_tag.get('curvature'))
            elif spiral_tag is not None:
                seg['type'] = 'spiral'
                seg['curvStart'] = float(spiral_tag.get('curvStart'))
                seg['curvEnd'] = float(spiral_tag.get('curvEnd'))
            else:
                seg['type'] = 'line'
            ref_line.geometries.append(seg)
        # 遍历全局 s 值，插值计算参考线点和局部切向角
        road_length = ref_line.geometries[-1]['s'] + ref_line.geometries[-1]['length']
        num = max(int(road_length/0.1), 2)
        s_vals = np.linspace(0, road_length, num=num)
        for s in s_vals:
            # 找到 s 所在的 geometry
            seg = None
            for g in ref_line.geometries:
                if s >= g['s'] and s <= g['s'] + g['length']:
                    seg = g
                    break
            if seg is None:
                seg = ref_line.geometries[-1]
                ds = seg['length']
            else:
                ds = s - seg['s']
            if seg['type'] == 'line':
                x_val = seg['x'] + ds * np.cos(seg['hdg'])
                y_val = seg['y'] + ds * np.sin(seg['hdg'])
                hdg_val = seg['hdg']
            elif seg['type'] == 'arc':
                k = seg['k']
                if abs(k) > 1e-8:
                    x_val = seg['x'] + (np.sin(seg['hdg'] + k*ds) - np.sin(seg['hdg']))/k
                    y_val = seg['y'] - (np.cos(seg['hdg'] + k*ds) - np.cos(seg['hdg']))/k
                else:
                    x_val = seg['x'] + ds*np.cos(seg['hdg'])
                    y_val = seg['y'] + ds*np.sin(seg['hdg'])
                hdg_val = seg['hdg'] + k*ds
            elif seg['type'] == 'spiral':
                curvStart = seg['curvStart']
                curvEnd = seg['curvEnd']
                theta = lambda u: seg['hdg'] + curvStart*u + 0.5*(curvEnd-curvStart)/seg['length'] * u**2
                n_sub = 50
                I_x = composite_simpson(lambda u: np.cos(theta(u)), 0, ds, n_sub)
                I_y = composite_simpson(lambda u: np.sin(theta(u)), 0, ds, n_sub)
                x_val = seg['x'] + I_x
                y_val = seg['y'] + I_y
                hdg_val = theta(ds)
            ref_line.sampled_points.append((x_val, y_val))
            ref_line.headings.append(hdg_val)
            ref_line.s_values.append(s)
            
    def parse_driving_lanes(self, road_elem, road_obj):
        """
        从 road 的 <lanes> 部分解析车道信息，
        累积所有车道宽度（包括类型不为 "driving" 的，如 shoulder、median 等），
        但仅对类型为 "driving" 的车道生成 Lane 对象以供可视化。

        利用 road_obj.reference_line 中的 sampled_points、headings 与 s_values，
        以及全局 laneOffset 多项式（a, b, c, d）计算基础偏移，
        对于左侧车道：
            offset_left(s) = global_offset(s) + cum_width_left(s) + [w_current(s)]/2,
        对于右侧车道：
            offset_right(s) = global_offset(s) - (cum_width_right(s) + [w_current(s)]/2),
        其中：
            global_offset(s) = offset_poly(s, sl_offset, a, b, c, d)
            w_current(s) = offset_poly(s, sw_offset, a_w, b_w, c_w, d_w)
            
        本函数在解析 lane 宽度信息的同时，也读取各 lane 内的连接关系信息，
        如果 lane 元素包含 <link> 节点，则提取其中的 <predecessor> 和 <successor> 节点，
        将其 id 属性记录到 Lane 对象的对应字段中（初步存为字符串）。
        
        针对非junction的 Road，如果 Road 的 predecessor/successor 的 elementType 为 "junction"，
        则通过查询 junction 节点，查找对应 connection 中 laneLink 的信息来更新 Lane 的连接关系，
        例如：若 road1 的 predecessor 为 ("junction", "841")，且当前 lane 的 lane_id 为 "3"，
        则在 junction id="841" 中查找 connection 中 incomingRoad 为 road1 且 contactPoint 为 "end"，
        从中查找 laneLink whose from=="3"，将其 to 属性作为 lane1 的 predecessor。
        
        结果直接更新 road_obj.lanes（仅保留 type 为 "driving" 的车道）。
        """
        root = self.tree.getroot()
        junction_dict = {junc.get('id'): junc for junc in root.findall('./junction')}
        # 将 s_values 转为 numpy 数组
        s_arr = np.array(road_obj.reference_line.s_values)
        baseline = road_obj.reference_line.sampled_points
        baseline_headings = road_obj.reference_line.headings

        driving_lanes = []
        lanes_elem = road_elem.find('lanes')
        if lanes_elem is None:
            road_obj.lanes = driving_lanes
            return

        # 提取全局 laneOffset 多项式系数
        lane_offset_elem = lanes_elem.find('laneOffset')
        if lane_offset_elem is not None:
            sl_offset = float(lane_offset_elem.get('s', '0'))
            a_val = float(lane_offset_elem.get('a', '0'))
            b_val = float(lane_offset_elem.get('b', '0'))
            c_val = float(lane_offset_elem.get('c', '0'))
            d_val = float(lane_offset_elem.get('d', '0'))
        else:
            sl_offset = 0.0
            a_val = b_val = c_val = d_val = 0.0

        # 计算全局偏移数组
        global_offset = offset_poly(s_arr, sl_offset, a_val, b_val, c_val, d_val)

        lane_section = lanes_elem.find('laneSection')
        if lane_section is None:
            road_obj.lanes = driving_lanes
            return


        # -------------------------
        # 解析左侧车道（left 节点），lane id 为正，按升序排列
        # -------------------------
        left_elem = lane_section.find('left')
        left_all = []
        if left_elem is not None:
            for lane in left_elem.findall('lane'):
                lane_id = int(lane.get('id'))
                lane_type = lane.get('type')
                width_elem = lane.find('width')
                user_data_elem = lane.find('userData')
                travel_dir = ""
                lane_change = ""
                line_style = ""
                line_color = ""
                if user_data_elem is not None:
                    vector_lane_elem = user_data_elem.find('vectorLane')
                    if vector_lane_elem is not None:
                        travel_dir = vector_lane_elem.get('travelDir','')
                road_mark_elem = lane.find('roadMark')
                if road_mark_elem is not None:
                    lane_change = road_mark_elem.get('laneChange','')
                    line_style = road_mark_elem.get('type','')
                    line_color = road_mark_elem.get('color','')
                    
                if width_elem is not None:
                    sw_offset = float(width_elem.get('sOffset', '0'))
                    a_w = float(width_elem.get('a', '0'))
                    b_w = float(width_elem.get('b', '0'))
                    c_w = float(width_elem.get('c', '0'))
                    d_w = float(width_elem.get('d', '0'))
                else:
                    sw_offset = 0.0; a_w = b_w = c_w = d_w = 0.0
                
                
                # print(f"解析 R {road_obj.road_id} L {lane_id}")
                road_pred = road_obj.predecessor
                road_succ = road_obj.successor
                # print(f"road_pred id {road_pred[1]} type {road_pred[0]}")
                # print(f"road_succ id {road_succ[1]} type {road_succ[0]}")
                
                lane_pred = []
                lane_succ = []
                if road_pred and road_pred[0]=='road':
                    # 读取 lane 的链接关系（若存在）
                    link_elem = lane.find('link')
                    if link_elem is not None:
                        pred_elem = link_elem.find('predecessor')
                        if pred_elem is not None:
                            lane_pred.append((road_pred[1],pred_elem.get('id')))
                            # print(f"找到前续 {(road_pred[1],pred_elem.get('id'))}")
                elif road_pred and road_pred[0]=='junction':
                    junction = junction_dict.get(road_pred[1])
                    for conn in junction.findall('connection'):
                        if conn.get('incomingRoad')==road_obj.road_id:
                            for link in conn.findall('laneLink'):
                                if lane.get('id') == link.get('from'):
                                    lane_pred.append((conn.get('connectingRoad'),link.get('to')))
                                    # print(f"找到前续 {(conn.get('connectingRoad'),link.get('to'))}")
                                    break
              
                                    
                if road_succ and road_succ[0]=='road':
                    # 读取 lane 的链接关系（若存在）
                    link_elem = lane.find('link')
                    if link_elem is not None:
                        succ_elem = link_elem.find('successor')
                        if succ_elem is not None:
                            lane_succ.append((road_succ[1],succ_elem.get('id')))
                            # print(f"找到后继 {(road_succ[1],succ_elem.get('id'))}")
                elif road_succ and road_succ[0]=='junction':
                    junction = junction_dict.get(road_succ[1])
                    for conn in junction.findall('connection'):
                        if conn.get('incomingRoad')==road_obj.road_id:
                            for link in conn.findall('laneLink'):
                                if lane.get('id') == link.get('from'):
                                    lane_succ.append((conn.get('connectingRoad'),link.get('to')))
                                    # print(f"找到后继 {(conn.get('connectingRoad'),link.get('to'))}")
                                    break


                left_all.append((lane_id, sw_offset, a_w, b_w, c_w, d_w, lane_type, lane_pred, lane_succ, travel_dir, lane_change, line_style, line_color))
            left_all.sort(key=lambda x: x[0])
        cum_width_left = np.zeros_like(s_arr)

        for info in left_all:
            lane_id, sw_offset, a_w, b_w, c_w, d_w, lane_type, lane_pred, lane_succ ,travel_dir, lane_change, line_style, line_color = info
            w_current = offset_poly(s_arr, sw_offset, a_w, b_w, c_w, d_w)
            if lane_type == "driving":
                current_offset = global_offset + cum_width_left + w_current/2.0
                left_offset = global_offset + cum_width_left
                right_offset = global_offset + cum_width_left + w_current
                sampled_points = []
                headings = []
                for (pt, local_hdg, offset_val, l_offset_val, r_offset_val) in zip(baseline, baseline_headings, current_offset, left_offset, right_offset):
                    x_ref, y_ref = pt
                    x_lane = x_ref - offset_val * np.sin(local_hdg)
                    y_lane = y_ref + offset_val * np.cos(local_hdg)
                    l_x_lane = x_ref - l_offset_val * np.sin(local_hdg)
                    l_y_lane = y_ref + l_offset_val * np.cos(local_hdg)
                    r_x_lane = x_ref - r_offset_val * np.sin(local_hdg)
                    r_y_lane = y_ref + r_offset_val * np.cos(local_hdg)
                    sampled_points.append([(x_lane, y_lane),(l_x_lane, l_y_lane),(r_x_lane, r_y_lane)])
                    if travel_dir=='backward':
                        headings.append(local_hdg + np.pi)
                    else:
                        headings.append(local_hdg)
                if travel_dir=='backward':
                    w_current_ = w_current[::-1]
                    headings_ = headings[::-1]
                    s_arr_ = np.cumsum(np.diff(s_arr,prepend=0)[::-1])
                    sampled_points_ = sampled_points[::-1]
                else:
                    w_current_ = w_current
                    headings_ = headings
                    s_arr_ = s_arr
                    sampled_points_ = sampled_points
                lane_obj = Lane(
                    road_obj, lane_id, self.lane_code, "left", 
                    sampled_points_, headings=headings_, widths=w_current_, hauls=s_arr_, 
                    in_range=False, travel_dir=travel_dir, lane_change=lane_change, 
                    line_style=line_style, line_color=line_color
                    )
                self.lane_code += int(1)
                # 记录从 lane 自身获取的链接信息（可能只有 lane id，没有 Road 信息）
                if travel_dir=='backward':
                    if lane_pred is not None:
                        lane_obj.successor = lane_pred
                    if lane_succ is not None:
                        lane_obj.predecessor = lane_succ
                else:
                    if lane_pred is not None:
                        lane_obj.predecessor = lane_pred
                    if lane_succ is not None:
                        lane_obj.successor = lane_succ
                lane_obj.compute_midpoint()
                # print(f"road {road_obj.road_id} lane {lane_id} lane_obj:{lane_obj}")
                # print(f"lane_obj.successor:{lane_obj.successor}")
                driving_lanes.append(lane_obj)
            cum_width_left = cum_width_left + w_current

        # -------------------------
        # 解析右侧车道（right 节点），lane id 为负，按降序排列
        # -------------------------
        right_elem = lane_section.find('right')
        right_all = []
        if right_elem is not None:
            for lane in right_elem.findall('lane'):
                lane_id = int(lane.get('id'))
                lane_type = lane.get('type')
                width_elem = lane.find('width')
                user_data_elem = lane.find('userData')
                travel_dir = ""
                lane_change = ""
                line_style = ""
                line_color = ""
                if user_data_elem is not None:
                    vector_lane_elem = user_data_elem.find('vectorLane')
                    if vector_lane_elem is not None:
                        travel_dir = vector_lane_elem.get('travelDir','')
                road_mark_elem = lane.find('roadMark')
                if road_mark_elem is not None:
                    lane_change = road_mark_elem.get('laneChange','')
                    line_style = road_mark_elem.get('type','')
                    line_color = road_mark_elem.get('color','')

                if width_elem is not None:
                    sw_offset = float(width_elem.get('sOffset', '0'))
                    a_w = float(width_elem.get('a', '0'))
                    b_w = float(width_elem.get('b', '0'))
                    c_w = float(width_elem.get('c', '0'))
                    d_w = float(width_elem.get('d', '0'))
                else:
                    sw_offset = 0.0; a_w = b_w = c_w = d_w = 0.0
                    
                
                # print(f"解析 R {road_obj.road_id} L {lane_id}")
                road_pred = road_obj.predecessor
                road_succ = road_obj.successor
                # print(f"road_pred id {road_pred[1]} type {road_pred[0]}")
                # print(f"road_succ id {road_succ[1]} type {road_succ[0]}")
                
                lane_pred = []
                lane_succ = []
                if road_pred and road_pred[0]=='road':
                    # 读取 lane 的链接关系（若存在）
                    link_elem = lane.find('link')
                    if link_elem is not None:
                        pred_elem = link_elem.find('predecessor')
                        if pred_elem is not None:
                            lane_pred.append((road_pred[1],pred_elem.get('id')))
                            # print(f"找到前续 {(road_pred[1],pred_elem.get('id'))}")
                elif road_pred and road_pred[0]=='junction':
                    junction = junction_dict.get(road_pred[1])
                    for conn in junction.findall('connection'):
                        if conn.get('incomingRoad')==road_obj.road_id:
                            for link in conn.findall('laneLink'):
                                if lane.get('id') == link.get('from'):
                                    lane_pred.append((conn.get('connectingRoad'),link.get('to')))
                                    # print(f"找到前续 {(conn.get('connectingRoad'),link.get('to'))}")
                                    break


                if road_succ and road_succ[0]=='road':
                    # 读取 lane 的链接关系（若存在）
                    link_elem = lane.find('link')
                    if link_elem is not None:
                        succ_elem = link_elem.find('successor')
                        if succ_elem is not None:
                            lane_succ.append((road_succ[1],succ_elem.get('id')))
                            # print(f"找到后继 {(road_succ[1],succ_elem.get('id'))}")
                elif road_succ and road_succ[0]=='junction':
                    junction = junction_dict.get(road_succ[1])
                    for conn in junction.findall('connection'):
                        if conn.get('incomingRoad')==road_obj.road_id:
                            for link in conn.findall('laneLink'):
                                if lane.get('id') == link.get('from'):
                                    lane_succ.append((conn.get('connectingRoad'),link.get('to')))
                                    # print(f"找到后继 {(conn.get('connectingRoad'),link.get('to'))}")
                                    break

                                
                right_all.append((lane_id, sw_offset, a_w, b_w, c_w, d_w, lane_type, lane_pred, lane_succ, travel_dir, lane_change, line_style, line_color))
            right_all.sort(key=lambda x: x[0], reverse=True)
        cum_width_right = np.zeros_like(s_arr)

        for info in right_all:
            lane_id, sw_offset, a_w, b_w, c_w, d_w, lane_type, lane_pred, lane_succ ,travel_dir, lane_change, line_style, line_color = info
            w_current = offset_poly(s_arr, sw_offset, a_w, b_w, c_w, d_w)
            if lane_type == "driving":
                current_offset = global_offset - (cum_width_right + w_current/2.0)
                left_offset = global_offset - cum_width_right
                right_offset = global_offset - (cum_width_right + w_current)
                sampled_points = []
                headings = []
                for (pt, local_hdg, offset_val, l_offset_val, r_offset_val) in zip(baseline, baseline_headings, current_offset, left_offset, right_offset):
                    x_ref, y_ref = pt
                    x_lane = x_ref - offset_val * np.sin(local_hdg)
                    y_lane = y_ref + offset_val * np.cos(local_hdg)
                    l_x_lane = x_ref - l_offset_val * np.sin(local_hdg)
                    l_y_lane = y_ref + l_offset_val * np.cos(local_hdg)
                    r_x_lane = x_ref - r_offset_val * np.sin(local_hdg)
                    r_y_lane = y_ref + r_offset_val * np.cos(local_hdg)
                    sampled_points.append([(x_lane, y_lane),(l_x_lane, l_y_lane),(r_x_lane, r_y_lane)])
                    if travel_dir=='backward':
                        headings.append(local_hdg + np.pi)
                    else:
                        headings.append(local_hdg)
                if travel_dir=='backward':
                    w_current_ = w_current[::-1]
                    headings_ = headings[::-1]
                    s_arr_ = np.cumsum(np.diff(s_arr,prepend=0)[::-1])
                    sampled_points_ = sampled_points[::-1]
                else:
                    w_current_ = w_current
                    headings_ = headings
                    s_arr_ = s_arr
                    sampled_points_ = sampled_points
                lane_obj = Lane(
                    road_obj, lane_id, self.lane_code, "right", 
                    sampled_points_, headings=headings_, widths=w_current_, hauls=s_arr_, 
                    in_range=False, travel_dir=travel_dir, lane_change=lane_change, 
                    line_style=line_style, line_color=line_color
                    )
                self.lane_code += int(1)
                if travel_dir=='backward':
                    if lane_pred is not None:
                        lane_obj.successor = lane_pred
                    if lane_succ is not None:
                        lane_obj.predecessor = lane_succ
                else:
                    if lane_pred is not None:
                        lane_obj.predecessor = lane_pred
                    if lane_succ is not None:
                        lane_obj.successor = lane_succ
                lane_obj.compute_midpoint()
                # print(f"road {road_obj.road_id} lane {lane_id} lane_obj:{lane_obj}")
                # print(f"lane_obj.successor:{lane_obj.successor}")
                driving_lanes.append(lane_obj)
            cum_width_right = cum_width_right + w_current

        self.lanes.update({lane.unicode:lane for lane in driving_lanes})
        road_obj.lanes = driving_lanes



    




