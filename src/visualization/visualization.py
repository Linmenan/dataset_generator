import math
from typing import List
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from pyqtgraph import TextItem

from ..models.agent import TrafficAgent
from ..utils.geometry import Point2D
from ..sensor.traffic_light import get_lane_traffic_light

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

    def __init__(self, sim, size=(900, 600)):
        super().__init__()
        self.sim = sim
        # 根据 sim.mode 生成标题
        from ..scene_simulation.scene_simulator import Mode   # 防循环引用

        mode_title = {
            Mode.SYNC:  "Real-Time Traffic — 同步模式",
            Mode.ASYNC: "Real-Time Traffic — 异步模式",
            Mode.REPLAY: ""
        }.get(sim.mode, f"")

        self.setWindowTitle(mode_title)
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
        if self.sim.mode in (Mode.SYNC, Mode.ASYNC):          # ← 只有实时仿真才建曲线窗
            self._init_curve_dock(view_menu)
        else:                                                 # ← REPLAY：把控件放这里
            self._init_replay_controls(QtCore.Qt.RightDockWidgetArea, view_menu)
        
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
        # === 仅回放模式才绘制控制栏 ===
        if self.sim.mode is Mode.REPLAY:
            self.update_replay_slider()

    def _init_curve_dock(self, menu):
        self.data_plot_widget = pg.PlotWidget(title="")
        self.data_plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.data_plot_widget.setBackground((240, 240, 240, 255))
        self.legend = self.data_plot_widget.addLegend(offset=(500, 20))
        self.data_plot_widget.setLabel("left",   "Value")
        self.data_plot_widget.setLabel("bottom", "Time", "s")
        self.data_lines, self.color_idx, self.style_idx = {}, 0, 0

        self.dock_curve = QtWidgets.QDockWidget("数据曲线", self)
        self.dock_curve.setWidget(self.data_plot_widget)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock_curve)
        self.splitDockWidget(self.dock_info, self.dock_curve, QtCore.Qt.Vertical)
        menu.addAction(self.dock_curve.toggleViewAction())

    def _init_replay_controls(self, area, menu):
        """回放模式：构建两行控制栏（按钮行 + 进度条行）"""
        self.ctrl_dock = QtWidgets.QDockWidget("播放控制", self)
        self.addDockWidget(area, self.ctrl_dock)
        menu.addAction(self.ctrl_dock.toggleViewAction())

        root = QtWidgets.QWidget(); self.ctrl_dock.setWidget(root)
        vbox = QtWidgets.QVBoxLayout(root); vbox.setContentsMargins(6, 4, 6, 4)

        # ---------- 第 1 行：五个按钮 ----------
        h_btn = QtWidgets.QHBoxLayout()
        self.btn_first = QtWidgets.QPushButton("⏮")   # ↩ 回到第一帧
        self.btn_prev  = QtWidgets.QPushButton("⏪")   # ← 上一帧
        self.btn_play  = QtWidgets.QPushButton()       # ▶/❚❚ 由状态决定
        self.btn_next  = QtWidgets.QPushButton("⏩")   # → 下一帧
        self.btn_last  = QtWidgets.QPushButton("⏭")   # ↪ 跳到最后一帧
        for btn in (self.btn_first, self.btn_prev, self.btn_play, self.btn_next, self.btn_last):
            h_btn.addWidget(btn)
        vbox.addLayout(h_btn)

        # ---------- 第 2 行：进度条 + 倍速框 ----------
        h_seek = QtWidgets.QHBoxLayout()
        self.sld_prog = QtWidgets.QSlider(QtCore.Qt.Horizontal)   # 原生 QSlider
        self.sld_prog.setMinimum(0)
        self.sld_prog.setMaximum(max(1, getattr(self.sim, "_replay_frames", 1)) - 1)
        self.sld_prog.setPageStep(1)              # 单击滑槽走 1 帧（可保留，也可去掉）

        self.cmb_speed = QtWidgets.QComboBox()
        self.cmb_speed.addItems(
            ["0.5×", "0.75×", "1.0×", "1.25×", "1.5×", "2.0×", "5.0×", "10.0×"]
        )
        self.cmb_speed.setCurrentIndex(2)         # 默认 1.0×

        h_seek.addWidget(self.sld_prog, 1)        # 伸缩因子 1 → 进度条占满
        h_seek.addWidget(self.cmb_speed)
        vbox.addLayout(h_seek)

        # ---------- 初始播放状态 ----------
        timer_active = bool(getattr(self.sim, "_timer", None) and self.sim._timer.isActive())
        self._playing = timer_active
        self.btn_play.setText("❚❚" if timer_active else "▶")

        # ---------- 信号连接 ----------
        self.btn_first.clicked.connect(lambda: self._seek_frame_and_pause(0))
        self.btn_last .clicked.connect(lambda: self._seek_frame_and_pause(self.sim._replay_frames - 1))
        self.btn_prev .clicked.connect(lambda: self._seek_delta(-1))
        self.btn_next .clicked.connect(lambda: self._seek_delta(1))
        self.btn_play .clicked.connect(self._toggle_play)
        self.sld_prog.valueChanged.connect(self._seek_abs)
        self.cmb_speed.currentIndexChanged.connect(self._speed_changed)

    # def eventFilter(self, obj, ev):
    #     """让原生 QSlider 支持“点击任意位置 → 直接 Seek”"""
    #     if obj is getattr(self, "sld_prog", None) and ev.type() == QtCore.QEvent.MouseButtonPress:
    #         if ev.button() == QtCore.Qt.LeftButton and self.sld_prog.maximum() > self.sld_prog.minimum():
    #             # 兼容 PyQt5 / PyQt6 的 event.pos() / event.position()
    #             pos_x = ev.position().x() if hasattr(ev, "position") else ev.pos().x()
    #             ratio = pos_x / self.sld_prog.width()
    #             value = round(self.sld_prog.minimum()
    #                         + ratio * (self.sld_prog.maximum() - self.sld_prog.minimum()))
    #             self.sld_prog.setValue(value)   # 会触发 valueChanged → _seek_abs
    #             return True                     # 事件已处理
    #     # 交给父类处理其余事件
    #     return super().eventFilter(obj, ev)

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
                self.lane_curves[(road.road_id, lane.lane_id, lane.unicode, 'c', lane_change)] = curve_c

                mid_x, mid_y = centers[0]
                txt = TextItem('', color='black', anchor=(0.0, 0.0))
                txt.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
                txt.setPos(mid_x, mid_y)
                self.plot.addItem(txt)
                self.lane_countdowns[(road.road_id, lane.lane_id)] = txt

                curve_l = self.plot.plot(offset_lefts[:, 0], offset_lefts[:, 1],
                                         pen=pg.mkPen('grey', width=1, style=l_style))
                self.lane_curves[(road.road_id, lane.lane_id, lane.unicode, 'l', lane_change)] = curve_l

                curve_r = self.plot.plot(offset_rights[:, 0], offset_rights[:, 1],
                                         pen=pg.mkPen('grey', width=1, style=r_style))
                self.lane_curves[(road.road_id, lane.lane_id, lane.unicode, 'r', lane_change)] = curve_r
    
    def add_data(self, dataname:str, x:float, y:float)->None:
        if dataname not in self.data_lines:
            self.data_lines[dataname] = {"x": [x], "y": [y]}
        self.data_lines[dataname]["x"].append(x)
        self.data_lines[dataname]["y"].append(y)

    def update(self):
        """每步仿真后调用：更新车道颜色 & 智能体安全盒。"""
        # 1) 车道颜色
        # ---------- 1)  车道信号 ----------
        for (rid, lid, unicode, kind, lane_change), curve in self.lane_curves.items():
            road = self.sim.map_parser.lanes[unicode].belone_road
            if road.junction == "-1":      # 没有信号控制
                # curve.setPen(pg.mkPen('grey', width=1, style=QtCore.Qt.DashLine))
                self.lane_countdowns[(rid, lid)].setText('')   # 清空数字
                continue

            # 有信号灯
            color, countdown, _ = get_lane_traffic_light(self.sim.map_parser.lanes[unicode],self.sim.map_parser.traffic_lights.values(),sim_time=self.sim.sim_time)
            
            
            if color!='grey':
                if kind == 'c':      # 中心线
                    # curve.setPen(pg.mkPen('grey', width=0.1)) 
                    continue
                curve.setPen(pg.mkPen("#cccc00" if color == "yellow" else color))

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
                txt_item.setColor("#cccc00" if color == "yellow" else color)


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
        from ..scene_simulation.scene_simulator import Mode
        if self.sim.mode in (Mode.SYNC, Mode.ASYNC):
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
        
        if self.sim.mode is Mode.REPLAY and hasattr(self, "sld_prog"):
            # 防止 setValue 触发 valueChanged → _seek_abs → 再次 update 的循环
            self.sld_prog.blockSignals(True)
            self.sld_prog.setValue(self.sim._replay_index)
            self.sld_prog.blockSignals(False)


    # ---------- 控制栏槽函数 ----------
    def _toggle_play(self):
        # 1) 如果当前正在播放 → 执行暂停
        if self._playing:
            self._playing = False
            self.btn_play.setText("▶")
            if hasattr(self.sim, "_timer"):
                self.sim._timer.stop()
            return
        # 2) 当前处于暂停，需要开始播放
        self._playing = True
        self.btn_play.setText("❚❚")
        # 2-A 若已停在最后一帧，自动跳回第 0 帧并刷新画面/滑块
        if self.sim._replay_index >= self.sim._replay_frames - 1:
            self.sim._replay_index = 0
            self._show_replay_frame(0)            # 立即把首帧渲染出来

        # 2-B 复位时间基准：让下一 tick 从当前 sim_time 开始累积
        if hasattr(self.sim, "_vis_next_ts"):
            self.sim._vis_next_ts = self.sim._sim_time    # 关键行
        if hasattr(self.sim, "_replay_speed_accum"):
            self.sim._replay_speed_accum = 0.0            # 若用了小数累加器，也清零

        # 2-C 重设速度并启动计时器
        self.sim.set_replay_speed(self.sim.replay_speed)  # 内部会 stop→set→start

        # 2-D 启动定时器
        if hasattr(self.sim, "_timer"):
            self.sim._timer.start()
    
    def replay_finished(self):
        """
        回放帧序列播放到最后时由 SceneSimulator 调用。
        作用：暂停计时器、把 ▶/❚❚ 按钮恢复到“播放”状态，
        并允许用户拖动滑块或点击 ▶ 重新播放。
        """
        if hasattr(self, "_playing"):
            self._playing = False
        if hasattr(self, "btn_play"):
            self.btn_play.setText("▶")
        # 进度条已经在 _replay_step_once 中停在最后一帧，无需额外处理
        
    def _seek_delta(self, step: int):
        self._seek_frame_and_pause(self.sim._replay_index + step)

    def _seek_abs(self, idx: int):
        """
        跳到绝对帧 idx。
        供进度条 valueChanged 信号、以及 _seek_delta 调用。
        """
        idx = max(0, min(idx, self.sim._replay_frames - 1))
        self.sim._replay_index = idx
        self._show_replay_frame(idx)

        # 更新滑块而不触发递归信号
        self.sld_prog.blockSignals(True)
        self.sld_prog.setValue(idx)
        self.sld_prog.blockSignals(False)

    # ---------- 公用：暂停播放 ----------
    def _pause_playback(self):
        if getattr(self, "_playing", False):
            self._playing = False
            if hasattr(self.sim, "_timer"):
                self.sim._timer.stop()
            if hasattr(self, "btn_play"):
                self.btn_play.setText("▶")

    def _seek_frame_and_pause(self, idx: int):
        self._pause_playback()     # ① 先暂停
        self._seek_frame(idx)      # ② 再跳帧
    
    def _seek_frame(self, idx: int):
        idx = max(0, min(idx, self.sim._replay_frames - 1))
        self.sim._replay_index = idx
        self._show_replay_frame(idx)
        self.sld_prog.blockSignals(True)
        self.sld_prog.setValue(idx)
        self.sld_prog.blockSignals(False)

    def _speed_changed(self, idx: int):
        """
        处理倍速下拉框的变化。idx 为当前索引（0..4）。
        文本格式形如 '1.25×'，去掉末尾 '×' 后转成 float。
        """
        txt = self.cmb_speed.itemText(idx).rstrip("×")
        try:
            speed = float(txt)
        except ValueError:
            speed = 1.0
        self.sim.set_replay_speed(speed)


    # ---------- 仅做可视化刷新，不递增帧 ----------
    def _show_replay_frame(self, i: int):
        if not (0 <= i < self.sim._replay_frames):
            return
        for aid, ag in self.sim._replay_agents.items():
            row = self.sim._replay_data[aid]
            ag.pos.x   = row["PosX"][i]
            ag.pos.y   = row["PosY"][i]
            ag.pos.yaw = row["Yaw"][i]
            ag.speed   = row.get("Speed", [0]*self.sim._replay_frames)[i]

        self.sim._sim_time = self.sim._replay_data[self.sim.ego_vehicle.id]["SimTime"][i]
        self.update()

    def update_replay_slider(self):
        """回放文件载入后刷新进度条范围"""
        if hasattr(self, "sld_prog"):
            self.sld_prog.setMaximum(max(1, getattr(self.sim, "_replay_frames", 1))-1)