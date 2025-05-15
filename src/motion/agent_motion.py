import numpy as np
import math
from typing import List, Tuple
from scipy.optimize import minimize
from ..utils.geometry import Point2D
from ..models.agent import EgoVehicle


class LateralMPC_NL:
    def __init__(
        self,
        pred_horizon: float = 30.0,      # 预测长度（米）
        ctrl_horizon: float = 10.0,      # 控制长度（米）
        step_len: float = 1.0,           # 等路程步长（米）
        weights: dict = None             # 代价函数权重
    ):
        self.N = int(pred_horizon / step_len)
        self.M = int(ctrl_horizon / step_len)
        self.ds = step_len
        self.weights = weights if weights is not None else {
            'cte':   1.0,
            'ephi':   0.5,
            'curv':  0.05,
            'dcurv': 0.1
        }
        # 预测过程中会临时写入：
        self.kappa_min = None
        self.kappa_max = None

    def project_to_path(
        self,
        x: float,
        y: float,
        ref_line: List[Point2D]
    ) -> Tuple[Point2D, float, int]:
        """
        将 (x,y) 投影到 ref_line（折线）上，
        返回投影点、该段航向以及所在线段起点索引
        """
        best_dist2 = float('inf')
        best_proj: Point2D = ref_line[0]
        best_theta = 0.0
        best_idx = 0

        for i in range(len(ref_line) - 1):
            p0, p1 = ref_line[i], ref_line[i+1]
            vx, vy = p1.x - p0.x, p1.y - p0.y
            seg_len2 = vx*vx + vy*vy
            if seg_len2 < 1e-8:
                continue

            dx, dy = x - p0.x, y - p0.y
            u = (dx*vx + dy*vy) / seg_len2
            u_clamped = np.clip(u, 0.0, 1.0)

            proj_x = p0.x + u_clamped * vx
            proj_y = p0.y + u_clamped * vy
            dist2 = (proj_x - x)**2 + (proj_y - y)**2

            if dist2 < best_dist2:
                best_dist2 = dist2
                best_proj = Point2D(proj_x, proj_y)
                best_theta = math.atan2(vy, vx)
                best_idx = i

        return best_proj, best_theta, best_idx

    def trim_and_extend_ref(
        self,
        ego_pos: Point2D,
        ref_line: List[Point2D]
    ) -> List[Point2D]:
        """
        根据当前位置在 ref_line 上的投影点，截选后续预测长度的路径；
        若原始路径不足则按直线方向等间隔延长。
        """
        # 1) 获取投影点及所在线段索引
        p_proj, _, idx = self.project_to_path(ego_pos.x, ego_pos.y, ref_line)

        # 2) 截选从投影点开始到路径末尾
        new_line: List[Point2D] = [p_proj] + ref_line[idx+1:]

        # 3) 累计长度并截取满足预测长度的部分
        pred_len = self.N * self.ds
        cum_len = 0.0
        trimmed: List[Point2D] = [new_line[0]]
        for pt in new_line[1:]:
            last = trimmed[-1]
            seg = math.hypot(pt.x - last.x, pt.y - last.y)
            cum_len += seg
            trimmed.append(pt)
            if cum_len >= pred_len:
                break

        # 4) 若不够长度，则按最后一段方向延伸
        if cum_len < pred_len:
            # 方向取 trimmed 最后两点
            if len(trimmed) >= 2:
                p_last1, p_last2 = trimmed[-2], trimmed[-1]
                dx = p_last2.x - p_last1.x
                dy = p_last2.y - p_last1.y
                norm = math.hypot(dx, dy)
                if norm < 1e-8:
                    dir_x, dir_y = 1.0, 0.0
                else:
                    dir_x, dir_y = dx / norm, dy / norm
            else:
                # 若只有一个点，延长与 x 轴正方向
                dir_x, dir_y = 1.0, 0.0

            remain = pred_len - cum_len
            n_ext = int(math.ceil(remain / self.ds))
            last_pt = trimmed[-1]
            for i in range(1, n_ext + 1):
                new_x = last_pt.x + dir_x * self.ds * i
                new_y = last_pt.y + dir_y * self.ds * i
                trimmed.append(Point2D(new_x, new_y))

        return trimmed

    def compute_errors(
        self,
        x: float,
        y: float,
        theta: float,
        ref_line: List[Point2D]
    ) -> Tuple[float, float]:
        """
        对预测点 (x,y,theta)，基于 ref_line 实时投影计算横向误差 e_cte 和航向误差 e_phi。
        """
        p_proj, ref_theta, _ = self.project_to_path(x, y, ref_line)

        # 横向误差
        dx, dy = x - p_proj.x, y - p_proj.y
        lat_err = math.cos(ref_theta) * dy - math.sin(ref_theta) * dx

        # 航向误差
        dtheta = theta - ref_theta
        hdg_err = math.atan2(math.sin(dtheta), math.cos(dtheta))

        return lat_err, hdg_err

    def forward_simulate(
        self,
        x0: float,
        y0: float,
        theta0: float,
        kappa0: float,
        dkappa_seq: np.ndarray
    ) -> List[Tuple[float, float, float, float]]:
        """
        前向仿真 N 步：应用 M 步 dkappa，然后补零，并裁剪 kappa。
        返回每步的 (x, y, theta, kappa)。
        """
        delta_full = np.concatenate([
            dkappa_seq,
            np.zeros(self.N - self.M)
        ])

        states = []
        x, y, theta, kappa = x0, y0, theta0, kappa0
        for dk in delta_full:
            kappa = np.clip(kappa + dk, self.kappa_min, self.kappa_max)
            dtheta = kappa * self.ds
            if abs(kappa) > 1e-4:
                x += (np.sin(theta + dtheta) - np.sin(theta)) / kappa
                y += (np.cos(theta) - np.cos(theta + dtheta)) / kappa
            else:
                x += self.ds * math.cos(theta)
                y += self.ds * math.sin(theta)
            theta += dtheta
            states.append((x, y, theta, kappa))

        return states

    def objective_function(
        self,
        dkappa_vec: np.ndarray,
        ego: EgoVehicle,
        ref_line: List[Point2D]
    ) -> float:
        # 1) 前向仿真
        traj = self.forward_simulate(
            ego.pos.x, ego.pos.y, ego.pos.yaw,
            ego.curvature,
            dkappa_vec
        )

        # 2) 误差累积
        lat_errs, hdg_errs = [], []
        curv_cost = 0.0
        for i, (x, y, theta, kappa) in enumerate(traj):
            cte, ephi = self.compute_errors(x, y, theta, ref_line)
            lat_errs.append(cte)
            hdg_errs.append(ephi)
            if i < self.M:
                curv_cost += kappa**2

        # 3) 曲率变化率代价
        dcurv_cost = np.sum(np.square(dkappa_vec))

        # 4) 总代价
        cost = (
            self.weights['cte']   * np.sum(np.square(lat_errs)) +
            self.weights['ephi']  * np.sum(np.square(hdg_errs)) +
            self.weights['curv']  * curv_cost +
            self.weights['dcurv'] * dcurv_cost
        )
        return cost

    def solve(
        self,
        ego: EgoVehicle,
        ref_line: List[Point2D]
    ) -> Tuple[float, List[Point2D]]:
        """
        主函数：
          1) 根据当前位置截选并补充参考路径；
          2) 计算 kappa 限值；
          3) 优化 dkappa；
          4) 返回 kappa_cmd 和预测轨迹。
        """
        # 1) 截选并延长参考线
        sub_ref = self.trim_and_extend_ref(ego.pos, ref_line)

        # 2) 曲率限值
        delta_max, delta_min = ego.steer_angle_limit[1], ego.steer_angle_limit[0]
        L = ego.wheel_base
        self.kappa_max = math.tan(delta_max) / L
        self.kappa_min = math.tan(delta_min) / L

        # 3) 优化设置
        init_guess = np.zeros(self.M)
        max_dk = 0.02
        bounds = [(-max_dk, max_dk) for _ in range(self.M)]
        res = minimize(
            self.objective_function,
            init_guess,
            args=(ego, sub_ref),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        if not res.success:
            print("MPC 优化未收敛：", res.message)

        # 4) 生成并返回结果
        dkappa_opt = res.x if res.success else np.zeros(self.M)
        traj_opt = self.forward_simulate(
            ego.pos.x, ego.pos.y, ego.pos.yaw,
            ego.curvature,
            dkappa_opt
        )
        pred_traj = [Point2D(x, y) for x, y, _, _ in traj_opt]
        kappa_cmd = ego.curvature + float(dkappa_opt[0])
        return kappa_cmd, pred_traj, sub_ref
