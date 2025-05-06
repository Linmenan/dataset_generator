import numpy as np
from typing import List, Tuple
from scipy.optimize import minimize
from ..models.map_elements import Point2D
from ..models.agent import EgoVehicle


class LateralMPC_NL:
    def __init__(
        self,
        pred_horizon: float = 30.0,      # 总预测长度（米）
        ctrl_horizon: float = 10.0,      # 控制长度（米）
        step_len: float = 1.0,           # 步长（米）
        weights: dict = None             # 代价函数权重
    ):
        self.N = int(pred_horizon / step_len)
        self.M = int(ctrl_horizon / step_len)
        self.ds = step_len

        self.weights = weights if weights is not None else {
            'lat': 1.0,
            'hdg': 0.5,
            'curv': 0.05,
            'dcurv': 0.1
        }

    def compute_errors(self, x, y, theta, ref_point: Point2D, ref_theta: float) -> Tuple[float, float]:
        dx = x - ref_point.x
        dy = y - ref_point.y
        lat_err = -np.sin(ref_theta) * dx + np.cos(ref_theta) * dy
        hdg_err = np.arctan2(np.sin(theta - ref_theta), np.cos(theta - ref_theta))
        return lat_err, hdg_err

    def forward_simulate(self, x0, y0, theta0, curvature_seq) -> List[Tuple[float, float, float]]:
        traj = [(x0, y0, theta0)]
        x, y, theta = x0, y0, theta0
        for kappa in curvature_seq:
            if abs(kappa) > 1e-4:
                dtheta = kappa * self.ds
                x += (np.sin(theta + dtheta) - np.sin(theta)) / kappa
                y += (np.cos(theta) - np.cos(theta + dtheta)) / kappa
                theta += dtheta
            else:
                x += self.ds * np.cos(theta)
                y += self.ds * np.sin(theta)
            traj.append((x, y, theta))
        return traj

    def generate_reference(self, ref_line: List[Point2D]) -> List[Tuple[Point2D, float]]:
        ref = []
        for i in range(self.N):
            idx = min(i, len(ref_line) - 2)
            p0, p1 = ref_line[idx], ref_line[idx + 1]
            hdg = np.arctan2(p1.y - p0.y, p1.x - p0.x)
            ref.append((p0, hdg))
        return ref

    def objective_function(self, curv_seq: np.ndarray, ego: EgoVehicle, ref_line: List[Point2D]) -> float:
        # 填充后面的曲率为最后一个控制点
        if len(curv_seq) < self.N:
            curv_seq = np.concatenate([curv_seq, np.ones(self.N - self.M) * curv_seq[-1]])

        traj = self.forward_simulate(ego.pos.x, ego.pos.y, ego.hdg, curv_seq)
        ref = self.generate_reference(ref_line)

        lat_errs, hdg_errs, curv_cost, dcurv_cost = [], [], 0.0, 0.0
        for i in range(self.N):
            x, y, theta = traj[i]
            lat, hdg = self.compute_errors(x, y, theta, *ref[i])
            lat_errs.append(lat)
            hdg_errs.append(hdg)

        curv_cost = np.sum(np.square(curv_seq[:self.M]))
        dcurv = np.diff(curv_seq[:self.M])
        dcurv_cost = np.sum(np.square(dcurv))

        cost = (
            self.weights['lat'] * np.sum(np.square(lat_errs)) +
            self.weights['hdg'] * np.sum(np.square(hdg_errs)) +
            self.weights['curv'] * curv_cost +
            self.weights['dcurv'] * dcurv_cost
        )
        return cost

    def solve(self, ego: EgoVehicle, ref_line: List[Point2D]) -> float:
        """
        返回初始曲率控制值
        """
        init_guess = np.zeros(self.M)
        bounds = [(-0.6, 0.6)] * self.M

        res = minimize(
            self.objective_function,
            init_guess,
            args=(ego, ref_line),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-3, 'disp': False}
        )

        if not res.success:
            print("Optimization failed:", res.message)
        return res.x[0] if res.success else 0.0