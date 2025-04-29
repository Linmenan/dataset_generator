
from enum import Enum
from itertools import accumulate
from typing import Dict, Tuple

class Color(str, Enum):
    red = "red"; yellow = "yellow"; green = "green"; off = "off"

class Phase:
    def __init__(self, duration: float, signals: Dict[str, Color]):
        self.duration = duration
        self.signals = signals

class SignalController:
    def __init__(self, cfg):
        self.cycle = cfg["cycle"]
        self.offset = cfg.get("offset", 0)
        self.phases = [Phase(p["duration"], {str(k): Color(v)
                       for k, v in p["signals"].items()})
                       for p in cfg["phases"]]
        # 预计算累计时间，便于查表
        self._cum = list(accumulate(p.duration for p in self.phases))

    def state(self, t: float) -> Dict[str, Color]:
        """返回给定仿真时刻 t (s) 时，每个 signalId 的颜色"""
        t_rel = (t - self.offset) % self.cycle
        idx = next(i for i, end in enumerate(self._cum) if t_rel < end)
        return self.phases[idx].signals
    
    def _locate_phase(self, t: float) -> Tuple[int, float]:
        """返回 (相位索引, 距本相位结束的剩余时间)"""
        t_rel = (t - self.offset) % self.cycle
        for i, end in enumerate(self._cum):
            if t_rel < end:
                remaining = end - t_rel
                return i, remaining
        # 理论上不会到这里
        raise RuntimeError("time locating error")
    
    def state_with_countdown(self, t: float) -> Dict[str, Tuple[Color, float]]:
        """返回各信号颜色及距离下一相位切换的倒计时（秒）"""
        idx, remaining = self._locate_phase(t)
        return {sid: (clr, remaining) for sid, clr in self.phases[idx].signals.items()}