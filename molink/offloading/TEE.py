# TEE.py
from __future__ import annotations

import time
from typing import Any, Callable, Dict

import torch


class TEESimulator:
    """
    一个简单的 TEE 模拟器：
      - run(fn, *args, **kwargs) 作为统一入口；
      - 目前只是计时 + 调用原函数；
      - 后续如果你想真的搬到 CPU 或加密，只改这里即可。
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

        self.total_calls: int = 0
        self.total_time_ns: int = 0
        self.last_call_time_ns: int = 0

    def _to_device(self, obj: Any, device: torch.device) -> Any:
        """如需把参数搬到特定 device，可用这个递归工具（目前没启用）"""
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._to_device(x, device) for x in obj)
        if isinstance(obj, dict):
            return {k: self._to_device(v, device) for k, v in obj.items()}
        return obj

    def run(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """
        在“TEE 中”执行 fn(*args, **kwargs)。

        当前实现：
          - 不强制搬 device；
          - 只做时间统计 + 统一入口。
        """
        self.total_calls += 1
        t0 = time.perf_counter_ns()

        # 如需强制在 CPU 上跑，可以在这里启用：
        # args = self._to_device(args, self.device)
        # kwargs = self._to_device(kwargs, self.device)

        out = fn(*args, **kwargs)

        t1 = time.perf_counter_ns()
        self.last_call_time_ns = t1 - t0
        self.total_time_ns += self.last_call_time_ns
        return out

    def reset_stats(self) -> None:
        self.total_calls = 0
        self.total_time_ns = 0
        self.last_call_time_ns = 0

    def get_stats(self) -> Dict[str, Any]:
        avg = self.total_time_ns / self.total_calls if self.total_calls > 0 else 0
        return {
            "total_calls": self.total_calls,
            "total_time_ns": self.total_time_ns,
            "last_call_time_ns": self.last_call_time_ns,
            "avg_time_ns": avg,
        }
