# TEE.py
from __future__ import annotations

import time
from typing import Any, Callable, Dict

import torch


class TEESimulator:
    def __init__(self, device: torch.device) -> None:
        # TEE 计算所在 device（CPU）
        self.device = device

        self.total_calls: int = 0
        self.total_time_ns: int = 0
        self.last_call_time_ns: int = 0

    def _move(self, obj: Any, device: torch.device) -> Any:
        """递归把张量搬到指定 device。"""
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move(x, device) for x in obj)
        if isinstance(obj, dict):
            return {k: self._move(v, device) for k, v in obj.items()}
        return obj

    def run(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """
        在“TEE(=CPU)”中执行 fn(*args, **kwargs)：
          - 将输入搬到 CPU；
          - 调用 fn；
          - 将输出搬回原 device；
          - 打印进入/退出日志。
        """

        layer = kwargs.pop("__tee_layer", None)
        modname = kwargs.pop("__tee_modname", None)

        # 找出原始 device（任意一个 tensor 的 device）
        original_device = None
        for x in args:
            if torch.is_tensor(x):
                original_device = x.device
                break
        if original_device is None:
            original_device = torch.device("cuda:0")

        self.total_calls += 1
        t0 = time.perf_counter_ns()

        # ---------- LOG: 进入 TEE ----------
        # if layer is not None:
            # print(f"[TEE] ENTER layer={layer}, module={modname}")
            # for i, x in enumerate(args):
                # if torch.is_tensor(x):
                    # print(f"       input[{i}] shape={tuple(x.shape)} device={x.device}")
        # -------------------------------

        # 搬到 CPU
        args_cpu = self._move(args, self.device)
        kwargs_cpu = self._move(kwargs, self.device)

        # 在 CPU 上执行
        out_cpu = fn(*args_cpu, **kwargs_cpu)

        # 搬回原来的 device
        out = self._move(out_cpu, original_device)

        t1 = time.perf_counter_ns()
        self.last_call_time_ns = t1 - t0
        self.total_time_ns += self.last_call_time_ns

        # ---------- LOG: 退出 TEE ----------
        # if layer is not None:
        #     print(f"[TEE] EXIT layer={layer}, module={modname}, time={self.last_call_time_ns/1e6:.3f} ms")
        #     if torch.is_tensor(out):
        #         print(f"       output shape={tuple(out.shape)} device={out.device}")
        #     elif isinstance(out, tuple):
        #         shapes = [tuple(t.shape) for t in out if torch.is_tensor(t)]
        #         print(f"       output tuple shapes={shapes}")
        # -------------------------------

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
