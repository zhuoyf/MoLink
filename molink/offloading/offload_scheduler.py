from __future__ import annotations
import time
from typing import Optional, List, Dict, Tuple
import torch
import torch.nn as nn
from torch.func import functional_call
from molink.config import MolinkConfig
from vllm.model_executor.models.utils import LayerFn, PPMissingLayer
from vllm.utils import (get_cuda_view_from_cpu_tensor, is_pin_memory_available,
                        is_uva_available)
import vllm.envs as envs
from vllm.config import VllmConfig


class MolinkOffloadScheduler:
    _CPU_OFFLOAD_BYTES: int = 0
    _CPU_OFFLOAD_MAX_BYTES: int = 0

    def __init__(self,
                 vllm_config: VllmConfig) -> None:
        MolinkOffloadScheduler._CPU_OFFLOAD_BYTES = 0
        MolinkOffloadScheduler._CPU_OFFLOAD_MAX_BYTES = 0

        self.vllm_config = vllm_config
        serving_layers = vllm_config.pipeline_config.serving_layers
        self.start_layer = serving_layers[0]
        self.end_layer = serving_layers[1]
        num_layers = self.end_layer - self.start_layer + 1

        self.layer_managers: List[Optional[MolinkLayerManager]] = [None] * int(num_layers)

    # todo 便捷：外部可调用把某一层权重复制到 GPU（得到 device_state 映射）
    def materialize_layer_to_gpu(self, global_idx: int, include_buffers: bool = True) -> Dict[str, torch.Tensor]:
        rel = global_idx - self.start_layer
        mgr = self.layer_managers[rel]
        assert mgr is not None, f"Layer manager for index {global_idx} not registered"
        return mgr.materialize_to_gpu(include_buffers=include_buffers)

    def make_layers(
        self,
        num_hidden_layers: int,
        config: MolinkConfig,
        layer_fn: LayerFn,
        prefix: str,
    ) -> Tuple[int, int, torch.nn.ModuleList]:
        start_layer = self.start_layer
        end_layer = self.end_layer

        layers = []
        for idx in range(start_layer, end_layer):
            rel = idx - start_layer
            self.layer_managers[rel] = MolinkLayerManager(index=idx)
            layer_module = layer_fn(prefix=f"{prefix}.{idx}")
            layers.append(self.layer_managers[rel].maybe_offload_to_cpu(layer_module))

        modules = torch.nn.ModuleList(
            [PPMissingLayer() for _ in range(start_layer)] + layers
            + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
        )

        return start_layer, end_layer, modules


class MolinkLayerManager:
    def __init__(self, index: int) -> None:
        self.index = index

        # 该层原“目标设备”（offload 前所在设备），用于回迁/临时 materialize
        self.target_device: Optional[torch.device] = None

        # 保存 CPU 上的真实权重位置（包含参数；buffers 会在 materialize 时统一处理）
        self.cpu_weights: Dict[str, torch.Tensor] = {}

        # 计时数据（纳秒）
        self.fwd_calls: int = 0
        self.compute_time_ns_total: int = 0
        self.last_compute_time_ns: int = 0

        # 该层对应的 nn.Module（maybe_offload_to_cpu 里设置）
        self.module: Optional[nn.Module] = None

    def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:
        """尽量不改你的逻辑，仅补充：
           - 记录 target_device
           - 在 offload 时把 CPU 张量保存到 self.cpu_weights
           - 注入的 forward 里计算后清理 device_state 并记录计算时长
        """
        self.module = module

        if (params := next(module.parameters(), None)) is None:
            return module

        device = params.device
        self.target_device = device  # 记录原目标设备

        if device == torch.device("cpu"):
            # 已在 CPU，无需处理；但也记录一下 CPU 张量
            for k, v in module.named_parameters():
                if v.data.device.type == "cpu":
                    self.cpu_weights[k] = v.data
            return module

        # 全局预算检查
        print(MolinkOffloadScheduler._CPU_OFFLOAD_BYTES, MolinkOffloadScheduler._CPU_OFFLOAD_MAX_BYTES)
        if MolinkOffloadScheduler._CPU_OFFLOAD_BYTES >= MolinkOffloadScheduler._CPU_OFFLOAD_MAX_BYTES:
            return module

        pin_memory = is_pin_memory_available()
        uva_available = is_uva_available()

        if envs.VLLM_USE_V1:
            assert uva_available, ("V1 CPU offloading requires uva (pin memory) support")
            uva_offloading = True
        else:
            uva_offloading = False

        # offload parameters to CPU（逐参数，受预算限制）
        offloaded_parameters = False
        # 使用 named_parameters 便于记录到 cpu_weights（尽量少改动你的逻辑）
        for name, p in module.named_parameters():
            if MolinkOffloadScheduler._CPU_OFFLOAD_BYTES >= MolinkOffloadScheduler._CPU_OFFLOAD_MAX_BYTES:
                break

            cpu_data = torch.empty_strided(size=p.data.size(),
                                           stride=p.data.stride(),
                                           dtype=p.data.dtype,
                                           layout=p.data.layout,
                                           device='cpu',
                                           pin_memory=pin_memory)
            cpu_data.copy_(p.data)

            if not uva_offloading:
                p.data = cpu_data
                # 保存 CPU 上真实位置
                self.cpu_weights[name] = cpu_data
            else:
                # UVA：保持 CPU 数据生命期，并把参数设为 CUDA 视图
                p._vllm_offloaded_cpu_data = cpu_data
                p.data = get_cuda_view_from_cpu_tensor(cpu_data)
                # 同样保存 CPU 位置
                self.cpu_weights[name] = cpu_data

            MolinkOffloadScheduler._CPU_OFFLOAD_BYTES += cpu_data.numel() * cpu_data.element_size()
            offloaded_parameters = True

        # 非 UVA：包装 forward，在调用时临时将该层权重搬到 GPU，结束后删除，并记录计算时间
        if offloaded_parameters and not uva_offloading:
            original_forward = module.forward
            mgr = self  # 捕获到闭包

            def forward(*args, **kwargs):
                module.forward = original_forward

                # 复制整层 state_dict（参数+buffers）到 target_device
                device_state = {
                    k: v.to(device, non_blocking=True)
                    for k, v in module.state_dict().items()
                }

                # 仅统计“计算耗时”（不含拷贝）
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t2 = time.perf_counter_ns()

                try:
                    output = functional_call(module, device_state, args=args, kwargs=kwargs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t3 = time.perf_counter_ns()

                    # 记录计算时间
                    mgr.last_compute_time_ns = t3 - t2
                    mgr.compute_time_ns_total += mgr.last_compute_time_ns
                    mgr.fwd_calls += 1
                finally:
                    # 释放 GPU 端临时拷贝
                    device_state.clear()
                    del device_state  # 确保引用计数归零，进入 CUDA allocator 缓存

                    # 恢复包装
                    module.forward = forward

                return output

            module.forward = forward

        return module

    # 提供给 scheduler/外部：构造并返回“本层”的 device_state（参数+buffers）的 GPU 拷贝
    def materialize_to_gpu(self, include_buffers: bool = True, non_blocking: bool = True) -> Dict[str, torch.Tensor]:
        assert self.module is not None, "Layer module not initialized"
        assert self.target_device is not None, "Target device unknown"

        # 直接基于 state_dict 逐项搬运：既能覆盖参数，也能包含 buffers（如有）
        device_state: Dict[str, torch.Tensor] = {}
        for k, v in self.module.state_dict().items():
            # 若你只想用 self.cpu_weights，可替换为：
            # if (k in self.cpu_weights): src = self.cpu_weights[k]; else: src = v
            src = v
            device_state[k] = src.to(self.target_device, non_blocking=non_blocking)
        return device_state