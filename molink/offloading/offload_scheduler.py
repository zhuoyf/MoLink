from __future__ import annotations
import time
from typing import Optional, List, Dict, Tuple
import torch
import torch.nn as nn
from torch.func import functional_call
from molink.config import MolinkConfig
from vllm.model_executor.models.utils import LayerFn, PPMissingLayer
from vllm.utils import is_pin_memory_available
from vllm.config import VllmConfig


class MolinkOffloadScheduler:
    _CPU_OFFLOAD_BYTES: int = 0
    _CPU_OFFLOAD_MAX_BYTES: int = 0

    def __init__(self,
                 vllm_config: VllmConfig) -> None:
        MolinkOffloadScheduler._CPU_OFFLOAD_BYTES = 0
        MolinkOffloadScheduler._CPU_OFFLOAD_MAX_BYTES = 10

        self.vllm_config = vllm_config
        serving_layers = vllm_config.pipeline_config.serving_layers
        self.start_layer = serving_layers[0]
        self.end_layer = serving_layers[1]
        num_layers = self.end_layer - self.start_layer + 1

        self.layer_managers: List[Optional[MolinkLayerManager]] = [None] * int(num_layers)

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
        for idx in range(start_layer, end_layer + 1):
            rel = idx - start_layer
            print(f"layer {idx} init")
            self.layer_managers[rel] = MolinkLayerManager(index=idx)
            layer_module = layer_fn(prefix=f"{prefix}.{idx}")
            layers.append(self.layer_managers[rel].maybe_offload_to_cpu(layer_module))

        modules = torch.nn.ModuleList(
            [PPMissingLayer() for _ in range(start_layer)] + layers
            + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
        )

        # MoLink: [start_layer, end_layer] --> vLLM: [start_layer, end_layer)
        return start_layer, end_layer + 1, modules


class MolinkLayerManager:
    def __init__(self, index: int) -> None:
        self.index = index
        self.target_device: Optional[torch.device] = None
        self.cpu_weights: Dict[str, torch.Tensor] = {}
        self.device_state = Dict[str, torch.Tensor]
        self.module: Optional[nn.Module] = None
        self.is_on_gpu: bool = False

        # 计时数据: fwd_calls:前向传播次数；compute_time_ns_total:总共耗时；last_compute_time_ns:最后一次耗时
        self.fwd_calls: int = 0
        self.compute_time_ns_total: int = 0
        self.last_compute_time_ns: int = 0


    def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:
        """执行 offload 到 CPU 的操作，并包装 forward"""
        self.module = module

        if (params := next(module.parameters(), None)) is None:
            return module
        device = params.device
        self.target_device = device

        if device == torch.device("cpu"):
            for k, v in module.named_parameters():
                if v.data.device.type == "cpu":
                    self.cpu_weights[k] = v.data
            return module

        pin_memory = is_pin_memory_available()
        offloaded_parameters = False
        for name, p in module.named_parameters():
            cpu_data = torch.empty_strided(size=p.data.size(),
                                           stride=p.data.stride(),
                                           dtype=p.data.dtype,
                                           layout=p.data.layout,
                                           device='cpu',
                                           pin_memory=pin_memory)
            cpu_data.copy_(p.data)
            p.data = cpu_data

            # 保存 CPU 上的权重
            self.cpu_weights[name] = cpu_data

            MolinkOffloadScheduler._CPU_OFFLOAD_BYTES += cpu_data.numel() * cpu_data.element_size()
            print(f"移动{MolinkOffloadScheduler._CPU_OFFLOAD_BYTES}字节到CPU,name:{name}")
            offloaded_parameters = True

        if offloaded_parameters:
            original_forward = module.forward
            mgr = self

            def forward(*args, **kwargs):
                module.forward = original_forward

                mgr.check_layer()  # 在每次计算前调用 check_layer()，确保权重在GPU

                # 记录计算耗时
                t2 = time.perf_counter_ns()

                try:
                    output = functional_call(module, mgr.device_state, args=args, kwargs=kwargs)
                    t3 = time.perf_counter_ns()

                    # 记录计算时间
                    mgr.last_compute_time_ns = t3 - t2
                    mgr.compute_time_ns_total += mgr.last_compute_time_ns
                    mgr.fwd_calls += 1
                finally:
                    mgr.device_state.clear()
                    del mgr.device_state  # 确保引用计数归零，进入 CUDA allocator 缓存
                    mgr.device_state = None
                    self.is_on_gpu = False

                    module.forward = forward

                return output

            module.forward = forward

        return module


    def check_layer(self):
        if not self.is_on_gpu:
            self.device_state = self.materialize_to_gpu()
            self.is_on_gpu = True
        

    def materialize_to_gpu(self) -> Dict[str, torch.Tensor]:
        """将保存在 CPU 的权重复制到 GPU"""
        assert self.module is not None, "Layer module not initialized"
        assert self.target_device is not None, "Target device unknown"

        device_state: Dict[str, torch.Tensor] = {}
        for k, v in self.module.state_dict().items():
            src = self.cpu_weights.get(k, v)
            device_state[k] = src.to(self.target_device, non_blocking=True)
        return device_state