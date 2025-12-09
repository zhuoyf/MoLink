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
from molink.offloading.TEE import TEESimulator
import functools
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.activation import SiluAndMul

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
        self.num_layers = self.end_layer - self.start_layer + 1
        self.layer_managers: List[Optional[MolinkLayerManager]] = [None] * int(self.num_layers)
        self.prefetch_distance: int = 5

        self.tee = TEESimulator(device=torch.device("cpu"))


    def _prefetch_layer(self, global_idx: int) -> None:
        assert global_idx <= self.end_layer and global_idx >= self.start_layer, "Illegal prefetch index"

        rel = global_idx - self.start_layer
        mgr = self.layer_managers[rel]
        
        assert mgr is not None, f"Layer manager{rel} not been initialize"

        mgr.device_state = mgr.materialize_to_gpu()
        mgr.is_on_gpu = True


    def _prefetch_initial_layers(self) -> None:
        # prefetch self.prefetch_distance layers to GPU
        if self.prefetch_distance == 0:
            return
        max_idx = min(self.end_layer, self.start_layer + self.prefetch_distance - 1)
        for idx in range(self.start_layer, max_idx + 1):
            self._prefetch_layer(idx % self.num_layers)

    def layer_finished(self, idx: int) -> None:
        # todo
        """
        某一层 forward 完成后由 layer manager 调用。
        调度策略：
          - 当前层 index = i 完成后，尝试预取 i+3 层到 GPU。
        """
        target_idx = idx + self.prefetch_distance
        self._prefetch_layer(target_idx % self.num_layers)

    def materialize_layer_to_gpu(self, global_idx: int,
                                 include_buffers: bool = True) -> Dict[str, torch.Tensor]:
        rel = global_idx - self.start_layer
        mgr = self.layer_managers[rel]
        assert mgr is not None, f"Layer manager for index {global_idx} not registered"
        return mgr.materialize_to_gpu()

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
            self.layer_managers[rel] = MolinkLayerManager(index=idx, scheduler=self, tee=self.tee)
            layer_module = layer_fn(prefix=f"{prefix}.{idx}")
            layers.append(self.layer_managers[rel].maybe_offload_to_cpu(layer_module))

        modules = torch.nn.ModuleList(
            [PPMissingLayer() for _ in range(start_layer)] + layers
            + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
        )

        # 初始化完成后，预取前三个层到 GPU
        self._prefetch_initial_layers()

        # MoLink: [start_layer, end_layer] --> vLLM: [start_layer, end_layer)
        return start_layer, end_layer + 1, modules


class MolinkLayerManager:
    def __init__(self, index: int, scheduler: MolinkOffloadScheduler, tee: TEESimulator) -> None:
        self.index = index
        self.target_device: Optional[torch.device] = None
        self.cpu_weights: Dict[str, torch.Tensor] = {}
        self.device_state: Optional[Dict[str, torch.Tensor]] = None
        self.module: Optional[nn.Module] = None
        self.is_on_gpu: bool = False
        self.scheduler = scheduler

        self.tee = tee

        # todo 计时数据: fwd_calls:前向传播次数；compute_time_ns_total:总共耗时；last_compute_time_ns:最后一次耗时
        self.fwd_calls: int = 0
        self.compute_time_ns_total: int = 0
        self.last_compute_time_ns: int = 0

    def forward_finished(self) -> None:
        # report to scheduler
        self.scheduler.layer_finished(self.index)

    def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:
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

        # todo 调试if
        if MolinkOffloadScheduler._CPU_OFFLOAD_MAX_BYTES == 0:
            return module

        pin_memory = is_pin_memory_available()
        offloaded_parameters = False
        # todo 调试变量
        cnt = 0
        for name, p in module.named_parameters():
            cpu_data = torch.empty_strided(size=p.data.size(),
                                           stride=p.data.stride(),
                                           dtype=p.data.dtype,
                                           layout=p.data.layout,
                                           device='cpu',
                                           pin_memory=pin_memory)
            cpu_data.copy_(p.data)
            p.data = cpu_data

            self.cpu_weights[name] = cpu_data

            MolinkOffloadScheduler._CPU_OFFLOAD_BYTES += cpu_data.numel() * cpu_data.element_size()
            offloaded_parameters = True

            cnt += cpu_data.numel() * cpu_data.element_size()
        print(f"layer {self.index}({cnt / (1024**3)} GB) 成功移动到内存。")

        if offloaded_parameters:
            original_forward = module.forward
            mgr = self

            def forward(*args, **kwargs):
                module.forward = original_forward

                # make sure that layer in GPU
                mgr.check_layer()

                t2 = time.perf_counter_ns()
                try:
                    output = mgr._forward_with_tee_nonlinear(module, mgr.device_state, *args, **kwargs)

                    t3 = time.perf_counter_ns()
                    mgr.last_compute_time_ns = t3 - t2
                    mgr.compute_time_ns_total += mgr.last_compute_time_ns
                    mgr.fwd_calls += 1
                finally:
                    if mgr.device_state is not None:
                        mgr.device_state.clear()
                    mgr.device_state = None
                    mgr.is_on_gpu = False
                    mgr.forward_finished()

                    module.forward = forward

                return output

            module.forward = forward

        return module

    def check_layer(self):
        if self.is_on_gpu and self.device_state is not None:
            # successfully prefetch
            return
        else:
            # TODO 调度策略
            self.device_state = self.materialize_to_gpu()

    def materialize_to_gpu(self) -> Dict[str, torch.Tensor]:
        '''
        move weight to GPU
        '''
        assert self.module is not None, "Layer module not initialized"
        assert self.target_device is not None, "Target device unknown"

        if self.is_on_gpu and self.device_state is not None:
            return self.device_state

        self.is_on_gpu = True
        device_state: Dict[str, torch.Tensor] = {}
        for k, v in self.module.state_dict().items():
            src = self.cpu_weights.get(k, v)
            device_state[k] = src.to(self.target_device, non_blocking=True)
        return device_state


    def _forward_with_tee_nonlinear(
        self,
        module: nn.Module,
        params: Dict[str, torch.Tensor],
        *args,
        **kwargs,
    ):
        """
        使用 functional_call 执行整个 layer 的前向，但在本次调用期间：
          - 对 SiluAndMul：调用原 forward，但在 TEE 中、CPU 设备上执行；
          - 对 RMSNorm：绕过原 forward，使用我们手写的 CPU 实现，在 TEE 中执行；
          - 调用结束后恢复所有子模块的 forward。
        """

        # 如果没有 TEE，就直接走原逻辑
        if self.tee is None:
            return functional_call(module, params, args=args, kwargs=kwargs)

        nonlinear_types = (RMSNorm, SiluAndMul)
        orig_forwards = []

        for submodule in module.modules():
            # ---------- 1. RMSNorm：走我们自己写的 CPU 实现 ----------
            if isinstance(submodule, RMSNorm):
                orig = submodule.forward

                def rms_cpu_forward(*a, _mod=submodule):
                    """
                    纯 PyTorch 实现的 RMSNorm：
                      - 支持两种调用：
                          input_layernorm(x)
                          input_layernorm(x, residual)
                    """
                    if len(a) == 1:
                        x = a[0]
                        residual = None
                    elif len(a) == 2:
                        x, residual = a
                    else:
                        raise RuntimeError(
                            f"Unexpected RMSNorm inputs len={len(a)}, expect 1 or 2."
                        )

                    eps = _mod.variance_epsilon
                    weight = _mod.weight

                    # 假设 x 已经在 CPU（TEE 会把参数搬过去）
                    # 确保 weight 的 device / dtype 与 x 一致
                    if torch.is_tensor(weight) and weight.device != x.device:
                        w = weight.to(x.device)
                    else:
                        w = weight

                    x2 = x.to(w.dtype)
                    # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
                    var = x2.pow(2).mean(dim=-1, keepdim=True)
                    rms = torch.sqrt(var + eps)
                    y = x2 / rms * w

                    if residual is None:
                        return y
                    else:
                        return y, residual

                @functools.wraps(orig)
                def wrapped_forward(*a, _fn=rms_cpu_forward, _tee=self.tee, _layer=self.index, **k):
                    # 注意：我们不再调用 orig，而是调用自定义 CPU 实现
                    return _tee.run(
                        _fn,
                        *a,
                        **k,
                        __tee_layer=_layer,
                        __tee_modname="RMSNorm",
                    )

                orig_forwards.append((submodule, orig))
                submodule.forward = wrapped_forward

            # ---------- 2. SiluAndMul：直接用原 forward，在 TEE + CPU 执行 ----------
            elif isinstance(submodule, SiluAndMul):
                @functools.wraps(submodule.forward)
                def wrapped_forward(x, _tee=self.tee, _layer=self.index):
                    return _tee.run(
                        silu_and_mul_cpu, 
                        x,
                        __tee_layer=_layer,
                        __tee_modname="SiluAndMul"
                    )

                orig_forwards.append((submodule, submodule.forward))
                submodule.forward = wrapped_forward

        try:
            # 🚀 真正执行这一层的前向（会触发我们上面的 wrapped_forward）
            output = functional_call(module, params, args=args, kwargs=kwargs)
        finally:
            # 恢复所有子模块的原始 forward
            for submodule, orig in orig_forwards:
                submodule.forward = orig

        return output
    
def silu_and_mul_cpu(x: torch.Tensor) -> torch.Tensor:
        # vLLM 的 MergedColumnParallelLinear 输出 layout 保持一致
        # x shape: (B, 2*I)

        # 按照最后一维一分为二
        gate, up = x.chunk(2, dim=-1)

        # SiluAndMul 正确定义： silu(gate) * up
        return torch.nn.functional.silu(gate) * up