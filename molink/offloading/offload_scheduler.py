from __future__ import annotations
import time
from typing import Optional, List, Dict, Tuple
import torch
import torch.nn as nn
from molink.config import MolinkConfig
from vllm.model_executor.models.utils import LayerFn, PPMissingLayer
from vllm.utils import is_pin_memory_available
from vllm.config import VllmConfig
from molink.offloading.TEE import TEESimulator

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

        self.tee = TEESimulator()


    def _prefetch_layer(self, global_idx: int) -> None:
        assert global_idx <= self.end_layer and global_idx >= self.start_layer, "Illegal prefetch index"

        rel = global_idx - self.start_layer
        mgr = self.layer_managers[rel]
        
        assert mgr is not None, f"Layer manager{rel} not been initialize"

        mgr.materialize_to_gpu()


    def _prefetch_initial_layers(self) -> None:
        if self.prefetch_distance == 0:
            return

        max_idx = min(self.end_layer, self.start_layer + self.prefetch_distance - 1)
        for idx in range(self.start_layer, max_idx + 1):
            self._prefetch_layer(idx)

    def layer_finished(self, idx: int) -> None:
        target_idx = idx + self.prefetch_distance
        if target_idx <= self.end_layer:
            self._prefetch_layer(target_idx)


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

        # 初始化完成后，预取层到 GPU
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

    def forward_finished(self) -> None:
        # report to scheduler
        self.scheduler.layer_finished(self.index)

    def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:
        self.module = module

        if (params := next(module.parameters(), None)) is None:
            return module
        device = params.device
        self.target_device = device

        # todo 调试if
        if MolinkOffloadScheduler._CPU_OFFLOAD_MAX_BYTES == 0:
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

            self.cpu_weights[name] = cpu_data

            MolinkOffloadScheduler._CPU_OFFLOAD_BYTES += cpu_data.numel() * cpu_data.element_size()
            offloaded_parameters = True

        print(f"layer {self.index} 成功移动到内存。")

        if offloaded_parameters:
            original_forward = module.forward
            mgr = self

            def forward(*args, **kwargs):
                module.forward = original_forward

                # make sure that layer in GPU
                mgr.check_layer()

                try:
                    # forward with TEE
                    output = mgr.tee.forward_with_nonlinear(module, mgr.device_state, layer_idx=mgr.index, *args, **kwargs)

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
        self.device_state = device_state
        
        return self.device_state
