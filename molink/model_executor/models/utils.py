from typing import Any, Deque, Dict, Optional, Sequence, Tuple
import inspect
import warnings
import torch
from torch import nn
from torch.func import functional_call
from transformers import PretrainedConfig
from vllm.model_executor.models.utils import LayerFn, PPMissingLayer
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.model_loader.utils import configure_quant_config
from molink.config import MolinkConfig
from molink.model_executor.model_loader.utils import get_model_architecture
from vllm.utils import (get_cuda_view_from_cpu_tensor, is_pin_memory_available,
                        is_uva_available)
import vllm.envs as envs

def get_pp_indices(config: MolinkConfig) -> Tuple[int, int]:
    
    serving_layers = config.pipeline_config.serving_layers
    assert len(serving_layers) >= 1, 'serving layers no specified'
    start_layer = serving_layers[0]
    # to be compatible with vLLM's impl, the right side should be close
    end_layer = serving_layers[-1] + 1
    return (start_layer, end_layer)


_CPU_OFFLOAD_BYTES = 0
_CPU_OFFLOAD_MAX_BYTES = 0

def set_cpu_offload_max_bytes(max_bytes: int) -> None:
    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES
    _CPU_OFFLOAD_BYTES = 0
    _CPU_OFFLOAD_MAX_BYTES = max_bytes
    
def maybe_offload_to_cpu(module: torch.nn.Module, prefix: str) -> torch.nn.Module:
    print(f"prefix: {prefix}")
    module._layer_idx = prefix

    if (params := next(module.parameters(), None)) is None:
        return module

    device = params.device

    if device == torch.device("cpu"):
        return module

    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES
    print(_CPU_OFFLOAD_BYTES, _CPU_OFFLOAD_MAX_BYTES)
    if _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
        return module

    pin_memory = is_pin_memory_available()
    uva_available = is_uva_available()

    if envs.VLLM_USE_V1:
        assert uva_available, ("V1 CPU offloading requires"
                               " uva (pin memory) support")
        uva_offloading = True
    else:
        uva_offloading = False

    # offload parameters to CPU
    # use pin_memory if possible, which helps cudagraph capture speed
    offloaded_parameters = False
    for p in module.parameters():
        if _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
            # we use per-parameter offloading
            # one module might have some parameters offloaded and some not
            break

        # `torch.empty_like` does not support `pin_memory` argument
        cpu_data = torch.empty_strided(size=p.data.size(),
                                       stride=p.data.stride(),
                                       dtype=p.data.dtype,
                                       layout=p.data.layout,
                                       device='cpu',
                                       pin_memory=pin_memory)
        cpu_data.copy_(p.data)
        if not uva_offloading:
            p.data = cpu_data
        else:
            # keep the cpu data alive
            p._vllm_offloaded_cpu_data = cpu_data
            p.data = get_cuda_view_from_cpu_tensor(cpu_data)
        _CPU_OFFLOAD_BYTES += p.data.numel() * p.data.element_size()
        offloaded_parameters = True

    if offloaded_parameters and not uva_offloading:
        original_forward = module.forward

        def forward(*args, _layer_idx=prefix, **kwargs):
            module.forward = original_forward
            # print(type(module.state_dict().items()))
            # print(type(module.state_dict()))
            device_state = {
                # here we blindly call `to(device)`
                # if the parameter is already on the device, it will be a no-op
                k: v.to(device, non_blocking=True)
                for k, v in module.state_dict().items()
            }

            # print("device_state devices:", {k: v.device for k, v in device_state.items()})
            print(f"开始计算层{module._layer_idx}")
            output = functional_call(module,
                                     device_state,
                                     args=args,
                                     kwargs=kwargs)
            print(f"结束计算层{module._layer_idx}")
            module.forward = forward
            return output

        module.forward = forward

    return module

def make_layers(
    num_hidden_layers: int,
    config: MolinkConfig,
    layer_fn: LayerFn,
    prefix: str,
) -> Tuple[int, int, torch.nn.ModuleList]:
    """Make a list of layers with the given layer function, taking
    pipeline parallelism into account.
    """
    start_layer, end_layer = get_pp_indices(config)
    modules = torch.nn.ModuleList(
        [PPMissingLayer() for _ in range(start_layer)] + [
            maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}"), f"{prefix}.{idx}")
            for idx in range(start_layer, end_layer)
        ] + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)])
    return start_layer, end_layer, modules

def _initialize_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
) -> nn.Module:
    """Initialize a model with the given configurations."""
    model_config = vllm_config.model_config
    model_class, _ = get_model_architecture(model_config)

    if vllm_config.quant_config is not None:
        configure_quant_config(vllm_config.quant_config, model_class)

    signatures = inspect.signature(model_class.__init__)
    all_params = [param.name for param in signatures.parameters.values()]
    if "vllm_config" in all_params and "prefix" in all_params:
        # new-style model class
        with set_current_vllm_config(vllm_config, check_compile=True):
            return model_class(vllm_config=vllm_config, prefix=prefix)

    msg = ("vLLM model class should accept `vllm_config` and `prefix` as "
           "input arguments. Possibly you have an old-style model class"
           " registered from out of tree and it is used for new vLLM version. "
           "Check https://docs.vllm.ai/en/latest/design/arch_overview.html "
           "for the design and update the model class accordingly.")
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    # try to be compatible with old-style model class
    kwargs = {}
    if "prefix" in all_params:
        kwargs["prefix"] = prefix
    if "config" in all_params:
        kwargs["config"] = model_config.hf_config
    if "cache_config" in all_params:
        kwargs["cache_config"] = vllm_config.cache_config
    if "quant_config" in all_params:
        kwargs["quant_config"] = vllm_config.quant_config
    if "lora_config" in all_params:
        kwargs["lora_config"] = vllm_config.lora_config
    if "scheduler_config" in all_params:
        kwargs["scheduler_config"] = vllm_config.scheduler_config
    with set_current_vllm_config(vllm_config, check_compile=True):
        return model_class(**kwargs)

def init_vllm_registered_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
    hf_config: Optional[PretrainedConfig] = None,
    architectures: Optional[list[str]] = None,
) -> nn.Module:
    """
    Helper function to initialize an inner model registered to vLLM,
    based on the arguments passed to the outer vLLM model.
    """

    if hf_config is None and architectures is not None:
        # So that the architectures field is overridden
        hf_config = vllm_config.model_config.hf_config

    if hf_config is not None:
        vllm_config = vllm_config.with_hf_config(hf_config,
                                                 architectures=architectures)

    return _initialize_model(vllm_config=vllm_config, prefix=prefix)
