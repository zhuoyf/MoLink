from __future__ import annotations
import functools
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.activation import SiluAndMul


class TEESimulator:
    def __init__(self):
        self.cpu = torch.device("cpu")

    def forward_with_nonlinear(
        self,
        module: nn.Module,
        params: dict,
        *args,
        layer_idx: int,
        **kwargs,
    ):
        """
        Run layer forward via functional_call(module, params, ...),
        but override RMSNorm and SiluAndMul to execute on CPU (TEE).
        """

        # without TEE
        return functional_call(module, params, args=args, kwargs=kwargs)

        orig_forwards: list[tuple[nn.Module, callable]] = []

        # ---- Patch submodule forwards ----
        for submodule in module.modules():
            # RMSNorm -> CPU implementation
            if isinstance(submodule, RMSNorm):
                orig = submodule.forward

                @functools.wraps(orig)
                def wrapped_forward(x, residual=None, _mod=submodule):
                    out_dev = x.device

                    x_cpu = _to_cpu(x)
                    residual_cpu = None if residual is None else _to_cpu(residual)

                    # IMPORTANT: use _mod.weight (NOT .data) to keep functional_call param substitution working
                    if _mod.has_weight:
                        w_cpu = _to_cpu(_mod.weight)
                    else:
                        w_cpu = torch.empty(0, device="cpu")

                    y_cpu = rmsnorm_cpu(
                        x_cpu=x_cpu,
                        weight_cpu=w_cpu,
                        eps=_mod.variance_epsilon,
                        residual_cpu=residual_cpu,
                        variance_size_override=_mod.variance_size_override,
                        has_weight=_mod.has_weight,
                    )

                    if isinstance(y_cpu, tuple):
                        return (_to_dev(y_cpu[0], out_dev), _to_dev(y_cpu[1], out_dev))
                    return _to_dev(y_cpu, out_dev)

                orig_forwards.append((submodule, orig))
                submodule.forward = wrapped_forward

            # SiluAndMul -> CPU implementation
            elif isinstance(submodule, SiluAndMul):
                orig = submodule.forward

                @functools.wraps(orig)
                def wrapped_forward(x, _mod=submodule):
                    out_dev = x.device
                    x_cpu = _to_cpu(x)
                    y_cpu = silu_and_mul_cpu(x_cpu)
                    return _to_dev(y_cpu, out_dev)

                orig_forwards.append((submodule, orig))
                submodule.forward = wrapped_forward

        try:
            # 🚀 Actually run the layer forward
            out = functional_call(module, params, args=args, kwargs=kwargs)
            return out
        finally:
            # Restore original forwards (must always happen)
            for sm, orig in orig_forwards:
                sm.forward = orig


def _to_cpu(t: torch.Tensor) -> torch.Tensor:
    return t if t.device.type == "cpu" else t.to("cpu")


def _to_dev(t: torch.Tensor, dev: torch.device) -> torch.Tensor:
    return t if t.device == dev else t.to(dev)


def silu_and_mul_cpu(x: torch.Tensor) -> torch.Tensor:
    """
    CPU implementation of SwiGLU: silu(x[..., :d]) * x[..., d:],
    with numerically stable float32 compute and dtype preserved.

    Args:
        x: (..., 2 * d)

    Returns:
        (..., d)
    """

    # print("in cpu silu and mul")

    assert x.device.type == "cpu", "This function is CPU-only"
    assert x.shape[-1] % 2 == 0, "Last dim must be even"

    orig_dtype = x.dtype
    d = x.shape[-1] // 2

    # split
    gate = x[..., :d]
    up = x[..., d:]

    # compute in float32 for stability
    gate_f = gate.float()
    up_f = up.float()

    out = F.silu(gate_f) * up_f

    # cast back
    return out.to(orig_dtype)


def rmsnorm_cpu(
    x_cpu: torch.Tensor,
    weight_cpu: torch.Tensor,
    eps: float,
    residual_cpu: Optional[torch.Tensor] = None,
    variance_size_override: Optional[int] = None,
    has_weight: bool = True,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Strictly match the semantics of vLLM RMSNorm.forward_native you posted:
      - x -> float32
      - if residual: x = x + residual(float32), residual_out = x(orig_dtype)
      - variance on x or x[..., :override]
      - x = x * rsqrt(var + eps)
      - x -> orig_dtype
      - if has_weight: x = x * weight
      - return x or (x, residual_out)
    """

    # print("in cpu rmsnorm")

    orig_dtype = x_cpu.dtype
    x = x_cpu.to(torch.float32)

    if residual_cpu is not None:
        x = x + residual_cpu.to(torch.float32)
        residual_out = x.to(orig_dtype)
    else:
        residual_out = None

    hidden_size = x.shape[-1]
    if variance_size_override is None:
        x_var = x
    else:
        if hidden_size < variance_size_override:
            raise ValueError(
                f"Expected hidden_size >= {variance_size_override}, but got {hidden_size}"
            )
        x_var = x[..., :variance_size_override]

    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype)

    if has_weight:
        x = x * weight_cpu

    return x if residual_out is None else (x, residual_out)
