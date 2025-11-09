from typing import Optional
import torch

def default_unquantized_gemm(layer: torch.nn.Module,
                             x: torch.Tensor,
                             weight: torch.Tensor,
                             bias: Optional[torch.Tensor] = None):
    
    print("here in my monkey patching")
    return torch.nn.functional.linear(x, weight, bias)