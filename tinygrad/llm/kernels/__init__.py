from __future__ import annotations
from tinygrad import Tensor, UOp
from tinygrad.llm.kernels.cuda import Linear, cuda_custom_kernels_supported
from tinygrad.llm.kernels.cuda import cuda_fused_norm_linears as _cuda_fused_norm_linears
from tinygrad.llm.kernels.amd import flash_attention as _amd_flash, gated_delta_prefill as _amd_scan, amd_custom_kernels_supported

# Backend dispatch for LLM kernels. Callers ask for a capability by device.
# They never branch on backends. To add a backend, import its kernels and
# extend the supported checks plus the dispatchers below.

__all__ = ["Linear", "gated_delta_prefill", "gated_delta_supported", "flash_attention", "flash_supported", "fused_norm_linears"]

def fused_norm_linears(norm, x:Tensor, layers:list) -> list[Tensor]:
  # shared RMSNorm+quantize on CUDA, generic norm then one linear each elsewhere (or when incompatible)
  out = _cuda_fused_norm_linears(norm, x, layers)
  if out is not None: return out
  xn = norm(x)
  return [lin(xn) for lin in layers]

def gated_delta_supported(device:str|tuple[str, ...]|None) -> bool:
  # fused scan handles padded chunk sizes, so chunked prefill also keys off this
  return amd_custom_kernels_supported(device)

def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor, start_pos:Tensor|None=None) -> Tensor:
  dev = q.device
  assert gated_delta_supported(dev), f"fused scan not supported on {dev}"
  return _amd_scan(q, k, v, beta, alpha, state, start_pos)

def flash_supported(device:str|tuple[str, ...]|None) -> bool:
  return amd_custom_kernels_supported(device)

def flash_attention(q:Tensor, assigned_kv:Tensor, valid_end:int|UOp) -> Tensor:
  assert flash_supported(q.device), f"flash attention not supported on {q.device}"
  return _amd_flash(q, assigned_kv, valid_end)
