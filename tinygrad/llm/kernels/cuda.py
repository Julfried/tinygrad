from __future__ import annotations
import functools
from tinygrad import Tensor, UOp, nn, Device, Context
from tinygrad.llm.kernels.amd import Linear as AMDLinear, IQ4_XS, GGML_BLOCK_SIZE, Q8_GROUP_SIZE
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.renderer.cstyle import CUDARenderer

# Int8 CUDA GEMV over packed IQ4_XS weights. Mirrors the amd q8 decode path:
# quantize activations to int8 once, dot with dp4a, scale once at the end.
# Weights stream with nontemporal loads, activations stay cached.

IQ4_WORDS = 34  # 136 bytes per 256-weight block as u32 words

@functools.cache
def cuda_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "CUDA": return False
  with Context(ALLOW_DEVICE_USAGE=1):
    return isinstance(getattr(Device[device], "renderer", None), CUDARenderer)

def cuda_warp_reduce(val:UOp, maximum:bool=False, full_wave:bool=False) -> UOp:
  for offset in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)):
    if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
    other = UOp(Ops.CUSTOM, src=(val,), arg=(f"__shfl_xor_sync(0xffffffff, {{0}}, {offset})", val.dtype))
    val = val.maximum(other) if maximum else val + other
  return val

def _cuda_dp4a(a:UOp, b:UOp, c:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, src=(a, b, c), arg=("__dp4a({}, {}, {})", dtypes.int32))

def _iq4_scales(raw:UOp, base:UOp, subgroup:UOp) -> tuple[UOp, UOp]:
  low = ((raw[base+1] >> ((subgroup//2)*8).cast(dtypes.uint32)) & 255)
  scale_l = (subgroup & 1).eq(0).where(low & 15, (low >> 4) & 15)
  scale_h = ((((raw[base] >> 16) & 0xffff) >> (subgroup*2).cast(dtypes.uint32)) & 3)
  d = ((raw[base] & 0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).float())
  return d, (scale_l | (scale_h << 4)).float() - 32

@functools.cache
def _cuda_q8_quantize_kernel(q:UOp, scale:UOp, x:UOp, tokens:int, in_features:int) -> UOp:
  groups = in_features//Q8_GROUP_SIZE
  token_group, lane = UOp.range(tokens*groups, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, group = token_group//groups, token_group%groups
  value = x.reshape(tokens, groups, 32)[token, group, lane].float()
  d = (cuda_warp_reduce(value.abs(), maximum=True, full_wave=True)/127).maximum(1e-8)
  quant = UOp(Ops.CUSTOM, src=(value/d,), arg=("rintf({0})", dtypes.float)).clip(-127, 127).cast(dtypes.int8)
  word = quant.cast(dtypes.uint8).cast(dtypes.uint32) << ((lane%4)*8).cast(dtypes.uint32)
  for offset in (1, 2):
    word |= UOp(Ops.CUSTOM, src=(word,), arg=(f"__shfl_xor_sync(0xffffffff, {{0}}, {offset})", dtypes.uint32))
  stores = (q[token, group, (lane//4).valid((lane%4).eq(0))].store(word),
            scale[token, group.valid(lane.eq(0))].store(d))
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="q8_quantize_cuda", opts_to_apply=()))

def cuda_q8_quantize(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor]:
  groups = in_features//Q8_GROUP_SIZE
  q = Tensor.empty(tokens, groups, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, groups, dtype=dtypes.float32, device=x.device)
  q, scale = Tensor.custom_kernel(q, scale, x, fxn=functools.partial(_cuda_q8_quantize_kernel, tokens=tokens, in_features=in_features))[:2]
  return q, scale

@functools.cache
def _cuda_iq4_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, lut:UOp, out_features:int, in_features:int) -> UOp:
  group_count = in_features // Q8_GROUP_SIZE
  chunks = out.shape[2]
  token_output = UOp.range(out.shape[0]*out_features, 0, axis_type=AxisType.GLOBAL)
  chunk, lane = UOp.range(chunks, 1, axis_type=AxisType.GLOBAL), UOp.range(32, 2, axis_type=AxisType.LOCAL)
  token, output = token_output // out_features, token_output % out_features
  group = (lane+chunk*32).minimum(group_count-1)
  block, subgroup = group // 8, group % 8
  base = (output * in_features//GGML_BLOCK_SIZE + block) * IQ4_WORDS
  xwords = tuple(xq[token, group, w].load().bitcast(dtypes.int32) for w in range(8))
  dot = UOp.const(0, dtypes.int32)
  for w in range(8):
    qw = raw[base + 2 + subgroup*4 + w%4].load(arg="nontemporal")
    nib = tuple(((qw >> (4*(w//4) + p*8)) & 15) for p in range(4))
    # mask to 8 bits: int8->uint32 sign-extends, the high 1s would corrupt the packed neighbors
    wword = (((lut[nib[0]].load().cast(dtypes.uint32) & 255) | ((lut[nib[1]].load().cast(dtypes.uint32) & 255) << 8) |
              ((lut[nib[2]].load().cast(dtypes.uint32) & 255) << 16) | ((lut[nib[3]].load().cast(dtypes.uint32) & 255) << 24)
             ).bitcast(dtypes.int32))
    dot = _cuda_dp4a(wword, xwords[w], dot)
  d, scale = _iq4_scales(raw, base, subgroup)
  value = dot.float() * xd[token, group].load() * d * scale
  if chunks*32 != group_count: value = (lane+chunk*32 < group_count).where(value, UOp.const(0, dtypes.float32))
  total = cuda_warp_reduce(value, full_wave=True)
  return out[token, output, chunk.valid(lane.eq(0))].store(total.cast(out.dtype)).end(token_output, chunk, lane).sink(
    arg=KernelInfo(name="linear_iq4_xs_cuda", opts_to_apply=()))

def cuda_iq4_linear(layer:nn.Linear, x:Tensor) -> Tensor:
  from tinygrad.runtime.autogen.ggml_common import kvalues_iq4nl
  tokens = int(x.numel()) // layer.in_features
  raw, out_features, in_features = layer.weight.uop, layer.out_features, layer.in_features
  lut = Tensor(list(kvalues_iq4nl), dtype=dtypes.int8, device=x.device)
  xq, xd = cuda_q8_quantize(x.contiguous(), tokens, in_features)
  out = Tensor.empty(tokens, out_features, (in_features+1023)//1024, dtype=dtypes.float32, device=x.device).uop
  params = tuple(UOp.placeholder_like(src, slot=i) for i, src in enumerate((out, raw, xq.uop, xd.uop, lut.uop)))
  kernel = _cuda_iq4_decode_kernel(*params, out_features=out_features, in_features=in_features).call(out, raw, xq.uop, xd.uop, lut.uop)
  result = Tensor(out.after(kernel)).sum(-1).reshape(*x.shape[:-1], out_features)
  return result if layer.bias is None else result + layer.bias

class Linear(AMDLinear):
  def _generic_quant(self, x:Tensor) -> Tensor:
    from tinygrad.llm.gguf import ggml_data_to_tensor
    assert self.ggml_type == IQ4_XS
    w = ggml_data_to_tensor(self.weight.bitcast(dtypes.uint8), self.in_features*self.out_features, self.ggml_type)
    w = w.reshape(self.out_features, self.in_features).cast(dtypes.float32)
    return x.float().linear(w.T.contiguous(), self.bias)
  def __call__(self, x:Tensor) -> Tensor:
    cuda_supported = self.use_custom_quant and cuda_custom_kernels_supported(self.weight.device)
    if cuda_supported:
      if self.ggml_type is None:
        orig = self.weight
        self.set_quantized(self.weight)
        if self.ggml_type is None:
          self.use_custom_quant = False
          return nn.Linear.__call__(self, x)
        if self.ggml_type != IQ4_XS:
          self.weight, self.ggml_type, self.use_custom_quant = orig, None, False
          return nn.Linear.__call__(self, x)
      if self.ggml_type == IQ4_XS and self.in_features % Q8_GROUP_SIZE == 0:
        if isinstance(x.numel(), int): return cuda_iq4_linear(self, x)
        out = cuda_iq4_linear(self, x.pad_to(x.max_shape))
        return out.shrink(tuple((0, s) for s in (*x.shape[:-1], self.out_features)))
      return self._generic_quant(x)
    return super().__call__(x)
