# =============================================================================
# build.py  ─  CaiT-S24 QKV-split Dense + Sparse TensorRT engine build
#
# 기존 cait_s24_build.py 대비 변경사항:
#   - TalkingHeadAttn의 combined qkv [1152,384] → q/k/v 3개 [384,384]로 분리
#   - 분리 후 각 [384,384] weight에 2:4 sparsity 적용
#   - [384,384]는 128×3 타일로 나눠지므로 TensorRT가 sparse_gemm 선택 가능
#   - builder_optimization_level = 5 로 tactic 탐색 범위 확대
#
# 실행:
#   cd /home/hambugy/yc/cait_s24_qkv_split
#   python build.py
# =============================================================================

import os
import re
import types
import torch
import torch.nn as nn
import timm
import onnx
import tensorrt as trt

# ── 설정 ─────────────────────────────────────────────────────────────────────
MODEL_NAME         = "cait_s24_384"
INPUT_SHAPE        = (1, 3, 384, 384)
MODEL_DIR          = "cait_s24_model"
DENSE_ONNX         = f"{MODEL_DIR}/cait_s24_dense.onnx"
SPARSE_ONNX        = f"{MODEL_DIR}/cait_s24_sparse.onnx"
DENSE_ENGINE_PATH  = f"{MODEL_DIR}/cait_s24_dense_fp16_b64.engine"
SPARSE_ENGINE_PATH = f"{MODEL_DIR}/cait_s24_sparse_fp16_b64.engine"

os.makedirs(MODEL_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


# ── QKV 분리 ─────────────────────────────────────────────────────────────────
def split_qkv_in_model(model: nn.Module):
    """
    TalkingHeadAttn의 self.qkv (Linear [3D, D]) 를
    self.q / self.k / self.v (각 Linear [D, D]) 로 분리하고
    forward를 monkey-patch한다.

    ClassAttn은 이미 q/k/v가 분리되어 있으므로 건드리지 않는다.
    """
    from timm.models.cait import TalkingHeadAttn

    for module in model.modules():
        if not isinstance(module, TalkingHeadAttn):
            continue

        qkv: nn.Linear = module.qkv
        dim = qkv.in_features
        has_bias = qkv.bias is not None

        q_w, k_w, v_w = qkv.weight.data.chunk(3, dim=0)  # 각 [384, 384]

        module.q = nn.Linear(dim, dim, bias=has_bias).to(device)
        module.k = nn.Linear(dim, dim, bias=has_bias).to(device)
        module.v = nn.Linear(dim, dim, bias=has_bias).to(device)

        module.q.weight.data.copy_(q_w)
        module.k.weight.data.copy_(k_w)
        module.v.weight.data.copy_(v_w)

        if has_bias:
            q_b, k_b, v_b = qkv.bias.data.chunk(3, dim=0)
            module.q.bias.data.copy_(q_b)
            module.k.bias.data.copy_(k_b)
            module.v.bias.data.copy_(v_b)

        del module.qkv

        def _new_forward(self, x):
            B, N, C = x.shape
            head_dim = C // self.num_heads
            q = self.q(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3) * self.scale
            k = self.k(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

            attn = q @ k.transpose(-2, -1)
            attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            attn = attn.softmax(dim=-1)
            attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        module.forward = types.MethodType(_new_forward, module)

    return model


# ── 2:4 Sparsity ──────────────────────────────────────────────────────────────
@torch.no_grad()
def make_2_4_sparse_weight(weight: torch.Tensor):
    out_features, in_features = weight.shape
    if in_features % 4 != 0:
        raise ValueError(f"in_features must be divisible by 4, got {in_features}")
    w_view = weight.data.view(out_features, in_features // 4, 4)
    top2_idx = torch.topk(w_view.abs(), k=2, dim=-1).indices
    mask = torch.zeros_like(w_view, dtype=torch.bool)
    mask.scatter_(dim=-1, index=top2_idx, value=True)
    w_view.mul_(mask)
    return weight


def is_target_linear(name: str) -> bool:
    patterns = [
        r"^blocks\.\d+\.attn\.q$",
        r"^blocks\.\d+\.attn\.k$",
        r"^blocks\.\d+\.attn\.v$",
        r"^blocks\.\d+\.attn\.proj$",
        r"^blocks\.\d+\.mlp\.fc1$",
        r"^blocks\.\d+\.mlp\.fc2$",
        r"^blocks_token_only\.\d+\.attn\.q$",
        r"^blocks_token_only\.\d+\.attn\.k$",
        r"^blocks_token_only\.\d+\.attn\.v$",
        r"^blocks_token_only\.\d+\.attn\.proj$",
        r"^blocks_token_only\.\d+\.mlp\.fc1$",
        r"^blocks_token_only\.\d+\.mlp\.fc2$",
    ]
    return any(re.match(p, name) for p in patterns)


@torch.no_grad()
def apply_sparsity(model: nn.Module):
    applied, skipped = [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and is_target_linear(name):
            try:
                make_2_4_sparse_weight(module.weight)
                applied.append((name, tuple(module.weight.shape)))
            except Exception as e:
                skipped.append((name, tuple(module.weight.shape), str(e)))
    return applied, skipped


@torch.no_grad()
def check_2_4_sparsity(weight: torch.Tensor):
    out_features, in_features = weight.shape
    if in_features % 4 != 0:
        return None
    groups = weight.detach().view(out_features, in_features // 4, 4)
    return ((groups != 0).sum(dim=-1) == 2).float().mean().item()


# ── 1. 모델 로드 ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("1. 모델 로드")
print("=" * 60)

print("Dense CaiT-S24 로드 중...")
model_dense = timm.create_model(MODEL_NAME, pretrained=True).to(device).eval()

print("Sparse CaiT-S24 로드 중...")
model_sparse = timm.create_model(MODEL_NAME, pretrained=True).to(device).eval()

params = sum(p.numel() for p in model_dense.parameters())
print(f"모델명: {MODEL_NAME}  |  파라미터 수: {params / 1e6:.2f}M")

# Dense도 QKV 분리 (동일 구조로 공정 비교)
model_dense = split_qkv_in_model(model_dense)
model_sparse = split_qkv_in_model(model_sparse)

from timm.models.cait import TalkingHeadAttn
split_count = sum(1 for m in model_sparse.modules()
                  if isinstance(m, TalkingHeadAttn) and hasattr(m, 'q'))
print(f"QKV 분리 완료: TalkingHeadAttn {split_count}개 블록")

with torch.no_grad():
    dummy_check = torch.randn(1, 3, 384, 384).to(device)
    _ = model_dense(dummy_check)
    _ = model_sparse(dummy_check)
print("sanity check: 추론 통과")


# ── 2. 2:4 Sparsity 적용 ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. 2:4 Sparsity 적용")
print("=" * 60)

applied, skipped = apply_sparsity(model_sparse)

print(f"2:4 적용 레이어: {len(applied)}개")
print(f"\n{'#':>4}  {'layer name':<60}  {'weight shape'}")
print("-" * 85)
for idx, (name, shape) in enumerate(applied, 1):
    print(f"{idx:>4}  {name:<60}  {str(shape)}")

if skipped:
    print("\n=== 스킵 ===")
    for name, shape, reason in skipped:
        print(f"  SKIP  {name:<60}  {shape}  ({reason})")

print("\n=== 2:4 valid ratio ===")
print(f"{'layer name':<60}  {'ratio':>8}  {'status'}")
print("-" * 75)
for name, module in model_sparse.named_modules():
    if isinstance(module, nn.Linear) and is_target_linear(name):
        ratio = check_2_4_sparsity(module.weight)
        status = "OK" if ratio == 1.0 else f"WARN {ratio}"
        print(f"{name:<60}  {ratio:>8.4f}  {status}")


# ── 3. ONNX Export ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. ONNX Export (FP16)")
print("=" * 60)

dummy = torch.randn(*INPUT_SHAPE).to(device).half()
_export_kwargs = dict(
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    do_constant_folding=True,
)

print("Dense ONNX export 중...")
torch.onnx.export(model_dense.half().eval(), dummy, DENSE_ONNX, **_export_kwargs)
print(f"  → {DENSE_ONNX}  ({os.path.getsize(DENSE_ONNX)/1024/1024:.1f} MB)")

print("Sparse ONNX export 중...")
torch.onnx.export(model_sparse.half().eval(), dummy, SPARSE_ONNX, **_export_kwargs)
print(f"  → {SPARSE_ONNX}  ({os.path.getsize(SPARSE_ONNX)/1024/1024:.1f} MB)")

print("\n=== ONNX 검증 ===")
for f in [DENSE_ONNX, SPARSE_ONNX]:
    onnx.checker.check_model(onnx.load(f))
    print(f"  {f}: 검증 완료")


# ── 4. TensorRT Engine Build ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. TensorRT Engine Build")
print("=" * 60)

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


def build_engine(onnx_path, engine_path, sparse=False, batch_size=16,
                 input_hw=384, fp16=True, workspace_gb=4):
    print("=" * 80)
    print(f"ONNX   : {onnx_path}")
    print(f"Engine : {engine_path}")
    print(f"sparse={sparse}, fp16={fp16}, batch_size={batch_size}")
    print("=" * 80)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config  = builder.create_builder_config()

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if sparse:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        print("TensorRT SPARSE_WEIGHTS 활성화")

    try:
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
        )
    except Exception as e:
        print("workspace limit 설정 스킵:", e)

    try:
        config.builder_optimization_level = 3
        print("builder_optimization_level = 3")
    except Exception:
        pass

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    input_name = network.get_input(0).name
    fixed_shape = (batch_size, 3, input_hw, input_hw)
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min=fixed_shape, opt=fixed_shape, max=fixed_shape)
    config.add_optimization_profile(profile)

    print("TensorRT 엔진 빌드 중...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"엔진 저장 완료: {engine_path}  ({os.path.getsize(engine_path)/1024/1024:.1f} MB)")


build_engine(DENSE_ONNX,  DENSE_ENGINE_PATH,  sparse=False, batch_size=64, input_hw=384)
build_engine(SPARSE_ONNX, SPARSE_ENGINE_PATH, sparse=True,  batch_size=64, input_hw=384)

print("\n빌드 완료.")
print(f"  Dense  engine : {DENSE_ENGINE_PATH}")
print(f"  Sparse engine : {SPARSE_ENGINE_PATH}")
