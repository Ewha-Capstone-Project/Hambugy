# =============================================================================
# cait_s24_build.py  ─  CaiT-S24 ONNX export + TensorRT engine build
#
# 실행:
#   python cait_s24_build.py
# =============================================================================

import os
import re
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
DENSE_ENGINE_PATH  = f"{MODEL_DIR}/cait_s24_dense_fp16.engine"
SPARSE_ENGINE_PATH = f"{MODEL_DIR}/cait_s24_sparse_fp16.engine"

os.makedirs(MODEL_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


# ── 2:4 Sparsity 함수 ────────────────────────────────────────────────────────
@torch.no_grad()
def make_2_4_sparse_weight(weight: torch.Tensor):
    if weight.ndim != 2:
        raise ValueError(f"Only 2D Linear weight is supported, got {weight.shape}")
    out_features, in_features = weight.shape
    if in_features % 4 != 0:
        raise ValueError(f"in_features must be divisible by 4, got {in_features}")
    w_view = weight.data.view(out_features, in_features // 4, 4)
    top2_idx = torch.topk(w_view.abs(), k=2, dim=-1).indices
    mask = torch.zeros_like(w_view, dtype=torch.bool)
    mask.scatter_(dim=-1, index=top2_idx, value=True)
    w_view.mul_(mask)
    return weight


def is_target_cait_linear(name: str, include_token_only: bool = True) -> bool:
    normal_patterns = [
        r"^blocks\.\d+\.attn\.qkv$",
        r"^blocks\.\d+\.attn\.proj$",
        r"^blocks\.\d+\.mlp\.fc1$",
        r"^blocks\.\d+\.mlp\.fc2$",
    ]
    token_only_patterns = [
        r"^blocks_token_only\.\d+\.attn\.q$",
        r"^blocks_token_only\.\d+\.attn\.k$",
        r"^blocks_token_only\.\d+\.attn\.v$",
        r"^blocks_token_only\.\d+\.attn\.proj$",
        r"^blocks_token_only\.\d+\.mlp\.fc1$",
        r"^blocks_token_only\.\d+\.mlp\.fc2$",
    ]
    patterns = normal_patterns + (token_only_patterns if include_token_only else [])
    return any(re.match(p, name) for p in patterns)


@torch.no_grad()
def apply_2_4_sparsity_to_cait(model: nn.Module, include_token_only: bool = True):
    applied, skipped = [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and is_target_cait_linear(name, include_token_only):
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


# ── 2. 2:4 Sparsity 적용 ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. 2:4 Sparsity 적용")
print("=" * 60)

applied, skipped = apply_2_4_sparsity_to_cait(model_sparse, include_token_only=True)

print(f"2:4 적용된 Linear layer 수: {len(applied)}")
print(f"\n{'#':>4}  {'layer name':<70}  {'weight shape'}")
print("-" * 100)
for idx, (name, shape) in enumerate(applied, start=1):
    print(f"{idx:>4}  {name:<70}  {str(shape)}")
print(f"\n총 {len(applied)}개 레이어에 2:4 sparsity 적용 완료")

if skipped:
    print("\n=== 스킵된 layer ===")
    for name, shape, reason in skipped:
        print(f"  SKIP  {name:<70}  {shape}  ({reason})")

print("\n=== 2:4 sparsity valid ratio 확인 ===")
print(f"{'layer name':<70}  {'valid_ratio':>12}  {'status'}")
print("-" * 90)
for name, module in model_sparse.named_modules():
    if isinstance(module, nn.Linear) and is_target_cait_linear(name, include_token_only=True):
        ratio = check_2_4_sparsity(module.weight)
        status = "✓ OK" if ratio is not None and ratio == 1.0 else f"⚠ {ratio}"
        print(f"{name:<70}  {ratio:>12.4f}  {status}")


# ── 3. ONNX Export (FP16) ────────────────────────────────────────────────────
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

print("Dense ONNX export 중 (fp16)...")
torch.onnx.export(model_dense.half().eval(), dummy, DENSE_ONNX, **_export_kwargs)
print(f"  → {DENSE_ONNX}  ({os.path.getsize(DENSE_ONNX)/1024/1024:.1f} MB)")

print("Sparse ONNX export 중 (fp16)...")
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

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    input_name = network.get_input(0).name
    print(f"Network input  : {input_name}  {network.get_input(0).shape}")

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


build_engine(DENSE_ONNX,  DENSE_ENGINE_PATH,  sparse=False, batch_size=16, input_hw=384, fp16=True, workspace_gb=4)
build_engine(SPARSE_ONNX, SPARSE_ENGINE_PATH, sparse=True,  batch_size=16, input_hw=384, fp16=True, workspace_gb=4)

print("\n빌드 완료.")
print(f"  Dense  engine : {DENSE_ENGINE_PATH}")
print(f"  Sparse engine : {SPARSE_ENGINE_PATH}")
