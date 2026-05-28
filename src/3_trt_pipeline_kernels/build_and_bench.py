"""
DeiT-small / DeiT-base  x  batch=16,256  x  Dense/Sparse(2:4)
TRT 엔진 빌드 + 벤치마크 일괄 실행
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification
import tensorrt as trt

DEVICE   = "cuda"
DTYPE    = torch.float16
OUT_DIR  = "./deit_trt"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {
    "small": "facebook/deit-small-patch16-224",
    "base":  "facebook/deit-base-patch16-224",
}
BATCHES   = [16, 256]
N_WARMUP  = 50
N_RUNS    = 200


# ─── 2:4 Magnitude Pruning ────────────────────────────────────────────────────
def apply_24_pruning(weight: torch.Tensor) -> torch.Tensor:
    if weight.shape[-1] % 4 != 0:
        return weight
    shape = weight.shape
    w = weight.view(-1, 4)
    zero_idx = torch.argsort(torch.abs(w), dim=1)[:, :2]
    mask = torch.ones_like(w)
    mask.scatter_(1, zero_idx, 0)
    return (w * mask).view(shape)


def sparsify(model: nn.Module) -> nn.Module:
    m = copy.deepcopy(model)
    for module in m.modules():
        if isinstance(module, nn.Linear):
            module.weight.data = apply_24_pruning(module.weight.data)
    return m


# ─── ONNX export ─────────────────────────────────────────────────────────────
class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def export_onnx(model: nn.Module, path: str, ref_batch: int = 16):
    wrapped = Wrapper(model).to(DEVICE).half().eval()
    dummy   = torch.randn(ref_batch, 3, 224, 224, device=DEVICE, dtype=torch.float16)
    torch.onnx.export(
        wrapped, dummy, path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"    ONNX: {path}")


# ─── TRT 빌드 ─────────────────────────────────────────────────────────────────
def build_trt(onnx_path: str, engine_path: str, batch: int, use_sparse: bool):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder    = trt.Builder(TRT_LOGGER)
    network    = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors()):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    if use_sparse:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        network.get_input(0).name,
        (1,     3, 224, 224),
        (batch, 3, 224, 224),
        (batch, 3, 224, 224),
    )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT build failed")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"    TRT: {engine_path}")


# ─── 벤치마크 ─────────────────────────────────────────────────────────────────
def benchmark(engine_path: str, batch: int):
    logger  = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context  = engine.create_execution_context()
    in_name  = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)

    dummy  = torch.randn(batch, 3, 224, 224, dtype=torch.float16, device="cuda")
    context.set_input_shape(in_name, tuple(dummy.shape))
    out_shape = context.get_tensor_shape(out_name)
    output = torch.zeros(tuple(out_shape), dtype=torch.float16, device="cuda")
    context.set_tensor_address(in_name,  int(dummy.data_ptr()))
    context.set_tensor_address(out_name, int(output.data_ptr()))

    stream = torch.cuda.Stream()
    for _ in range(N_WARMUP):
        context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    lats  = []
    for _ in range(N_RUNS):
        start.record()
        context.execute_async_v3(stream.cuda_stream)
        end.record()
        torch.cuda.synchronize()
        lats.append(start.elapsed_time(end))

    lats = np.array(lats)
    return lats.mean(), np.percentile(lats, 50), np.percentile(lats, 99)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # ── Step 1: ONNX export (모델당 한 번만) ─────────────────────────────────
    print("=" * 60)
    print("  [Step 1] ONNX Export")
    print("=" * 60)

    onnx_paths = {}
    for name, model_id in MODELS.items():
        dense_onnx  = f"{OUT_DIR}/deit_{name}_dense.onnx"
        sparse_onnx = f"{OUT_DIR}/deit_{name}_sparse.onnx"
        onnx_paths[name] = (dense_onnx, sparse_onnx)

        if os.path.exists(dense_onnx) and os.path.exists(sparse_onnx):
            print(f"  [{name}] ONNX 이미 존재, 스킵")
            continue

        print(f"  [{name}] {model_id} 로드 중...")
        model = AutoModelForImageClassification.from_pretrained(
            model_id, dtype=DTYPE
        ).to(DEVICE)
        sparse_model = sparsify(model)

        print(f"  [{name}] ONNX export...")
        export_onnx(model,        dense_onnx)
        export_onnx(sparse_model, sparse_onnx)

        total, zeros = 0, 0
        for m in sparse_model.modules():
            if isinstance(m, nn.Linear):
                total += m.weight.numel()
                zeros += (m.weight == 0).sum().item()
        print(f"  [{name}] sparsity: {100 * zeros / total:.1f}%")

        del model, sparse_model
        torch.cuda.empty_cache()

    # ── Step 2: TRT 빌드 ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  [Step 2] TRT Engine Build")
    print("=" * 60)

    engine_paths = {}
    for name in MODELS:
        dense_onnx, sparse_onnx = onnx_paths[name]
        engine_paths[name] = {}
        for batch in BATCHES:
            dense_eng  = f"{OUT_DIR}/deit_{name}_dense_b{batch}.trt"
            sparse_eng = f"{OUT_DIR}/deit_{name}_sparse_b{batch}.trt"
            engine_paths[name][batch] = (dense_eng, sparse_eng)

            if not os.path.exists(dense_eng):
                print(f"  [{name} b{batch}] Dense 빌드...")
                build_trt(dense_onnx,  dense_eng,  batch, use_sparse=False)
            else:
                print(f"  [{name} b{batch}] Dense 이미 존재, 스킵")

            if not os.path.exists(sparse_eng):
                print(f"  [{name} b{batch}] Sparse 빌드 (SPARSE_WEIGHTS)...")
                build_trt(sparse_onnx, sparse_eng, batch, use_sparse=True)
            else:
                print(f"  [{name} b{batch}] Sparse 이미 존재, 스킵")

    # ── Step 3: 벤치마크 ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  [Step 3] Benchmark Results")
    print("=" * 60)
    print(f"  {'Model':<6} {'Batch':>5}  {'Dense(ms)':>10}  {'Sparse(ms)':>10}  {'Speedup':>8}")
    print("  " + "-" * 50)

    for name in MODELS:
        for batch in BATCHES:
            dense_eng, sparse_eng = engine_paths[name][batch]
            d_mean, _, _  = benchmark(dense_eng,  batch)
            s_mean, _, _  = benchmark(sparse_eng, batch)
            speedup = d_mean / s_mean
            print(f"  {name:<6} {batch:>5}  {d_mean:>10.3f}  {s_mean:>10.3f}  {speedup:>8.3f}x")

    print("=" * 60)


if __name__ == "__main__":
    main()
