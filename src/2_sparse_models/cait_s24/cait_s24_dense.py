# =============================================================================
# cait_s24_dense.py  ─  CaiT-S24 Dense TensorRT 엔진 실행 / 속도 측정
#
# 실행:
#   python cait_s24_dense.py
#
# Nsight Systems 프로파일링:
#   nsys profile --trace=cuda,nvtx -o dense_profile python cait_s24_dense.py
# =============================================================================

import numpy as np
import torch
import tensorrt as trt

# ── 설정 ─────────────────────────────────────────────────────────────────────
ENGINE_PATH = "cait_s24_model/cait_s24_dense_fp16.engine"
BATCH_SIZE  = 16
INPUT_HW    = 384
WARMUP      = 50
REPEAT      = 200

# ── 유틸 ─────────────────────────────────────────────────────────────────────
def load_engine(engine_path):
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to load engine: {engine_path}")
    return engine


def get_tensor_dtype_torch(engine, tensor_name):
    dtype = engine.get_tensor_dtype(tensor_name)
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF:  torch.float16,
        trt.DataType.INT8:  torch.int8,
    }[dtype]


# ── 실행 ─────────────────────────────────────────────────────────────────────
print(f"Dense 엔진 로드: {ENGINE_PATH}")
engine  = load_engine(ENGINE_PATH)
context = engine.create_execution_context()

input_name, output_name = "input", "output"
input_shape = (BATCH_SIZE, 3, INPUT_HW, INPUT_HW)
context.set_input_shape(input_name, input_shape)

input_tensor = torch.randn(
    *input_shape, device="cuda",
    dtype=get_tensor_dtype_torch(engine, input_name)
).contiguous()
output_tensor = torch.empty(
    BATCH_SIZE, 1000, device="cuda",
    dtype=get_tensor_dtype_torch(engine, output_name)
)

context.set_tensor_address(input_name,  input_tensor.data_ptr())
context.set_tensor_address(output_name, output_tensor.data_ptr())

stream  = torch.cuda.Stream()
starter = torch.cuda.Event(enable_timing=True)
ender   = torch.cuda.Event(enable_timing=True)

# Warmup
print(f"Warmup {WARMUP}회...")
with torch.cuda.stream(stream):
    for _ in range(WARMUP):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("Dense warmup 실행 실패")
torch.cuda.synchronize()

# Timing
print(f"측정 {REPEAT}회...")
times = []
with torch.cuda.stream(stream):
    for _ in range(REPEAT):
        starter.record(stream)
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("Dense 측정 실행 실패")
        ender.record(stream)
        ender.synchronize()
        times.append(starter.elapsed_time(ender))
torch.cuda.synchronize()

times = np.array(times)
print("\n=== Dense 결과 ===")
print(f"  평균      : {times.mean():.3f} ms")
print(f"  중앙값    : {np.median(times):.3f} ms")
print(f"  최소      : {times.min():.3f} ms")
print(f"  p90       : {np.percentile(times, 90):.3f} ms")
print(f"  p95       : {np.percentile(times, 95):.3f} ms")
print(f"  std       : {times.std():.3f} ms")
print(f"  throughput: {BATCH_SIZE * 1000.0 / times.mean():.2f} samples/s")
