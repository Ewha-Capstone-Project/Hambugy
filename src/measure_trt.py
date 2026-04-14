import tensorrt as trt
import numpy as np
import torch
import sys

logger = trt.Logger(trt.Logger.WARNING)

def run_engine(engine_path):
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    input_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float16).cuda()
    output_tensor = torch.zeros((1, 1000), dtype=torch.float32).cuda()
    context.set_tensor_address("input", input_tensor.data_ptr())
    context.set_tensor_address("output", output_tensor.data_ptr())
    stream = torch.cuda.Stream()
    for _ in range(3):
        context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

mode = sys.argv[1] if len(sys.argv) > 1 else "dense"

if mode == "dense":
    print("Dense 모델 실행 중...")
    run_engine("engines/dense.trt")
else:
    print("Sparse 모델 실행 중...")
    run_engine("engines/sparse.trt")

print("완료!")