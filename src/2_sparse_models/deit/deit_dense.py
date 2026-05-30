# benchmark_dense.py  (benchmark_sparse.py는 engine_path/tag만 다름)
import torch
import numpy as np
import tensorrt as trt
import torch.cuda.nvtx as nvtx
#from cuda import cudart  # pycuda 대신 cuda-python 사용
import ctypes
_cudart = ctypes.CDLL("libcudart.so")

ENGINE_PATH = "./deit_trt/dense_fp16.engine"
TAG         = "DENSE"
BATCH_SIZE  = 16
N_WARMUP    = 20
N_RUNS      = 200

def main():
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime    = trt.Runtime(TRT_LOGGER)

    with open(ENGINE_PATH, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context     = engine.create_execution_context()
    input_name  = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    dummy = torch.randn(BATCH_SIZE, 3, 224, 224,
                        dtype=torch.float16, device="cuda")
    context.set_input_shape(input_name, tuple(dummy.shape))

    out_shape = context.get_tensor_shape(output_name)
    output    = torch.zeros(tuple(out_shape),
                            dtype=torch.float16, device="cuda")

    ptrs = [int(dummy.data_ptr()), int(output.data_ptr())]

    # ── warmup (프로파일러 캡처 범위 밖) ──────────────────────
    for _ in range(N_WARMUP):
        context.execute_v2(ptrs)
    torch.cuda.synchronize()

    # ── 여기서부터만 Nsight가 캡처 ────────────────────────────
    _cudart.cudaProfilerStart()
    nvtx.range_push(f"{TAG}_INFERENCE")

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)

    lat = []
    for _ in range(N_RUNS):
        start_evt.record()
        context.execute_v2(ptrs)
        end_evt.record()
        torch.cuda.synchronize()
        lat.append(start_evt.elapsed_time(end_evt))  # ms

    nvtx.range_pop()
    _cudart.cudaProfilerStop()
    # ─────────────────────────────────────────────────────────

    print(f"{TAG}  mean={np.mean(lat):.3f} ms  "
          f"p50={np.percentile(lat,50):.3f} ms  "
          f"p99={np.percentile(lat,99):.3f} ms")

if __name__ == "__main__":
    main()
