'''
nsys profile \
  --trace=cuda,nvtx,cublas,cudnn \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o reports/cait_m48_dense_sparse \
  python3 profile_dense_sparse.py \
    --dense-engine cait_m48_model/cait_m48_dense_fp16.engine \
    --sparse-engine cait_m48_model/cait_m48_sparse_fp16.engine \
    --batch-size 1 \
    --input-hw 448 \
    --warmup 50 \
    --repeat 300
'''


import argparse
import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        engine_bytes = f.read()

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)

    if engine is None:
        raise RuntimeError(f"Failed to load engine: {engine_path}")

    return engine


def trt_dtype_to_torch(dtype):
    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.INT8:
        return torch.int8
    raise ValueError(f"Unsupported dtype: {dtype}")


def run_trt_engine(engine_path, label, batch_size=1, input_hw=448, warmup=50, repeat=300):
    print(f"\n=== Load engine: {label} ===")
    print(engine_path)

    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    input_name = "input"
    output_name = "output"

    input_shape = (batch_size, 3, input_hw, input_hw)
    context.set_input_shape(input_name, input_shape)

    input_dtype = trt_dtype_to_torch(engine.get_tensor_dtype(input_name))
    output_dtype = trt_dtype_to_torch(engine.get_tensor_dtype(output_name))

    x = torch.randn(
        *input_shape,
        device="cuda",
        dtype=input_dtype
    ).contiguous()

    y = torch.empty(
        batch_size,
        1000,
        device="cuda",
        dtype=output_dtype
    )

    context.set_tensor_address(input_name, x.data_ptr())
    context.set_tensor_address(output_name, y.data_ptr())

    stream = torch.cuda.Stream()

    # Warmup 구간
    torch.cuda.nvtx.range_push(f"{label}_WARMUP")
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            ok = context.execute_async_v3(stream.cuda_stream)
            if not ok:
                raise RuntimeError(f"{label} failed during warmup")
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # 실제 inference 구간
    torch.cuda.nvtx.range_push(f"{label}_INFERENCE_REPEAT")
    with torch.cuda.stream(stream):
        for _ in range(repeat):
            ok = context.execute_async_v3(stream.cuda_stream)
            if not ok:
                raise RuntimeError(f"{label} failed during inference")
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    print(f"{label} profiling finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-engine", type=str, required=True)
    parser.add_argument("--sparse-engine", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-hw", type=int, default=448)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=300)
    args = parser.parse_args()

    # 같은 실행 안에서 Dense와 Sparse를 순서대로 실행
    run_trt_engine(
        engine_path=args.dense_engine,
        label="DENSE",
        batch_size=args.batch_size,
        input_hw=args.input_hw,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    run_trt_engine(
        engine_path=args.sparse_engine,
        label="SPARSE",
        batch_size=args.batch_size,
        input_hw=args.input_hw,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    print("\nAll profiling finished.")


if __name__ == "__main__":
    main()