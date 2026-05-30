import os
import copy
import torch
from transformers import AutoModelForImageClassification
import tensorrt as trt

MODEL_ID   = "facebook/deit-base-patch16-224"
DEVICE     = "cuda"
DTYPE      = torch.float16

RESULT_DIR = "./deit_trt"
os.makedirs(RESULT_DIR, exist_ok=True)

# ── CLI로 배치 크기를 받아서 덮어쓸 수 있도록 전역 변수로 선언
# default는 256; 기존 16짜리 엔진은 deit_trt/dense_fp16_b16.engine 으로 보존됨
BATCH_SIZE = 256


def log(msg):
    print(msg)


# ───────────────────────────────
# 2:4 sparsity
# ───────────────────────────────
def apply_2_4_sparsity(weight):

    shape = weight.shape

    if shape[-1] % 4 != 0:
        return weight

    w = weight.view(-1, 4)

    idx = torch.argsort(torch.abs(w), dim=1)

    mask = torch.ones_like(w)
    mask.scatter_(1, idx[:, :2], 0)

    return (w * mask).view(shape)


def sparsify(model):

    m = copy.deepcopy(model)

    for module in m.modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = apply_2_4_sparsity(
                module.weight.data
            )

    return m


# ───────────────────────────────
# Wrapper
# ───────────────────────────────
class Wrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(
            pixel_values=pixel_values
        ).logits


# ───────────────────────────────
# ONNX export
# ───────────────────────────────
def export_onnx(model, path):

    model = Wrapper(model).to(DEVICE).half().eval()

    dummy = torch.randn(
        16, 3, 224, 224  # ONNX export dummy; dynamic_axes로 batch 가변
    ).to(DEVICE).half()

    torch.onnx.export(
        model,
        dummy,
        path,
        opset_version=17,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits":       {0: "batch"},
        }
    )

    log(f"ONNX export: {path}")


# ───────────────────────────────
# TensorRT build
# ───────────────────────────────
def build_trt(onnx_path, engine_path, use_sparse=False, max_batch=None):

    if max_batch is None:
        max_batch = BATCH_SIZE

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)

    network = builder.create_network(
        1 << int(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
        )
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

    profile      = builder.create_optimization_profile()
    input_tensor = network.get_input(0)

    log(f"  profile: min=1  opt/max={max_batch}  sparse={use_sparse}")
    profile.set_shape(
        input_tensor.name,
        (1,         3, 224, 224),   # min
        (max_batch, 3, 224, 224),   # opt
        (max_batch, 3, 224, 224),   # max
    )

    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)

    if serialized is None:
        raise RuntimeError("TensorRT build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    log(f"TensorRT engine: {engine_path}")


# ───────────────────────────────
# Main — 빌드만 수행
# ───────────────────────────────
def main():

    log("=" * 60)
    log("DeiT TensorRT FP16 — engine build")
    log("=" * 60)

    dense_onnx  = f"{RESULT_DIR}/dense.onnx"
    sparse_onnx = f"{RESULT_DIR}/sparse.onnx"

    dense_engine  = f"{RESULT_DIR}/dense_fp16_b{BATCH_SIZE}.engine"
    sparse_engine = f"{RESULT_DIR}/sparse_fp16_b{BATCH_SIZE}.engine"

    # ── ONNX export ─────────────────────────────────────────
    if not os.path.exists(dense_onnx):

        log("\n[1/2] Loading model...")
        model = AutoModelForImageClassification.from_pretrained(
            MODEL_ID, torch_dtype=DTYPE
        ).to(DEVICE)

        sparse_model = sparsify(model)

        log("\n[2/2] Exporting ONNX...")
        export_onnx(model,        dense_onnx)
        export_onnx(sparse_model, sparse_onnx)

    else:
        log("ONNX files already exist, skipping export.")

    # ── TensorRT engine build ────────────────────────────────
    if not os.path.exists(dense_engine):
        log("\n[3/3] Building dense engine...")
        build_trt(dense_onnx, dense_engine, use_sparse=False)
    else:
        log(f"dense engine already exists: {dense_engine}")

    if not os.path.exists(sparse_engine):
        log("\n[3/3] Building sparse engine...")
        build_trt(sparse_onnx, sparse_engine, use_sparse=True)
    else:
        log(f"sparse engine already exists: {sparse_engine}")

    log("\n" + "=" * 60)
    log("Build complete. Generated engines:")
    log(f"  dense  : {dense_engine}")
    log(f"  sparse : {sparse_engine}")
    log("=" * 60)
    log("\n  python deit_layer_profile.py --mode dense")
    log("  python deit_layer_profile.py --mode sparse")
    log("  python deit_layer_profile.py")


if __name__ == "__main__":
    main()
