"""
ViT-Large Dense vs Sparse — Nsight Systems 커널 분석 라이브 데모
실행: streamlit run demo_vit.py
"""

import os, re, copy, subprocess, sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import torch, torch.nn as nn
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="ViT-Large Kernel Analysis",
    layout="wide",
)

st.markdown("""
<style>
.section-header {
    background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
    color: white; padding: 10px 20px; border-radius: 8px;
    font-size: 1.2rem; font-weight: bold; margin: 16px 0 8px 0;
}
.finding-box {
    background: #d1ecf1; border-left: 4px solid #0dcaf0;
    padding: 12px 16px; border-radius: 4px; margin: 8px 0;
}
.warning-box {
    background: #f8d7da; border-left: 4px solid #dc3545;
    padding: 12px 16px; border-radius: 4px; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("ViT-Large: Dense vs Sparse — Kernel Analysis")
st.caption("Team Hambugy · Ewha Womans University · Capstone Design 2026")

# ──────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────
MODEL_NAME    = "vit_large_patch16_224"
INPUT_SHAPE   = (1, 3, 224, 224)
DENSE_ENGINE  = "vit_large_model/vit_large_dense_fp16.engine"
SPARSE_ENGINE = "vit_large_model/vit_large_sparse_fp16.engine"
DENSE_ONNX    = "vit_large_model/vit_large_dense.onnx"
SPARSE_ONNX   = "vit_large_model/vit_large_sparse.onnx"
NSYS_OUT_DIR  = "nsys_profiles"
device        = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(NSYS_OUT_DIR, exist_ok=True)
os.makedirs("vit_large_model", exist_ok=True)

# ──────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def make_2_4_sparse_weight(w: torch.Tensor):
    out_f, in_f = w.shape
    wv = w.data.view(out_f, in_f // 4, 4)
    mask = torch.zeros_like(wv, dtype=torch.bool)
    mask.scatter_(-1, torch.topk(wv.abs(), 2, dim=-1).indices, True)
    wv.mul_(mask)
    return w

def is_target_linear(name):
    return any(re.match(p, name) for p in [
        r"^blocks\.\d+\.attn\.qkv$", r"^blocks\.\d+\.attn\.proj$",
        r"^blocks\.\d+\.mlp\.fc1$",  r"^blocks\.\d+\.mlp\.fc2$",
    ])

@torch.no_grad()
def apply_sparsity(model):
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and is_target_linear(name):
            make_2_4_sparse_weight(m.weight)

def build_engine(onnx_path, engine_path, sparse=False, fp16=True, workspace_gb=8):
    import tensorrt as trt
    L   = trt.Logger(trt.Logger.WARNING)
    b   = trt.Builder(L)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    p   = trt.OnnxParser(net, L)
    cfg = b.create_builder_config()
    if fp16:   cfg.set_flag(trt.BuilderFlag.FP16)
    if sparse: cfg.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    try:
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb*(1<<30)))
    except Exception: pass
    with open(onnx_path, "rb") as f:
        if not p.parse(f.read()): raise RuntimeError("ONNX parse failed")
    inp   = net.get_input(0)
    prof  = b.create_optimization_profile()
    shape = (1, 3, 224, 224)
    prof.set_shape(inp.name, shape, shape, shape)
    cfg.add_optimization_profile(prof)
    ser = b.build_serialized_network(net, cfg)
    if ser is None: raise RuntimeError("TRT build failed")
    with open(engine_path, "wb") as f: f.write(ser)

def load_engine(path):
    import tensorrt as trt
    with open(path, "rb") as f:
        return trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())

def trt_dtype_torch(engine, name):
    import tensorrt as trt
    return {trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF:  torch.float16}.get(engine.get_tensor_dtype(name), torch.float32)

def run_benchmark(engine_path, warmup=50, repeat=200):
    engine = load_engine(engine_path)
    ctx    = engine.create_execution_context()
    ctx.set_input_shape("input", (1, 3, 224, 224))
    inp = torch.randn(1,3,224,224, device="cuda",
                      dtype=trt_dtype_torch(engine,"input")).contiguous()
    out = torch.empty(1, 1000, device="cuda",
                      dtype=trt_dtype_torch(engine,"output")).contiguous()
    ctx.set_tensor_address("input",  inp.data_ptr())
    ctx.set_tensor_address("output", out.data_ptr())
    stream  = torch.cuda.Stream()
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        for _ in range(warmup): ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    times = []
    with torch.cuda.stream(stream):
        for _ in range(repeat):
            starter.record(stream)
            ctx.execute_async_v3(stream.cuda_stream)
            ender.record(stream)
            ender.synchronize()
            times.append(starter.elapsed_time(ender))
    torch.cuda.synchronize()
    t = np.array(times)
    return {"mean": float(t.mean()), "median": float(np.median(t)),
            "min":  float(t.min()),  "p90":    float(np.percentile(t,90)),
            "p95":  float(np.percentile(t,95)), "std":  float(t.std()),
            "throughput": float(1000/t.mean())}

# ──────────────────────────────────────────────────────────
# Nsight Systems
# ──────────────────────────────────────────────────────────
PROFILE_SCRIPT = """\
import torch, sys
engine_path = sys.argv[1]
import tensorrt as trt
with open(engine_path, "rb") as f:
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()
ctx.set_input_shape("input", (1,3,224,224))
def dtype(name):
    return {trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF:  torch.float16}.get(engine.get_tensor_dtype(name), torch.float32)
inp = torch.randn(1,3,224,224, device="cuda", dtype=dtype("input")).contiguous()
out = torch.empty(1,1000,       device="cuda", dtype=dtype("output")).contiguous()
ctx.set_tensor_address("input",  inp.data_ptr())
ctx.set_tensor_address("output", out.data_ptr())
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    for _ in range(10): ctx.execute_async_v3(stream.cuda_stream)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
with torch.cuda.stream(stream):
    for _ in range(20): ctx.execute_async_v3(stream.cuda_stream)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
"""

def run_nsys(engine_path, tag):
    script_path = os.path.join(NSYS_OUT_DIR, "_runner.py")
    with open(script_path, "w") as f: f.write(PROFILE_SCRIPT)
    out_pfx = os.path.join(NSYS_OUT_DIR, tag)
    rep     = out_pfx + ".nsys-rep"
    sqlite  = out_pfx + ".sqlite"
    subprocess.run([
        "nsys", "profile",
        "--output", out_pfx,
        "--force-overwrite", "true",
        "--capture-range", "cudaProfilerApi",
        "--trace", "cuda,nvtx",
        "python", script_path, engine_path,
    ], capture_output=True, text=True)
    # returncode 143(SIGTERM)은 nsys 정상 종료 — rep 파일 존재 여부로 성공 판단
    if os.path.exists(rep):
        subprocess.run(["nsys","export","--type=sqlite","--output", sqlite, rep],
                       capture_output=True)
    return {"sqlite": sqlite if os.path.exists(sqlite) else None}

def parse_kernels_from_sqlite(sqlite_path: str) -> pd.DataFrame:
    con        = sqlite3.connect(sqlite_path)
    string_map = dict(zip(*pd.read_sql("SELECT id, value FROM StringIds", con).values.T))
    df         = pd.read_sql("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 10000", con)
    con.close()
    df["kernel_name"] = df["demangledName"].map(string_map).fillna(
                        df["shortName"].map(string_map)).fillna("unknown")
    df["duration_ns"] = df["end"] - df["start"]
    df["duration_us"] = df["duration_ns"] / 1000.0
    return df

def classify_kernel(name) -> str:
    if not isinstance(name, str): name = str(name)
    n = name.lower()
    if any(k in n for k in ["fused","flash","epilogue","myl_fc","myl2"]): return "fused"
    if any(k in n for k in ["spmma","sparse","spgemm","sp_gemm"]):        return "sparse_gemm"
    if any(k in n for k in ["gemm","cublas","wmma","hgemm","sgemm"]):     return "dense_gemm"
    if any(k in n for k in ["softmax","layernorm","norm"]):               return "norm/softmax"
    if any(k in n for k in ["elementwise","vectorized","cast","convert"]): return "elementwise"
    if any(k in n for k in ["reduce","sum","pool"]):                      return "reduction"
    return "other"

def summarize_kernels(df: pd.DataFrame) -> dict:
    if df.empty or "kernel_name" not in df.columns: return {}
    df         = df.copy()
    df["type"] = df["kernel_name"].apply(classify_kernel)
    total_us   = df["duration_us"].sum()
    by_type = (df.groupby("type")["duration_us"]
               .agg(["sum","count","mean"])
               .rename(columns={"sum":"total_us","count":"calls","mean":"mean_us"})
               .reset_index()
               .assign(pct=lambda d: (d["total_us"]/total_us*100).round(2))
               .sort_values("total_us", ascending=False))
    top_kernels = (df.groupby("kernel_name")["duration_us"]
                   .agg(["sum","count","mean"])
                   .rename(columns={"sum":"total_us","count":"calls","mean":"mean_us"})
                   .reset_index()
                   .assign(type=lambda d: d["kernel_name"].apply(classify_kernel))
                   .sort_values("total_us", ascending=False).head(20))
    return {
        "by_type":     by_type,
        "top_kernels": top_kernels,
        "fused_pct":   float(by_type.loc[by_type["type"]=="fused","pct"].sum()),
        "sparse_pct":  float(by_type.loc[by_type["type"]=="sparse_gemm","pct"].sum()),
    }

# ──────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────
COLOR_MAP = {
    "fused":        "#e74c3c",
    "sparse_gemm":  "#f39c12",
    "dense_gemm":   "#3498db",
    "norm/softmax": "#2ecc71",
    "elementwise":  "#9b59b6",
    "reduction":    "#1abc9c",
    "other":        "#95a5a6",
}

def plot_latency_bar(dr, sr):
    metrics = ["mean", "median", "min", "p90", "p95"]
    labels  = ["Mean", "Median", "Min", "P90", "P95"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Dense",        x=labels, y=[dr[m] for m in metrics],
                         marker_color="#3498db",
                         text=[f"{dr[m]:.2f}" for m in metrics],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="Sparse (2:4)", x=labels, y=[sr[m] for m in metrics],
                         marker_color="#e67e22",
                         text=[f"{sr[m]:.2f}" for m in metrics],
                         textposition="outside"))
    speedup = dr["mean"] / sr["mean"]
    fig.update_layout(
        barmode="group",
        title=f"ViT-Large TensorRT FP16 — Dense vs Sparse  (Speedup: {speedup:.3f}x / 이론치 ~2.0x)",
        yaxis_title="Latency (ms)",
        yaxis=dict(range=[0, max(dr["p95"], sr["p95"]) * 1.3]),
        height=380, margin=dict(t=60, b=40),
    )
    return fig

def plot_kernel_type_pie(summary: dict, tag: str):
    bt     = summary["by_type"]
    colors = [COLOR_MAP.get(t, "#95a5a6") for t in bt["type"]]
    fig = go.Figure(go.Pie(
        labels=bt["type"], values=bt["total_us"],
        marker_colors=colors, textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value:.0f} µs (%{percent})<extra></extra>",
    ))
    fig.update_layout(title=f"{tag} — 커널 타입별 GPU 시간",
                      height=360, margin=dict(t=50, b=20))
    return fig

def plot_top_kernels(summary: dict, tag: str, n=15):
    tk = summary["top_kernels"].head(n).copy()
    tk["color"]   = tk["type"].apply(lambda t: COLOR_MAP.get(t, "#95a5a6"))
    tk["opacity"] = tk["type"].apply(lambda t: 1.0 if t == "fused" else 0.75)
    fig = go.Figure(go.Bar(
        x=tk["total_us"],
        y=tk["kernel_name"],
        orientation="h",
        marker_color=tk["color"],
        marker_opacity=tk["opacity"],
        hovertemplate="<b>%{y}</b><br>%{x:.0f} µs (%{customdata}회)<extra></extra>",
        customdata=tk["calls"],
    ))
    fig.update_layout(
        title=f"{tag} — Top {n} 커널  (빨강=Fused, 노랑=Sparse GEMM, 파랑=Dense GEMM)",
        xaxis_title="Total GPU Time (µs)",
        height=n * 52,
        margin=dict(t=50, b=40, r=160, autoexpand=True),
        yaxis=dict(
            autorange="reversed",
            automargin=True,
            tickfont=dict(size=10),
        ),
    )
    return fig

# ──────────────────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("설정")
    warmup = st.slider("Warmup 횟수", 10, 100, 50, 10)
    repeat = st.slider("측정 반복",   50, 500, 200, 50)

    st.divider()
    st.header("Nsight Systems")
    nsys_ok = subprocess.run(["which","nsys"], capture_output=True).returncode == 0
    if nsys_ok:
        st.success("nsys 감지됨")
        ver = subprocess.run(["nsys","--version"], capture_output=True,
                              text=True).stdout.strip().split("\n")[0]
        st.caption(ver)
    else:
        st.error("nsys 미설치")
    use_cached = st.checkbox("캐시된 nsys 결과 사용", value=True)

    st.divider()
    trt_ok = False
    try:
        import tensorrt; trt_ok = True
        st.success("TensorRT 감지됨")
    except ImportError:
        st.warning("TensorRT 없음")

    st.caption(f"Device: `{device}`")
    run_btn = st.button("데모 실행", type="primary", use_container_width=True)

if not run_btn:
    st.info("사이드바에서 **데모 실행** 버튼을 누르세요.")
    st.stop()

# ──────────────────────────────────────────────────────────
# 섹션 1: 모델 로드 & 엔진 준비
# ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">① 모델 로드 & TensorRT 엔진 준비</div>',
            unsafe_allow_html=True)

with st.status("모델 준비 중...", expanded=True) as s1:
    import timm, onnx
    st.write("ViT-Large 로드...")
    model  = timm.create_model(MODEL_NAME, pretrained=True).to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    st.write(f"파라미터: {params/1e6:.1f}M")
    model_dense  = copy.deepcopy(model).to(device).eval()
    model_sparse = copy.deepcopy(model).to(device).eval()
    apply_sparsity(model_sparse)
    st.write("2:4 Sparsity 적용 완료 (blocks.*.attn.qkv/proj, mlp.fc1/fc2)")
    if trt_ok:
        dummy = torch.randn(*INPUT_SHAPE).to(device)
        for tag, mdl, opath in [("Dense", model_dense, DENSE_ONNX),
                                 ("Sparse", model_sparse, SPARSE_ONNX)]:
            if not os.path.exists(opath):
                st.write(f"{tag} ONNX export...")
                torch.onnx.export(
                    mdl.float().eval(), dummy, opath, opset_version=17,
                    input_names=["input"], output_names=["output"],
                    dynamic_axes={"input":{0:"b"},"output":{0:"b"}},
                    do_constant_folding=True,
                )
            st.write(f"{tag} ONNX: {os.path.getsize(opath)/1024/1024:.1f} MB")
        for tag, op, ep, sp_flag in [
            ("Dense",  DENSE_ONNX,  DENSE_ENGINE,  False),
            ("Sparse", SPARSE_ONNX, SPARSE_ENGINE, True),
        ]:
            if not os.path.exists(ep):
                st.write(f"{tag} TRT 엔진 빌드 중...")
                build_engine(op, ep, sparse=sp_flag)
            st.write(f"{tag} 엔진: {os.path.getsize(ep)/1024/1024:.1f} MB")
    s1.update(label="① 준비 완료", state="complete")

# 모델 기본 정보
i1, i2, i3, i4, i5, i6 = st.columns(6)
i1.metric("Model",       "ViT-Large/16")
i2.metric("Parameters",  f"{params/1e6:.0f}M")
i3.metric("Hidden Dim",  "1024")
i4.metric("Depth",       "24 blocks")
i5.metric("Patch Size",  "16x16")
i6.metric("Batch Size",  "1")

# ──────────────────────────────────────────────────────────
# 섹션 2: 속도 비교
# ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">② Dense vs Sparse 속도 비교</div>',
            unsafe_allow_html=True)

with st.status("벤치마크 실행 중...", expanded=True) as s2:
    if trt_ok and os.path.exists(DENSE_ENGINE) and os.path.exists(SPARSE_ENGINE):
        st.write(f"TensorRT FP16 측정 (warmup={warmup}, repeat={repeat})...")
        dr = run_benchmark(DENSE_ENGINE,  warmup, repeat)
        sr = run_benchmark(SPARSE_ENGINE, warmup, repeat)
    else:
        @torch.no_grad()
        def pt_bench(mdl, warmup, repeat):
            mdl = mdl.to(device).eval()
            x   = torch.randn(*INPUT_SHAPE).to(device)
            for _ in range(warmup): mdl(x)
            if device == "cuda": torch.cuda.synchronize()
            times = []
            for _ in range(repeat):
                e1 = torch.cuda.Event(enable_timing=True)
                e2 = torch.cuda.Event(enable_timing=True)
                e1.record(); mdl(x); e2.record()
                torch.cuda.synchronize()
                times.append(e1.elapsed_time(e2))
            t = np.array(times)
            return {"mean":float(t.mean()), "median":float(np.median(t)),
                    "min":float(t.min()),   "p90":float(np.percentile(t,90)),
                    "p95":float(np.percentile(t,95)), "std":float(t.std()),
                    "throughput":float(1000/t.mean())}
        dr = pt_bench(model_dense,  warmup, repeat)
        sr = pt_bench(model_sparse, warmup, repeat)

    speedup = dr["mean"] / sr["mean"]
    st.write(f"Dense: {dr['mean']:.2f} ms  |  Sparse: {sr['mean']:.2f} ms  |  Speedup: {speedup:.3f}x")
    s2.update(label="② 벤치마크 완료", state="complete")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dense Mean",   f"{dr['mean']:.2f} ms")
c2.metric("Sparse Mean",  f"{sr['mean']:.2f} ms",
          delta=f"-{dr['mean']-sr['mean']:.2f} ms", delta_color="inverse")
c3.metric("Speedup",      f"{speedup:.3f}x")
c4.metric("이론 최대",    "~2.0x")

st.plotly_chart(plot_latency_bar(dr, sr), use_container_width=True)

st.markdown(f"""
<div class="warning-box">
이론치 2.0x 대비 실제 <b>{speedup:.2f}x</b> — Nsight Systems로 어떤 커널이 병목인지 확인합니다.
</div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# 섹션 3: nsys 프로파일링 & 커널 분석
# ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">③ Nsight Systems — 커널 분석</div>',
            unsafe_allow_html=True)

dense_sum = sparse_sum = None
d_sqlite  = os.path.join(NSYS_OUT_DIR, "dense.sqlite")
s_sqlite  = os.path.join(NSYS_OUT_DIR, "sparse.sqlite")
need_dense  = not (use_cached and os.path.exists(d_sqlite))
need_sparse = not (use_cached and os.path.exists(s_sqlite))

if nsys_ok and (need_dense or need_sparse):
    with st.status("nsys 프로파일링 실행 중...", expanded=True) as s3:
        if need_dense:
            st.write("Dense 엔진 프로파일링...")
            r = run_nsys(DENSE_ENGINE, "dense")
            if r["sqlite"]: st.write("Dense 완료")
            else: st.error("Dense 프로파일링 실패 — .nsys-rep 파일이 생성되지 않았습니다.")
        if need_sparse:
            st.write("Sparse 엔진 프로파일링...")
            r = run_nsys(SPARSE_ENGINE, "sparse")
            if r["sqlite"]: st.write("Sparse 완료")
            else: st.error("Sparse 프로파일링 실패 — .nsys-rep 파일이 생성되지 않았습니다.")
        s3.update(label="③ 프로파일링 완료", state="complete")
elif not nsys_ok:
    st.warning("nsys가 없습니다. nsys_profiles/ 폴더에 .sqlite 파일이 있으면 자동으로 사용합니다.")

if os.path.exists(d_sqlite) and os.path.exists(s_sqlite):
    dense_df   = parse_kernels_from_sqlite(d_sqlite)
    sparse_df  = parse_kernels_from_sqlite(s_sqlite)
    dense_sum  = summarize_kernels(dense_df)
    sparse_sum = summarize_kernels(sparse_df)

if dense_sum and sparse_sum:
    col_l, col_r = st.columns(2)
    with col_l: st.plotly_chart(plot_kernel_type_pie(dense_sum,  "Dense"),  use_container_width=True)
    with col_r: st.plotly_chart(plot_kernel_type_pie(sparse_sum, "Sparse"), use_container_width=True)

    st.markdown(f"""
<div class="warning-box">
<code>fc1/MatMul + GELU + fc1/Add</code> 가 하나의 fusion 노드(<code>__myl_Fc_myl2_*</code>)로 합쳐짐<br>
TensorRT가 이 구간을 fused kernel로 최적화하면서 내부 GEMM이 <b>Sparse GEMM으로 교체되지 않음</b><br>
Dense Fused: <b>{dense_sum['fused_pct']:.1f}%</b> &nbsp;/&nbsp; Sparse Fused: <b>{sparse_sum['fused_pct']:.1f}%</b>
</div>""", unsafe_allow_html=True)

    st.plotly_chart(plot_top_kernels(dense_sum,  "Dense",  15), use_container_width=True)
    st.plotly_chart(plot_top_kernels(sparse_sum, "Sparse", 15), use_container_width=True)

    st.subheader("커널 타입별 상세 수치")
    col_l, col_r = st.columns(2)
    with col_l:
        st.caption("Dense")
        st.dataframe(dense_sum["by_type"].style.format(
            {"total_us":"{:.0f}","mean_us":"{:.1f}","pct":"{:.1f}%"}),
            use_container_width=True, hide_index=True)
    with col_r:
        st.caption("Sparse")
        st.dataframe(sparse_sum["by_type"].style.format(
            {"total_us":"{:.0f}","mean_us":"{:.1f}","pct":"{:.1f}%"}),
            use_container_width=True, hide_index=True)
else:
    st.info("nsys_profiles/ 폴더에 dense.sqlite, sparse.sqlite가 있으면 자동으로 분석합니다.")