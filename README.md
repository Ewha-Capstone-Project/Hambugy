# 2:4 Structured Sparsity for Faster ViT Inference on Edge Devices: TensorRT Fusion and Speed Bottlenecks

> **NVIDIA Jetson AGX Orin** 위에서 Vision Transformer 계열 모델에 2:4 Structured Sparsity를 적용하고,  
> TensorRT Fusion이 Sparse Tensor Core 활용을 어떻게 방해하는지 분석하여 커널 수준에서 해결하는 파이프라인

---

## Team

| 이름 | GitHub | 담당 |
|------|--------|------|
| 신성현 | [@cindyshin2211](https://github.com/cindyshin2211) | 모델 선정 분석 · 프로파일링 |
| 장수연 | [@ally010314](https://github.com/ally010314) | TensorRT 파이프라인 · 커스텀 커널 |
| 송영채 | [@syc031023](https://github.com/syc031023) | 다중 모델 Sparse 변환 · QKV 분리 |

---

## Background

Vision-Language Model(VLM)의 비전 인코더는 대규모 Linear weight와 반복적인 Attention+MLP 구조로 인해  
엣지 디바이스에서의 추론 속도가 심각한 병목이 된다.

NVIDIA Ampere 이상 아키텍처는 **2:4 Structured Sparsity**를 지원한다.  
연속 4개 가중치 중 정확히 2개를 0으로 만들면 Sparse Tensor Core가 non-zero 값만 선택적으로 연산하여  
이론상 **2배의 처리량**을 달성한다.

그러나 실제로 TensorRT로 엔진을 빌드하면 이 이론치에 도달하지 못하는 경우가 있다.  
**그 이유를 파고드는 것이 이 프로젝트의 핵심이다.**

---

## Story

### 1단계 — 어떤 모델이 의미 있는가? (`1_model_analysis/`)

2:4 Sparsity는 weight matrix가 충분히 커야 효과가 나온다.  
행렬이 작으면 연산 절감보다 메모리 IO 오버헤드가 커서 오히려 느려질 수 있다.

신성현이 다양한 모델을 Jetson 위에서 직접 프로파일링하여 후보군을 좁혔다.

| 모델 | 입력 해상도 | 파라미터 | 선정 여부 |
|------|------------|----------|-----------|
| DeiT-small | 224×224 | 22M | 기준선 |
| **DeiT-base** | 224×224 | 86M | ✅ 선정 |
| CaiT-S24 | 384×384 | 47M | ✅ 선정 |
| CaiT-S36 | 384×384 | 68M | ✅ 선정 |
| **CaiT-M48** | 448×448 | 356M | ✅ 선정 (최대 규모) |
| RT-DETR | — | — | ❌ (구조 제약) |
| Qwen-VL | — | — | ❌ (메모리 초과) |
| ViT-large / ViT-huge | 224×224 | 307M / 633M | 참고 비교용 |

**Key finding**: weight matrix 크기와 sparsity 효과 간 상관관계 확인.  
작은 모델(DeiT-small)은 Dense가 오히려 빠른 배치 구간이 존재함.

---

### 2단계 — 2:4 Sparse 변환 파이프라인 구축 (`2_sparse_models/`)

선정된 모델 전체에 일관된 2:4 변환 파이프라인을 구축했다.

**변환 방식**: magnitude 기반 pruning  
연속 4개 가중치 블록에서 절댓값 기준 하위 2개를 0으로 마스킹 → `to_sparse_semi_structured` 변환

```
모델 로드 (timm) → Linear weight에 2:4 pruning → ONNX export → TensorRT FP16 엔진 빌드
```

Jetson Orin은 cuSPARSELt를 지원하지 않으므로 **CUTLASS 백엔드**를 강제 사용:
```python
torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = True
```

**QKV Split 최적화** (`cait_s24_qkv_split/`): CaiT의 QKV projection을 단일 행렬 대신  
Q, K, V 세 개의 독립 행렬로 분리하여 각각에 2:4 패턴을 적용, sparsity 활용률 향상 시도.

---

### 3단계 — 왜 이론치에 못 미치는가? (`3_trt_pipeline_kernels/`)

2:4 Sparse 엔진을 빌드하고 측정했더니 fc1 레이어에서 예상보다 speedup이 낮게 나왔다.  
TensorRT 엔진 레이어 이름을 보면 그 이유가 드러난다:

```
Name: __myl_Fc_myl2_7
LayerType: fusion
→ fc1/MatMul + GELU(Mul, Add, Div, Erf) + fc1/Add 가 하나의 fusion 노드로 합쳐짐
```

**TensorRT가 MatMul과 GeLU를 자동으로 fuse하기 때문에**  
fc1 MatMul이 독립적인 Sparse Tensor Core 연산으로 분리되지 않는다.

#### 해결책 1 — ONNX Graph Surgery (`unfused.py`)

ONNX 그래프에서 `fc1/MatMul` 뒤에 `Identity` 노드를 삽입하여  
TensorRT가 MatMul-GeLU를 하나로 합치지 못하도록 강제로 끊는다.

```python
# MatMul 출력 뒤에 Identity 삽입 → TRT fusion 차단
identity_node = gs.Node(op="Identity", inputs=[matmul_out], outputs=[identity_out])
```

#### 해결책 2 — Custom Triton Sparse Kernel (`sparse_kernels.py`)

MVUE24(Minimum Variance Unbiased Estimator for 2:4) 알고리즘을 Triton으로 직접 구현.  
확률적 반올림 기반으로 2:4 마스크를 생성하여 기댓값 보존.

```python
@triton.jit
def _MVUE24_approx_triton(dense_ptr, sparse_ptr, ...):
    # 4개 원소씩 로드
    # 확률적 2:4 마스크 생성 (seed 기반 재현 가능)
    # non-zero 2개만 sparse 버퍼에 저장
```

---

## Results

Dense / 2:4 Sparse TRT / Unfused Sparse / Custom Kernel 4가지를 Jetson AGX Orin에서 비교.

> 측정 조건: FP16, Warmup 50회, 측정 200회, NVTX 마커로 구간 분리

| 구성 | 비고 |
|------|------|
| Dense FP16 TRT | 기준 |
| 2:4 Sparse TRT | fc1 fusion → Sparse TC 미활용 구간 존재 |
| Unfused Sparse TRT | ONNX surgery로 MatMul 분리 후 |
| Custom Triton Kernel | MVUE24 기반 직접 구현 |

---

## Repository Structure

```
.
├── 1_model_analysis/                    # 신성현
│   ├── 01_ViT_memory_analysis.ipynb
│   ├── 02_VLM_sparsity_test.ipynb
│   ├── 05_DEIT_speed.ipynb              ← fusion 문제 최초 발견
│   ├── 06_CaiT-M48_speed.ipynb
│   ├── 08_Vit_large_speed.ipynb
│   └── profile_dense_sparse.py
│
├── 2_sparse_models/                     # 송영채
│   ├── deit/
│   │   ├── deit_build.py
│   │   ├── deit_dense.py
│   │   └── deit_sparse.py
│   ├── cait_s24/
│   │   ├── cait_s24_build.py
│   │   ├── cait_s24_dense.py
│   │   └── cait_s24_sparse.py
│   ├── cait_s24_qkv_split/              ← QKV 분리 최적화
│   │   ├── build.py
│   │   ├── dense.py
│   │   └── sparse.py
│   └── cait_s36/
│       ├── cait_build.py
│       ├── cait_dense.py
│       └── cait_sparse.py
│
└── 3_trt_pipeline_kernels/              # 장수연
    ├── deit_24.py                       ← 2:4 pruning + CUTLASS 변환
    ├── unfused.py                       ← ONNX graph surgery (fusion 차단)
    ├── sparse_kernels.py                ← Triton MVUE24 custom kernel
    ├── build_and_bench.py
    ├── benchmark_orin_fin.py
    └── utils.py
```

> `.onnx`, `.engine`, `.trt` 등 빌드 결과물은 용량 문제로 포함하지 않습니다.  
> 각 `build.py`를 실행하면 로컬에 자동 생성됩니다.

---

## Reproducing Experiments

### Prerequisites

```bash
pip install torch torchvision timm transformers
pip install tensorrt onnx onnx-graphsurgeon
pip install triton
```

> Jetson AGX Orin에서는 JetPack SDK가 설치된 상태를 전제합니다.  
> TensorRT는 JetPack에 포함된 버전을 사용하세요 (pip 버전과 충돌 가능).

---

### Stage 1 — 모델 선정 분석 (`1_model_analysis/`)

Jupyter Notebook으로 순서대로 실행합니다.

```bash
jupyter notebook 1_model_analysis/
```

| 노트북 | 내용 |
|--------|------|
| `01_ViT_memory_analysis.ipynb` | ViT 메모리 사용량 분석, Torch Profiler 기반 |
| `02_VLM_sparsity_test.ipynb` | VLM에 2:4 적용 초기 실험 |
| `05_DEIT_speed.ipynb` | DeiT Dense vs Sparse TRT 비교 — **fc1 fusion 문제 발견** |
| `06_CaiT-M48_speed.ipynb` | 최대 규모 모델(356M) 속도 측정 |
| `08_Vit_large_speed.ipynb` | ViT-large 참고 비교 |

레이어별 dense/sparse 프로파일이 필요한 경우:

```bash
python 1_model_analysis/profile_dense_sparse.py
```

---

### Stage 2 — Sparse 변환 파이프라인 (`2_sparse_models/`)

각 모델 폴더 안의 `build.py` → `sparse.py` 순서로 실행합니다.

#### DeiT

```bash
cd 2_sparse_models/deit
python deit_build.py        # ONNX export + TRT 엔진 빌드 (dense / sparse)
python deit_dense.py        # Dense 엔진 벤치마크
python deit_sparse.py       # Sparse 엔진 벤치마크
```

#### CaiT-S24

```bash
cd 2_sparse_models/cait_s24
python cait_s24_build.py    # 2:4 sparsity 적용 레이어 목록 출력 + 엔진 빌드
python cait_s24_dense.py
python cait_s24_sparse.py
```

#### CaiT-S24 QKV Split (QKV 분리 최적화 실험)

```bash
cd 2_sparse_models/cait_s24_qkv_split
python build.py
python dense.py
python sparse.py
```

#### CaiT-S36

```bash
cd 2_sparse_models/cait_s36
python cait_build.py
python cait_dense.py
python cait_sparse.py
```

빌드가 끝나면 각 모델 폴더 아래 `*_model/` 디렉토리에 `.onnx`와 `.engine` 파일이 생성됩니다.

---

### Stage 3 — TRT 파이프라인 & 커널 실험 (`3_trt_pipeline_kernels/`)

#### 3-A. DeiT-base 일괄 빌드 + 벤치마크

small / base 두 모델, batch 16 / 256 조합을 한 번에 빌드하고 결과를 표로 출력합니다.

```bash
cd 3_trt_pipeline_kernels
python build_and_bench.py
```

DeiT-base만 따로 빌드하고 싶다면:

```bash
python deit_base_build.py   # ./deit_trt/ 아래 .trt 파일 생성
python deit_base_bench.py   # mean / p50 / p99 레이턴시 출력
```

#### 3-B. Sparse Tensor Core 단독 baseline

TRT 없이 PyTorch + cuSPARSELt로 matmul 수준 속도를 직접 측정합니다.

```bash
python benchmark_orin_fin.py
# 출력 예: [1] Dense  / [2] 2:4 Sparse 각각의 타이밍
```

#### 3-C. Fusion 차단 실험 (핵심)

1. 먼저 sparse ONNX가 있어야 합니다 (Stage 2 또는 3-A에서 생성).
2. `unfused.py` 내 경로가 실제 파일 위치와 맞는지 확인합니다:
   ```python
   onnx_path = "sunghyun/deit_onnx/deit_small_sparse.onnx"
   save_path = "sunghyun/deit_onnx/deit_small_sparse_unfused.onnx"
   ```
3. 실행:
   ```bash
   python unfused.py
   # → deit_small_sparse_unfused.onnx 생성 (fc1/MatMul 뒤 Identity 노드 삽입)
   ```
4. unfused ONNX로 TRT 엔진을 재빌드한 뒤 `deit_base_bench.py`에서 경로를 바꿔 비교합니다.

#### 3-D. Nsight Systems 프로파일링

fusion 차단 전후 커널 타임라인을 시각화합니다.

```bash
# Sparse (fusion 있음)
nsys profile --trace=cuda,nvtx -o sparse_fused   python deit_base_bench.py

# Unfused Sparse
nsys profile --trace=cuda,nvtx -o sparse_unfused python deit_base_bench.py
```

생성된 `.nsys-rep` 파일을 Nsight Systems GUI에서 열어 fc1 MatMul 커널 비중을 비교합니다.

#### 3-E. Custom Triton Kernel (MVUE24)

```bash
python sparse_kernels.py
```

dense 텐서 입력 → MVUE24 확률적 2:4 마스크 적용 → sparse 텐서 출력.  
`utils.py`의 `_MVUE24_approx`, `_sparse24`, `_soft_threshold`가 의존성입니다.

---

### 전체 실험 순서 요약

```
[분석]  1_model_analysis/05_DEIT_speed.ipynb   # fusion 문제 확인
[빌드]  2_sparse_models/cait_s24/cait_s24_build.py
[측정]  2_sparse_models/cait_s24/cait_s24_sparse.py
[빌드]  3_trt_pipeline_kernels/build_and_bench.py   # DeiT small/base 일괄
[fusion 차단]  3_trt_pipeline_kernels/unfused.py → TRT 재빌드 → 재측정
[커널]  3_trt_pipeline_kernels/sparse_kernels.py
```

---

## Environment

| 항목 | 사양 |
|------|------|
| 디바이스 | NVIDIA Jetson AGX Orin |
| CUDA | Ampere 아키텍처 (Sparse Tensor Core 지원) |
| Sparsity 백엔드 | CUTLASS (cuSPARSELt 미지원 환경) |
| 추론 런타임 | TensorRT FP16 |
| 모델 로드 | [timm](https://github.com/huggingface/pytorch-image-models) |
| 커스텀 커널 | [Triton](https://github.com/openai/triton) |

---

## Key Takeaway

> TensorRT의 자동 레이어 fusion은 성능 최적화를 위한 것이지만,  
> 2:4 Structured Sparsity 적용 시 **MatMul이 GeLU와 합쳐지면서 Sparse Tensor Core가 작동하지 않는** 역설적 상황이 발생한다.  
> ONNX graph surgery로 fusion을 차단하거나 커스텀 커널로 직접 우회하는 것이 실질적인 엣지 가속의 핵심이다.
