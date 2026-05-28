# HAMBUGY Project - CLAUDE.md

## 프로젝트 개요
- **프로젝트명**: HAMBUGY (Edge-Cloud Split Inference 및 2:4 Structured Sparsity 기반 Vision 가속 파이프라인)
- **목표**: NVIDIA Jetson AGX Orin 환경에서 비전 모델의 최적화 및 고속 추론 가속화

## 개발 및 빌드 환경 (Environment Specs)
- **JetPack**: 6.2.1 (최신 버전 계열 JetPack 6.x)
- **CUDA**: 12.6 (드라이버 및 컴파일러 모두 12.6 대응)
- **NVIDIA Driver**: 540.4.0 (L4T 기반 드라이버)
- **TensorRT**: 10.x 계열 (JetPack 6.2.1 기본 탑재 버전 및 CUDA 12.6 호환 버전)

## 평가 기준 및 제약 조건 (Evaluation Criteria & Constraints)

### 1. CUDA / TensorRT 버전 검증
- 모든 빌드 및 커널 실행 시 CUDA 12.6 환경 툴체인을 엄격히 준수해야 합니다.
- TensorRT 10.x의 최적화 기능(특히 2:4 Structured Sparsity 가속 기능)을 활용하도록 파이프라인이 구성되어야 합니다.

### 2. 입력 Shape (Input Shape)
- **기본 입력 포맷**: NCHW (Batch Size, Channels, Height, Width)
- **기본 해상도**: `1 x 3 x 640 x 640` (또는 모델 분석 단계에서 정의된 고정/동적 Shape)
- 파이프라인 최적화 및 커텀 커널 실행 시 정의된 입력 Shape 조건을 임의로 변경하지 마십시오.

### 3. 프로젝트 실행 순서 (Execution Order)
코드 수정 및 파이프라인 테스트는 반드시 아래 순서대로 진행 및 검증되어야 합니다.
1. **`src/1_model_analysis/`**: 원본 비전 모델 분석, 구조 파악 및 레이어별 프로파일링 수행
2. **`src/2_sparse_models/`**: 분석된 모델에 2:4 Structured Sparsity 알고리즘 적용 및 최적화된 희소(Sparse) 모델 내보내기 (ONNX 포맷 등)
3. **`src/3_trt_pipeline_kernels/`**: 내보낸 모델을 기반으로 TensorRT 엔진 빌드, 커스텀 가속 파이프라인 구성 및 Jetson AGX Orin 전용 하드웨어 커널 실행

### 4. 수정 금지 범위 (Non-modifiable Scope)
- **핵심 알고리즘 엔진**: `src/1_model_analysis/` 내부에 구현된 기본 분석 모듈 및 하드웨어 벤치마크 평가 핵심 기준 코드는 수정할 수 없습니다.
- **가속 가이드라인**: `src/3_trt_pipeline_kernels/` 내부의 저수준 TensorRT 파이프라인 초기화 로직 및 가속 레이어 바인딩 구조는 평가의 일관성을 위해 수정을 금지합니다.

## 주요 명령 가이드 (Commands)
- **모델 분석 실행**: `python3 src/1_model_analysis/analyze.py`
- **희소 최적화 적용**: `python3 src/2_sparse_models/apply_sparsity.py`
- **TensorRT 엔진 빌드 및 가속 테스트**: `cd src/3_trt_pipeline_kernels && make && ./trt_inference_profile`
