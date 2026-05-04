# Vision-Language Model의 실질적 엣지 가속을 위한 2:4 Structured Sparsity 최적화 파이프라인 구축

---

## 1. 문제

**대상 고객:** 엣지 디바이스(NVIDIA Jetson AGX Orin 등)에서 VLM을 실시간으로 구동해야 하는 개발자 및 연구자

**Pain Point:**

- BLIP, Qwen-VL 등 최신 Vision-Language Model(VLM)은 비전+언어를 동시에 처리하여 강력하지만, 파라미터가 방대하여 엣지 GPU에서 실시간 추론이 사실상 불가능
- 기존 Unstructured Sparsity는 모델 크기를 줄여도 실제 하드웨어에서 속도 향상으로 이어지지 않음 — "줄인 것 같지만 빨라지지 않는" 문제

---

## 2. 해결 아이디어

NVIDIA Ampere 아키텍처가 하드웨어 수준에서 직접 지원하는 **2:4 Structured Sparsity**를 VLM에 적용한다.

- 연속된 4개의 가중치 중 정확히 2개를 0으로 강제 → Sparse Tensor Core가 이를 인식하여 이론상 2배 처리량 달성
- 단순한 모델 경량화가 아니라, 하드웨어가 실제로 가속하는 구조적 희소화

---

## 3. 기술 / 구현

**핵심 파이프라인 3단계:**

| 단계 | 내용 |
| --- | --- |
| ① Magnitude-based Pruning | 가중치 크기 기준으로 2:4 패턴 결정 |
| ② Sparse-aware Fine-tuning | 정확도 손실 복구를 위한 재학습 (비전 인코더 / 언어 디코더 각각 적용) |
| ③ TensorRT 엔진 빌드 | 2:4 패턴 인식 TensorRT 엔진으로 AGX Orin 디바이스 최적화 실행파일 생성 |
- 비전 인코더 / 언어 디코더 각각의 희소성 효율성 비교
- 레이어별 2:4 Sparsity 적용 민감도 분석 (어느 레이어가 정확도에 영향을 주는지)
- 희소화 전후 정확도 손실 및 Fine-tuning을 통한 복원 능력 검증

**최종 목표:**
이미지+텍스트 입력 → Sparse 연산 → 추론까지, 엣지 환경에 최적화된 End-to-End 파이프라인 완성

---

## 4. MVP / 배포

**만들어질 결과물:**

- Dense 모델 vs. 2:4 Sparse 모델의 Latency · FPS 비교
- VLM 태스크에서 희소화 전후 정확도 손실 및 복원 가능성 실증
- NVIDIA Jetson AGX Orin 위에서 구동되는 최적화된 TensorRT 추론 파이프라인

---

## 5. 차별성

| 구분 | 기존 접근 | 본 프로젝트 |
| --- | --- | --- |
| 실제 가속 여부 | 이론적 경량화에 그침 | Sparse Tensor Core 실가속 |
| 대상 모델 | 단일 태스크 경량 모델 | VLM (BLIP, Qwen-VL) |
| 희소화 범위 | 비전 단일 모달 | 비전 인코더 + 언어 디코더 각각 분석 |

→ "실제로 빨라지는 VLM 경량화" — 하드웨어-소프트웨어 공동 최적화가 핵심 차별점

---

## Related Work

2:4 Structured Sparsity를 Vision Transformer에 적용하여 실제 GPU 가속을 달성한 선행 연구들을 참고한다. Transformer 사전학습 단계에서의 희소화 적용 가능성과, Deep Neural Network에서 구조적 희소성을 학습하는 방법론까지 포괄하여 본 프로젝트의 이론적 기반으로 삼는다.

- **Boost Vision Transformer with GPU-Friendly Sparsity and Quantization (CVPR 2023)**
ViT에 2:4 Sparsity와 Quantization을 결합한 압축 기법을 제안. AGX Orin에서 Latency 최대 1.69배, Throughput 최대 2.51배 향상을 실증하여, 본 프로젝트의 비전 인코더 희소화 전략의 핵심 참고 기반이 된다. [TheCVF](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Boost_Vision_Transformer_With_GPU-Friendly_Sparsity_and_Quantization_CVPR_2023_paper.html)
- **Accelerating Transformer Pre-training with 2:4 Sparsity (ICML 2024)**
사전학습 단계부터 2:4 Sparsity를 Transformer에 적용한 최초의 End-to-End 가속 연구. gradient 기반 masked decay와 dense fine-tuning을 결합하여 dense 모델과 유사한 수렴 성능을 달성한 점을 Fine-tuning 정확도 복원 전략에 활용한다. [arXiv](https://arxiv.org/abs/2404.01847)
- **Learning Structured Sparsity in Deep Neural Networks (NeurIPS 2016)** 
필터·채널·레이어 단위 구조적 희소성 학습 방법론(SSL)을 제안. 비구조적 희소화 대비 약 2배의 실제 가속을 달성하며, 레이어별 Sparsity 적용 민감도 분석의 이론적 토대가 된다. https://arxiv.org/pdf/1608.03665