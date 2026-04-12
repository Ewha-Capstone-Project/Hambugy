# Hambugy
| Name     | GitHub                                     |
| -------- | ------------------------------------------ |
| 신성현 | [@cindyshin2211](https://github.com/cindyshin2211) |
| 장수연 | [@ally010314](https://github.com/ally010314) |
| 송영채 | [@syc031023](https://github.com/syc031023) |

## 1. Problem Statement

최근 Vision Foundation Model(VFM)은 강력한 성능을 보여주지만, 엣지 환경에 배포하기에는 다음과 같은 치명적인 한계가 있다.

* **높은 연산 복잡도:** ViT 기반 모델은 파라미터 수가 방대하여 엣지 GPU에서도 실시간 추론이 어렵다.
* **메모리 대역폭 병목:** 대규모 가중치를 메모리에서 불러오는 과정에서 에너지가 소모되고 지연 시간이 발생한다.
* **하드웨어 최적화 미비:** 일반적인 Unstructured Sparsity(무작위 희소화)는 가중치를 줄여도 실제 하드웨어(GPU)에서 연산 가속으로 이어지지 않는 경우가 많다.

따라서 엣지 디바이스(NVIDIA Jetson AGX Orin 등)의 **Sparse Tensor Core**를 활용하여, 하드웨어 수준에서 성능을 끌어올릴 수 있는 구조적 최적화가 필요하다.

---

## 2. Proposed Idea

본 프로젝트에서는 NVIDIA Ampere 및 차세대 아키텍처에서 지원하는 **2:4 Structured Sparsity** 기법을 VFM에 적용하여 엣지 디바이스에서의 추론 성능을 극대화하는 파이프라인을 구축한다.

**2:4 Structured Sparsity 방식:**
* **패턴 제약:** 연속된 4개의 가중치 블록 중 정확히 2개를 0으로 설정한다.
* **하드웨어 가속:** Sparse Tensor Core가 0이 아닌 값만 선택적으로 연산하여 **이론상 2배의 처리량(Throughput)**을 달성한다.

**핵심 접근 방식:**
1.  **VFM Fine-tuning:** DINOv2나 CLIP 같은 모델에 `Sparse-refined` 학습을 적용하여 정확도 손실을 최소화한다.
2.  **TensorRT 가속:** 2:4 패턴을 인식하는 TensorRT 엔진을 빌드하여 Orin 디바이스에 최적화된 실행 파일을 생성한다.
3.  **End-to-End 파이프라인:** 이미지 입력부터 Sparse 연산을 거친 최종 추론까지의 전 과정을 자동화한다.

---

## 3. Key Idea

핵심 아이디어는 다음과 같다.

* **Hard-constrained Sparsity:** 하드웨어가 요구하는 2:4 구조를 강제하여 '무늬만 경량화'가 아닌 **실질적인 FPS 향상**을 꾀함.
* **Magnitude-based Pruning & Retraining:** 가중치 크기 기반으로 2:4 패턴을 정하고, 손실된 정확도를 복구하기 위한 미세 조정(Fine-tuning) 전략 수립.
* **VFM 확장성 검증:** 단일 태스크 모델을 넘어, 멀티모달 모델(CLIP)의 텍스트/이미지 엔코더 각각에 대한 희소성 효율성을 분석.

---

## 4. Related Research Area

본 프로젝트는 다음 연구 분야의 교차 영역에 해당한다.

* **Model Compression & Acceleration:** Sparsity, Quantization, Knowledge Distillation.
* **Edge AI & Hardware-Aware Optimization:** 특정 하드웨어 아키텍처(Ampere/Orin) 최적화.
* **Vision Foundation Models:** ViT, DINOv2, CLIP 등 대형 비전 모델 활용.

---

## 5. Expected Outcome

본 프로젝트를 통해 다음을 분석하고 구축할 수 있다.

* **성능 지표 확보:** Dense 모델 대비 2:4 Sparse 모델의 **Latency(지연 시간)** 및 **Throughput(초당 프레임 수)** 비교 데이터 도출.
* **정확도 유지 확인:** 모델 크기를 절반으로 줄이면서도 주요 비전 태스크(Classification, Detection)에서의 성능 저하가 미미함을 입증.
* **실용적 배포 가이드:** 엣지 환경에서 VFM을 실시간으로 구동하기 위한 **최적의 Sparsity-Aware 추론 파이프라인** 제시.
