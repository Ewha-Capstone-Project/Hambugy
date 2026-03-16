# Hambugy

| Name   | GitHub                                     |
| ------ | ------------------------------------------ |
| 신성현 | [@cindyshin2211](https://github.com/cindyshin2211) |
| 장수연 | [@ally010314](https://github.com/ally010314) |
| 송영채 | [@syc031023](https://github.com/syc031023) |

---

## Ideation: Edge–Cloud Split Inference for Vision Models

### 1. Problem Statement

최근 많은 **Computer Vision 서비스**가 스마트 카메라, 드론, 모바일 기기와 같은 **엣지 장치(Edge devices)** 에서 사용되고 있다.

하지만 이러한 장치들은 다음과 같은 제약을 가진다.
* 제한된 **연산 능력 (compute)**
* 낮은 **전력 효율**
* 제한된 **메모리**

또한 모든 데이터를 **클라우드 서버로 직접 전송**하는 방식에는 다음과 같은 문제가 있다.
* 높은 **네트워크 대역폭 사용**
* 증가하는 **지연(latency)**
* **프라이버시 문제**

따라서 **엣지 장치와 클라우드 서버 간의 효율적인 추론 구조**가 필요하다.

---

### 2. Proposed Idea

본 프로젝트에서는 **Edge–Cloud Split Inference** 구조를 활용한 비전 모델 추론 방식을 탐구한다.

Edge–Cloud Split Inference는 하나의 딥러닝 모델을 **엣지 장치와 클라우드 서버 사이에서 분할하여 실행하는 방식**이다.

**기존 방식:**
```text
Edge device
↓
Image 전송
↓
Cloud server
↓
전체 모델 실행
```

**제안 방식:**
```
Edge device
↓
초기 CNN layer 실행
↓
Feature map 생성
↓
Feature 전송
↓
Cloud server
↓
나머지 layer 실행
↓
Prediction
```

---

### 3. Key Idea

핵심 아이디어는 다음과 같다.

* 딥러닝 모델을 중간 layer에서 분할
* Edge에서 early layers 실행
* Cloud에서 remaining layers 실행
* feature map만 네트워크로 전송

이를 통해 다음과 같은 장점을 기대할 수 있다.

* 네트워크 bandwidth 감소
* latency 개선
* Edge device의 연산 부담 감소

---

### 4. Related Research Area

본 프로젝트는 다음 연구 분야의 교차 영역에 해당한다.
* Computer Vision
* Edge Computing
* Distributed Systems

---

### 5. Expected Outcome

본 프로젝트를 통해 다음을 분석할 수 있다.
* Edge–Cloud 분할 위치에 따른 latency 변화
* Feature 전송 시 bandwidth 사용량
* Edge vs Cloud 연산 분배 효과

이를 통해 효율적인 Split Inference 구조를 탐구하는 것을 목표로 한다.
