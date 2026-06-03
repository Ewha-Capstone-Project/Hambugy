# ViT-Large Dense vs Sparse — Self Demo 가이드

## 개요

이 데모는 두 가지 방법으로 재현할 수 있습니다.

1. **서버 접속** — Jetson AGX Orin에 직접 접속하여 속도 비교 및 커널 분석 데모 실행
2. **로컬 확인** — 서버 접속 없이 저장된 프로파일 파일로 Nsight Systems GUI에서 커널 타임라인 확인

---

## Part 1. 서버에 접속하여 데모 실행

### 요구사항

- SSH 접속 가능한 터미널
- 서버 접속 정보 (IP, 계정)

### 접속 및 실행

서버 접속 정보(IP, 계정, 포트)는 보안상 이 파일에 포함하지 않았습니다. 필요 시 팀에 문의해주세요.

**터미널 A** — 서버에 접속하여 Streamlit 먼저 실행:

```bash
ssh user@서버IP -p 포트번호
source ~/sunghyun/venv/bin/activate
cd ~/yc/live_demo
streamlit run demo_vit.py
```

**터미널 B** — 새 터미널을 열어 포트 포워딩 접속:

```bash
ssh -L 8501:localhost:8501 user@서버IP -p 포트번호
```

로컬 브라우저에서 `http://localhost:8501` 접속 후 **데모 실행** 버튼 클릭

> Streamlit이 실행된 이후에 포트 포워딩을 연결해야 합니다.

### 실행 흐름

1. ViT-Large 모델 로드 및 2:4 Sparsity 자동 적용
2. TensorRT FP16 엔진 파일이 있으면 빌드 스킵, 없으면 자동 빌드 (10~15분 소요)
3. Warmup 50회 / 측정 200회 기준으로 벤치마크 자동 실행
4. Dense Mean, Sparse Mean, Speedup 결과 확인
5. 사이드바의 **캐시된 nsys 결과 사용** 체크박스가 ON이면 저장된 프로파일을 바로 읽어 커널 분석 결과 표시

---

## Part 2. 서버 접속 없이 커널 타임라인 확인

서버에 접속할 수 없는 경우, GitHub 레포지토리에 업로드된 `.nsys-rep` 파일을
다운받아 로컬 Nsight Systems GUI에서 직접 열 수 있습니다.

### 1. Nsight Systems 설치

NVIDIA 공식 페이지에서 운영체제에 맞는 버전 설치:
https://developer.nvidia.com/nsight-systems

### 2. 프로파일 파일 다운로드

GitHub 레포지토리 `src/4_live_demo/nsys_profiles/` 폴더에서 다운로드:

- `dense.nsys-rep` — Dense 엔진 프로파일
- `sparse.nsys-rep` — Sparse 엔진 프로파일

### 3. GUI에서 열기

Nsight Systems 실행 → `File` → `Open` → 다운로드한 `.nsys-rep` 파일 선택

Dense와 Sparse를 각각 열어 커널 타임라인을 비교하면
`fc1/MatMul + GELU + fc1/Add`가 하나의 fusion 노드(`__myl_Fc_myl2_*`)로 합쳐져
내부 GEMM이 Sparse GEMM으로 교체되지 않는 것을 직접 확인할 수 있습니다.

> `.nsys-rep` 파일은 Jetson AGX Orin에서 캡처한 결과이며,
> GUI에서 열람하는 것은 어떤 운영체제에서도 가능합니다.