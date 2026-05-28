import torch
import torch.utils.benchmark as benchmark
from torch.nn import functional as F

# PyTorch의 cuSPARSELt 백엔드 호출 API (Ampere 아키텍처 지원)
from torch.sparse import to_sparse_semi_structured

if __name__ == "__main__":
    print("Orin 보드 추론 가속 검증: 일반 행렬 곱(Dense) vs 2:4 희소 행렬 곱(spMM)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 텐서 세팅: 트랜스포머의 FFN 레이어를 모사하기 위해 충분히 큰 사이즈로 설정
    # M(배치x시퀀스), K(입력 차원), N(출력 차원)
    m, k, n = 2048, 2048, 2048 

    # 1. 일반 입력 데이터와 가중치 (비교군)
    input_tensor = torch.randn(m, k, device=device, dtype=torch.float16)
    weight_dense = torch.randn(n, k, device=device, dtype=torch.float16)

    # 2. 2:4 형태로 깎인 가중치 임시 생성 
    # (주의: 실제 분할 추론 시나리오에서는 이 과정이 모델 배포 전 '오프라인 클라우드'에서 이미 끝난 상태임)
    weight_24 = weight_dense.clone()
    reshaped = weight_24.abs().view(-1, 4)
    _, indices = torch.topk(reshaped, 2, dim=-1, largest=False)
    mask = torch.ones_like(reshaped, dtype=torch.bool)
    mask.scatter_(1, indices, False)
    weight_24 = weight_24 * mask.view(n, k)

    # 3. cuSPARSELt 백엔드를 활용하기 위해 2:4 가중치를 압축(Compression)
    # TRT도 추론 시 메모리 대역폭을 줄이기 위해 내부적으로 이 압축된 포맷을 사용함
    weight_sparse_compressed = to_sparse_semi_structured(weight_24)

    print(f"테스트 행렬 크기 - 입력: {m}x{k}, 가중치: {n}x{k}\n")

    # [비교군] 기존 최적화된 Dense 행렬 곱 (일반 Tensor Core 연산기 사용)
    t0 = benchmark.Timer(
        stmt='F.linear(input_tensor, weight_dense)',
        setup='from torch.nn import functional as F',
        globals={'input_tensor': input_tensor, 'weight_dense': weight_dense}
    )
    print("[1] 일반 Dense 행렬 곱 (cuBLAS / Tensor Core):")
    print(t0.timeit(100))

    # [실험군] 2:4 희소 행렬 곱 (Sparse Tensor Core 전용 연산기 사용)
    t1 = benchmark.Timer(
        stmt='F.linear(input_tensor, weight_sparse_compressed)',
        setup='from torch.nn import functional as F',
        globals={'input_tensor': input_tensor, 'weight_sparse_compressed': weight_sparse_compressed}
    )
    print("[2] 2:4 Sparse 행렬 곱 (cuSPARSELt / Sparse Tensor Core):")
    print(t1.timeit(100))