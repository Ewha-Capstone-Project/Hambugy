"""
DeiT-base 모델 생성 + 2:4 Sparsity 적용 가이드

흐름:
  1. timm으로 DeiT-base 모델 로드
  2. Linear 레이어 weight에 magnitude 기반 2:4 pruning 적용
  3. to_sparse_semi_structured 로 CUDA Sparse Tensor 변환
  4. Dense vs Sparse 추론 속도 비교
"""

import copy

import torch
import torch.nn as nn
from torch.sparse import to_sparse_semi_structured
import timm

# cuSPARSELt 미지원 환경(Jetson Orin 등)에서는 CUTLASS 백엔드 사용
torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = True

# ─── 설정 ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda")
DTYPE  = torch.float16
BATCH  = 16

# ─── 1. 모델 로드 ─────────────────────────────────────────────────────────────
def load_deit():
    model = timm.create_model("deit_base_patch16_224", pretrained=False)
    model = model.to(DEVICE).to(DTYPE).eval()
    return model


# ─── 2. 2:4 Magnitude Pruning ─────────────────────────────────────────────────
def apply_24_pruning(weight: torch.Tensor) -> torch.Tensor:
    """
    weight를 [N, 4] 블록으로 reshape 후,
    절댓값 기준 하위 2개를 0으로 만들어 2:4 패턴 강제 적용.
    """
    if weight.shape[-1] % 4 != 0:
        return weight

    shape = weight.shape
    w = weight.view(-1, 4)

    # 절댓값 정렬 후 작은 2개 인덱스를 0으로 마스킹
    zero_idx = torch.argsort(torch.abs(w), dim=1)[:, :2]
    mask = torch.ones_like(w)
    mask.scatter_(1, zero_idx, 0)

    return (w * mask).view(shape)


def sparsify_model(model: nn.Module) -> nn.Module:
    """모든 Linear 레이어의 weight에 2:4 pruning 적용."""
    sparse_model = copy.deepcopy(model)
    pruned = 0
    for name, module in sparse_model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data = apply_24_pruning(module.weight.data)
            pruned += 1
    print(f"  2:4 pruning 적용 완료: Linear 레이어 {pruned}개")
    return sparse_model


# ─── 3. Semi-Structured Sparse 변환 ─────────────────────────────────────────
def convert_to_sparse_semi_structured(model: nn.Module) -> nn.Module:
    """
    2:4 패턴이 적용된 weight를 to_sparse_semi_structured로 변환.
    CUDA가 compressed format으로 저장해 Tensor Core 가속을 활성화.
    """
    converted = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            try:
                module.weight = nn.Parameter(
                    to_sparse_semi_structured(module.weight.data)
                )
                converted += 1
            except Exception as e:
                print(f"  [skip] {name}: {e}")
    print(f"  semi-structured 변환 완료: {converted}개")
    return model


# ─── 4. 벤치마크 ─────────────────────────────────────────────────────────────
def benchmark(model: nn.Module, label: str, n_warmup=50, n_iter=200):
    dummy = torch.randn(BATCH, 3, 224, 224, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for _ in range(n_iter):
            model(dummy)
    end.record()
    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / n_iter
    print(f"  [{label}] 평균 추론 시간: {avg_ms:.3f} ms  (batch={BATCH})")
    return avg_ms


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  DeiT-base  2:4 Sparsity 가이드")
    print("=" * 55)

    # Step 1: 모델 로드
    print("\n[Step 1] DeiT-base 모델 로드 중...")
    dense_model = load_deit()
    total_params = sum(p.numel() for p in dense_model.parameters())
    print(f"  파라미터 수: {total_params / 1e6:.1f}M")

    # Step 2: 2:4 Pruning
    print("\n[Step 2] 2:4 Magnitude Pruning 적용 중...")
    sparse_model = sparsify_model(dense_model)

    # sparsity 비율 확인
    with torch.no_grad():
        total_w, zero_w = 0, 0
        for m in sparse_model.modules():
            if isinstance(m, nn.Linear):
                total_w += m.weight.numel()
                zero_w  += (m.weight == 0).sum().item()
    print(f"  실제 sparsity: {100 * zero_w / total_w:.1f}%  (목표: 50%)")

    # Step 3: Semi-Structured 변환
    print("\n[Step 3] to_sparse_semi_structured 변환 중...")
    sparse_model = convert_to_sparse_semi_structured(sparse_model)

    # Step 4: 벤치마크
    print(f"\n[Step 4] 추론 속도 비교  (batch={BATCH}, dtype=fp16)")
    t_dense  = benchmark(dense_model,  "Dense ")
    t_sparse = benchmark(sparse_model, "Sparse")

    speedup = t_dense / t_sparse
    print(f"\n  → Speedup: {speedup:.2f}x")
    print("=" * 55)


if __name__ == "__main__":
    main()
