import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import Dict
from dataclasses import dataclass
import warnings


@dataclass
class BenchmarkConfig:
    num_query_tokens: int = 32
    qformer_hidden_size: int = 768
    qformer_num_layers: int = 12
    lm_hidden_size: int = 2048
    lm_intermediate_size: int = 5120
    lm_num_layers: int = 24
    batch_size: int = 8
    image_seq_length: int = 257
    text_seq_length: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================
# WANDA + 2:4 Sparsity
# =============================

def wanda_importance_score(weight, activation_stats=None):
    if activation_stats is None:
        return weight.abs()
    return weight.abs() * activation_stats.unsqueeze(0)


def apply_wanda_24_pruning(weight, activation_stats=None):
    importance = wanda_importance_score(weight, activation_stats)
    out_features, in_features = weight.shape
    pruned_weight = weight.clone()

    for i in range(out_features):
        for j in range(0, in_features - 3, 4):
            group = importance[i, j:j+4]
            _, idx = torch.topk(group, k=2, largest=False)
            pruned_weight[i, j + idx] = 0

    return pruned_weight


class WandaPrunedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        w = torch.randn(out_features, in_features) * 0.02
        act = torch.rand(in_features) + 0.5
        sparse_w = apply_wanda_24_pruning(w, act)
        self.register_buffer("weight", sparse_w)

    def forward(self, x):
        return F.linear(x, self.weight)


class HardwareAcceleratedSparseLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        w = torch.randn(out_features, in_features) * 0.02
        act = torch.rand(in_features) + 0.5
        sparse_w = apply_wanda_24_pruning(w, act)

        try:
            from torch.sparse import to_sparse_semi_structured
            cuda_w = sparse_w.to("cuda")
            self.weight = nn.Parameter(to_sparse_semi_structured(cuda_w), requires_grad=False)
            self.sparse_format = "semi_structured"
        except:
            self.register_buffer("weight", sparse_w)
            self.sparse_format = "dense"

    def forward(self, x):
        return F.linear(x, self.weight)


# =============================
# Q-Former Layer (SparseLoRA 제거)
# =============================

class BLIP2QFormerLayer(nn.Module):
    def __init__(self, hidden_size=768, use_sparse=False, use_hw_accel=False):
        super().__init__()
        inter = hidden_size * 4

        if use_hw_accel:
            Linear = HardwareAcceleratedSparseLinear
        elif use_sparse:
            Linear = WandaPrunedLinear
        else:
            Linear = nn.Linear

        self.query = Linear(hidden_size, hidden_size)
        self.key = Linear(hidden_size, hidden_size)
        self.value = Linear(hidden_size, hidden_size)
        self.fc1 = Linear(hidden_size, inter)
        self.fc2 = Linear(inter, hidden_size)

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        r = x
        q, k, v = self.query(x), self.key(x), self.value(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.size(-1))
        attn = F.softmax(attn, dim=-1)
        x = self.ln1(r + torch.matmul(attn, v))

        r = x
        x = self.fc2(F.gelu(self.fc1(x)))
        return self.ln2(r + x)


# =============================
# Benchmark
# =============================

def benchmark(layer, x, iters=200):
    x = x.to(next(layer.parameters()).device)
    layer.eval()

    for _ in range(50):
        _ = layer(x)

    torch.cuda.synchronize()
    times = []

    for _ in range(iters):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        _ = layer(x)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    return np.mean(times)


# =============================
# Main
# =============================

def main():
    cfg = BenchmarkConfig()

    x = torch.randn(cfg.batch_size, cfg.num_query_tokens, cfg.qformer_hidden_size).to(cfg.device)

    models = {
        "Dense1": BLIP2QFormerLayer(use_sparse=False, use_hw_accel=False),
        "Dense2": BLIP2QFormerLayer(use_sparse=False, use_hw_accel=True),
        "Sparse": BLIP2QFormerLayer(use_sparse=True, use_hw_accel=False),
        "HW Sparse": BLIP2QFormerLayer(use_sparse=False, use_hw_accel=True),
    }

    for name, m in models.items():
        m = m.to(cfg.device)
        t = benchmark(m, x)
        print(f"{name}: {t:.4f} ms")


if __name__ == "__main__":
    main()
