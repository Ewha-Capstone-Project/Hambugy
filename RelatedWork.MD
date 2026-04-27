# 1. Related Research & Papers

- Boost Vision Transformer with GPU-Friendly Sparsity and Quantization  
  https://arxiv.org/pdf/2305.10727

- Accelerating Transformer Pre-training with 2:4 Sparsity  
  https://arxiv.org/pdf/2404.01847

- Learning Structured Sparsity in Deep Neural Networks  
  https://arxiv.org/pdf/1608.03665


# 2. Differences from Prior Work

Compared to prior work such as *Boost Vision Transformer with GPU-Friendly Sparsity and Quantization* and *Accelerating Transformer Pre-training with 2:4 Sparsity*, which focus on leveraging structured sparsity and quantization to improve efficiency on GPU-oriented high-performance systems, and *Learning Structured Sparsity in Deep Neural Networks*, which mainly addresses general model compression through structured pruning, our work differs in that it explicitly targets edge devices with strict constraints on power, memory, and computational resources. While existing approaches are largely designed to align with GPU-friendly parallelism or general sparsity learning, our method emphasizes hardware-aware optimization tailored to edge platforms, considering not only model compression but also real-world deployment factors such as latency, energy efficiency, and limited on-device resources.
