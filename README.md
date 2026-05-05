# FlashAttention in MiniTorch

> A from-scratch CUDA implementation of FlashAttention integrated into the MiniTorch educational deep learning framework, with empirical benchmarks on a machine translation task.

A course project for **LLM Systems (CMU, 2025)**.  
Authors: Chi-Yeh Chen · Jonathan Tang

---

## Overview

Standard self-attention has O(N²) time and memory complexity in sequence length N, which quickly becomes a bottleneck as contexts grow. This project integrates **FlashAttention** — a tiling-and-recomputation kernel that reduces peak memory complexity and improves throughput — into [MiniTorch](https://minitorch.github.io/), a minimalist Python deep learning framework.

The key contributions are:

- A hand-crafted **CUDA kernel** implementing FlashAttention's tiling, streaming, and log-sum-exp softmax
- **Python bindings** via `ctypes` to expose the kernel to MiniTorch's Python API
- **Autograd integration** by subclassing MiniTorch's `Function` for both forward and recomputation-based backward passes
- **Empirical benchmarks** sweeping sequence length, batch size, and embedding dimension on a Portuguese→English translation task

---

## Background

FlashAttention (Dao et al., 2022) avoids materializing the full N×N attention matrix in HBM by splitting Q, K, V into tiles that fit in SRAM, computing softmax on-the-fly, and writing back only the final output. In the backward pass, the QK⊤ intermediates are **recomputed on demand** rather than stored, trading a modest compute increase for a significant reduction in peak memory.

| Memory Level | Bandwidth |
|---|---|
| SRAM (on-chip) | ~19 TB/s |
| HBM (GPU DRAM) | ~1.5 TB/s |
| Main Memory (CPU DRAM) | ~12.8 GB/s |

By keeping hot data in SRAM, FlashAttention avoids the costly round-trips to HBM that dominate naive attention.

---

## Implementation

The integration proceeds in five stages:

**1. CUDA kernel design** — `.cu` files implement FlashAttention at the thread-block level. Each kernel tiles Q and K across `threadIdx.x/y`, loads M×M blocks into shared memory, and fuses scaled dot-product + softmax in one pass with numerical stability via the log-sum-exp trick.

**2. Compilation** — CUDA sources are compiled with `nvcc` into a position-independent shared library (`.so`) using `-O3 -use_fast_math`. A `Makefile` automates builds and block-size tuning.

**3. Python binding** — A `ctypes`-based wrapper loads the `.so` at runtime, marshals device pointers, handles stream synchronization, and raises clear errors on shape/type mismatches.

**4. Framework integration** — MiniTorch's `Function` base class is subclassed. `forward()` dispatches to the CUDA kernel; `backward()` invokes it in recompute mode to regenerate intermediates on-the-fly, preserving the familiar autograd interface.

**5. Validation & benchmarking** — Correctness is verified against a NumPy reference and PyTorch baseline. Performance is swept across sequence lengths, batch sizes, and embedding dimensions.

---

## Experiments

All experiments were run on a single NVIDIA GPU with CUDA 11.7, using `TensorBackend(CudaKernelOps)`. FlashAttention is toggled via the `use_fused_kernel` flag.

**Dataset:** TED HRLR Portuguese→English validation split (1,213 sentence pairs; avg. ~21 source tokens, ~19 target tokens). Tokenized with `facebook/bart-base`.

**Speedup metric:**
```
Speedup = avg_forward_time(NoFlash) / avg_forward_time(Flash)
```
Each configuration is averaged over 5 batches × 3 training epochs with 2 warm-up passes before timing.

### Parameter Sweeps

| Sweep | Variable | Fixed |
|---|---|---|
| Sequence length | `max_len` ∈ {64, 128, 256, 512} | batch=8, d_model=512, heads=8 |
| Batch size | `batch_size` ∈ {4, 8, 16, 32} | max_len=128, d_model=512, heads=8 |
| Embedding dim | `n_embd` ∈ {64, 128, 512, 1024} | max_len=128, batch=8, heads=8 |

### Key Results

**Sequence length** — FlashAttention is slightly slower at short sequences (0.98× at len=64) but reaches ~1.06× speedup at len=512. The backward pass consistently benefits more, peaking near 1.08× at len=512.

**Batch size** — Highest speedup (~1.015×) at batch size 4; gains diminish and drop below parity at batch size 32 due to kernel launch and memory-pool overhead. Backward pass peaks at ~1.07× for batch size 16.

**Embedding dimension** — Moderate dimensions (d=128) yield the highest overall speedup (~1.028×). Both very small and very large dimensions show reduced or negative gains as overhead dominates. Backward pass peaks at ~1.039× at d=512.

### Why gains are modest

The TED HRLR dataset consists of short-to-medium sequences (avg. ~21 tokens), which limits workload size. FlashAttention's tiling benefits only fully manifest when sequences are long enough to saturate GPU compute and memory bandwidth. Additionally, hardware memory constraints prevented testing beyond 512 tokens.

---

## Conclusion

A compact, educational implementation of FlashAttention was successfully integrated into MiniTorch with correct autograd support. The fine-tuned model shows that speedup scales with sequence length and is most pronounced in the backward pass. Future work should use longer-sequence datasets (e.g., WMT En-De) and larger-memory GPUs to more fully demonstrate FlashAttention's advantages beyond 2048 tokens.

---

## Individual Contributions

- **Chi-Yeh Chen** — CUDA kernel design and implementation (tiling, streaming, kernel fusion), Python bindings, and MiniTorch integration
- **Jonathan Tang** — Validation and benchmarking scripts, comprehensive parameter-sweep experiments

---

## References

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *arXiv preprint*.
- Qi, Y. et al. (2018). When and Why Are Pre-Trained Word Embeddings Useful for Neural Machine Translation? *NAACL-HLT*.
