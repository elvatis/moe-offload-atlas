# Strata

- Repo: [Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT). Own C++/CUDA/HIP engine; borrows ggml's
  dequantization kernels, not llama.cpp's runtime.
- Model: Qwen3.8-Flash-Next only (and same-architecture variants: Coder, Swift 1.5, Unsloth UD-Q4_K_XL).
  The model geometry is fixed in `include/strata/core/layout.hpp` (48 layers, 512 experts per layer, top-10 router
  kernel `router_top10`). The maintainer: "Other MoE architectures (DeepSeek, GLM) aren't planned: the kernels and the
  pack format are built around this model" ([#310](https://github.com/Niko1221/Strata/issues/310)).

## What makes it fast

From Strata's [HOW_IT_WORKS.md](https://github.com/Niko1221/Strata/blob/main/docs/HOW_IT_WORKS.md):

- GPU: attention and DeltaNet mixers, routers, shared experts, head, MTP layer, hot KV, and an **adaptive expert cache**
  filling the rest of VRAM.
- RAM: all 24,576 experts, pinned; the CPU computes cache misses **in place, concurrently** with the GPU.
- SSD: the 28.8 GB n-gram table, a few rows per token.
- MTP drafts up to 3 tokens; 2.4-3.2 tokens per pass; prompt lookup for repeated text.
- Prefill in chunks up to 8,192 tokens while the next layer's experts stream over PCIe.

## What a port to another architecture would need

New layer kernels (attention type, mixers), a router for that top-k and expert count, kernels for the quant types the
target GGUFs use (e.g. MXFP4, Q3_K, Q6_K are not in Strata's set), the model's MTP, and a pack converter. The
expert storage, CPU expert compute, cache and server are the reusable parts.

## Measured here

[2026-10-02, RTX 2080 Ti + Threadripper 3960X](../bench/results/2026-10-02-rtx2080ti-tr3960x/).
