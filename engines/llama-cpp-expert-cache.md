# llama.cpp and its expert-cache forks

Stock llama.cpp offloads MoE experts with `--cpu-moe`, `--n-cpu-moe N` or `-ot <regex>=CPU`. Which experts stay
on the GPU is fixed at load time, so decode on the offloaded layers runs at RAM bandwidth while the GPU waits.

## Upstream threads

- [#24524](https://github.com/ggml-org/llama.cpp/pull/24524) (closed 2026-06-12): CUDA-side adaptive cache; the
  `MUL_MAT_ID` node stays on the CPU, thread 0 dispatches the hit rows to the GPU while the other threads compute the
  misses. Closed as too large; maintainer asked for an RFC.
- [RFC discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528): design, prior attempts
  (#20757, #21609, #21614, #21620, #23170) and many community measurements. The prior attempts moved `MUL_MAT_ID` to
  the GPU, which put every miss on the critical path as a synchronous PCIe copy.
- [#27861](https://github.com/ggml-org/llama.cpp/pull/27861) (open): per-layer LRU cache, `--moe-expert-cache N`,
  `--moe-expert-cache-inserts`; no new kernels (a second `mul_mat_id` chain over cache tensors plus a CPU path that
  skips cached ids); async throttled uploads; decode-only (`n_tokens == 1`), so MTP/speculation bypasses it.
- [#26563](https://github.com/ggml-org/llama.cpp/pull/26563), [#26824](https://github.com/ggml-org/llama.cpp/pull/26824)
  (closed): heat-map cache with many extras (sidecar heat-map persistence, mmap pinning, multi-GPU priority).
- Related open issues: [#27584](https://github.com/ggml-org/llama.cpp/issues/27584) (bandwidth-adaptive CPU-GPU
  co-execution), [#25859](https://github.com/ggml-org/llama.cpp/issues/25859) (offloaded-MoE prefill idles the GPU on
  serial H2D copies), [#26448](https://github.com/ggml-org/llama.cpp/issues/26448) (experts from host RAM via PCIe DMA).

## neurall/llama.cpp (`release` branch)

[neurall/llama.cpp](https://github.com/neurall/llama.cpp): built on #27861, adds VRAM-filling auto-sizing, pay-back
eviction (swap only if it pays for its upload), CPU/GPU overlap within a layer, prefill cache warming, autotune.

- `--moe-expert-cache N` (`-1` = size from free VRAM, `0` = off), `--moe KEY=VAL,...`, `-at off` disables autotune.
- Reported caveats in its README: prompt processing can be 0.4-0.7x of stock on long prompts; the first run is slower
  than stock while the cache warms.
- **Build on CUDA 12.0 / pre-Ampere (sm_75) fails** in `mmvf.cu` (`__bfloat1622float2` undefined). Fix:
  [fixes/neurall-mmvf-bf16-cuda12-sm75.patch](../fixes/neurall-mmvf-bf16-cuda12-sm75.patch), using upstream's guarded
  `ggml_cuda_cast<float2>`.

## Measured here

[2026-10-02, RTX 2080 Ti + Threadripper 3960X](../bench/results/2026-10-02-rtx2080ti-tr3960x/).
