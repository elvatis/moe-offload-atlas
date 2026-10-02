# MoE Offload Atlas

Running big mixture-of-experts models (MiMo-V2.6-Flash, GLM-5.3-Flash, Qwen3.8-Flash-Next, DeepSeek V4 Flash) on one
consumer GPU plus system RAM. The work to make this fast is spread over a dozen forks, pull requests and discussion
threads, each measured on different hardware with different tests. This repo collects it in one place and measures
it the same way.

- **[Engines](engines/)** - what each engine and fork does, its status, and the numbers its authors report.
- **[Models](models/)** - per model: which quant fits how much RAM, flags, draft/MTP files.
- **[Bench](bench/)** - one script that measures any OpenAI-compatible server the same way, and the results per
  machine.
- **[Fixes](fixes/)** - small patches we needed and sent upstream.

Rule for everything here: a number comes with the machine, model, quant and test it was measured on, and a link to
where it was measured. Numbers reported by others say so.

## Why it works at all

A MoE model only uses a few experts per token, so the experts can live in RAM while the GPU keeps everything that
every token needs. Decode speed is then roughly bounded by:

```
tokens/s  ≈  RAM bandwidth / bytes of active expert weights NOT in VRAM per token   x   tokens per pass (MTP/draft)
```

Three things move that number:

1. **Fewer bytes from RAM per token** - an expert cache in VRAM. Routing has strong *temporal* locality (the experts
   just used come back soon) but little static skew: in [#27861](https://github.com/ggml-org/llama.cpp/pull/27861) a
   fixed top-32 hot list covered ~10% of routing on held-out text, a per-layer LRU-64 ~67% and LRU-128 ~81%
   (Qwen3.8-Flash-Next, reported by the PR author).
2. **CPU and GPU working at the same time** on the same layer: cached experts on the GPU, misses on the CPU.
3. **More than one token per pass** - the model's own MTP layers or a draft model, checked by the big model.

[Strata](engines/strata.md) does all three for one model (Qwen3.8-Flash-Next) in its own engine. The llama.cpp forks
below bring 1 and 2 to every architecture llama.cpp supports.

## Expert cache in llama.cpp: the state of play

Not merged upstream as of 2026-10-02. Maintainers asked for an RFC because the first PR was too large to review.

| Work | Status (2026-10-02) | Reported result |
|---|---|---|
| [#24524](https://github.com/ggml-org/llama.cpp/pull/24524) + [RFC #24528](https://github.com/ggml-org/llama.cpp/discussions/24528) (leloch) | PR closed (too large), RFC open, 41 comments | GLM-5.1 754B IQ2_M +25%, Qwen3.5 397B +7%, 13 models +10% to +57% (4x RTX 3090, EPYC 7R13, 8-ch DDR4) |
| [#27861](https://github.com/ggml-org/llama.cpp/pull/27861) (csantiago78), GPU LRU cache, no new kernels | open | Qwen3.8-Flash-Next UD-Q4_K_XL 18.4 -> 24.2 tok/s (2x RTX 3090, dual Xeon) |
| [#26563](https://github.com/ggml-org/llama.cpp/pull/26563) / [#26824](https://github.com/ggml-org/llama.cpp/pull/26824) (miltos22), heat-map cache | closed, author paused | Qwen3.6-35B-A3B 1.7-2.1x on 8 GB VRAM |
| [neurall/llama.cpp](https://github.com/neurall/llama.cpp) `release`, built on #27861 + auto-sizing, pay-back eviction, CPU/GPU overlap, autotune | fork, active | MiMo-V2.6-Flash IQ3_XXS 4.6 -> 10.9 tok/s, GLM-5.3-Flash 3.0-bit 11.9 -> 22.4 (2x RTX 3090, Ryzen 7 3700X, 125 GB DDR4-3200) |
| Other forks named in the RFC: [GenerelSchwerz](https://github.com/GenerelSchwerz/llama.cpp), [borisk1/llama.cpp-fusion](https://github.com/borisk1/llama.cpp-fusion), [Atomic-Germ/Guanaco](https://github.com/Atomic-Germ/Guanaco), TheTom, thecodacus | forks | see the RFC thread; Volunteer-1 compared several on one machine there |

Gains depend on how much of the model fits in VRAM: the neurall README reports the largest gains at 33-55% of the
model in VRAM (1.7-2.4x), fading above that and 1.0x once the model fits. On a 2080 Ti 11 GB (PCIe 3.0) with
Qwen3.6-35B-A3B Q6_K, a user reported +16% in [#27861](https://github.com/ggml-org/llama.cpp/pull/27861).

## Other engines

| Engine | What it brings | Models |
|---|---|---|
| [Strata](engines/strata.md) | Own engine: adaptive VRAM expert cache, CPU+GPU overlap, MTP, n-gram table on SSD | Qwen3.8-Flash-Next only ([other architectures not planned](https://github.com/Niko1221/Strata/issues/310)) |
| [ik_llama.cpp](engines/ik_llama.md) | Faster CPU MoE kernels; no expert cache found; [#2259](https://github.com/ikawrakow/ik_llama.cpp/pull/2259) placement solver open | broad, incl. MiMo ([#1723](https://github.com/ikawrakow/ik_llama.cpp/pull/1723)) |
| [FreeToken](https://github.com/FlashML-org/FreeToken) | GPU-offload, CPU and hybrid MoE modes, pick by benchmark | see its docs |

## Our measurements

[RTX 2080 Ti 11 GB + Threadripper 3960X, 128 GB DDR4-3200](bench/results/2026-10-02-rtx2080ti-tr3960x/):

| Engine | Model | Decode tok/s | Prefill 32K tok/s |
|---|---|---:|---:|
| Strata 0.1.34 | Qwen3.8-Flash-Next IQ3_S | 56.0 | 783 |
| llama.cpp v0.5.0 + #27044 | MiMo-V2.6-Flash IQ2_M | 15.5 | 208 |
| neurall cache fork (auto) | MiMo-V2.6-Flash IQ2_M | 6.6 | 205 |

[RX 9070 XT 16 GB + the same CPU and RAM, Windows 11](bench/results/2026-10-02-rx9070xt-tr3960x/):

| Engine | Model | Decode tok/s | Prefill 32K tok/s |
|---|---|---:|---:|
| Strata 0.1.35 (Windows AMD engine) | Qwen3.8-Flash-Next IQ3_S | 45.2 | 600 |
| llama.cpp b11344 (Vulkan) | MiMo-V2.6-Flash IQ2_M | 8.5 | 196 |

Stock v0.5.0 crashes on MiMo prefill at `-ub 2048`; [#27044](https://github.com/ggml-org/llama.cpp/pull/27044) fixes it.

## License

MIT for the scripts and docs in this repo. Linked projects and model files keep their own licenses.
