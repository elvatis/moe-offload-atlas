# GLM-5.3-Flash (Z.ai)

320B total / 18B active, 288 experts top-8, 45 layers (34 linear attention, 11 sparse DSA attention), an MTP block,
1M context, images. Released in FP8 (322 GB). Model card: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).
llama.cpp architecture `glm5next`, merged upstream (b11516 has it).

## GGUFs that fit 128 GB of RAM

| Quant | Size | Notes |
|---|---|---|
| [unsloth UD-IQ2_XXS](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) | 102 GB | measured here: experts 88.6 GiB in 43 layers (2.06 GiB each), the rest 6.2 GiB |
| unsloth UD-Q2_K_XL | 109 GB | same repo |
| GSQ-RCO 3.0 bit | 118 GB | very tight next to the OS |

Larger quants (UD-Q3_K_XL 148 GB, UD-Q4_K_XL 200 GB) need two machines over llama.cpp RPC, which has an open issue
with this model ([#28360](https://github.com/ggml-org/llama.cpp/issues/28360)).

## Flags

`-fa on -ngl 99 --n-cpu-moe 46` (all experts in RAM on an 11 GB card), 24 threads (unlike smaller MoE models, more
threads help: 18B active at 2 bit costs the CPU more decoding work than RAM bandwidth). On 11 GB:

- `-ub 2048` reads prompts at 161-165 tok/s, `-ub 512` at 59; writing is about the same (11.5-14 tok/s, within 7 %).
- Load from a warm page cache: a run right after another big model writes up to 20 % slower while experts come back
  from disk.
- The built-in MTP (`--spec-type draft-mtp`) only fits at `-ub 512` and is slower here (10.7-11.0 tok/s): checking
  4 tokens needs the experts of 4 tokens, all read from RAM.
- `--moe-cache-mib` makes it 4x slower on 11 GB: see [the expert-cache page](../engines/llama-cpp-expert-cache.md).

## Known issues

- [#28144](https://github.com/ggml-org/llama.cpp/issues/28144): `SOFT_MAX failed` on Turing after 10 hours, on a
  pre-merge build. Not reproduced here in 4 hours on b11516 with the report's flags (138 requests, context up to
  218K, [comment](https://github.com/ggml-org/llama.cpp/issues/28144#issuecomment-6106433932)).
- [#30093](https://github.com/ggml-org/llama.cpp/issues/30093): the DFlash2 drafter aborts with glm5next.

## Elsewhere

sybil-solutions/glm53-flash-offload reports 20-28 tok/s with ExLlamaV3 on an RTX 3090 and an 8-channel EPYC
(reported, not measured here; ExLlamaV3 needs Ampere or newer).

## Measured here

[2026-10-11, RTX 2080 Ti + Threadripper 3960X](../bench/results/2026-10-11-rtx2080ti-tr3960x/).
