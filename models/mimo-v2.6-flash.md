# MiMo-V2.6-Flash (Xiaomi)

309B total / 15B active, 256 experts top-8 per layer, 48 layers, sliding-window + full attention, 3 MTP layers,
1M context, MIT. Experts ship natively in MXFP4. Model card:
[XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL).

## GGUFs that fit 128 GB of RAM

| Quant | Size | Notes |
|---|---|---|
| [fresherbz IQ2_XXS](https://huggingface.co/fresherbz/MiMo-V2.6-Flash-MOPD-IQ2_XXS-GGUF) (MOPD checkpoint) | 76.1 GiB | text only; card reports 10/14 agentic tasks right and 2 runaway generations |
| [Baekpica mixed](https://huggingface.co/Baekpica/MiMo-V2.6-Flash-RL-Mixed-Quant-GGUF) | 86.7 GiB | IQ2_XXS/IQ2_XS experts, Q8_0 shared + MTP, BF16 media; tested on its own runtime |
| [YanissAmz IQ2_M-class](https://huggingface.co/YanissAmz/MiMo-V2.6-Flash-RL-GGUF) | 100.4 GiB | per-tensor mix; KLD 0.164 vs MXFP4, 88.2% same top-1; stock llama.cpp; MTP layers included but not used by stock llama.cpp |

Larger: [ggml-org Q2_K 126 GB, MXFP4 167 GB](https://huggingface.co/ggml-org/MiMo-V2.6-Flash-RL-GGUF),
[ProCreations IQ3_XXS 131.7 GiB](https://huggingface.co/ProCreations/MiMo-V2.6-Flash-RL-IQ3_XXS-GGUF).

## Flags

From the YanissAmz card: `-fa on -c 32768 -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 --jinja`
(`-ub 2048` matters for MoE prefill). Recommended sampling (model card): temperature 1.0, top_p 0.95.

## Measured here

[2026-10-02, RTX 2080 Ti + Threadripper 3960X](../bench/results/2026-10-02-rtx2080ti-tr3960x/).
