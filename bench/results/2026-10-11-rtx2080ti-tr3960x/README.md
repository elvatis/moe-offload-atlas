# RTX 2080 Ti 11 GB + Threadripper 3960X, 128 GB DDR4-3200 (quad channel), 2026-10-11

Machine details: [MACHINE.md](MACHINE.md). llama.cpp b11516 (d4d82d67f), the official CUDA 12.8 Linux release.
Thinking off in every `bench.py` run (`{"chat_template_kwargs":{"enable_thinking":false}}`), medians of 3 runs,
`bench.py` defaults. Decode = a 256-token answer to a short coding prompt.

| Engine | Model / quant | Decode tok/s | Prefill 4K tok/s | Prefill 32K tok/s |
|---|---|---:|---:|---:|
| llama.cpp b11516, `--n-cpu-moe 32` | Qwen3.6-35B-A3B UD-Q4_K_M (22.7 GB) | 57.7 | 1101 | 1037 |
| same + built-in MTP (`--spec-type draft-mtp --spec-draft-n-max 3`) | same | **83.8** | 1065 | 1004 |
| llama.cpp b11516, `-cmoe --moe-cache-mib 6000` | same | 56.9 | **1168** | **1089** |
| llama.cpp b11516, `--n-cpu-moe 44` | Qwen3-Coder-Next Q4_K_M (48.5 GB) | **42.6** | 503 | 562 |
| llama.cpp b11516, `-cmoe --moe-cache-mib 6000` | same | 32.1 | 525 | 590 |
| llama.cpp b11516, `--n-cpu-moe 46 -ub 2048` | GLM-5.3-Flash UD-IQ2_XXS (102 GB) | **12.0** | **161** | **165** |
| same, `-ub 512` | same | 11.7 | 59 | - |
| same, `-ub 512 --moe-cache-mib 2048` | same | 3.5 | - | - |

## What this shows

- **Qwen3.6-35B-A3B writes 84 tok/s with its own MTP layer** on an 11 GB Turing card with the experts of 32 of 40
  layers in RAM: +45 % on this coding prompt, +21 % on the mixed text below.
- **llama.cpp's expert cache (`--moe-cache-mib`, merged as
  [#29887](https://github.com/ggml-org/llama.cpp/pull/29887)) does not help on PCIe 3.0.** On a short single-topic
  answer it ties the static placement for Qwen3.6 (56.9 vs 57.7), with more varied text it is slower (below).
  Prompt reading is 5-6 % faster with `-cmoe` (large batches skip the cache).
- **A cache smaller than one token's working set copies everything.** GLM-5.3-Flash on 11 GB has room for 2 GB of
  cache, 259 slots shared by 39 layers; one token needs 39 × 8 = 312. Every token evicts its own experts: 0.7 % hits,
  2.3 GiB over PCIe per token, 3.5 instead of 12 tok/s.
  [Reported on #29887](https://github.com/ggml-org/llama.cpp/pull/29887#issuecomment-6106433768).
- **GLM-5.3-Flash 320B-A18B runs at 12 tok/s** with every expert in RAM. Prompt reading is bound by PCIe 3.0: each
  batch of 2048 sends the 88.6 GiB of experts to the card once.

## Beyond bench.py

Measured with our own scripts on the same machine and builds (data files in this folder).

### llama.cpp's expert cache on real text

[moe-cache-real-text.jsonl](moe-cache-real-text.jsonl). One llama-server per setting continues four texts in a row
(German prose, code, agent transcripts, English prose), 384 tokens each (GLM 256), temperature 0.7, after a short
warm-up. Hit rates and copied bytes are llama.cpp's own numbers, printed at exit with `-lv 4`. `-fa on -c 16384
-ub 2048 -t 12 -tb 24`.

| Model | Setting | tok/s | Hit rate | Copied per token |
|---|---|---:|---:|---:|
| Qwen3.6-35B-A3B | `--n-cpu-moe 32` | **58.5** | | |
| | `-cmoe` | 52.2 | | |
| | `-cmoe --moe-cache-mib 2000` (11 % of experts) | 31.1 | 57 % | 255 MiB |
| | `-cmoe --moe-cache-mib 4000` (21 %) | 40.7 | 72 % | 164 MiB |
| | `-cmoe --moe-cache-mib 6000` (32 %) | 53.0 | 83 % | 98 MiB |
| | `--n-cpu-moe 32` + MTP | **70.9** | | |
| | `-cmoe --moe-cache-mib 5000` + MTP | 46.1 | 71 % | 164 MiB |
| Qwen3-Coder-Next | `--n-cpu-moe 44` | **42.7** | | |
| | `-cmoe --moe-cache-mib 3000` (7 %) | 24.8 | 62 % | 338 MiB |
| | `-cmoe --moe-cache-mib 6000` (13 %) | 33.0 | 75 % | 220 MiB |
| GLM-5.3-Flash | `--n-cpu-moe 46 -ub 512` | **14.2-15.1** | | |
| | same + `--moe-cache-mib 1024` / `2048` | 3.7 / 3.6 | 0 % / 0.7 % | 2272 / 2387 MiB |

- Every miss is copied: 100-340 MiB per token, at ~10 GB/s that is 10-35 ms of a token's 17-25 ms.
- Code is the hardest topic for the cache: Coder-Next writes 47 tok/s on German and 24 on code at 6 GB.
- MTP misses more (a pass checks 4 tokens, up to 4 × 8 experts per layer).

### Cache policies on a routing trace

[cache-policy-sim-qwen36.json](cache-policy-sim-qwen36.json). The router's top-8 per token and layer of
Qwen3.6-35B-A3B (transformers, BF16) over 8,192 tokens (2,048 each of German, code, agent transcripts, English),
replayed against cache policies. Hits · experts copied per token:

| Policy | 12 % | 25 % | 36 % | 50 % |
|---|---|---|---|---|
| static, profiled on other text | 35 % | 57 % | 71 % | 83 % |
| LRU (every miss comes in) | 67 % · 107 | 83 % · 54 | 91 % · 30 | 96 % · 14 |
| LFU, halved every 512 tokens, admit only if hotter than the coldest | 56 % · 3 | 76 % · 4 | 86 % · 4 | 93 % · 4 |

llama.cpp's LRU measured 8-10 points below the simulated LRU (57/72/83 % at 11/21/32 %). An admission rule copies
8-38× less for a few points of hit rate, which matters on a slow link.

### Expert usage and placement

- **A static profile is worth something, a random set is not.** The share of routings a static set of experts on the
  card would serve (the experts that fit in 11 GB after the rest): Qwen3.6-35B-A3B 36 % random, 74 % profiled,
  83-89 % with the right topic's profile; Qwen3-Coder-Next 17 / 54 / 67-77 %. Code and prose share under 10 % of
  their top experts. Counted with `llama-imatrix` (its per-expert counts).
- **Writing is bound by RAM bandwidth:** with every expert in RAM Qwen3.6 writes 51 tok/s, Coder-Next 37; the expert
  layers on the card add 16-24 %.
- **Threads:** 12 write as fast as 24. **48 (all SMT threads) collapse:** Qwen3.6 from 62.9 to 1.2 tok/s, Coder-Next
  from 42.9 to 2.0 (llama-bench `-t 48`).
- **Prompt reading** needs `-ub 2048` with op offload (the default), which streams experts to the card: 2.8× the
  speed of `-ub 512`.

### Dropping cold experts (Qwen3.6-35B-A3B)

[pruning-kl-qwen36.jsonl](pruning-kl-qwen36.jsonl). Per layer, keep the experts the router picked most (German,
English, code, agent counts weighted equally); the MTP layer by router-row norm. Built BF16 → Q8_0 → German
importance matrix → Q4_K_M, measured against the full Q8_0 (`llama-perplexity --kl-divergence`, 60 × 512 tokens):

| Model | Q4_K_M | KL de / en / code | Same top token |
|---|---|---|---|
| full (256 experts) | 20.7 GB | 0.024 / 0.027 / 0.050 | 93 / 93 / 96 % |
| keep 75 % (192) | 15.9 GB | 0.100 / 0.072 / 0.170 | 89 / 90 / 93 % |
| keep 50 % (128) | 11.1 GB | 0.382 / 0.208 / 0.349 | 80 / 84 / 89 % |

Dropping a quarter costs 3-4× what 4-bit quantization costs; half breaks the model.

### GLM-5.3-Flash: 4 hours as an agent

[soak-glm53flash.json](soak-glm53flash.json). For [#28144](https://github.com/ggml-org/llama.cpp/issues/28144)
(`SOFT_MAX failed` on Turing): the report's flags, 256K context (512K did not fit), agent chat with tool calls.
138 requests, context up to 218K, answers up to 4096 tokens, no crash. Writing falls with the context: 14.9 tok/s
under 50K, 11.0 at 50-100K, 9.9 at 100-150K, 8.0 above. ngram speculation accepted 12-17 % of its drafts.
[Reported on #28144](https://github.com/ggml-org/llama.cpp/issues/28144#issuecomment-6106433932).

## Command lines

Common llama-server arguments: `-fa on -c 40960 -b 2048 --jinja --parallel 1`, plus per model:

- **Qwen3.6-35B-A3B** ([unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF),
  UD-Q4_K_M): `-ub 2048 -t 12 -tb 24 --no-mmproj`, then what the table names.
- **Qwen3-Coder-Next** ([unsloth/Qwen3-Coder-Next-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF),
  Q4_K_M): `-ub 2048 -t 12 -tb 24`, then what the table names.
- **GLM-5.3-Flash** ([unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF), UD-IQ2_XXS):
  `-t 24 -tb 24 --n-cpu-moe 46`, then what the table names. The `-ub 512` row ran prefill 4K only, the cache row
  decode only.

Soak: `-c 262144 -ngl 99 --cpu-moe -fa on -ctk q4_0 -ctv q4_0 --kv-offload --reasoning-preserve --spec-type
ngram-map-k4v --chat-template-kwargs '{"reasoning_effort":"max"}' --temp 1.0 --top-p 0.95 -t 24 --parallel 1`.

GLM runs right after another big model start from a partly cold page cache and write up to 20 % slower until the
experts are back in RAM; the numbers above are warm.
