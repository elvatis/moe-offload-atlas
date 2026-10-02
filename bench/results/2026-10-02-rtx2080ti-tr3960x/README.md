# RTX 2080 Ti 11 GB + Threadripper 3960X, 128 GB DDR4-3200 (quad channel)

Machine details: [MACHINE.md](MACHINE.md). Thinking off in every run. Medians of 3 runs (5 for `default` and `pin`),
`bench.py` defaults. Decode = a 256-token answer to a short coding prompt.

| Engine | Model / quant | Decode tok/s | Prefill 4K tok/s | Prefill 32K tok/s |
|---|---|---:|---:|---:|
| Strata 0.1.34 (main @ 1678de3) | Qwen3.8-Flash-Next IQ3_S | **56.0** | **839** | **783** |
| llama.cpp v0.5.0 + [#27044](https://github.com/ggml-org/llama.cpp/pull/27044) | MiMo-V2.6-Flash-RL IQ2_M (100.4 GiB) | **15.5** | 192 | 208 |
| neurall fork 9a77361 + fixes, `default` (cache auto: 6 slots/layer, 2.6 GiB, hit ~21%) | same | 6.6 | - | - |
| same, `--load-mode pin` (hit ~22%) | same | 6.6 | - | - |
| same, `--moe inserts=1` | same | 5.6 | - | - |
| same, `--moe inserts=0` | same | 6.2 | - | - |
| same, `--fit on --moe-expert-cache -1` (1 slot/layer, 0.7 GiB, hit 1-5%) | same | 5.9 | 187 | 205 |

## What this shows

- **Strata on its one model is 3.6x faster to decode and ~4x faster to read than llama.cpp on MiMo** on this machine.
  MiMo moves about 2.5x more bytes per token (15B active at 2.76 bpw vs 6B active) and has no working MTP in llama.cpp.
- **Stock llama.cpp v0.5.0 crashes** on MiMo prefill at `-ub 2048` (CUDA illegal memory access, 3 of 3 runs, always at
  the same position); `-ub 1024` gives 109 tok/s, `-ub 512` 68 tok/s. The one-line fix in
  [#27044](https://github.com/ggml-org/llama.cpp/pull/27044) removes the crash (4 of 4) and gives the 192-208 tok/s
  above. Report: [#27792](https://github.com/ggml-org/llama.cpp/issues/27792#issuecomment-5950525456).
- **The neurall expert-cache fork is 2.3x slower than stock here** (6.6 vs 15.5 tok/s), with its automatic placement
  ("model 100.4 GiB exceeds free VRAM 10.2 GiB: experts in RAM + GPU expert cache"). Only ~10% of this model fits in
  11 GB; the fork's README reports its gains at 33-55% in VRAM. Upload rate is not the cause: 0, 1 and 2 inserts per
  step all give 5.6-6.6 tok/s. Stock `--fit on` keeps ~9 GiB of experts permanently on the GPU instead.
- The fork's probe measured 86 GB/s CPU read bandwidth (saturated at 8 threads) and 12.9 GB/s host-to-GPU.

## Command lines

Common llama-server arguments (MiMo):
`-m MiMo-V2.6-Flash-RL-IQ2_M-00001-of-00003.gguf -fa on -c 32768 -b 2048 -ub 2048 --cache-type-k q8_0
--cache-type-v q8_0 --jinja --parallel 1 --threads 24`, plus `--fit on` for stock; the fork rows add what the table
names. Request extra: `{"chat_template_kwargs":{"enable_thinking":false}}`. Builds: CUDA 12.0, `-DCMAKE_CUDA_ARCHITECTURES=75`.
The fork needed [neurall/llama.cpp#1](https://github.com/neurall/llama.cpp/pull/1) to build on sm_75 and got #27044 for
the same crash.

**Strata** (`strata-iq3_s.json`, set up by Strata's installer): `serve/server.py --engine strata --config strata-iq3_s.json`
with `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <rt> --max-context 262144 --kv int8
--kv-resident 32768 --vision --vram-reserve-mib 700`. Request extra: `{"reasoning_effort":"none"}`.
