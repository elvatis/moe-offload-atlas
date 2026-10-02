# RX 9070 XT 16 GB + Threadripper 3960X, 128 GB DDR4-3200 (quad channel), Windows 11

Machine details: [MACHINE.md](MACHINE.md). Same CPU, RAM and `bench.py` as
[the RTX 2080 Ti run](../2026-10-02-rtx2080ti-tr3960x/); the GPU, OS and engines differ. Thinking off, medians of 3 runs.

| Engine | Model / quant | Decode tok/s | Prefill 4K tok/s | Prefill 32K tok/s |
|---|---|---:|---:|---:|
| Strata 0.1.35, ready-made Windows AMD engine (HIP, gfx1201) | Qwen3.8-Flash-Next IQ3_S, 64K context | **45.2** | **368** | **600** |
| llama.cpp b11344 (ec7630a), Windows Vulkan release | MiMo-V2.6-Flash-RL IQ2_M (100.4 GiB) | 8.5 | 175 | 196 |

## Next to the RTX 2080 Ti (same CPU and RAM)

| | RTX 2080 Ti 11 GB, Linux | RX 9070 XT 16 GB, Windows |
|---|---:|---:|
| Strata Flash-Next IQ3_S: decode / 4K / 32K | 56.0 / 839 / 783 (CUDA, 0.1.34, 256K context) | 45.2 / 368 / 600 (HIP, 0.1.35, 64K context) |
| MiMo IQ2_M on llama.cpp: decode / 4K / 32K | 15.5 / 192 / 208 (CUDA, v0.5.0 + #27044) | 8.5 / 175 / 196 (Vulkan, b11344) |

- The card with more VRAM is slower in both engines here. The runs differ in more than the card (OS, backend, engine
  version, Strata's context setting), so this table does not say which of them causes it.
- Strata's first decode run was 37.2 tok/s, then 45.2 and 46.5 (the expert cache filling). MiMo's first runs were
  also slower (decode 7.5, prefill 4K 135).
- The Vulkan backend does not use the CUDA MMQ code with the `-ub 2048` crash
  ([#27792](https://github.com/ggml-org/llama.cpp/issues/27792)); no crash here.

## Command lines

Run by `F:\benchmarking\bench-windows.ps1` (steps `strata-setup`, `strata-bench`, `llama-vulkan`, `mimo-bench`).

- **Strata:** `START-HERE.bat --yes --family qwen --model IQ3_S --no-start --data-dir F:\benchmarking\Strata-data`,
  then `.venv\Scripts\python.exe serve\server.py --engine strata --config strata-iq3_s.json --port 8080`.
  Request extra: `bench/extra/strata-no-thinking.json`.
- **llama.cpp:** `llama-server.exe -m MiMo-V2.6-Flash-RL-IQ2_M-00001-of-00003.gguf --fit on -fa on -c 32768 -b 2048
  -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 --jinja --parallel 1 --threads 24`.
  Request extra: `bench/extra/llamacpp-no-thinking.json`.
