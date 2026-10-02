# RX 9070 XT 16 GB + Threadripper 3960X, 128 GB DDR4-3200 (quad channel), Windows 11

Machine details: [MACHINE.md](MACHINE.md). Same CPU, RAM and `bench.py` as
[the RTX 2080 Ti run](../2026-10-02-rtx2080ti-tr3960x/); the GPU, OS and engines differ. Thinking off, medians of 3 runs.

| Engine | Model / quant | Decode tok/s | Prefill 4K tok/s | Prefill 32K tok/s |
|---|---|---:|---:|---:|
| Strata 0.1.35, ready-made Windows AMD engine (HIP, gfx1201) | Qwen3.8-Flash-Next IQ3_S, 64K context | **45.2** | **368** | **600** |
| llama.cpp b11344 (ec7630a), Windows Vulkan release | MiMo-V2.6-Flash-RL IQ2_M (100.4 GiB) | 8.5 | 175 | 196 |
| llama.cpp v0.5.0, HIP (built here, see below) | same | 9.1 | 129 | 138 |
| llama.cpp v0.5.0 + [#27044](https://github.com/ggml-org/llama.cpp/pull/27044), HIP | same | 9.2 | 132 | 138 |
| neurall fork 9a77361 + fixes, HIP, defaults (cache auto: 21 slots/layer, 8.1 GiB, hit ~37%) | same | 9.1 | **206** | **241** |

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

## MiMo: Vulkan vs HIP vs the expert-cache fork

- **HIP stock vs Vulkan:** decode 9.1 vs 8.5 tok/s, prompt reads 30% slower on HIP (138 vs 196 tok/s at 28K).
- **No crash on HIP at `-ub 2048`:** stock v0.5.0 compiles the same MMQ code as the CUDA build that crashes on the
  RTX 2080 Ti ([#27792](https://github.com/ggml-org/llama.cpp/issues/27792)), but ran 3 of 3 prefills here. The
  out-of-bounds read only crashes when it leaves mapped memory, so this does not show the bug is absent on HIP.
  #27044 changes nothing measurable here (9.2 / 132 / 138).
- **The expert-cache fork helps here, the opposite of the RTX 2080 Ti:** on this 16 GB card (about 15% of the
  model in VRAM) its automatic placement (experts in RAM + 21 cache slots per layer, 8.1 GiB, ~37% hits) matches
  stock decode (9.1) and reads prompts 75% faster than HIP stock (241 vs 138 tok/s at 28K) and 23% faster than
  Vulkan. On the 2080 Ti (11 GB, ~10% in VRAM, 6 slots/layer, ~21% hits) the same fork was 2.3x slower than stock
  at decode and about equal at prefill.
- None of the MiMo builds reaches the RTX 2080 Ti's 15.5 tok/s decode with CUDA on the same CPU and RAM.

## Command lines

Run by `F:\benchmarking\bench-windows.ps1` (steps `strata-setup`, `strata-bench`, `llama-vulkan`, `mimo-bench`).

- **Strata:** `START-HERE.bat --yes --family qwen --model IQ3_S --no-start --data-dir F:\benchmarking\Strata-data`,
  then `.venv\Scripts\python.exe serve\server.py --engine strata --config strata-iq3_s.json --port 8080`.
  Request extra: `bench/extra/strata-no-thinking.json`.
- **llama.cpp HIP builds:** HIP SDK 7.2 (clang 21) with VS 2026 Build Tools (MSVC 14.51), `-G Ninja -DGPU_TARGETS=gfx1201
  -DGGML_HIP=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++`. This SDK's clang lacks LLVM #201563, so with
  MSVC >= 14.40 every HIP file fails (`'isgreater' cannot overload`, [#22570](https://github.com/ggml-org/llama.cpp/issues/22570));
  the builds used a copy of clang's resource folder with that one-line header fix, via
  `-DCMAKE_CXX_FLAGS=-resource-dir=<copy>` (llama.cpp compiles `.cu` as CXX with `-x hip` on Windows, so
  `CMAKE_HIP_FLAGS` is ignored). The installed SDK was not changed. Build time 2-5 min per tree.
- **llama.cpp:** `llama-server.exe -m MiMo-V2.6-Flash-RL-IQ2_M-00001-of-00003.gguf --fit on -fa on -c 32768 -b 2048
  -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 --jinja --parallel 1 --threads 24`.
  Request extra: `bench/extra/llamacpp-no-thinking.json`.
