# Windows (AMD) benchmark script

[bench-windows.ps1](bench-windows.ps1) runs the measurements behind
[2026-10-02-rx9070xt-tr3960x](../results/2026-10-02-rx9070xt-tr3960x/): Strata with its ready-made Windows AMD
engine, and MiMo-V2.6-Flash on llama.cpp (Vulkan release, and HIP builds of three source trees). Logs go to
`logs\`, results to `moe-offload-atlas\bench\results\<date>-<Label>\`.

## Folder layout

Put the script in a folder laid out like this (it uses the folder it is in; `-Root` points elsewhere):

```
<folder>\
  bench-windows.ps1
  moe-offload-atlas\                       a clone of this repo (bench.py, bench\extra\*.json)
  Strata\                                  a clone of github.com/Niko1221/Strata
  Strata-data\                             models, packs, MTP (setup downloads them, or copy an existing set)
  llama-cpp\
    models\MiMo-V2.6-Flash-RL-IQ2_M\       huggingface.co/YanissAmz/MiMo-V2.6-Flash-RL-GGUF (3 shards)
    src\llama.cpp-v0.5.0\                  ggml-org/llama.cpp tag v0.5.0
    src\llama.cpp-v0.5.0+pr27044\          the same + ggml-org/llama.cpp#27044
    src\llama.cpp-neurall-9a77361+fixes\   neurall/llama.cpp release 9a77361 + neurall/llama.cpp#1 + #27044
```

Only the steps you run need their folders: the Vulkan steps need no source trees, the Strata steps no llama.cpp.

## Steps

```powershell
powershell -ExecutionPolicy Bypass -File bench-windows.ps1 -Step <step> [-Label rx9070xt-tr3960x]
```

| Step | What it does |
|---|---|
| `sysinfo` | `MACHINE.md` for the results folder (CPU, RAM, GPU, driver) |
| `strata-setup` | Strata's installer for Qwen3.8-Flash-Next IQ3_S with `--data-dir <folder>\Strata-data` |
| `strata-device` | `strata-device --list-devices` and `--selftest` (what Strata's AMD_HIP.md asks for in a Windows report) |
| `strata-bench` | starts Strata, runs `bench.py`, stops it |
| `llama-vulkan` | downloads the llama.cpp Windows Vulkan release (b11344) |
| `mimo-bench` | MiMo on the Vulkan release |
| `hip-check` | lists the HIP SDK, Visual Studio Build Tools, cmake and ninja it found |
| `llama-hip-build` | builds `llama-server` for gfx1201 from each source tree into `build-hip\` |
| `mimo-hip-bench` | MiMo on each HIP build; a crash is recorded and the next build runs |
| `all` | sysinfo, strata-setup, strata-bench, llama-vulkan, mimo-bench |

`bench.py` needs only the Python standard library; the script uses the Python in Strata's `.venv`, and cmake/ninja
from there for the HIP builds.

## Things that bit us, handled in the script

- **Strata downloaded the model again** although `Strata-data` was complete: `%APPDATA%\Strata\settings.json` from an
  earlier install remembered another `data_dir`, which wins. The script passes `--data-dir` explicitly.
- **HIP SDK 7.2 + Visual Studio 2026 Build Tools (MSVC 14.51)**: every HIP file fails with
  `__device__ function 'isgreater' cannot overload __host__ __device__ function 'isgreater'` (MSVC >= 14.40 made these
  `<cmath>` functions `constexpr`; the SDK's clang 21 lacks the fix, LLVM #201563; see ggml-org/llama.cpp#22570).
  The script copies clang's resource folder into `llama-cpp\clang-resource-hip-patched`, applies that one-line
  header change to the copy and builds with `-DCMAKE_CXX_FLAGS=-resource-dir=<copy>`. The installed SDK is not
  changed. llama.cpp compiles the `.cu` files on Windows as CXX with `-x hip`, so `CMAKE_HIP_FLAGS` would be ignored.
- **Windows PowerShell 5.1** turns every stderr line of a native program into an error record. The script runs with
  `$ErrorActionPreference = "Continue"`, prints stderr as plain text, and checks exit codes itself.
- **JSON on the command line** breaks between PowerShell versions; `bench.py --extra @file.json` reads it from a file.
