# bench-windows.ps1 - phase 1 on the Windows PC (RX 9070 XT): Strata (Qwen3.8-Flash-Next IQ3_S) and MiMo on llama.cpp Vulkan.
# Put it in the benchmark folder (layout: bench/windows/README.md) and run it from PowerShell, one step at a time or all:
#   powershell -ExecutionPolicy Bypass -File bench-windows.ps1 -Step all
#   steps: sysinfo, strata-device, hip-check, llama-hip-build, mimo-hip-bench, strata-setup, strata-bench, llama-vulkan, mimo-bench
# Every step logs to <Root>\logs\ and results go to <Root>\moe-offload-atlas\bench\results\<date>-<Label>\.
param([ValidateSet("all","sysinfo","strata-device","hip-check","llama-hip-build","mimo-hip-bench","strata-setup","strata-bench","llama-vulkan","mimo-bench")][string]$Step = "all",
      [string]$Root = $PSScriptRoot,            # the folder this script is in
      [string]$Label = "windows-amd")           # results folder suffix, e.g. rx9070xt-tr3960x
$ErrorActionPreference = "Continue"   # PS 5.1: "Stop" would abort on every stderr line of python/llama-server
$Strata  = "$Root\Strata"
$Logs    = "$Root\logs"
$Results = "$Root\moe-offload-atlas\bench\results\$(Get-Date -Format yyyy-MM-dd)-$Label"
$Py      = "$Strata\.venv\Scripts\python.exe"     # exists after strata-setup; bench.py only needs the standard library
$Bench   = "$Root\moe-offload-atlas\bench\bench.py"
$LlamaTag = "b11344"
$LlamaDir = "$Root\llama-cpp\bin-vulkan-$LlamaTag"
$MiMo    = "$Root\llama-cpp\models\MiMo-V2.6-Flash-RL-IQ2_M\MiMo-V2.6-Flash-RL-IQ2_M-00001-of-00003.gguf"
New-Item -ItemType Directory -Force -Path $Logs, $Results | Out-Null

function Wait-Health($Url, $Proc, $Log, $Minutes = 20) {
  $deadline = (Get-Date).AddMinutes($Minutes)
  while ((Get-Date) -lt $deadline) {
    if ($Proc.HasExited) { throw "server exited (code $($Proc.ExitCode)), see $Log" }
    try { if ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 $Url).StatusCode -eq 200) { return } } catch {}
    Start-Sleep 5
  }
  throw "server not ready after $Minutes min, see $Log"
}
function Stop-Tree($Proc) {   # the server and everything it started (Strata's engine runs as a child process)
  Get-CimInstance Win32_Process -Filter "ParentProcessId=$($Proc.Id)" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue
}

function Step-Sysinfo {
  $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
  $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
  $mem = Get-CimInstance Win32_PhysicalMemory
  $gpu = Get-CimInstance Win32_VideoController | Where-Object Name -match "Radeon|AMD"
  $os  = Get-CimInstance Win32_OperatingSystem
  @("# Machine", "",
    "- Date: $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mmZ'))",
    "- OS: $($os.Caption) $($os.Version)",
    "- CPU: $($cpu.Name.Trim()), $($cpu.NumberOfLogicalProcessors) threads",
    "- RAM: $ram GiB total, $($mem.Count) modules at $(($mem | Select-Object -First 1).ConfiguredClockSpeed) MT/s",
    "- GPU: $($gpu.Name), driver $($gpu.DriverVersion) (AdapterRAM is capped at 4 GB by Windows; the RX 9070 XT has 16 GB)"
  ) | Set-Content -Encoding utf8 "$Results\MACHINE.md"
  Get-Content "$Results\MACHINE.md"
}

function Step-StrataDevice {   # what docs/AMD_HIP.md asks for in a Windows AMD report (needs the engine from strata-setup)
  $env:PATH = "$Strata\engine\rocm\bin;$env:PATH"
  Push-Location $Strata
  & "$Strata\engine\strata-device.exe" --list-devices 2>&1 | ForEach-Object { "$_" } | Tee-Object "$Logs\strata-device.log"
  & "$Strata\engine\strata-device.exe" --selftest 2>&1 | ForEach-Object { "$_" } | Tee-Object -Append "$Logs\strata-device.log"
  Pop-Location
}

# ---------------------------------------------------------------------------------------------- phase 2: llama.cpp on HIP
# the three source trees from the server (see llama-cpp\src\SOURCES.md), each built into its own build-hip folder
$HipTrees = [ordered]@{
  "v0.5.0"            = "$Root\llama-cpp\src\llama.cpp-v0.5.0"
  "v0.5.0+pr27044"    = "$Root\llama-cpp\src\llama.cpp-v0.5.0+pr27044"
  "neurall+fixes"     = "$Root\llama-cpp\src\llama.cpp-neurall-9a77361+fixes"
}

function Find-HipPath {
  if ($env:HIP_PATH -and (Test-Path "$env:HIP_PATH\bin\clang.exe")) { return $env:HIP_PATH.TrimEnd('\') }
  $c = Get-ChildItem "C:\Program Files\AMD\ROCm" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending |
       Where-Object { Test-Path "$($_.FullName)\bin\clang.exe" } | Select-Object -First 1
  if ($c) { return $c.FullName }
  return $null
}
function Find-VsPath {
  $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
  if (-not (Test-Path $vswhere)) { return $null }
  return (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
}
function Enter-BuildEnv {   # what "x64 Native Tools Command Prompt for VS" does, plus the HIP SDK and Strata's cmake/ninja
  $hip = Find-HipPath; $vs = Find-VsPath
  if (-not $hip) { throw "HIP SDK not found (no HIP_PATH, nothing under C:\Program Files\AMD\ROCm)" }
  if (-not $vs)  { throw "Visual Studio Build Tools with 'Desktop development with C++' not found (vswhere)" }
  cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] } }
  $env:HIP_PATH = $hip
  $env:PATH = "$hip\bin;$Strata\.venv\Scripts;$env:PATH"
  return $hip
}

function Get-PatchedClangResource($hip) {
  # HIP SDK 7.2's clang 21 predates LLVM #201563: with MSVC >= 14.40 (VS 2026 / v145) every HIP file fails with
  # "__device__ function 'isgreater' cannot overload __host__ __device__ function" (ggml-org/llama.cpp#22570,
  # ROCm/llvm-project#2669). The fix is one line in __clang_hip_runtime_wrapper.h. We patch a COPY of clang's resource
  # folder and point only our builds at it (-resource-dir); the installed SDK is not touched. llama.cpp compiles the .cu
  # files on Windows as CXX with -x hip (not CMake's HIP language), so the flag goes into CMAKE_CXX_FLAGS.
  $src = (& "$hip\bin\clang.exe" -print-resource-dir).Trim()
  $dst = "$Root\llama-cpp\clang-resource-hip-patched"
  $hdr = "$dst\include\__clang_hip_runtime_wrapper.h"
  if (-not (Test-Path "$dst\.patched")) {
    if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
    Write-Host "copying clang resource folder $src -> $dst"
    Copy-Item -Recurse -Force $src $dst -ErrorAction Stop
    $text = [IO.File]::ReadAllText($hdr)
    if ($text -match 'math_forward_declares\.h>\r?\n#include <cmath>') { Write-Host "header already fixed in this SDK" }
    else {
      $re = [regex]'#if !defined\(__HIPCC_RTC__\)(\r?\n)#include <cmath>'
      if (-not $re.IsMatch($text)) { throw "unexpected $hdr layout: the #include <cmath> after #if !defined(__HIPCC_RTC__) was not found" }
      $text = $re.Replace($text, '#if !defined(__HIPCC_RTC__)$1// LLVM #201563: forward declarations before <cmath> (MSVC >= 14.40 constexpr cmath)$1#include <__clang_cuda_math_forward_declares.h>$1#include <cmath>', 1)
      [IO.File]::WriteAllText($hdr, $text)
      Write-Host "patched $hdr (LLVM #201563)"
    }
    Set-Content "$dst\.patched" "from $src"
  }
  return ($dst -replace '\\','/')
}

function Step-HipCheck {
  $hip = Find-HipPath; $vs = Find-VsPath
  @("HIP SDK:        $(if ($hip) { $hip } else { 'NOT FOUND' })",
    "Visual Studio:  $(if ($vs) { $vs } else { 'NOT FOUND (needs Build Tools with Desktop development with C++)' })",
    "cmake (Strata): $(if (Test-Path "$Strata\.venv\Scripts\cmake.exe") { 'ok' } else { 'NOT FOUND - run strata-setup first' })",
    "ninja (Strata): $(if (Test-Path "$Strata\.venv\Scripts\ninja.exe") { 'ok' } else { 'NOT FOUND - run strata-setup first' })"
  ) | Tee-Object "$Logs\hip-check.log"
  if ($hip) { & "$hip\bin\clang.exe" --version 2>&1 | ForEach-Object { "$_" } | Tee-Object -Append "$Logs\hip-check.log"
              if (Test-Path "$hip\bin\hipconfig.exe") { & "$hip\bin\hipconfig.exe" --version 2>&1 | ForEach-Object { "hipconfig: $_" } | Tee-Object -Append "$Logs\hip-check.log" } }
}

function Step-LlamaHipBuild {
  $hip = Enter-BuildEnv
  $res = Get-PatchedClangResource $hip
  foreach ($name in $HipTrees.Keys) {
    $src = $HipTrees[$name]; $log = "$Logs\build-hip-$($name -replace '[^\w.+-]','_').log"
    if (Test-Path "$src\build-hip\bin\llama-server.exe") { Write-Host "already built: $name"; continue }
    if (Test-Path "$src\build-hip") { Remove-Item -Recurse -Force "$src\build-hip" }   # a failed earlier attempt: configure fresh
    Write-Host "== building $name (about 10-20 min) -> $log"
    & cmake -S $src -B "$src\build-hip" -G Ninja -DGPU_TARGETS=gfx1201 -DGGML_HIP=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF `
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_FLAGS=-resource-dir=$res" 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 $log
    & cmake --build "$src\build-hip" --target llama-server 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
    if (Test-Path "$src\build-hip\bin\llama-server.exe") { Write-Host "   ok" } else { Write-Host "   FAILED, see $log"; Get-Content $log -Tail 15 }
  }
}

function Step-MimoHipBench {
  $hip = Find-HipPath; $env:PATH = "$hip\bin;$env:PATH"
  foreach ($name in $HipTrees.Keys) {
    $bin = "$($HipTrees[$name])\build-hip\bin"; $tag = $name -replace '[^\w.+-]','_'
    if (-not (Test-Path "$bin\llama-server.exe")) { Write-Host "skip $name (not built)"; continue }
    $log = "$Logs\mimo-hip-$tag-server.log"
    $srvArgs = @("-m", $MiMo, "--alias", "mimo-v2.6-flash-iq2m", "--fit", "on", "-fa", "on", "-c", "32768", "-b", "2048", "-ub", "2048",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--jinja", "--parallel", "1", "--threads", "24",
                 "--host", "127.0.0.1", "--port", "8812")
    Write-Host "== MiMo on HIP: $name"
    $p = Start-Process -FilePath "$bin\llama-server.exe" -ArgumentList $srvArgs -WorkingDirectory $bin `
         -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -WindowStyle Minimized -ErrorAction Stop
    try {
      Wait-Health "http://127.0.0.1:8812/health" $p "$log.err" 30
      & $Py $Bench --url http://127.0.0.1:8812/v1 --label "llamacpp-hip-$tag-mimo26flash-iq2m" `
         --extra "@$Root\moe-offload-atlas\bench\extra\llamacpp-no-thinking.json" --out $Results 2>&1 | ForEach-Object { "$_" } | Tee-Object "$Logs\mimo-hip-$tag-bench.log"
      if ($LASTEXITCODE -ne 0) { Write-Host "   bench FAILED for $name (server crash?), see $log.err - continuing" }
    } catch { Write-Host "   $name failed: $_ - continuing" }
    finally { Stop-Tree $p; Start-Sleep 5 }
  }
}

function Step-StrataSetup {
  # --data-dir explicitly: without it, a data_dir remembered in %APPDATA%\Strata\settings.json from an earlier install
  # wins over Strata-data next to this folder, and setup downloads the models again. With it, only the AMD engine
  # (~600 MB) is downloaded; the copied GGUFs carry their .done marks and are skipped.
  $settings = "$env:APPDATA\Strata\settings.json"
  if (Test-Path $settings) { Write-Host "earlier Strata settings ($settings):"; Get-Content $settings | Write-Host }
  Push-Location $Strata
  cmd /c "START-HERE.bat --yes --family qwen --model IQ3_S --no-start --data-dir $Root\Strata-data" 2>&1 | ForEach-Object { "$_" } | Tee-Object "$Logs\strata-setup.log"
  Pop-Location
  if (-not (Test-Path "$Strata\strata-iq3_s.json")) { throw "setup did not write strata-iq3_s.json, see $Logs\strata-setup.log" }
}

function Step-StrataBench {
  $log = "$Logs\strata-server.log"
  $p = Start-Process -FilePath $Py -ArgumentList "serve\server.py","--engine","strata","--config","strata-iq3_s.json","--port","8080" `
       -WorkingDirectory $Strata -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -WindowStyle Minimized -ErrorAction Stop
  try {
    Wait-Health "http://127.0.0.1:8080/health" $p $log 30
    & $Py $Bench --url http://127.0.0.1:8080/v1 --label strata-0.1.35-qwen38-flashnext-iq3s `
       --extra "@$Root\moe-offload-atlas\bench\extra\strata-no-thinking.json" --out $Results 2>&1 | ForEach-Object { "$_" } | Tee-Object "$Logs\strata-bench.log"
    if ($LASTEXITCODE -ne 0) { throw "bench failed, see $Logs\strata-bench.log" }
  } finally { Stop-Tree $p }
}

function Step-LlamaVulkan {
  if (Test-Path "$LlamaDir\llama-server.exe") { Write-Host "already there: $LlamaDir"; return }
  $zip = "$Root\llama-cpp\llama-$LlamaTag-bin-win-vulkan-x64.zip"
  Invoke-WebRequest -UseBasicParsing "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaTag/llama-$LlamaTag-bin-win-vulkan-x64.zip" -OutFile $zip -ErrorAction Stop
  Expand-Archive $zip -DestinationPath $LlamaDir -Force -ErrorAction Stop
  Remove-Item $zip
  & "$LlamaDir\llama-server.exe" --version 2>&1 | Tee-Object "$Logs\llama-vulkan-version.log"
  & "$LlamaDir\llama-server.exe" --list-devices 2>&1 | Tee-Object -Append "$Logs\llama-vulkan-version.log"
}

function Step-MimoBench {
  $log = "$Logs\mimo-vulkan-server.log"
  $srvArgs = @("-m", $MiMo, "--alias", "mimo-v2.6-flash-iq2m", "--fit", "on", "-fa", "on", "-c", "32768", "-b", "2048", "-ub", "2048",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--jinja", "--parallel", "1", "--threads", "24",
            "--host", "127.0.0.1", "--port", "8812")
  $p = Start-Process -FilePath "$LlamaDir\llama-server.exe" -ArgumentList $srvArgs -WorkingDirectory $LlamaDir `
       -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -WindowStyle Minimized -ErrorAction Stop
  try {
    Wait-Health "http://127.0.0.1:8812/health" $p "$log.err" 30
    & $Py $Bench --url http://127.0.0.1:8812/v1 --label llamacpp-$LlamaTag-vulkan-mimo26flash-iq2m `
       --extra "@$Root\moe-offload-atlas\bench\extra\llamacpp-no-thinking.json" --out $Results 2>&1 | ForEach-Object { "$_" } | Tee-Object "$Logs\mimo-vulkan-bench.log"
    if ($LASTEXITCODE -ne 0) { throw "bench failed, see $Logs\mimo-vulkan-bench.log" }
  } finally { Stop-Tree $p }
}

switch ($Step) {
  "sysinfo"      { Step-Sysinfo }
  "strata-device" { Step-StrataDevice }
  "hip-check"     { Step-HipCheck }
  "llama-hip-build" { Step-LlamaHipBuild }
  "mimo-hip-bench"  { Step-MimoHipBench }
  "strata-setup" { Step-StrataSetup }
  "strata-bench" { Step-StrataBench }
  "llama-vulkan" { Step-LlamaVulkan }
  "mimo-bench"   { Step-MimoBench }
  "all"          { Step-Sysinfo; Step-StrataSetup; Step-StrataBench; Step-LlamaVulkan; Step-MimoBench }
}
Write-Host "Done: $Step. Logs: $Logs  Results: $Results"
