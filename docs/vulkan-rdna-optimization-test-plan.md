# Vulkan Optimization Test Plan — AMD Ryzen AI MAX+ 395 (Windows)

## Target Hardware

| Spec | Value |
|------|-------|
| CPU | AMD Ryzen AI MAX+ 395 |
| iGPU | Radeon 8060S (RDNA 3.5, gfx1151) |
| Compute Units | 40 CU (20 WGP) |
| Memory | LPDDR5X-8000, 256-bit, up to 128 GB |
| GPU VRAM (configurable) | Up to 96 GB (shared UMA) |
| Memory Bandwidth | ~256 GB/s theoretical, ~212 GB/s measured |
| Infinity Cache | 32 MB |
| L2 Cache | 2 MB |
| Subgroup Size | 64 (RADV) / 32 (AMDVLK / AMD official driver) |
| Vulkan Device ID | `0x1586` |
| Driver String (AMDVLK) | `AMD Radeon(TM) 8060S Graphics` |

## Prerequisites

### Software

- Windows 10/11 (latest updates)
- AMD Adrenalin driver >= 24.12 (for gfx1151 Vulkan support)
- Vulkan SDK >= 1.3 (install from https://vulkan.lunarg.com/)
- CMake >= 3.21
- Visual Studio 2022 (with C++ desktop workload) or MinGW-w64
- Git for Windows

### Verify GPU Detection

Open **cmd** or **PowerShell**:

```cmd
vulkaninfo --summary 2>nul | findstr "deviceName driverVersion apiVersion"
```

Expected output (approximately):
```
deviceName    = AMD Radeon(TM) 8060S Graphics
driverVersion = 24.x.x (or newer)
apiVersion    = 1.3.x (or newer)
```

### Models

With up to 96 GB GPU VRAM, this system can run large models fully offloaded.
Choose models appropriate for your VRAM allocation.

| Model | Size | Quant | Purpose |
|-------|------|-------|---------|
| Qwen2.5-0.5B | ~0.4 GB | Q4_0 | Quick smoke test |
| Llama-3.2-3B | ~1.8 GB | Q4_K_M | Medium baseline |
| Llama-3.1-8B | ~4.6 GB | Q4_K_M | Typical workload |
| Qwen2.5-32B | ~18 GB | Q4_K_M | Large model (if VRAM allows) |
| Llama-3.1-70B | ~40 GB | Q4_K_M | Stress test (needs ~48 GB VRAM) |

## Build

Use **Developer Command Prompt for VS 2022** or regular **cmd** with CMake on PATH:

```cmd
REM Baseline: build from upstream main (parent of optimization commits)
git checkout 3571565
cmake -B build-baseline -DGGML_VULKAN=ON
cmake --build build-baseline --config Release -j %NUMBER_OF_PROCESSORS%

REM Optimized: build from our branch
git checkout claude/vulkan-ryzen-ai-optimization-Sl23N
cmake -B build-opt -DGGML_VULKAN=ON
cmake --build build-opt --config Release -j %NUMBER_OF_PROCESSORS%
```

> **Note:** On Windows with MSVC, binaries are placed in `build-opt\bin\Release\`.
> If using MinGW or Ninja, they may be in `build-opt\bin\` directly.

## Test 1: Correctness

Run the full backend ops test suite to ensure no regressions:

```cmd
build-opt\bin\Release\test-backend-ops.exe -b Vulkan0 test > correctness.txt 2>&1
findstr /C:"FAIL" correctness.txt
```

`findstr` should return no matches (errorlevel 1 means no FAIL found — that is correct).

## Test 2: Operator-Level Performance (test-backend-ops)

Compare per-kernel performance between baseline and optimized builds.

```cmd
REM Baseline
build-baseline\bin\Release\test-backend-ops.exe -b Vulkan0 perf ^
    -o SOFT_MAX,RMS_NORM,NORM,GROUP_NORM,L2_NORM,MUL_MAT,SUM_ROWS,ARGMAX,COUNT_EQUAL ^
    --output baseline-ops.sqlite

REM Optimized
build-opt\bin\Release\test-backend-ops.exe -b Vulkan0 perf ^
    -o SOFT_MAX,RMS_NORM,NORM,GROUP_NORM,L2_NORM,MUL_MAT,SUM_ROWS,ARGMAX,COUNT_EQUAL ^
    --output optimized-ops.sqlite

REM Compare
python scripts\compare-llama-bench.py ^
    -t test-backend-ops ^
    baseline-ops.sqlite optimized-ops.sqlite
```

### Expected Improvements

| Operation | Change | Expected Speedup |
|-----------|--------|-----------------|
| SOFT_MAX | Subgroup reduction + wave64 | 10-30% |
| RMS_NORM | Subgroup reduction (2 barriers vs 10) | 15-40% |
| NORM | Subgroup vec2 reduction | 15-40% |
| GROUP_NORM | Subgroup reduction | 10-25% |
| L2_NORM | Subgroup reduction | 15-35% |
| SUM_ROWS | Subgroup reduction + wave64 | 10-25% |
| ARGMAX | Subgroup max + ballot + wave64 | 20-40% |
| COUNT_EQUAL | Subgroup atomic reduction | 5-15% |
| MUL_MAT (vec) | Cross-subgroup subgroupAdd | 5-15% |
| MUL_MAT (large) | 128x128 tile tuning for 40 CU iGPU | 5-20% |

## Test 3: End-to-End Inference (llama-bench)

### Prompt Processing (prefill)

```cmd
build-baseline\bin\Release\llama-bench.exe ^
    -m <model.gguf> -ngl 99 ^
    -p 512 -n 0 -r 5 ^
    --output baseline-pp.sqlite

build-opt\bin\Release\llama-bench.exe ^
    -m <model.gguf> -ngl 99 ^
    -p 512 -n 0 -r 5 ^
    --output optimized-pp.sqlite
```

### Token Generation (decode)

```cmd
build-baseline\bin\Release\llama-bench.exe ^
    -m <model.gguf> -ngl 99 ^
    -p 0 -n 128 -r 5 ^
    --output baseline-tg.sqlite

build-opt\bin\Release\llama-bench.exe ^
    -m <model.gguf> -ngl 99 ^
    -p 0 -n 128 -r 5 ^
    --output optimized-tg.sqlite
```

### Compare

```cmd
python scripts\compare-llama-bench.py ^
    baseline-pp.sqlite baseline-tg.sqlite ^
    optimized-pp.sqlite optimized-tg.sqlite
```

### Model Sweep

```cmd
@echo off
for %%M in (qwen2.5-0.5b-q4_0.gguf llama-3.2-3b-q4_k_m.gguf llama-3.1-8b-q4_k_m.gguf) do (
    for %%P in (128 512 2048) do (
        for %%G in (32 128) do (
            echo === %%M pp=%%P tg=%%G ===
            build-opt\bin\Release\llama-bench.exe -m %%M -ngl 99 -p %%P -n %%G -r 3
        )
    )
)
```

## Test 4: Perplexity (optional)

Verify numerical accuracy has not degraded:

```cmd
build-baseline\bin\Release\llama-perplexity.exe -m <model.gguf> -ngl 99 ^
    -f wikitext-2-raw\wiki.test.raw --chunks 32 > baseline-ppl.txt 2>&1

build-opt\bin\Release\llama-perplexity.exe -m <model.gguf> -ngl 99 ^
    -f wikitext-2-raw\wiki.test.raw --chunks 32 > optimized-ppl.txt 2>&1
```

Perplexity difference should be < 0.01.

## Results Template

### Operator-Level

| Op | Params | Baseline (us) | Optimized (us) | Speedup |
|----|--------|---------------|----------------|---------|
| RMS_NORM | [4096] | | | |
| RMS_NORM | [8192] | | | |
| SOFT_MAX | [4096,1,32] | | | |
| SOFT_MAX | [4096,1,128] | | | |
| NORM | [4096] | | | |
| GROUP_NORM | [4096,32] | | | |
| L2_NORM | [4096] | | | |
| SUM_ROWS | [4096] | | | |
| ARGMAX | [32000] | | | |
| COUNT_EQUAL | [32000] | | | |
| MUL_MAT | [4096,4096,1] | | | |
| MUL_MAT | [4096,11008,1] | | | |

### End-to-End

| Model | Quant | pp512 base (t/s) | pp512 opt (t/s) | tg128 base (t/s) | tg128 opt (t/s) | pp Speedup | tg Speedup |
|-------|-------|-------------------|------------------|-------------------|------------------|-----------|-----------|
| Qwen2.5-0.5B | Q4_0 | | | | | | |
| Llama-3.2-3B | Q4_K_M | | | | | | |
| Llama-3.1-8B | Q4_K_M | | | | | | |

## Environment Info to Report

```cmd
REM GPU info
vulkaninfo --summary 2>nul | findstr "deviceName driverVersion apiVersion"

REM System info
systeminfo | findstr /B /C:"OS Name" /C:"OS Version"

REM VRAM allocation (PowerShell)
powershell -Command "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM"

REM llama.cpp version
git log --oneline -1
```
