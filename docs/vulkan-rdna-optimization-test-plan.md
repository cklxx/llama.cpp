# Vulkan Optimization Test Plan — AMD Ryzen AI MAX+ 395

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
| Subgroup Size (RADV) | 64 |
| Vulkan Device ID | `0x1586` |
| Driver String | `Radeon 8060S Graphics (RADV GFX1151)` |

## Prerequisites

### Software

- Linux with Vulkan support (kernel >= 6.8 recommended for Strix Halo)
- Mesa RADV >= 24.3 (for gfx1151 support and `VK_KHR_cooperative_matrix`)
- Vulkan SDK >= 1.3
- CMake >= 3.21
- C++17 compiler (GCC >= 11 or Clang >= 14)

### Verify GPU Detection

```bash
vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverVersion|apiVersion"
```

Expected output (approximately):
```
deviceName    = Radeon 8060S Graphics (RADV GFX1151)
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

```bash
# Baseline: build from upstream main (parent of optimization commits)
git checkout 3571565  # upstream main before our changes
cmake -B build-baseline -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-baseline --config Release -j$(nproc)

# Optimized: build from our branch
git checkout claude/vulkan-ryzen-ai-optimization-Sl23N
cmake -B build-opt -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-opt --config Release -j$(nproc)
```

## Test 1: Correctness

Run the full backend ops test suite to ensure no regressions:

```bash
./build-opt/bin/test-backend-ops -b Vulkan0 test 2>&1 | tee correctness.txt
grep -c "FAIL" correctness.txt  # Must be 0
```

## Test 2: Operator-Level Performance (test-backend-ops)

Compare per-kernel performance between baseline and optimized builds.

```bash
# Baseline
./build-baseline/bin/test-backend-ops -b Vulkan0 perf \
    -o SOFT_MAX,RMS_NORM,NORM,GROUP_NORM,L2_NORM,MUL_MAT,SUM_ROWS,ARGMAX,COUNT_EQUAL \
    --output baseline-ops.sqlite

# Optimized
./build-opt/bin/test-backend-ops -b Vulkan0 perf \
    -o SOFT_MAX,RMS_NORM,NORM,GROUP_NORM,L2_NORM,MUL_MAT,SUM_ROWS,ARGMAX,COUNT_EQUAL \
    --output optimized-ops.sqlite

# Compare
python3 scripts/compare-llama-bench.py \
    -t test-backend-ops \
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

```bash
./build-baseline/bin/llama-bench \
    -m <model.gguf> -ngl 99 \
    -p 512 -n 0 -r 5 \
    --output baseline-pp.sqlite

./build-opt/bin/llama-bench \
    -m <model.gguf> -ngl 99 \
    -p 512 -n 0 -r 5 \
    --output optimized-pp.sqlite
```

### Token Generation (decode)

```bash
./build-baseline/bin/llama-bench \
    -m <model.gguf> -ngl 99 \
    -p 0 -n 128 -r 5 \
    --output baseline-tg.sqlite

./build-opt/bin/llama-bench \
    -m <model.gguf> -ngl 99 \
    -p 0 -n 128 -r 5 \
    --output optimized-tg.sqlite
```

### Compare

```bash
python3 scripts/compare-llama-bench.py \
    baseline-pp.sqlite baseline-tg.sqlite \
    optimized-pp.sqlite optimized-tg.sqlite
```

### Model Sweep

```bash
for MODEL in qwen2.5-0.5b-q4_0.gguf llama-3.2-3b-q4_k_m.gguf llama-3.1-8b-q4_k_m.gguf; do
    for NP in 128 512 2048; do
        for NG in 32 128; do
            echo "=== $MODEL pp=$NP tg=$NG ==="
            ./build-opt/bin/llama-bench -m $MODEL -ngl 99 -p $NP -n $NG -r 3
        done
    done
done
```

## Test 4: Perplexity (optional)

Verify numerical accuracy has not degraded:

```bash
./build-baseline/bin/llama-perplexity -m <model.gguf> -ngl 99 \
    -f wikitext-2-raw/wiki.test.raw --chunks 32 2>&1 | tee baseline-ppl.txt

./build-opt/bin/llama-perplexity -m <model.gguf> -ngl 99 \
    -f wikitext-2-raw/wiki.test.raw --chunks 32 2>&1 | tee optimized-ppl.txt
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

```bash
# GPU info
vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverVersion|apiVersion"

# System info
uname -r
cat /etc/os-release | head -3

# VRAM allocation
cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null

# llama.cpp version
git log --oneline -1
```
