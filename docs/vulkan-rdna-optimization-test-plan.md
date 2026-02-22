# Vulkan RDNA iGPU Optimization - Test & Comparison Plan

## Overview

This document describes the testing methodology for evaluating the Vulkan
backend optimizations targeting AMD RDNA iGPUs (Ryzen AI). The changes span
four commits:

1. **Baseline optimizations**: RDNA iGPU tile tuning, UMA-aware matmul config
2. **Subgroup reductions (Phase 1)**: Replace shared-memory tree reductions with
   `subgroupAdd`/`subgroupMax` in 14 shaders (norm, soft_max, argmax, etc.)
3. **RDNA3 pipeline config**: Wave64 for reduction shaders, subgroup_reduce.glsl
   helper library
4. **Subgroup reductions (Phase 2)**: `count_equal` atomic reduction,
   `mul_mat_vec` cross-subgroup reduction, extended RDNA2/RDNA3 pipeline configs

## Target Hardware

| Device | Architecture | CUs | Subgroup Size | UMA | Notes |
|--------|-------------|-----|---------------|-----|-------|
| Ryzen AI 9 HX 370 (Radeon 890M) | RDNA 3.5 | 16 | 32/64 | Yes | Primary target |
| Ryzen 7 8845HS (Radeon 780M) | RDNA 3 | 12 | 32/64 | Yes | Primary target |
| Ryzen 7 7840U (Radeon 780M) | RDNA 3 | 12 | 32/64 | Yes | Primary target |
| Ryzen 5 7640U (Radeon 760M) | RDNA 3 | 8 | 32/64 | Yes | Budget iGPU |
| Ryzen 7 6800U (Radeon 680M) | RDNA 2 | 12 | 32/64 | Yes | Previous gen |
| Discrete: RX 7900 XTX | RDNA 3 | 96 | 32/64 | No | Regression check |
| Discrete: RX 6800 XT | RDNA 2 | 72 | 32/64 | No | Regression check |

## Prerequisites

### Software
- Vulkan SDK >= 1.3 (with `VK_KHR_shader_subgroup_arithmetic` support)
- RADV (Mesa) driver >= 24.0 or AMDVLK >= 2024.Q1
- Python 3.8+ with `GitPython` and `tabulate` (`pip install GitPython tabulate`)

### Models (recommended test set)

| Model | Size | Quant | Use Case |
|-------|------|-------|----------|
| Qwen2.5-0.5B | ~0.4 GB | Q4_0 | Fast smoke test |
| Llama-3.2-1B | ~0.7 GB | Q4_0 | Small model baseline |
| Llama-3.2-3B | ~1.8 GB | Q4_K_M | Medium model, fits iGPU VRAM |
| Phi-3-mini-4k (3.8B) | ~2.2 GB | Q4_K_M | Common iGPU workload |
| Llama-3.1-8B | ~4.6 GB | Q4_K_M | Large model stress test |
| Mistral-7B-v0.3 | ~4.1 GB | Q4_K_M | Alternative architecture |

## Test Procedure

### Step 1: Build Baseline and Optimized Versions

```bash
# Baseline: build from parent of first optimization commit
git checkout <parent-of-first-commit>
cmake -B build-baseline -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-baseline --config Release -j$(nproc)

# Optimized: build from branch head
git checkout claude/vulkan-ryzen-ai-optimization-Sl23N
cmake -B build-opt -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-opt --config Release -j$(nproc)
```

### Step 2: Operator-Level Benchmarks (test-backend-ops)

This validates individual kernel performance and correctness.

```bash
# Baseline
./build-baseline/bin/test-backend-ops -b Vulkan0 perf \
    -o SOFT_MAX,RMS_NORM,NORM,GROUP_NORM,L2_NORM,MUL_MAT,SUM_ROWS,ARGMAX,COUNT_EQUAL \
    2>&1 | tee baseline-ops.txt

# Optimized
./build-opt/bin/test-backend-ops -b Vulkan0 perf \
    -o SOFT_MAX,RMS_NORM,NORM,GROUP_NORM,L2_NORM,MUL_MAT,SUM_ROWS,ARGMAX,COUNT_EQUAL \
    2>&1 | tee optimized-ops.txt
```

**Database mode** (for automated comparison):
```bash
./build-baseline/bin/test-backend-ops -b Vulkan0 perf \
    -o SOFT_MAX,RMS_NORM,NORM,GROUP_NORM,L2_NORM,MUL_MAT,SUM_ROWS,ARGMAX,COUNT_EQUAL \
    --output baseline-ops.sqlite

./build-opt/bin/test-backend-ops -b Vulkan0 perf \
    -o SOFT_MAX,RMS_NORM,NORM,GROUP_NORM,L2_NORM,MUL_MAT,SUM_ROWS,ARGMAX,COUNT_EQUAL \
    --output optimized-ops.sqlite

python3 scripts/compare-llama-bench.py \
    -t test-backend-ops \
    baseline-ops.sqlite optimized-ops.sqlite
```

**Key metrics**: `time_us`, `flops`, `bandwidth_gb_s`

**Expected improvements for optimized ops**:

| Operation | Mechanism | Expected Speedup |
|-----------|-----------|-----------------|
| SOFT_MAX | Subgroup reduction + wave64 | 10-30% |
| RMS_NORM | Subgroup reduction (2 barriers vs 10) | 15-40% |
| NORM | Subgroup vec2 reduction | 15-40% |
| GROUP_NORM | Subgroup reduction | 10-25% |
| L2_NORM | Subgroup reduction | 15-35% |
| SUM_ROWS | Subgroup reduction + wave64 | 10-25% |
| ARGMAX | Subgroup max + ballot + wave64 | 20-40% |
| COUNT_EQUAL | Subgroup atomic reduction | 5-15% |
| MUL_MAT (vec) | Cross-subgroup subgroupAdd | 5-15% |

### Step 3: End-to-End Inference Benchmarks (llama-bench)

```bash
# Prompt processing (prefill) - tests MUL_MAT heavy path
./build-baseline/bin/llama-bench \
    -m <model.gguf> -ngl 99 \
    -p 512 -n 0 -r 5 \
    --output baseline-pp.sqlite

./build-opt/bin/llama-bench \
    -m <model.gguf> -ngl 99 \
    -p 512 -n 0 -r 5 \
    --output optimized-pp.sqlite

# Token generation (decode) - tests MUL_MAT_VEC + SOFT_MAX + RMS_NORM path
./build-baseline/bin/llama-bench \
    -m <model.gguf> -ngl 99 \
    -p 0 -n 128 -r 5 \
    --output baseline-tg.sqlite

./build-opt/bin/llama-bench \
    -m <model.gguf> -ngl 99 \
    -p 0 -n 128 -r 5 \
    --output optimized-tg.sqlite

# Compare results
python3 scripts/compare-llama-bench.py \
    baseline-pp.sqlite baseline-tg.sqlite \
    optimized-pp.sqlite optimized-tg.sqlite
```

**Sweep across models and batch sizes**:
```bash
for MODEL in qwen2.5-0.5b-q4_0.gguf llama-3.2-1b-q4_0.gguf llama-3.2-3b-q4_k_m.gguf phi-3-mini-q4_k_m.gguf; do
    for NP in 128 512 2048; do
        for NG in 32 128; do
            echo "=== $MODEL pp=$NP tg=$NG ==="
            ./build-opt/bin/llama-bench -m $MODEL -ngl 99 -p $NP -n $NG -r 3
        done
    done
done
```

### Step 4: Correctness Validation

Ensure no numerical regressions:

```bash
# Full backend ops correctness tests
./build-opt/bin/test-backend-ops -b Vulkan0 test 2>&1 | tee correctness.txt

# Check for failures
grep -c "FAIL" correctness.txt  # Should be 0

# Perplexity comparison (optional, thorough)
./build-baseline/bin/llama-perplexity -m <model.gguf> -ngl 99 -f wikitext-2-raw/wiki.test.raw \
    --chunks 32 2>&1 | tee baseline-ppl.txt
./build-opt/bin/llama-perplexity -m <model.gguf> -ngl 99 -f wikitext-2-raw/wiki.test.raw \
    --chunks 32 2>&1 | tee optimized-ppl.txt
# Perplexity values should be identical or within floating-point tolerance (<0.01 difference)
```

### Step 5: Regression Testing on Discrete GPUs

Run on a discrete RDNA2/RDNA3 GPU to ensure no regressions:

```bash
# Discrete GPU test
./build-opt/bin/test-backend-ops -b Vulkan0 test
./build-opt/bin/llama-bench -m llama-3.2-3b-q4_k_m.gguf -ngl 99 -p 512 -n 128 -r 5
```

Compare against baseline to confirm no performance degradation on discrete.

### Step 6: Multi-Driver Validation

If available, test with multiple Vulkan drivers:

```bash
# RADV (Mesa open-source)
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
    ./build-opt/bin/test-backend-ops -b Vulkan0 test

# AMDVLK (AMD proprietary)
VK_ICD_FILENAMES=/etc/vulkan/icd.d/amd_icd64.json \
    ./build-opt/bin/test-backend-ops -b Vulkan0 test
```

## Metrics Collection Template

### Per-Operation Results Table

| Op | Params | Baseline (us) | Optimized (us) | Speedup | Notes |
|----|--------|---------------|----------------|---------|-------|
| RMS_NORM | [4096] | | | | |
| RMS_NORM | [8192] | | | | |
| SOFT_MAX | [4096,1,32] | | | | |
| SOFT_MAX | [4096,1,128] | | | | |
| NORM | [4096] | | | | |
| GROUP_NORM | [4096,32] | | | | |
| L2_NORM | [4096] | | | | |
| SUM_ROWS | [4096] | | | | |
| ARGMAX | [32000] | | | | |
| COUNT_EQUAL | [32000] | | | | |
| MUL_MAT | [4096,4096,1] | | | | |
| MUL_MAT | [4096,11008,1] | | | | |

### End-to-End Results Table

| Model | Quant | pp512 base (t/s) | pp512 opt (t/s) | tg128 base (t/s) | tg128 opt (t/s) | pp Speedup | tg Speedup |
|-------|-------|-------------------|------------------|-------------------|------------------|-----------|-----------|
| Qwen2.5-0.5B | Q4_0 | | | | | | |
| Llama-3.2-1B | Q4_0 | | | | | | |
| Llama-3.2-3B | Q4_K_M | | | | | | |
| Phi-3-mini | Q4_K_M | | | | | | |
| Llama-3.1-8B | Q4_K_M | | | | | | |

## What to Report

When reporting results, include:

1. **Hardware**: CPU model, iGPU model, total system RAM, Vulkan driver version
   ```bash
   vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverVersion|apiVersion"
   ```
2. **Software**: Mesa/AMDVLK version, kernel version, llama.cpp commit hashes
3. **Operator-level**: Table with per-op speedups (test-backend-ops perf)
4. **End-to-end**: Table with tokens/sec for prompt processing and generation
5. **Correctness**: Pass/fail status of test-backend-ops, perplexity delta

## Optimization Summary

### Changes Made

| File | Change | Rationale |
|------|--------|-----------|
| `ggml-vulkan.cpp` | RDNA iGPU tile tuning (m_warptile) | Smaller tiles for L2 residency on 4-12 CU iGPUs |
| `ggml-vulkan.cpp` | RDNA2/3 pipeline wave64 configs | Wider subgroups for reduction-heavy kernels |
| `subgroup_reduce.glsl` | New shared helper library | Reusable 2-phase subgroup reduction macros |
| `soft_max*.comp` | Subgroup reduction | 2 barriers vs log2(N)+1 |
| `norm.comp` | Subgroup vec2 reduction | 2 barriers vs 10 |
| `rms_norm.comp` | Subgroup reduction | 2 barriers vs 10 |
| `rms_norm_back.comp` | Subgroup reduction | 2 barriers vs 10 |
| `group_norm.comp` | Subgroup reduction | 2 barriers vs log2(N)+1 |
| `l2_norm.comp` | Subgroup reduction | 2 barriers vs 10 |
| `sum_rows.comp` | Subgroup reduction | 2 barriers vs log2(N)+1 |
| `argmax.comp` | Subgroup max+ballot | 2 barriers vs log2(N)+1 |
| `count_experts.comp` | Subgroup reduction | 2 barriers vs log2(N)+1 |
| `count_equal.comp` | Subgroup atomic reduction | N atomics → N/subgroup_size |
| `mul_mat_vec_nc.comp` | Subgroup + vec4 loads | Vectorized memory + single subgroupAdd |
| `mul_mat_vec_base.glsl` | Cross-subgroup subgroupAdd | Replace serial loop with hardware reduction |

### Why Subgroup Ops Help on RDNA

1. **Hardware-accelerated**: `subgroupAdd`/`subgroupMax` map directly to RDNA's
   DPP (Data Parallel Primitives) and cross-lane instructions
2. **Zero extra barriers**: Intra-subgroup phase needs no synchronization
3. **Fewer shared memory accesses**: Only subgroup leaders write partial results
4. **Wave64 amplification**: On RDNA, wave64 mode doubles the lanes covered per
   subgroup intrinsic, halving the cross-subgroup partial count
5. **Better occupancy on iGPU**: Fewer barriers means less warp stall time,
   which matters when CU count is low (4-12 on iGPU vs 60-96 on discrete)
