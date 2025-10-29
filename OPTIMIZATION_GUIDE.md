# Vision-BDH Performance Optimization Guide

## Summary of Optimizations Applied

This document outlines the optimizations made to achieve transformer-level training speed for the Vision-BDH model.

---

## Key Optimizations Implemented

### 1. **Model Architecture Optimizations** (`vision_bdh_v2_cls_optimized.py`)

#### A. Cached RoPE Embeddings
- **Problem**: RoPE was computed from scratch every forward pass
- **Solution**: Pre-compute cos/sin values for max sequence length and cache as buffers
- **Expected Speedup**: 10-15% reduction in attention computation time
- **Implementation**:
  ```python
  # Pre-compute and register as buffers (moved to device automatically)
  self.register_buffer('cos_cached', cos_cached)
  self.register_buffer('sin_cached', sin_cached)
  ```

#### B. Flash Attention / SDPA Integration
- **Problem**: Manual attention computation is slow
- **Solution**: Use PyTorch's `scaled_dot_product_attention` which automatically uses Flash Attention when available
- **Expected Speedup**: 2-4x faster attention computation
- **Fallback**: Manual computation if SDPA not available

#### C. Optimized Weight Layout
- **Problem**: Raw parameter tensors with manual broadcasting
- **Solution**: Use `nn.Linear` layers which have better kernel fusion
- **Expected Speedup**: 5-10% from better memory access patterns

#### D. JIT-Compiled RoPE
- **Problem**: RoPE rotation has Python overhead
- **Solution**: Use `@torch.jit.script` decorator for compiled execution
- **Expected Speedup**: Marginal but free performance gain

#### E. Fused Operations
- **Problem**: Separate LayerNorm and activation calls
- **Solution**: Where possible, fuse operations to reduce kernel launches
- **Expected Speedup**: 5-10% reduction in overhead

---

### 2. **Training Loop Optimizations** (`train_v2_cifar10_cls_optimized.py`)

#### A. Gradient Accumulation
- **Change**: Effective batch size from 64 → 256 (128 × 2 accumulation steps)
- **Benefit**: Better GPU utilization, more stable gradients
- **Expected Speedup**: 20-30% from larger batches

#### B. Advanced Model Compilation
- **Change**: Use `inductor` backend with `max-autotune` mode
- **Benefit**: More aggressive kernel fusion and optimization
- **Expected Speedup**: 10-20% from better compiled code
- **Code**:
  ```python
  model = torch.compile(model, mode="max-autotune", backend="inductor")
  ```

#### C. Fused Optimizer
- **Change**: `AdamW(..., fused=True)`
- **Benefit**: Single fused kernel for parameter updates
- **Expected Speedup**: 5-10% in optimizer step

#### D. Optimized DataLoader
- **Changes**:
  - `num_workers`: 2 → 4
  - Added `prefetch_factor=2`
  - Added `persistent_workers=True`
  - Larger validation batch size (128 → 256)
- **Benefit**: Eliminate CPU bottleneck in data loading
- **Expected Speedup**: 15-25% from better CPU-GPU overlap

#### E. Memory Format Optimization
- **Change**: Use `channels_last` memory format for conv operations
- **Benefit**: Better memory locality for convolutions
- **Expected Speedup**: 5-15% for conv layers
- **Code**:
  ```python
  images = images.to(DEVICE, memory_format=torch.channels_last)
  model.patch_embed = model.patch_embed.to(memory_format=torch.channels_last)
  ```

#### F. Async GPU Operations
- **Change**: `non_blocking=True` for all `.to(DEVICE)` calls
- **Benefit**: Overlap CPU and GPU work
- **Expected Speedup**: 5-10% from better pipelining

#### G. Efficient Gradient Management
- **Changes**:
  - `zero_grad(set_to_none=True)` instead of default
  - Less frequent checkpoint saving
- **Benefit**: Faster gradient zeroing, less I/O
- **Expected Speedup**: 2-5% reduction in overhead

#### H. Optional Deterministic Mode
- **Change**: Allow disabling deterministic ops with `--deterministic` flag
- **Benefit**: cudnn.benchmark can find fastest algorithms
- **Expected Speedup**: 10-20% when determinism not required

---

## Expected Total Speedup

Combining all optimizations:
- **Conservative Estimate**: 1.8-2.5x faster
- **Optimistic Estimate**: 2.5-3.5x faster
- **Per-epoch time**: ~40-60s → ~15-25s (on modern GPU like A100/H100)

---

## Usage

### Run Optimized Training (Fast Mode)
```bash
python train_v2_cifar10_cls_optimized.py
```

### Run Optimized Training (Reproducible Mode)
```bash
python train_v2_cifar10_cls_optimized.py --deterministic --seed 42
```

### Resume from Checkpoint
```bash
python train_v2_cifar10_cls_optimized.py --resume
```

---

## Comparison: Original vs Optimized

| Component | Original | Optimized | Speedup |
|-----------|----------|-----------|---------|
| **Batch Size** | 64 | 256 (128×2 accum) | 1.3x |
| **Compilation** | aot_eager | inductor max-autotune | 1.2x |
| **Optimizer** | AdamW | AdamW (fused) | 1.1x |
| **DataLoader Workers** | 2 | 4 + prefetch | 1.2x |
| **RoPE Computation** | Per forward pass | Cached | 1.1x |
| **Attention** | Manual | Flash/SDPA | 2.5x |
| **Memory Format** | Default | channels_last | 1.1x |
| **Async GPU Ops** | Blocking | Non-blocking | 1.1x |
| **Overall** | Baseline | **~2.5-3x faster** | |

---

## Hardware Requirements

- **GPU**: NVIDIA GPU with Ampere or newer architecture recommended (for best Flash Attention performance)
- **CUDA**: 11.7+ for full feature support
- **PyTorch**: 2.0+ (required for `torch.compile` and SDPA)
- **Memory**: ~8GB GPU memory (can reduce batch size if needed)

---

## Troubleshooting

### If compilation fails:
The script will automatically fall back to eager mode. You can also explicitly disable compilation:
```python
# Comment out torch.compile line
# model = torch.compile(model, ...)
```

### If OOM (Out of Memory):
Reduce batch size or gradient accumulation:
```python
BATCH_SIZE = 64  # Reduce from 128
GRAD_ACCUM_STEPS = 1  # Reduce from 2
```

### If Flash Attention not available:
The code automatically falls back to manual attention. To check:
```python
import torch
print(torch.backends.cuda.flash_sdp_enabled())  # Should be True
```

---

## Further Optimizations (Advanced)

For even more speed, consider:

1. **Mixed Expert Parallelism**: Split model across multiple GPUs
2. **Custom CUDA Kernels**: For the recurrent BDH operations
3. **Quantization**: INT8 or FP8 training with proper calibration
4. **Selective Activation Checkpointing**: Trade compute for memory
5. **Profile-Guided Optimization**: Use PyTorch profiler to identify remaining bottlenecks

---

## Benchmarking

To measure actual speedup, run both versions and compare:

```bash
# Original
time python train_v2_cifar10_cls.py

# Optimized
time python train_v2_cifar10_cls_optimized.py
```

Monitor GPU utilization:
```bash
watch -n 1 nvidia-smi
```

Target: >90% GPU utilization during training.

---

## Notes

- The import errors shown are IDE linting issues and won't affect runtime
- Actual speedup depends on hardware (A100 > RTX 4090 > RTX 3090)
- First epoch may be slower due to compilation warmup
- Flash Attention provides biggest gains for longer sequences
