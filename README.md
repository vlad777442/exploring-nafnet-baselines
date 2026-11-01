# Exploring NAFNet Baselines for Image Denoising

## Overview
This project evaluates various NAFBlock (Neural Architecture Factorization Block) variants for image denoising, analyzing how different normalization layers, attention mechanisms, and architectural modifications affect performance on corrupted images.

## Features
- **Multiple NAFBlock Variants**: Comparing different architectural designs
- **Comprehensive Evaluation**: Using PSNR, SSIM, and LPIPS metrics
- **Visualizations**: Comparing noisy inputs with denoised outputs
- **Extensible Design**: Framework for testing new variants

## Architecture
The base architecture includes:
- Input convolutional layer
- Multiple NAFBlock modules
- Output convolutional layer

NAFBlocks typically contain:
- Normalization
- Pointwise and depthwise convolutions
- Activation/gating mechanisms
- Attention modules
- Residual connections

## Dataset
CIFAR-10 with artificial corruptions:
- Gaussian blur with random radius
- Random brightness adjustments

## Block Variants
- **Baseline**: Standard NAFBlock with LayerNorm, SimpleGate, and SCA attention
- **Variant A1**: GELU activation instead of SimpleGate
- **Variant A2**: ECA attention mechanism
- **Variant A3**: GroupNorm instead of LayerNorm
- **Variant A4**: No attention mechanism
- **Variant A5**: InstanceNorm with CBAM attention

### Variant A5 implementation
A5 retains the baseline convolution, SimpleGate, and residual path. It replaces
LayerNorm with `InstanceNorm2d(affine=True, track_running_stats=False)` and SCA
with CBAM after the output pointwise convolution. CBAM uses shared channel MLP
weights for global average/max pooling (reduction ratio 16, at least one hidden
channel), followed by spatial attention over channel average/max maps with a
7×7 convolution. Both attention stages use sigmoid gates.

These are explicit choices for this newly added implementation; the earlier
README did not specify these details. The committed results cover only the
baseline and A1–A4; A5 has not yet been benchmarked.

Run A5 with the same dependencies used by the existing training runner:

```bash
python train_eval_runner.py --variant a5 --epochs 10 --output ./results/a5
```

Run the A5 shape, gradient, attention, residual, and runner integration checks
with PyTorch installed (no dataset or pretrained weights are downloaded):

```bash
python -m unittest discover -s tests -v
```

## Evaluation Metrics
- **PSNR**: Higher is better
- **SSIM**: Higher is better
- **LPIPS**: Lower is better

Results are stored in CSV format for variant comparison.

## License
MIT License
