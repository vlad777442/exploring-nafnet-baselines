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

## Evaluation Metrics
- **PSNR**: Higher is better
- **SSIM**: Higher is better
- **LPIPS**: Lower is better

Results are stored in CSV format for variant comparison.

## License
MIT License