# UOMI Network GPU Benchmark

A tool for evaluating GPU hardware suitability for AI inference in the UOMI Network.


## Overview

This benchmark evaluates NVIDIA GPUs to determine if they meet the requirements for participating in the UOMI Network's distributed AI inference system. It performs comprehensive testing of:

- Computational performance (TFLOPS)
- Memory capacity and bandwidth
- Multi-GPU scaling efficiency
- Overall suitability for LLM inference

## Quick Start

Run the benchmark with a single command:

```bash
curl -sSL https://raw.githubusercontent.com/Uomi-network/uomi-gpu-benchmark/refs/heads/main/benchmark.sh | bash
```

The benchmark will:

1. Check system requirements
2. Install necessary Python dependencies (in a virtual environment if needed)
3. Run comprehensive GPU performance tests
4. Generate a detailed report with your hardware score and classification

## Requirements

- NVIDIA GPU with CUDA support (minimum 16GB VRAM recommended)
- Ubuntu/Debian Linux (other distros should work but are less tested)
- Python 3.6+ with pip
- Internet connection to download dependencies

For optimal performance in the UOMI Network, we recommend:
- 2 x NVIDIA RTX 4090 or better ( score: 800)
- 48GB+ VRAM

## Scoring System

The benchmark uses a 1000-point scoring system to evaluate hardware:

| Tier | Score Range | Classification | Examples |
|------|-------------|----------------|----------|
| S+ | 2000+ | Data Center | H100 SXM5, Multiple A100s |
| S | 1100-1999 | Professional | H100 PCIe, Multiple RTX 4090s |
| A+ | 900-1099 | High-End | 2x RTX 4090, H100 PCIe |
| A | 750-899 | Premium | A100 80GB, 2x RTX 3090 |
| B+ | 550-749 | Performance | A100 40GB, RTX 4090 |
| B | 450-549 | Mainstream | A6000, RTX 3090 |
| C | 300-449 | Entry | A4000, RTX 3080 |
| D | 0-299 | Minimal | Consumer GPUs <16GB |

## LLM Inference Capabilities

The benchmark also estimates large language model (LLM) capabilities:

| Classification | Parameter Range | Example Models |
|----------------|-----------------|----------------|
| SUPERB | 180B+ | BLOOM 176B, GPT-3 175B |
| EXCELLENT | 100-180B | Llama 2 70B (high throughput) |
| GREAT | 70-100B | Claude Opus, GPT-4, Llama 2 70B |
| VERY GOOD | 40-70B | Falcon 40B, Llama 2 70B (quantized) |
| GOOD | 20-40B | Llama 2 30B, Claude Sonnet |
| FAIR | 10-20B | Llama 2 13B, MPT 30B (quantized) |
| BASIC | <10B | Mistral 7B, Phi-2 |

## Interpreting Results

After running the benchmark, you'll see output similar to:

```
📊 GPU SCORE: 877/1000
🏆 TIER: B+ (Performance)
💾 Total GPU Memory: 47.29 GB
⚡ Combined Performance: 102.01 TFLOPS
🧠 Estimated LLM Parameters: up to 31B

🏁 FINAL RESULT: 877/1000 - TIER B+ (Performance)
✅ HARDWARE ACCEPTED for UOMI network
```

A detailed JSON file `gpu_benchmark_result.json` will also be saved with comprehensive metrics.

## Multi-GPU Support

The benchmark automatically detects and tests all available NVIDIA GPUs. It provides:

- Individual performance metrics for each GPU
- Combined performance estimates with appropriate scaling factors
- Optimized scoring for both homogeneous (same model) and heterogeneous (mixed) setups

## Troubleshooting

### Dependency Issues

If you encounter Python package installation problems:

```bash
# On Ubuntu/Debian
sudo apt install python3-venv python3-pip
```

### Permission Errors

If you get permission errors:

```bash
# Run with sudo privileges
sudo curl -sSL https://raw.githubusercontent.com/uomi-network/gpu-benchmark/main/benchmark.sh | sudo bash
```

### CUDA Not Found

If the benchmark can't detect your GPU:

1. Ensure you have the NVIDIA drivers installed
2. Check that CUDA is installed and in your PATH
3. Verify the GPU is properly seated and powered

## Contributing

Contributions are welcome! Please feel free to submit a pull request.
