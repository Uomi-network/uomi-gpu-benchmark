#!/bin/bash

set -e
echo "========================================"
echo "🚀 GPU Benchmark for UOMI Network"
echo "========================================"

# Settings
MIN_ACCEPTABLE_SCORE=80
TMP_DIR=$(mktemp -d)
PYTHON_SCRIPT="${TMP_DIR}/benchmark.py"
RESULT_FILE="${TMP_DIR}/result.json"
PYTHON_CMD=""

# Verify requirements
check_requirements() {
  echo "🔍 Checking system requirements..."
  
  # Check if python is installed (try both python3 and python)
  if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
  elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
  else
    echo "❌ Python not found. This benchmark requires Python 3.6+"
    exit 1
  fi
  
  # Verify Python version is at least 3.6
  PY_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
  PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)
  
  if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 6 ]); then
    echo "❌ Python 3.6+ required, found $PY_VERSION"
    exit 1
  fi
  
  # Check pip
  if $PYTHON_CMD -m pip --version &> /dev/null; then
    PIP_CMD="$PYTHON_CMD -m pip"
  else
    echo "❌ pip not found. The benchmark requires pip to install dependencies"
    exit 1
  fi
  
  echo "✅ Basic requirements met (using $PYTHON_CMD)"
}

# Install Python dependencies if needed
install_dependencies() {
  echo "📦 Checking/installing Python dependencies..."
  
  # Required packages
  PACKAGES="torch numpy psutil json5 tqdm"
  
  # Install packages
  for package in $PACKAGES; do
    if ! $PYTHON_CMD -c "import $package" &> /dev/null; then
      echo "🔄 Installing $package..."
      $PIP_CMD install --quiet $package
    fi
  done
  
  echo "✅ Dependencies installed/verified"
}

# Create the Python script
create_benchmark_script() {
  echo "🔧 Preparing GPU benchmark..."

  cat > "${PYTHON_SCRIPT}" << 'EOF'
#!/usr/bin/env python3
"""
GPU Benchmark for UOMI Network
------------------------------------------
This script tests the GPU to determine if it's suitable
for inference in UOMI network.
"""

import os
import time
import json
import platform
import sys
import multiprocessing
import numpy as np
import psutil

# Constants
MIN_ACCEPTABLE_SCORE = 50
BENCHMARK_VERSION = "1.0.0"

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False
    print("⚠️  PyTorch not found. GPU benchmark will not be possible.")
    sys.exit(1)

def get_system_info():
    """Collect basic system information"""
    system_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "cpu_model": platform.processor(),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
    }
    
    # GPU information via PyTorch
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_gb": round(props.total_memory / (1024**3), 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "cuda_cores": props.multi_processor_count * (
                    64 if props.major <= 2 else 
                    128 if props.major <= 6 else 
                    64  # Approximated for newer architectures
                )
            })
    system_info["gpus"] = gpu_info
    
    return system_info

def benchmark_gpu():
    """Run the GPU benchmark, if available"""
    print("🔥 Running GPU benchmark...")
    
    if not torch.cuda.is_available():
        print("❌ No CUDA-compatible GPU found")
        return {
            "gpu_score": 0,
            "raw_metrics": {
                "gpu": {
                    "error": "No CUDA GPU available"
                }
            }
        }
    
    results = {}
    gpu_count = torch.cuda.device_count()
    total_gpu_memory = 0
    best_tflops = 0
    size_results = []

    for device_id in range(gpu_count):
        print(f"\n📊 Testing GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.synchronize(device)

        # Test tensor calculations on GPU with increasing dimensions
        tensor_sizes = [2000, 4000, 8000, 12000, 16000]
        iterations = 10

        # Warm-up
        warmup = torch.rand(1000, 1000, device=device)
        warmup = torch.matmul(warmup, warmup)
        torch.cuda.synchronize(device)

        for tensor_size in tensor_sizes:
            try:
                print(f"  🧮 Testing with tensor size: {tensor_size}x{tensor_size}")
                
                times = []
                
                for i in range(iterations):
                    # Generate tensors
                    a = torch.rand(tensor_size, tensor_size, device=device)
                    b = torch.rand(tensor_size, tensor_size, device=device)
                    
                    # Synchronize for accurate measurement
                    torch.cuda.synchronize(device)
                    t0 = time.time()
                    
                    # Perform matrix multiplication
                    c = torch.matmul(a, b)
                    
                    torch.cuda.synchronize(device)
                    t1 = time.time()
                    
                    times.append(t1 - t0)
                
                avg_time = sum(times) / len(times)
                tflops = 2 * (tensor_size**3) / avg_time / 1e12  # TeraFLOPS
                
                size_results.append({
                    "gpu_index": device_id,
                    "tensor_size": tensor_size,
                    "avg_time": avg_time,
                    "tflops": tflops
                })
                
                # Update best TFLOPS
                if tflops > best_tflops:
                    best_tflops = tflops
                
                # Free memory
                del a, b, c
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                # Likely OOM, stop here
                print(f"  ⚠️  Unable to complete test with size {tensor_size}: {e}")
                break

        # Accumulate total GPU memory
        total_gpu_memory += torch.cuda.get_device_properties(device_id).total_memory / (1024**3)

    # Additional test: inference on a small model (optional)
    # (Keep the existing inference test code here)

    # Calculate GPU score
    if size_results:
        # Score based on performance and memory
        reference_tflops = 15.0  # 15 TFLOPS as reference for max score
        reference_memory = 24.0 * gpu_count  # 24 GB per GPU as reference for max score
        
        perf_score = min(70, (best_tflops / reference_tflops) * 70)
        mem_score = min(30, (total_gpu_memory / reference_memory) * 30)
        
        gpu_score = perf_score + mem_score
    else:
        gpu_score = 0

    # Parameters for LLM inference
    memory_per_param = 2.0  # Bytes per parameter (FP16)
    estimated_max_params = int((total_gpu_memory * 1024**3 * 0.7) / memory_per_param / 1e9)  # In billions

    # Final results
    results["gpu_score"] = gpu_score
    results["raw_metrics"] = {
        "gpu": {
            "best_tflops": best_tflops if size_results else 0,
            "total_gpu_memory_gb": total_gpu_memory,
            "gpu_count": gpu_count,
            "perf_score": perf_score if size_results else 0,
            "mem_score": mem_score if size_results else 0,
            "size_results": size_results,
            "estimated_max_params_billions": estimated_max_params
        }
    }
    
    print(f"📊 GPU SCORE: {gpu_score:.2f}/100")
    print(f"💾 Total GPU Memory: {total_gpu_memory:.2f} GB")
    print(f"⚡ Max Performance: {best_tflops:.2f} TFLOPS") if size_results else None
    print(f"🧠 Estimated LLM Parameters: up to {estimated_max_params}B")
    
    return results

def check_gpu_specs():
    """Check if the GPU meets the minimum specs (comparable to RTX 4090)"""
    if not torch.cuda.is_available():
        return False
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        if props.total_memory < 23 * 1024**3:  # Min 24 GB di memoria
            print(f"❌ GPU {i} does not meet the minimum memory requirement: {props.total_memory / 1024**3:.2f} GB")
            return False
    
    return True
    
def main():
    """Main function"""
    
    print("\n" + "="*50)
    print("GPU BENCHMARK FOR UOMI NETWORK")
    print("="*50 + "\n")
    
    system_info = get_system_info()
    
    # Verify hardware
    if not torch.cuda.is_available():
        print("❌ ERROR: No CUDA GPU detected")
        print("This benchmark requires an NVIDIA GPU with CUDA support")
        return 1
    
    # Show system information
    print("\n📋 SYSTEM INFORMATION:")
    print(f"OS: {system_info['os']} {system_info['os_version']}")
    print(f"CPU: {system_info['cpu_model']} ({system_info['cpu_cores_logical']} logical cores)")
    print(f"RAM: {system_info['memory_total_gb']} GB")
    
    # Show GPU information
    print("\n🖥️  DETECTED GPUS:")
    for i, gpu in enumerate(system_info["gpus"]):
        print(f"  GPU {i}: {gpu['name']}")
        print(f"     Memory: {gpu['memory_total_gb']:.2f} GB")
        print(f"     CUDA Cores (est.): {gpu['cuda_cores']}")
        print(f"     Compute Capability: {gpu['compute_capability']}")
    
    # Run GPU benchmark
    print("\n🚀 STARTING BENCHMARK...")
    if not check_gpu_specs():
        print("❌ This benchmark requires GPUs with at least 24 GB of memory.")
        sys.exit(1)
    results = benchmark_gpu()
    
    # Determine acceptability
    is_acceptable = results["gpu_score"] >= MIN_ACCEPTABLE_SCORE
    total_score = results["gpu_score"]
    
    # Show final result
    print("\n" + "="*50)
    print(f"🏁 FINAL RESULT: {total_score:.2f}/100")
    
    if is_acceptable:
        print("✅ HARDWARE ACCEPTED for UOMI network")
    else:
        print("❌ HARDWARE NOT ACCEPTED for UOMI network")
        print(f"   The minimum required score is {MIN_ACCEPTABLE_SCORE}")
    
    print("="*50)
    
    # Additional LLM data if accepted
    if is_acceptable and "gpu" in results["raw_metrics"]:
        metrics = results["raw_metrics"]["gpu"]
        print("\n🧠 LLM INFERENCE CLASSIFICATION:")
        
        # Classification based on estimated parameters
        max_params = metrics.get("estimated_max_params_billions", 0)
        
        if max_params >= 13:
            print("👍 GOOD GPU")
     
    
    # Save results to file
    results["system_info"] = system_info
    results["is_acceptable"] = is_acceptable
    results["benchmark_timestamp"] = time.time()
    results["benchmark_version"] = BENCHMARK_VERSION
    
    with open("gpu_benchmark_result.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to gpu_benchmark_result.json")
    
    return 0 if is_acceptable else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during benchmark execution: {e}")
        sys.exit(1)
EOF

  echo "✅ Script generated"
}

# Run the benchmark
run_benchmark() {
  echo "🚀 Starting GPU benchmark..."
  $PYTHON_CMD "${PYTHON_SCRIPT}"
  BENCHMARK_EXIT_CODE=$?
  
  if [ -f "gpu_benchmark_result.json" ]; then
    echo "📋 Results saved"
  else
    echo "❌ Error: Results file not found"
  fi
  
  return $BENCHMARK_EXIT_CODE
}

# Cleanup
cleanup() {
  echo "🧹 Cleaning up temporary files..."
  rm -rf "${TMP_DIR}"
  echo "✅ Done"
}

# Main function
main() {
  check_requirements
  install_dependencies
  create_benchmark_script
  run_benchmark
  RESULT=$?
  cleanup
  
  echo ""
  echo "========================================"
  echo "🏁 Benchmark completed!"
  echo "========================================"
  
  exit $RESULT
}

# Execute main
main