## Introduction

This project is a CUDA-accelerated implementation of "Ray Tracing in One Weekend" with advanced optimizations for GPU memory management and scene traversal. It demonstrates various CUDA memory strategies and acceleration structures for high-performance ray tracing.

### Key Features

- **GPU-accelerated Ray Tracing**: Parallelized rendering using CUDA kernels with per-thread random sampling
- **BVH Acceleration**: Optional Bounding Volume Hierarchy for faster scene intersection tests
- **Multiple Memory Modes**: Four different CUDA memory management strategies to explore performance characteristics:
  - **Explicit Mode (0)**: Traditional `cudaMalloc` with explicit `cudaMemcpy` operations
  - **Unified Memory (1)**: CUDA Unified Memory with automatic page migration
  - **UM + Prefetch (2)**: Unified Memory with explicit prefetch hints for better locality
  - **UM + Advise (3)**: Unified Memory with memory advice hints (preferred location, read-mostly)
- **Configurable Rendering**: Adjustable samples per pixel and scene traversal methods
- **Performance Monitoring**: Detailed timing for scene creation, BVH building, rendering, and image output

Originally based on Roger Allen's CUDA port (May 2018). See the [original repository](https://github.com/rogerallen/raytracinginoneweekend) for more information.

---

## Build Instructions

### Prerequisites
- CUDA Toolkit (tested with compute capability 8.6)
- NVCC compiler
- C++ compiler (g++)

### Compilation

```bash
make
```

This will compile the project with optimizations (`--use_fast_math`) and generate the `cudart` executable.

For debug builds, edit the Makefile to uncomment the debug flags:
```makefile
NVCC_DBG = -g -G  # Debug
```

---

## Usage

### Command Line Options

```bash
./cudart [options]
```

**Available Options:**

| Short | Long | Description | Default |
|-------|------|-------------|---------|
| `-b` | `--use-bvh` | Enable BVH acceleration structure for scene traversal | Linear traversal |
| `-s <num>` | `--samples <num>` | Set samples per pixel for anti-aliasing | 20 |
| `-m <mode>` | `--mem-mode <mode>` | Set memory management mode (0-3) | 0 |

**Memory Modes:**
- `0` - **Explicit**: `cudaMalloc` + explicit `cudaMemcpy`
- `1` - **UM**: Unified Memory without hints
- `2` - **UM+Prefetch**: Unified Memory with `cudaMemPrefetchAsync`
- `3` - **UM+Advise**: Unified Memory with `cudaMemAdvise`

### Examples

```bash
# Default: Linear traversal, 20 samples, explicit memory mode
./cudart

# High quality render with BVH and 100 samples
./cudart -b -s 100

# Test different memory modes
./cudart -m 0  # Explicit mode
./cudart -m 1  # Unified Memory
./cudart -m 2  # UM + Prefetch
./cudart -m 3  # UM + Advise

# BVH with UM+Prefetch mode and 50 samples
./cudart --use-bvh --samples 50 --mem-mode 2
```

### Output

The program generates `out.ppm` in PPM (P3) format, which can be viewed with image viewers.