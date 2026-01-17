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
make clean && make
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
| `-o <num>` | `--objects <num>` | Set number of small spheres in the scene | 484 |

**Memory Modes:**
- `0` - **Explicit**: `cudaMalloc` + explicit `cudaMemcpy`
- `1` - **UM**: Unified Memory without hints
- `2` - **UM+Prefetch**: Unified Memory with `cudaMemPrefetchAsync`
- `3` - **UM+Advise**: Unified Memory with `cudaMemAdvise`

### Examples

```bash
# Default: Linear traversal, 20 samples, explicit memory mode, 484 small spheres
./cudart

# High quality render with BVH and 100 samples
./cudart -b -s 100

# Custom scene with 1000 small spheres
./cudart -o 1000

# Small scene with 100 spheres for faster testing
./cudart -b -o 100 -s 10

# Test different memory modes
./cudart -m 0  # Explicit mode
./cudart -m 1  # Unified Memory
./cudart -m 2  # UM + Prefetch
./cudart -m 3  # UM + Advise

# BVH with UM+Prefetch mode, 50 samples, and 500 spheres
./cudart --use-bvh --samples 50 --mem-mode 2 --objects 500
```

### Output

The program generates `out.ppm` in PPM (P3) format, which can be viewed with image viewers.

**Scene Composition:**
- 1 large ground sphere (radius 1000)
- Configurable number of small random spheres (radius 0.2, default 484)
  - Small spheres are distributed in a grid pattern with random offsets
  - Materials: 80% diffuse (lambertian), 15% metallic, 5% dielectric (glass)
- 3 large decorative spheres (radius 1.0):
  - Center: Glass sphere at (0, 1, 0)
  - Left: Diffuse sphere at (-4, 1, 0)
  - Right: Metallic sphere at (4, 1, 0)