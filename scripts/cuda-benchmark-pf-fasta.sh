#!/usr/bin/env bash
set -euo pipefail

readonly source_dir="${VRNA_SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
readonly image="viennarna-cuda-dev:13.1"
readonly gpu="${VRNA_GPU_DEVICE:-0}"
readonly mode="${1:-cpu-bpp}"
readonly fasta="${2:?usage: cuda-benchmark-pf-fasta.sh MODE FASTA [COUNT] [ITERATIONS] [EXACT_LENGTH]}"
readonly count="${3:-256}"
readonly iterations="${4:-3}"
readonly exact_length="${5:-900}"
readonly cpu_threads="${VRNA_CPU_THREADS:-32}"
readonly precision="${VRNA_CUDA_PF_PRECISION:-$([[ "${mode}" == "cuda-bpp" ]] && printf fp64 || printf ffloat)}"

exec docker run --rm \
  --device="nvidia.com/gpu=${gpu}" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e OMP_NUM_THREADS="${cpu_threads}" \
  -e VRNA_PF_BACKEND=cuda \
  -e VRNA_CUDA_LIBRARY=/src/install-cuda/lib/libRNA_cuda.so \
  -e VRNA_CUDA_PF_REFERENCE_DAG="${VRNA_CUDA_PF_REFERENCE_DAG:-0}" \
  -e VRNA_CUDA_PF_ENGINE="${VRNA_CUDA_PF_ENGINE:-blocked}" \
  -e VRNA_CUDA_PF_GEMM="${VRNA_CUDA_PF_GEMM:-auto}" \
  -e VRNA_CUDA_PF_GEMM_CROSSOVER="${VRNA_CUDA_PF_GEMM_CROSSOVER:-}" \
  -e VRNA_CUDA_PF_PRECISION="${precision}" \
  -e VRNA_CUDA_PF_ACCURACY_FALLBACK="${VRNA_CUDA_PF_ACCURACY_FALLBACK:-0}" \
  -e VRNA_CUDA_PF_SCALE_ADJUSTMENT="${VRNA_CUDA_PF_SCALE_ADJUSTMENT:-}" \
  -e VRNA_CUDA_PF_PROFILE="${VRNA_CUDA_PF_PROFILE:-0}" \
  -v "${source_dir}:/src" \
  -v "${fasta}:/benchmark.fasta:ro" \
  -w /src \
  "${image}" \
  bash -lc '
    set -euo pipefail
    gcc -O3 -fopenmp -I/src/install-cuda/include \
      /src/benchmarks/cuda/benchmark_pf_fasta.c \
      /src/install-cuda/lib/libRNA.a \
      -lm -lpthread -lstdc++ -ldl \
      -o /tmp/benchmark_pf_fasta
    /tmp/benchmark_pf_fasta "$@"
  ' -- "${mode}" /benchmark.fasta "${count}" "${iterations}" "${exact_length}"
