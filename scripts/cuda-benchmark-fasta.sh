#!/usr/bin/env bash
set -euo pipefail

readonly source_dir="${VRNA_SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
readonly image="viennarna-cuda-dev:13.1"
readonly gpu="${VRNA_GPU_DEVICE:-0}"
readonly mode="${1:-cuda}"
readonly fasta="${2:?usage: cuda-benchmark-fasta.sh MODE FASTA [COUNT] [ITERATIONS]}"
readonly count="${3:-256}"
readonly iterations="${4:-3}"
readonly exact_length="${5:-900}"
readonly cpu_threads="${VRNA_CPU_THREADS:-$(nproc)}"

exec docker run --rm \
  --device="nvidia.com/gpu=${gpu}" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e OMP_NUM_THREADS="${cpu_threads}" \
  -e VRNA_MFE_BACKEND=cuda \
  -e VRNA_CUDA_LIBRARY=/src/install-cuda/lib/libRNA_cuda.so \
  -e VRNA_CUDA_LANES="${VRNA_CUDA_LANES:-}" \
  -e VRNA_CUDA_PAIRED_LANES="${VRNA_CUDA_PAIRED_LANES:-}" \
  -e VRNA_CUDA_PROFILE="${VRNA_CUDA_PROFILE:-}" \
  -e VRNA_CUDA_PROFILE_COUNTERS="${VRNA_CUDA_PROFILE_COUNTERS:-}" \
  -e VRNA_CUDA_BATCH_CHUNK="${VRNA_CUDA_BATCH_CHUNK:-}" \
  -e VRNA_CUDA_SPARSE_M2="${VRNA_CUDA_SPARSE_M2:-}" \
  -e VRNA_CUDA_CANDIDATE_CAPACITY="${VRNA_CUDA_CANDIDATE_CAPACITY:-}" \
  -e VRNA_CUDA_VALIDATE_SPARSE_M2="${VRNA_CUDA_VALIDATE_SPARSE_M2:-}" \
  -e VRNA_CUDA_M2_RING="${VRNA_CUDA_M2_RING:-}" \
  -e VRNA_CUDA_TRACEBACK="${VRNA_CUDA_TRACEBACK:-}" \
  -e VRNA_CUDA_ASYNC_ALLOC="${VRNA_CUDA_ASYNC_ALLOC:-}" \
  -e VRNA_CUDA_PAIR_BITS="${VRNA_CUDA_PAIR_BITS:-}" \
  -e VRNA_CUDA_PRECOMPUTE_HAIRPIN="${VRNA_CUDA_PRECOMPUTE_HAIRPIN:-}" \
  -e VRNA_CUDA_DERIVE_PAIR_TYPES="${VRNA_CUDA_DERIVE_PAIR_TYPES:-}" \
  -e VRNA_CUDA_SKIP_DP_INIT="${VRNA_CUDA_SKIP_DP_INIT:-}" \
  -e VRNA_CUDA_PACKED_DP="${VRNA_CUDA_PACKED_DP:-}" \
  -e VRNA_CUDA_PRECOMPUTE_OUTER_CONTEXT="${VRNA_CUDA_PRECOMPUTE_OUTER_CONTEXT:-}" \
  -e VRNA_CUDA_CANDIDATE_LOWER_BOUND="${VRNA_CUDA_CANDIDATE_LOWER_BOUND:-}" \
  -e VRNA_CUDA_CONTEXTUAL_LOWER_BOUND="${VRNA_CUDA_CONTEXTUAL_LOWER_BOUND:-}" \
  -v "${source_dir}:/src" \
  -v "${fasta}:/benchmark.fasta:ro" \
  -w /src \
  "${image}" \
  bash -lc '
    set -euo pipefail
    gcc -O3 -fopenmp -I/src/install-cuda/include \
      /src/benchmarks/cuda/benchmark_mfe_fasta.c \
      /src/install-cuda/lib/libRNA.a \
      -lm -lpthread -lstdc++ -ldl \
      -o /tmp/benchmark_mfe_fasta
    /tmp/benchmark_mfe_fasta "$@"
  ' -- "${mode}" /benchmark.fasta "${count}" "${iterations}" "${exact_length}"
